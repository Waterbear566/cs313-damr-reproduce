import os
import torch
import numpy as np
import time
import utils
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn.utils import clip_grad_value_
from torch.nn.init import xavier_uniform_
from augmentation import DataTransform
from utils import get_features
from thop import profile, clever_format

def trainer(model, adaptive_temp, train_loader, val_loader, test_loader, n_sensor, device, args, logger, experiment_log_dir):

    init = torch.zeros([n_sensor, n_sensor])
    init = xavier_uniform_(init).abs()
    init = init.fill_diagonal_(0.0)
    A = torch.tensor(init, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([
        {'params':model.parameters(), 'weight_decay':args.weight_decay},
        {'params': [A]}], lr=args.lr, weight_decay=0.0)

    best_auc = 0
    logger.debug("Training started ...")
    start_time = time.time()

    for epoch in range(0, args.epochs):

        epoch_start_time = time.time()
        train_loss = model_train(model, adaptive_temp, A, optimizer, train_loader, device, args.num_heads)
        logger.debug('Epoch: {}/{}, Loss: {}'.format(epoch + 1, args.epochs, train_loss))
        # val_auc, _ = model_evaluate(model, A, train_loader, val_loader, device)
        # logger.debug('Epoch: {}/{}, valid AUROC is: {}'.format(epoch + 1, args.epochs, val_auc))
        test_auc, _ = model_evaluate(model, args, A, train_loader, test_loader, device)
        logger.debug('Epoch: {}/{}, test AUROC is: {}'.format(epoch + 1, args.epochs, test_auc))

        if best_auc < test_auc:
            best_auc = test_auc

            # Save the model with the best test AUROC.
            os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
            chkpoint = {'model_state_dict': model.state_dict()
            }
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'model.pt'))

        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time
        logger.debug('Epoch: {}/{}, Time: {:.3f}'
                    .format(epoch + 1, args.epochs, epoch_train_time))

        logger.debug('======================================================')

    train_time = time.time() - start_time

    logger.debug('Training time: {}'.format(train_time))
    logger.debug('Finished training.')
    logger.debug(f'best Auc : {best_auc:2.4f}')

def model_train(model, adaptive_temp, A, optimizer, train_loader, device, num_heads):

    total_loss, total_num = 0.0, 0
    model.train()
    adaptive_temp.train()

    for (data, labels) in train_loader:
    # for (data, labels) in tqdm(train_loader, desc='Train...'):

        aug1, aug2 = DataTransform(data)
        # print(data.shape)
        aug1, aug2 = torch.tensor(aug1).float().to(device), torch.tensor(aug2).float().to(device)

        optimizer.zero_grad()
        out_1 = model(aug1, A)
        out_2 = model(aug2, A)

        scaled_similarities, temps = adaptive_temp(out_1, out_2)

        batch_loss = 0
        # Sum temperature-scaled contrastive loss across heads.
        for head_idx in range(num_heads):
            scaled_similarities[head_idx] = torch.exp(scaled_similarities[head_idx])
            scaled_similarities[head_idx] = torch.cat([scaled_similarities[head_idx], scaled_similarities[head_idx]], dim=0)
            loss = (-torch.log(scaled_similarities[head_idx] /
                              (sim_matrix(out_1[head_idx], out_2[head_idx], temps[head_idx]).sum(dim=-1)))).mean()
            batch_loss += loss

        batch_loss.backward()
        optimizer.step()

        clip_grad_value_(model.parameters(), 1)
        A.data.copy_(torch.clamp(A.data, min=0, max=1))

        total_num += data.size(0)
        total_loss += batch_loss.item() * data.size(0)

    return total_loss / (total_num)


def model_evaluate(model, args, A, train_loader, test_loader, device):

    # Build a normal feature bank for KNN scoring.
    train_feature_space = []
    model.eval()
    with torch.no_grad():
        # for (data, _) in tqdm(train_loader, desc='Train set feature extracting'):
        for (data, _) in train_loader:
            data = data.to(device)
            features = model(data, A)
            features = get_features(features)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    # Subsample the feature bank to reduce KNN cost.
    sampling_ratio = 0.1
    num_samples = len(train_feature_space)
    num_to_keep = max(1, int(num_samples * sampling_ratio))

    # Randomly select samples
    indices = np.random.choice(num_samples, num_to_keep, replace=False)
    train_feature_space = train_feature_space[indices] # Retain selected features

    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        # for (data, labels) in tqdm(test_loader, desc='Test set feature extracting'):
        for (data, labels) in test_loader:
            data = data.to(device)
            features = model(data, A)
            features = get_features(features)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    # Use nearest-normal distance as the anomaly score.
    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space


def sim_matrix(out_1, out_2, temp):

    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)

    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    return sim_matrix
