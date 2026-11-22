import os
import re
import argparse
import logging
import torch
import random
import warnings
import numpy as np
from datetime import datetime
from train import trainer, model_evaluate
from utils import _logger, str2bool
from data_loader import get_loader_segment
from model import DAMR, AdaptiveTemperature

def main(args):

    warnings.filterwarnings("ignore")
    cuda = torch.cuda.is_available()
    device = torch.device(args.device)
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    experiment_log_dir = os.path.join('log', args.dataset + f"_seed_{SEED}")

    os.makedirs(experiment_log_dir, exist_ok=True)
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 50)
    logger.debug("=" * 50)
    logger.debug(args)
    logger.debug("Learning rate: {}".format(args.lr))
    logger.debug("Computation device: {}".format(device))
    logger.debug("Loading dataset")

    train_loader, val_loader, n_sensor = get_loader_segment(args.data_path,
                                                  batch_size=args.batch_size,
                                                  win_size=args.win_size,
                                                  step=args.step,
                                                  mode='train',
                                                  dataset=args.dataset)

    test_loader = get_loader_segment(args.data_path,
                                      batch_size=args.batch_size,
                                      win_size=args.win_size,
                                      step=args.step,
                                      mode='test',
                                      dataset=args.dataset)

    args.seq_len = args.win_size

    model = DAMR(args.seq_len, args.hidden_dim, args.dropout_rate,
                  args.depth, n_sensor, args.out_channels, args.num_heads, args.MSASB, args.adaptive_filter, args.input_channels, args.conv_hidden)
    adaptive_temp = AdaptiveTemperature(args.emb_dim, args.num_heads)
    model = model.to(device)
    adaptive_temp = adaptive_temp.to(device)
    path = os.path.join(experiment_log_dir, "saved_models", "model.pt")

    # if os.path.exists(path):
    #     logger.debug(f"Found existing model checkpoint at {path}, loading weights...")
    #     chkpoint = torch.load(path, map_location=device)
    #     model.load_state_dict(chkpoint["model_state_dict"])
    #     logger.debug("Model weights loaded successfully.")
    #     auc, _ = model_evaluate(model, args, None, train_loader, test_loader, device)
    #     logger.debug(f'AUROC is : {auc:2.4f}')
    # else:
    #     trainer(model, adaptive_temp, train_loader, val_loader, test_loader, n_sensor, device, args, logger, experiment_log_dir)

    trainer(model, adaptive_temp, train_loader, val_loader, test_loader, n_sensor, device, args, logger, experiment_log_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # files
    parser.add_argument('--dataset', type=str, default='SWaT')
    parser.add_argument('--data_path', type=str, default='./data/swat')
    parser.add_argument('--device', default='cuda:0', type=str, help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=16, help='Random seed to use.')

    # made parameters
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--conv_hidden', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=16)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=3)

    # training params
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--win_size', type=int, default=60)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--MSASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    args = parser.parse_known_args()[0]
    main(args)
