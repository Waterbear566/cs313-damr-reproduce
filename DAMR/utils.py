import sys
import torch
import faiss
import logging
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import auc
from sklearn.cluster import KMeans


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def knn_score(train_set, test_set, n_neighbours=1):
    """
    Calculates the KNN distance
    """
    # Score anomalies with exact L2 distance to normal features.
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def h(A):
    return torch.trace(torch.matrix_exp(A*A)) - A.shape[0]

def normalize(A):
    D = A.sum(dim=0)
    D_inv = D.pow_(-1)
    D_inv.masked_fill_(D_inv == float('inf'), 0)

    return A * D_inv

def thresholding(A, thre):
    return torch.where(A.abs()>thre, A, torch.scalar_tensor(0.0, dtype=torch.float32, device=A.device))

def binarize(A, thre):
    return torch.where(A.abs()>thre, 1.0, 0.0)
# %%
def get_timestamp(stamps):
    return (stamps - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
# %%

def roc_auc(label_time, pred, negative_sample, sigma):
    negative_sample = np.sort(negative_sample)[::-1]
    thresholds = list(negative_sample[::int(len(negative_sample)/50)])
    thresholds.append(negative_sample[-1])
    tps=[]
    fps=[]

    for thre in thresholds:
        pred_pos = pred[pred>thre]

        tp = 0
        for i in range(len(label_time)):
            start_time = label_time[i] - pd.Timedelta(30, unit='min')
            end_time = label_time[i] + pd.Timedelta(30, unit='min')

            detected_event = pred_pos[str(start_time): str(end_time)]
            if len(detected_event)>0:
                timestamps = get_timestamp(detected_event.index)
                delta_t = np.min(np.abs(timestamps.values - get_timestamp(label_time[i])))
                tp += np.exp(-np.power(delta_t/sigma,2))
        tp = tp/len(label_time)
        tps.append(tp)

        fp = (negative_sample>thre).sum()/len(negative_sample)
        fps.append(fp)
    return auc(fps,tps), (fps,tps)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_features(features):
    # Average projection heads into one window representation.
    features = torch.stack(features, dim=0)
    return features.mean(dim=0)
