"""Phần dùng chung cho baselines/DeepSVDD và baselines/ALOCC: nạp dữ liệu thành tensor, chia batch, chống ghi trùng CSV."""
import csv
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from common.data import load_split  # noqa: E402

DATA_DIRS = {"PSM": "DAMR/data/psm", "MSL": "DAMR/data/msl", "SMD": "DAMR/data/smd"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tensors(dataset, data_path, seed):
    """Dữ liệu chung (common.data.load_split) gom thành tensor CPU (n, sensor, 60), đúng shape đầu vào của
    code MTGFlow sau bước transpose(2,3)+reshape."""
    sp = load_split(dataset, data_path or os.path.join(ROOT, DATA_DIRS[dataset]), seed=seed)
    tensors = {part: torch.from_numpy(sp.to_numpy(part, layout="CT")) for part in ("train", "val", "test")}
    return sp, tensors


def batches(x, batch_size, shuffle=False, drop_last=False):
    """Chia tensor CPU thành batch; shuffle dùng bộ sinh ngẫu nhiên toàn cục của torch (đã đặt seed)."""
    n = len(x)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    stop = n - n % batch_size if drop_last else n
    for i in range(0, stop, batch_size):
        yield x[order[i:i + batch_size]]


def already_recorded(csv_path, model, variant, dataset, seed):
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, newline="", encoding="utf-8") as f:
        return any(r["model"] == model and r["variant"] == variant and r["dataset"] == dataset
                   and r["seed"] == str(seed) for r in csv.DictReader(f))
