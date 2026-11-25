"""Dữ liệu dùng chung cho baseline: cùng file, cùng chuẩn hóa, cùng cửa sổ, cùng nhãn và cùng cách chia
validation như DAMR.

Cách dùng trong một baseline:

    from common.data import load_split
    sp = load_split("PSM", "DAMR/data/psm", seed=16)
    sp.train, sp.val, sp.test          # torch Dataset, mỗi phần tử (x, label), x có dạng (sensor, thời gian)
    sp.test_labels                     # nhãn cửa sổ test, dạng (n_test,)
    X = sp.to_numpy("train", layout="TC")   # mảng (n, thời gian, sensor) nếu baseline cần

Chuẩn hóa và cắt cửa sổ do chính các lớp *SegLoader trong DAMR/data_loader.py làm, nên giống DAMR từng bit.
Validation: một đoạn liên tục 20% số cửa sổ train, vị trí bắt đầu = random.Random(seed).randrange(...),
đúng bằng vị trí DAMR chọn khi chạy main.py với --seed=seed (main.py gọi random.seed(seed) rồi chia ngay).
"""
import os
import random
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAMR_DIR = os.path.join(ROOT, "DAMR")

WIN_SIZE = 60
STEP = 10
VAL_RATIO = 0.2
SEEDS = (16, 17, 18)        # seed chung của nhóm; DAMR chạy thêm 19–25
DATASETS = ("PSM", "MSL", "SMD", "SWaT", "WADI")


def damr_val_split(n_windows, seed, val_ratio=VAL_RATIO):
    """Chỉ số cửa sổ train/val giống get_loader_segment trong DAMR/data_loader.py."""
    train_use_len = int(n_windows * (1 - val_ratio))
    val_use_len = int(n_windows * val_ratio)
    start = random.Random(seed).randrange(train_use_len)
    idx = np.arange(n_windows)
    val_idx = idx[start:start + val_use_len]
    train_idx = np.concatenate([idx[:start], idx[start + val_use_len:]])
    return train_idx, val_idx


def _loader_class(dataset):
    if DAMR_DIR not in sys.path:
        sys.path.insert(0, DAMR_DIR)
    import data_loader as dl
    classes = {"PSM": dl.PSMSegLoader, "MSL": dl.MSLSegLoader, "SMD": dl.SMDSegLoader,
               "SWaT": dl.SWaTSegLoader, "WADI": dl.WADISegLoader}
    if dataset not in classes:
        raise ValueError(f"Dataset lạ: {dataset}. Chọn một trong {sorted(classes)}")
    return classes[dataset]


@dataclass
class Split:
    dataset: str
    seed: int
    n_sensor: int
    train: object        # torch.utils.data.Subset
    val: object          # torch.utils.data.Subset
    test: object         # torch.utils.data.Dataset
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_labels: np.ndarray

    def to_numpy(self, part, layout="CT"):
        """Gom cả một phần thành mảng float32. layout 'CT' = (n, sensor, thời gian) như DAMR,
        'TC' = (n, thời gian, sensor) như phần lớn baseline Transformer. SMD tốn khoảng 0.6 GB."""
        ds = getattr(self, part)
        x = np.stack([ds[i][0] for i in range(len(ds))]).astype(np.float32)
        return x.transpose(0, 2, 1) if layout == "TC" else x


def load_split(dataset, data_path, seed, win_size=WIN_SIZE, step=STEP, val_ratio=VAL_RATIO):
    from torch.utils.data import Subset

    cls = _loader_class(dataset)
    train_full = cls(data_path, win_size, step, mode="train")
    test = cls(data_path, win_size, step, mode="test")
    train_idx, val_idx = damr_val_split(len(train_full), seed, val_ratio)
    # Nhãn cửa sổ = 1 nếu có ít nhất một điểm bất thường (cùng quy tắc với __getitem__ ở chế độ test).
    point = np.asarray(test.test_labels).reshape(-1)
    test_labels = np.array([point[i * step:i * step + win_size].sum() > 0 for i in range(len(test))],
                           dtype=np.int8)
    return Split(dataset=dataset, seed=seed, n_sensor=train_full.channels,
                 train=Subset(train_full, train_idx.tolist()), val=Subset(train_full, val_idx.tolist()),
                 test=test, train_idx=train_idx, val_idx=val_idx, test_labels=test_labels)
