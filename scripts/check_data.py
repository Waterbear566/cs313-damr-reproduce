"""Kiểm tra dữ liệu đúng định dạng mà DAMR/data_loader.py đọc, và in thống kê kiểu Bảng 1 (E1).

Cách dùng:  python scripts/check_data.py --root DAMR/data
Cấu trúc mong đợi (giống TSLib / Anomaly-Transformer):
  psm/train.csv, psm/test.csv, psm/test_label.csv
  msl/MSL_train.npy, msl/MSL_test.npy, msl/MSL_test_label.npy
  smd/SMD_train.npy, smd/SMD_test.npy, smd/SMD_test_label.npy
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

EXPECTED_DIMS = {"PSM": 25, "MSL": 55, "SMD": 38}  # số chiều chuẩn trong các bài báo MTSAD


def load(name, folder):
    if name == "PSM":
        tr = pd.read_csv(os.path.join(folder, "train.csv")).values[:, 1:]
        te = pd.read_csv(os.path.join(folder, "test.csv")).values[:, 1:]
        lb = pd.read_csv(os.path.join(folder, "test_label.csv")).values[:, 1:]
    else:
        tr = np.load(os.path.join(folder, f"{name}_train.npy"))
        te = np.load(os.path.join(folder, f"{name}_test.npy"))
        lb = np.load(os.path.join(folder, f"{name}_test_label.npy"))
    return tr, te, np.asarray(lb).reshape(-1)


def window_stats(labels, win=60, step=10):
    n = (len(labels) - win) // step + 1
    w = np.array([labels[i * step:i * step + win].sum() > 0 for i in range(n)])
    return n, w.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="DAMR/data")
    ap.add_argument("--datasets", nargs="+", default=["PSM", "MSL", "SMD"])
    args = ap.parse_args()
    ok = True
    print(f"{'dataset':8}{'train':>10}{'test':>10}{'dim':>6}{'% bất thường (điểm)':>22}{'số cửa sổ test':>16}{'% cửa sổ bất thường':>22}")
    for name in args.datasets:
        folder = os.path.join(args.root, name.lower())
        try:
            tr, te, lb = load(name, folder)
        except FileNotFoundError as e:
            print(f"{name:8} THIẾU FILE: {e.filename}")
            ok = False
            continue
        problems = []
        if tr.shape[1] != te.shape[1]:
            problems.append(f"train {tr.shape[1]} chiều nhưng test {te.shape[1]} chiều")
        if tr.shape[1] != EXPECTED_DIMS[name]:
            problems.append(f"mong đợi {EXPECTED_DIMS[name]} chiều")
        if len(lb) != len(te):
            problems.append(f"nhãn {len(lb)} dòng nhưng test {len(te)} dòng")
        if not set(np.unique(lb)) <= {0, 1}:
            problems.append(f"nhãn không phải 0/1: {np.unique(lb)[:5]}")
        n_nan = int(pd.isna(tr).sum() + pd.isna(te).sum())
        if n_nan:
            problems.append(f"{n_nan} giá trị NaN (PSM: code gốc thay bằng 0; MSL/SMD: code gốc không xử lý)")
        nw, wr = window_stats(lb)
        print(f"{name:8}{len(tr):>10}{len(te):>10}{tr.shape[1]:>6}{100 * lb.mean():>21.2f}%{nw:>16}{100 * wr:>21.2f}%")
        for p in problems:
            print(f"         ! {p}")
        ok = ok and not [p for p in problems if "NaN" not in p]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
