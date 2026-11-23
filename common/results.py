"""Ghi kết quả vào CSV chung của nhóm (schema cố định, xem README mục "Quy ước kết quả")."""
import csv
import math
import os

import numpy as np

COLUMNS = ["model", "variant", "dataset", "seed", "noise", "epoch",
           "auroc", "auprc", "f1", "flagged_rate", "runtime_s"]


def append_result(csv_path, **row):
    """Thêm một dòng vào csv_path; tạo file kèm header nếu chưa có.

    Cột không truyền vào sẽ để trống. Truyền cột lạ sẽ báo lỗi để tránh lệch schema.
    """
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Cột không có trong schema chung: {sorted(unknown)}")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in COLUMNS})


def _fmt(x, digits=6):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{digits}f}"


def record_epoch(run_dir, csv_path, *, model, variant, dataset, seed, epoch,
                 test_scores, test_labels, val_scores=None, runtime_s=None, noise=0):
    """Một lệnh cho mọi phương pháp: lưu điểm bất thường của epoch, tính metric, ghi một dòng CSV.

    File lưu trong <run_dir>/scores/:
      epoch_XXX.npy      điểm test (float32), một điểm cho mỗi cửa sổ
      val_epoch_XXX.npy  điểm validation (nếu có), dùng để đặt ngưỡng F1
      labels.npy         nhãn cửa sổ test (int8), ghi một lần
    noise != 0 (E5) thì thêm hậu tố _noise<giá trị> vào tên file điểm test.

    csv_path rỗng thì chỉ lưu điểm, không ghi CSV. Trả về dict metric.
    """
    from common.metrics import compute_metrics, threshold_from_val

    scores_dir = os.path.join(run_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    test_scores = np.asarray(test_scores).reshape(-1)
    test_labels = np.asarray(test_labels).reshape(-1)
    suffix = f"_noise{noise}" if noise else ""
    np.save(os.path.join(scores_dir, f"epoch_{epoch:03d}{suffix}.npy"), test_scores.astype(np.float32))
    labels_path = os.path.join(scores_dir, "labels.npy")
    if not os.path.exists(labels_path):
        np.save(labels_path, test_labels.astype(np.int8))

    threshold = None
    if val_scores is not None:
        val_scores = np.asarray(val_scores).reshape(-1)
        if not noise:
            np.save(os.path.join(scores_dir, f"val_epoch_{epoch:03d}.npy"), val_scores.astype(np.float32))
        threshold = threshold_from_val(val_scores)

    m = compute_metrics(test_scores, test_labels, threshold)
    if csv_path:
        append_result(csv_path, model=model, variant=variant, dataset=dataset, seed=seed, noise=noise,
                      epoch=epoch, auroc=_fmt(m["auroc"]), auprc=_fmt(m["auprc"]), f1=_fmt(m["f1"]),
                      flagged_rate=_fmt(m["flagged_rate"]),
                      runtime_s="" if runtime_s is None else f"{runtime_s:.3f}")
    return m
