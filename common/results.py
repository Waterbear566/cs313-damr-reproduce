"""Ghi kết quả vào CSV chung của nhóm (schema cố định, xem README mục "Quy ước kết quả")."""
import csv
import os

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
