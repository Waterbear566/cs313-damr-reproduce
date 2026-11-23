"""Hàm metric chung cho mọi phương pháp (DAMR và baseline).

Mọi metric tính theo CỬA SỔ (một nhãn cho mỗi cửa sổ, như bài báo), không dùng point adjustment.

- AUROC, AUPRC: không cần ngưỡng.
- F1, flagged_rate: cần ngưỡng. Ngưỡng lấy từ điểm trên tập validation (chỉ chứa dữ liệu bình thường),
  KHÔNG lấy từ tập test, để tránh tinh chỉnh theo đáp án. Mặc định là phân vị 99% của điểm validation,
  tức chấp nhận khoảng 1% cửa sổ bình thường bị báo động nhầm.
"""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

DEFAULT_VAL_QUANTILE = 0.99


def threshold_from_val(val_scores, q=DEFAULT_VAL_QUANTILE):
    """Ngưỡng = phân vị q của điểm bất thường trên validation (điểm càng cao càng bất thường)."""
    val_scores = np.asarray(val_scores, dtype=np.float64).reshape(-1)
    if val_scores.size == 0:
        raise ValueError("Không có điểm validation để đặt ngưỡng")
    return float(np.quantile(val_scores, q))


def compute_metrics(test_scores, test_labels, threshold=None):
    """Trả về dict auroc, auprc, f1, flagged_rate.

    threshold=None thì f1 và flagged_rate là NaN (không có ngưỡng hợp lệ thì không báo cáo).
    Cửa sổ bị gán bất thường khi điểm > threshold.
    """
    s = np.asarray(test_scores, dtype=np.float64).reshape(-1)
    y = np.asarray(test_labels).astype(int).reshape(-1)
    if s.shape != y.shape:
        raise ValueError(f"Số điểm ({s.size}) khác số nhãn ({y.size})")
    if not np.isfinite(s).all():
        raise ValueError("Điểm bất thường có NaN hoặc inf")
    out = {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "f1": float("nan"),
        "flagged_rate": float("nan"),
    }
    if threshold is not None:
        pred = s > threshold
        out["f1"] = float(f1_score(y, pred, zero_division=0))
        out["flagged_rate"] = float(pred.mean())
    return out
