# Pipeline thí nghiệm

Mọi phương pháp đi qua cùng một đường: cùng dữ liệu → cùng cửa sổ và cách chia → điểm bất thường mỗi epoch →
cùng hàm metric → một schema CSV → một script tổng hợp. Chỉ phần mô hình là khác nhau.

```
 DAMR/data/<ds>/  ──►  [1] Kiểm tra dữ liệu ───────────────────────────►  Bảng 1 (E1)
        │               scripts/check_data.py
        ▼
 [2] Chuẩn hóa + cửa sổ 60/10 + chia val 20% theo seed
     DAMR: DAMR/data_loader.py      Baseline: common/data.py (gọi lại đúng các lớp đó)
        │
        ▼
 [3] Huấn luyện, chấm điểm test + val mỗi epoch
     DAMR: DAMR/main.py --log_extra=True (scripts/run_e2.sh)
     Baseline: baselines/<Tên>/run.py (mẫu: baselines/_template/run.py)
        │
        ▼
 [4] common/results.record_epoch  ──►  checkpoints/<E>/.../scores/*.npy   (dùng lại cho E5, E6)
     (metric: common/metrics.py)   ──►  results/<E>_<model>.csv            (mỗi epoch một dòng)
        │
        ▼
 [5] scripts/summarize.py  ──►  results/summary.csv  (mean ± std 3 seed, best_test và last, cạnh số bài báo)
        │
        ▼
 [6] Script vẽ hình/bảng  ──►  figures/   (chưa có)
```

## Trạng thái từng khâu

| Khâu | File | Trạng thái |
|---|---|---|
| 1. Kiểm tra dữ liệu, thống kê Bảng 1 | `scripts/check_data.py` | Đã chạy trên PSM, MSL, SMD (bản TSLib), khớp số chuẩn |
| 2. Tiền xử lý cho DAMR | `DAMR/data_loader.py` (code gốc) | Có |
| 2. Tiền xử lý cho baseline | `common/data.py` | Có. Đã kiểm tra vị trí validation trùng code gốc với seed 0, 1, 2, 16 |
| 3. Chạy DAMR (E2) | `scripts/run_e2.sh`, `notebooks/e2_kaggle_run.ipynb` | Seed 0, 1, 2 đã chạy (thí nghiệm phụ); seed 16–25 chạy lại trên Kaggle |
| 3. Biến thể ablation (E4) | `scripts/run_e4.sh`, cờ `--global_filter`, `--fixed_temp` | Có cờ cho đủ 5 biến thể; cách hiểu MSASB(G)/(L) và τ của biến thể cuối cần đối chiếu bài báo |
| 3. Baseline (E3a, E3b) | `baselines/_template/run.py`, `baselines/README.md` | Có mẫu, chưa có baseline nào |
| 3. Nhiễu (E5), t-SNE (E6) | — | Chưa có |
| 4. Metric và ghi kết quả | `common/metrics.py`, `common/results.py` | Có, dùng thật trong E2 |
| 5. Tổng hợp | `scripts/summarize.py` (`--seeds`), `configs/paper_results.csv` | Có, dùng thật trong E2. Số bài báo mới có AUROC của DAMR |
| 6. Hình và bảng | — | Chưa có (Trung Hiếu) |

## Quy ước seed

| Mục | Quy ước |
|---|---|
| Seed chung (bảng so sánh, ablation) | **16, 17, 18** cho mọi phương pháp và dataset. 16 là seed tác giả dùng trong `DAMR/run/run_*.sh` |
| DAMR (E2) | Chạy **16–25** cho đủ 10 lần như bài báo; bảng chính lấy 16, 17, 18 (`summarize.py --seeds 16 17 18`) |
| Thí nghiệm phụ | E2 với seed 0, 1, 2 (chạy trước khi đổi quy ước): `results/E2_DAMR_seeds012.csv`, dùng để xem độ nhạy theo seed |
| Seed điều khiển | `random`, `numpy`, `torch` (CPU + CUDA), đặt trước khi nạp dữ liệu; vị trí đoạn validation cũng do seed quyết định |
| Tên | thư mục `<dataset>_seed_<k>`, cột CSV `seed` |

Cùng seed vẫn lệch nhỏ giữa các lần chạy (code không bật `cudnn.deterministic`; đo được khoảng 0.02 điểm AUROC trên PSM).

## Quy ước metric

- Theo **cửa sổ**, không dùng point adjustment. AUROC và AUPRC là chính.
- F1 và `flagged_rate` dùng ngưỡng = **phân vị 99% (P99) của điểm validation** (validation chỉ có dữ liệu bình
  thường), tính riêng cho từng mô hình. Không dùng tập test để chọn ngưỡng. **Nhóm đã chốt P99 (26/11/2026).**
  Không dùng ngưỡng cố định (0.5, 0.8…): mỗi mô hình có thang điểm khác nhau (điểm DAMR là khoảng cách kNN, cỡ 0.01).
  Với DAMR, P99 cho F1 rất thấp trên PSM (điểm test thấp hơn điểm đoạn validation nên ngưỡng nằm quá cao);
  báo cáo cần giải thích điều này.
- `summarize.py` báo cáo hai cách chọn epoch: `best_test` (như code gốc, nhìn tập test) và `last` (epoch cuối).
  **Bảng so sánh chính dùng `last` cho mọi mô hình**; `best_test` của DAMR đưa vào bảng phụ.
- Std là độ lệch chuẩn mẫu (ddof = 1) trên các seed. Metric nhân 100.

## Chạy cả đường cho DAMR

```bash
python scripts/check_data.py --root DAMR/data
bash scripts/run_e2.sh PSM 0
python scripts/summarize.py results/E2_DAMR.csv
```

Chạy lại một lần chạy thì xóa các dòng cũ của nó trong CSV trước, nếu không `summarize.py` sẽ báo trùng epoch.
