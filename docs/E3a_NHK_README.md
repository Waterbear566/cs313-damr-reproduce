# Bàn giao thực nghiệm E1 và E3a — Hoàng Khang

**Người thực hiện:** Ngô Hoàng Khang — MSSV 24520748 — CS313, Nhóm 5.

Bàn giao thống kê dữ liệu (E1) và kết quả 5 baseline nhóm A (E3a): **MTGFlow, DeepSVDD, USAD, ALOCC, GANF** trên **PSM, MSL, SMD**, mỗi model–dataset **3 seed: 16, 17, 18**. Bảng chính gồm **45 lượt**, lấy **epoch cuối**, F1/flagged rate theo **ngưỡng P99 validation**.

## 1. Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| E1: thống kê PSM/MSL/SMD | Hoàn tất (`results/E1_*.csv`, `configs/data_manifest_MSL_SMD.json` đã có trong repo) |
| E3a: 45 lượt AUROC/AP | Hoàn tất; checkpoint, scaler, split, score lớn giữ trên Drive |
| DeepSVDD | Bảng chính dùng `upstream_eval_bn` (đúng hành vi BatchNorm code gốc), chạy lại 26/11/2026. Bản cũ `mtgflow_repo_cnn_no_pretrain` giữ làm tham khảo |
| Threshold | **Nhóm đã chốt P99 validation (26/11/2026)** |
| F1 và flagged rate | Đã tính cho 45 lượt chính từ score đã lưu, không train lại. Bản DeepSVDD cũ không tính (chỉ tham khảo) |
| Metric | Chính: AUROC và AP (cột `auprc`). Phụ: F1 và flagged rate. Định nghĩa trùng `common/metrics.py` |
| Đối chiếu đầy đủ Table 2/cấu hình paper | Chưa hoàn tất |
| Review chéo và demo | Chưa hoàn tất |

## 2. File trong repo

PR này cập nhật `results/`, thêm `configs/`, `scripts/`, `docs/`, code Colab `baselines/<tên>/colab_E3a/` và notebook `notebooks/e3a_*.ipynb`.

| File | Nội dung |
|---|---|
| `results/E3_MTGFlow.csv`, `E3_GANF.csv`, `E3_USAD.csv`, `E3_ALOCC.csv` | 9 dòng mỗi file (3 dataset × 3 seed), epoch cuối, đủ 11 cột chung kể cả `f1`, `flagged_rate` |
| `results/E3_DeepSVDD.csv` | 9 dòng `upstream_eval_bn` (bảng chính) + 9 dòng `mtgflow_repo_cnn_no_pretrain` (tham khảo, F1 để trống) |
| `results/E3a_thresholds_P99.csv` | Ngưỡng P99 từng lượt, số window validation/test, SHA256 score |
| `results/E3a_run_inventory.csv` | Đường dẫn Drive của 45 lượt chính (không phải link tải trực tiếp) |
| `results/E3a_score_audit.csv` | AUROC/AP tính lại từ score đã lưu, khớp 45/45 với số đã báo cáo |
| `configs/E3a_threshold_policy_P99.json` | Định nghĩa ngưỡng P99, phạm vi kiểm tra và giới hạn |
| `baselines/<tên>/colab_E3a/*.py` | Code Colab đã dùng chạy E3a (không phải CLI độc lập, xem mục 3) |
| `scripts/e3a_merge_p99_colab.py` | Ô Colab gộp 36 lượt cũ + 9 lượt DeepSVDD mới và tính F1/flagged rate P99 |
| `notebooks/e3a_NHK_TK1.ipynb` | E1 và 43 lượt đầu (tài khoản Colab 1) |
| `notebooks/e3a_NHK_TK2.ipynb` | 2 lượt GANF–SMD seed 17/18 (tài khoản Colab 2) |
| `notebooks/e3a_deepsvdd_upstream.ipynb` | 9 lượt DeepSVDD `upstream_eval_bn` và bước gộp P99 |

Tên notebook gốc: TK1 = `NHK_code (10).ipynb` (bản `(11)` có các ô giống hệt); TK2 = `Untitled0 (1).ipynb`, ô khôi phục nạp định nghĩa từ `NHK_code (9).ipynb`; notebook DeepSVDD = `SVDD (1).ipynb`.

Tổng hợp mean ± std: `python scripts/summarize.py results/E3_*.csv` (bảng chính lấy dòng `last`).
Bảng chính chỉ lấy selection=last; riêng DeepSVDD chỉ lấy
variant=upstream_eval_bn. Biến thể mtgflow_repo_cnn_no_pretrain
chỉ để tham khảo, không cộng vào 45 lượt chính.

## 3. Code baseline

| File | Phạm vi |
|---|---|
| `baselines/MTGFlow/colab_E3a/mtgflow_extra_colab.py` | Runner MTGFlow cho MSL/SMD; phần PSM nằm trong notebook TK1 |
| `baselines/DeepSVDD/colab_E3a/deepsvdd_upstream_colab.py` | Runner DeepSVDD **bảng chính** (`upstream_eval_bn`) |
| `baselines/DeepSVDD/colab_E3a/deepsvdd_colab_cells.py` | Runner DeepSVDD bản cũ (`net.train()` mỗi epoch) |
| `baselines/USAD/colab_E3a/usad_colab_cells.py` | USAD và helper dữ liệu dùng chung `load_usad_shared_data` |
| `baselines/ALOCC/colab_E3a/alocc_colab_cells.py` | ALOCC: R/D, scoring, kiểm tra lượt hoàn tất |
| `baselines/GANF/colab_E3a/ganf_colab_cells.py` | GANF: model + đồ thị A, scoring, exporter |

**Đây không phải chương trình CLI độc lập.** Các file nối với định nghĩa trong notebook và đường dẫn Google Drive (`/content/drive/MyDrive/...`). `run.py` trong `baselines/DeepSVDD` và `baselines/ALOCC` là bản Mạnh Hùng viết lại theo pipeline chung, không thay thế các file này.

## 4. Cách sử dụng an toàn

### Chỉ đọc kết quả
- Mở CSV hoặc output đã lưu trong notebook; **không cần GPU, không cần chạy notebook**.
- Mọi số AUROC/AP trong CSV dùng thang 0–1; nhân 100 nếu trình bày phần trăm.
- Không nhập lại số bằng tay để tạo bảng/hình; đọc từ CSV bằng script.

### Nạp code hoặc chạy inference
- Xem các định nghĩa/source, cấu hình và hướng dẫn trong notebook trước.
- Cần quyền truy cập data, source cache và checkpoint qua các link Drive gửi trong nhóm.
- Đường dẫn mặc định của các runner là `/content/drive/MyDrive/...`. Tài khoản khác cần mount đúng Drive/lối tắt hoặc chỉnh đường dẫn có kiểm soát.
- Chỉ nạp import/định nghĩa cần thiết. **Không bấm Run all:** notebook có các ô gọi train và tạo lượt mới.
- Các hàm `run_*` là huấn luyện, không phải chỉ xem kết quả. Hàm `run_*_if_missing` có kiểm tra bỏ qua lượt hoàn tất nhưng vẫn bắt đầu train nếu thiếu lượt tương thích.
- Nhiều runner chưa có cơ chế tự resume train chính xác từ checkpoint giữa chừng. Không suy rằng có `.pt` là tiếp tục train được ngay.
- Inference cần model code, scaler và checkpoint đúng lượt; DeepSVDD cần tâm `c`, GANF cần graph `A`, ALOCC lưu cả R và D. Không chỉ giao một file trọng số rồi bỏ config/scaler.

## 5. Quy ước đánh giá

Schema kết quả:

`model,variant,dataset,seed,noise,epoch,auroc,auprc,f1,flagged_rate,runtime_s`

- Đánh giá **theo window**, nhãn 1 nếu có ít nhất một điểm bất thường; không point adjustment.
- Score lớn hơn biểu thị bất thường hơn theo scoring đã chốt của mỗi model.
- `auroc`: `sklearn.metrics.roc_auc_score`.
- `auprc`: **average precision**, tính bằng `average_precision_score`; không phải PR-AUC nội suy hình thang.
- Tính metric riêng từng seed rồi tổng hợp mean ± sample std (`ddof=1`), không chọn best seed.
- `noise=0`: không thêm corruption ngoài vào dữ liệu benchmark. ALOCC vẫn có noise nội bộ std=0,2 khi train R.
- `runtime_s` của các runner E3a là thời gian cả lượt (train + đánh giá cuối, không tính khâu chuẩn bị), không phải một epoch. MTGFlow không ghi `runtime_definition` trong `run_config.json`. DeepSVDD `upstream_eval_bn` có chấm test sau mỗi epoch nên chậm hơn bản cũ. Không so runtime giữa các model hoặc với runtime mỗi epoch của E2.

## 6. Thiết lập và các điểm nhóm đã thống nhất

- Cùng StandardScaler, window 60/stride 10, actual split IDs và file dữ liệu theo protocol đã lưu. Scaler fit toàn file train trước split; train/val khác window IDs nhưng có thể chung điểm thời gian. MSL/SMD là mảng gộp thiếu metadata ranh giới chuỗi/máy.
- **Seed:** 16/17/18 là seed chung đã thống nhất cho mọi phương pháp (repo: `common/data.py`, `docs/pipeline.md`). Vị trí validation tính bằng `random.Random(seed).randrange(...)`, trùng với DAMR.
- **Checkpoint:** kết quả E3a lấy epoch cuối theo lịch chốt, không chọn best-test. Nhóm đã chốt **bảng so sánh chính dùng epoch cuối (`last`) cho mọi model**; `best_test` của DAMR đưa vào bảng phụ.
- Epoch: MTGFlow 40; DeepSVDD 20; USAD 100; ALOCC 40. GANF dùng lịch học đồ thị rồi 30 epoch cuối, thực tế 38–45 epoch trong 9 lượt; đây là ngoại lệ cần nhóm ghi nhận.
- USAD giữ Sigmoid dù dùng StandardScaler chung; đây là hạn chế miền giá trị, không đổi activation/clip input để tăng điểm test.
- DeepSVDD là CNN width16, không pretrain. ALOCC width16 và safe shape probe. Xem `adapter_notes.json` từng lượt cho sửa đổi chi tiết; chưa xác minh tất cả cấu hình trùng paper DAMR.
- **DeepSVDD trong bộ 45 lượt cũ gọi `net.train()` đầu mỗi epoch**, tương ứng biến thể `bn_train_each_epoch` trong `baselines/DeepSVDD` của repo. Code gốc chỉ gọi một lần, nên từ epoch 2 BatchNorm chạy ở eval mode (biến thể `full` của repo). Bản chạy lại đúng hành vi code gốc là `upstream_eval_bn` (`deepsvdd_upstream_colab.py`), **nay là bảng chính**: epoch 1 train mode, epoch 2–20 eval mode vẫn cập nhật gradient. Runner chấm test sau mỗi epoch như code gốc nhưng chỉ để chẩn đoán; kết quả lấy epoch 20. Kết quả: PSM 44,6 → 67,5; MSL 54,0 → 46,9; SMD 75,2 → 74,5 (AUROC, %). Epoch tốt nhất trên test chỉ hơn epoch cuối 0,1–1 điểm, nên việc chọn epoch không giải thích khoảng cách với paper.
- **ALOCC** train với `drop_last=False`; bản `baselines/ALOCC` của repo dùng `drop_last=True` và có thêm cách chấm −D(R(x)). Bản trong 45 lượt chỉ chấm −D(x) như repo MTGFlow.

## 7. Threshold P99 (đã chốt)

Mỗi lượt lấy **P99 validation score**, `np.quantile(..., 0.99, method="linear")`; test score **lớn hơn** ngưỡng mới được dự đoán bất thường. Giá trị ngưỡng khác nhau giữa các lượt vì thang score khác nhau, nhưng quy tắc percentile giống nhau cho mọi model. Định nghĩa trùng `threshold_from_val` và `compute_metrics` trong `common/metrics.py`.

- Không dùng nhãn test để chọn ngưỡng; P99 không bảo đảm flagged rate hay FPR trên test bằng 1%.
- F1/flagged rate tính từ `validation_scores.csv` và `test_scores.csv` đã lưu trong từng lượt, bằng CPU, không train lại. Script đọc lại score và kiểm tra AUROC/AP khớp với số đã báo cáo trước khi tính (`results/E3a_score_audit.csv`).
- MSL chỉ có 1165 window validation, nên P99 dựa vào khoảng 12 score cao nhất; ngưỡng MSL kém ổn định hơn PSM (2648) và SMD (14167).
- Ngưỡng P95 trước đây chỉ là đề xuất, nay không dùng; file P95 giữ làm lịch sử.
- Validation lấy từ file train, vốn không có nhãn bất thường. Window validation mang nhãn −1 (không có nhãn gốc), nên không có window nhãn 1 nào. Như mọi phương pháp, ngưỡng giả định dữ liệu train là bình thường.

## 8. Artifact lớn

Checkpoint, scaler, split và score gốc nằm trên các thư mục Drive (`CS313_Nhom5/hoang_khang/E3a` ở tài khoản 1; `NHK_GANF_TAI_KHOAN_2/E3a` ở tài khoản 2; `NHK_DeepSVDD_UPSTREAM/E3a` ở tài khoản đã chạy notebook DeepSVDD). Link chia sẻ gửi riêng trong nhóm, không commit lên repo. Không commit raw data, checkpoint `.pt`, score lớn hay thông tin đăng nhập.

## 9. Hạn chế đã biết

- **PSM seed 17, ngưỡng P99 không ổn định.** MTGFlow, GANF, USAD và ALOCC chỉ đạt F1 1,8–2,8% và gắn cờ 0,3–0,4% window test, trong khi seed 16 và 18 đạt F1 khoảng 20–45% (trừ ALOCC). DeepSVDD không bị ảnh hưởng (F1 32%). Có thể khối validation 20% mà seed 17 chọn chứa vài window score cao, đẩy ngưỡng lên; chưa kiểm chứng. AUROC/AP không phụ thuộc ngưỡng nên không bị ảnh hưởng.
- **ALOCC AUROC quanh 0,5** trên cả ba dataset (trung bình 49,2 / 50,7 / 51,0 cho PSM / MSL / SMD). Điểm −D(x) gần như không phân biệt được window bất thường, có thể vì discriminator bão hòa (D loss giảm về gần 0 ở cuối train). Đây là hạn chế đã biết, không tinh chỉnh thêm để tăng điểm test.
- **DeepSVDD `upstream_eval_bn` trên MSL dưới 0,5** (44,1–49,5, trung bình 46,9), thấp hơn bản cũ (54,0) và paper (52,9). Bảng chính vẫn dùng `upstream_eval_bn` vì đúng hành vi code gốc; không chọn biến thể theo điểm test.
- Chỉ có kết quả epoch cuối cho 36 lượt không phải DeepSVDD, nên không tính được `best_test` cho các model này.
