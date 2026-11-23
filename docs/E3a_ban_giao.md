# E3a và E1: kết quả bàn giao của Hoàng Khang

Nguồn: gói `Data_mining_NHK.zip` (Ngô Hoàng Khang, 23/11/2026). Code chạy trên Colab và checkpoint nằm trên Drive của Khang, xem `artifact_locations.md` trong gói. Repo chỉ giữ CSV.

## File đã đưa vào repo

| File trong repo | Lấy từ file nào trong gói | Ghi chú |
|---|---|---|
| `results/E3_MTGFlow.csv`, `E3_GANF.csv`, `E3_USAD.csv`, `E3_DeepSVDD.csv`, `E3_ALOCC.csv` | `results_45_runs.csv` | Tách theo model, giá trị giữ nguyên không làm tròn. Mỗi file 9 dòng (3 dataset × 3 seed) |
| `results/E1_dataset_stats.csv` | `E1_.../dataset_stats_all.csv` | Bảng 1 |
| `results/E1_missing_values_by_feature.csv` | `E1_.../missing_values_by_feature_all.csv` | NaN theo từng feature (PSM) |
| `results/E3a_thresholds_P95_PROPOSED.csv` | `thresholds_PROPOSED_P95_45_runs.csv` | Ngưỡng **đề xuất**, chưa được duyệt |
| `results/E3a_run_inventory.csv` | `run_inventory.csv` | Đường dẫn Drive của từng lượt |
| `configs/data_manifest_MSL_SMD.json` | `E1_.../data_manifest_MSL_SMD.json` | SHA256 và revision HuggingFace của dữ liệu |
| `configs/paper_results.csv` (thêm 15 dòng) | Bảng trong PDF của Khang | Số bài báo của 5 baseline, **cần đối chiếu lại với bài báo** |

Tổng hợp bằng: `python scripts/summarize.py results/E3_*.csv`. Kết quả khớp với `summary_15_groups.csv` của Khang.

## Những chỗ khác quy ước repo (chưa sửa, cần nhóm quyết)

1. **Seed 16, 17, 18** thay vì 0, 1, 2. Không đổi nhãn seed trong CSV, vì seed còn quyết định vị trí validation.
2. **Chỉ có epoch cuối.** Quy ước của repo là mỗi epoch một dòng. Vì thế trong `summary.csv`, dòng `best_test` của E3a chính là `last`: **không** được đem so với `best_test` của DAMR.
3. **Ngưỡng:** Khang đề xuất P95 của validation, còn `common/metrics.py` đang dùng P99. Nhóm cần chọn một mức cho mọi phương pháp.
4. **`runtime_s`** là thời gian của cả lượt chạy, không phải của một epoch.
5. **`variant`** là tên cách cài đặt (ví dụ `shared_standard_sigmoid`), không phải `full`. `summarize.py` lấy số bài báo của bản `full` cho các baseline này. Riêng DAMR không áp dụng cách này, vì biến thể của DAMR là ablation.
6. USAD dùng StandardScaler nhưng vẫn giữ lớp Sigmoid ở đầu ra, nên không tái tạo được giá trị âm. GANF chạy 38–45 epoch thay vì một số cố định.

## Vì sao ALOCC và DeepSVDD thấp

Kết quả của nhóm so với bài báo: ALOCC lệch −19,7 (PSM), −6,6 (MSL), −7,3 (SMD); DeepSVDD trên PSM lệch −30,5, AUROC dưới 0,5.

### Các phần đã kiểm tra và không thấy lỗi

Đối chiếu với `zqhang/MTGFLOW@f8cb9ea` (SHA256 khớp với bản Khang ghim): `models/GAN.py`, `models/DeepSAD.py`, `utils.py`.

- **Dữ liệu:** window 60, bước 10; nhãn window bằng 1 nếu có điểm bất thường; validation 20% liên tục, vị trí theo seed; scaler fit trên train. Tất cả trùng với `DAMR/data_loader.py`.
- **Shape đầu vào:** `x.squeeze(-1)` cho kết quả giống `transpose(2,3)` rồi `reshape` của code gốc, ra `(B, sensor, 60)`.
- **DeepSVDD:** tâm `c` (forward ở eval mode, kẹp ±0,1, lấy trung bình theo thời gian), loss, điểm bằng khoảng cách tới `c`, lr 1e-4 rồi 1e-5 ở epoch 20, weight decay 5e-4, không pretrain. Tất cả giống code gốc.
- **ALOCC:** R_Net, D_Net, R_Loss, D_Loss, thứ tự cập nhật D trước R, lr 1e-3, λ = 0,2, nhiễu 0,2, điểm = −D(x). Tất cả giống code gốc. Khang chỉ sửa phần dò kích thước, không ảnh hưởng tới việc học.

### Nguyên nhân 1, có bằng chứng trong code: code gốc chọn epoch theo tập test

- DeepSVDD trong repo MTGFlow chấm test sau **mỗi epoch** (`DeepSAD.py:140-141`) và giữ AUROC lớn nhất (`utils.py:20-21`, `roc_auc_max`).
- ALOCC cũng chấm test và in AUROC sau mỗi epoch (`GAN.py:220`, `GAN.py:329-330`).

Như vậy số của code gốc là AUROC ở **epoch tốt nhất trên test**, giống cách DAMR tự chọn checkpoint. Khang lấy **epoch cuối**, nên kết quả thấp hơn là có hệ thống. Hai model càng dao động mạnh thì chênh lệch càng lớn: độ lệch chuẩn giữa các seed của ALOCC là 4–7 điểm, DeepSVDD trên PSM là 5,8 điểm.

Chưa chắc bài báo DAMR lấy số baseline theo cách này. Cần đọc mục Experiments của bài báo.

### Nguyên nhân 2, ALOCC: discriminator thắng áp đảo nên điểm không còn phân biệt được

Log huấn luyện trong `NHK_code_TK1.ipynb`, so giữa epoch 1 và epoch 40:

| | D loss | R loss |
|---|---|---|
| PSM | 1,25 → 0,03 | 1,08 → 6,6 |
| MSL | 1,33 → 0,14 | 0,88 → 4,6 |
| SMD | 0,66 → 0,06 | 2,17 → 6,7 |

D học phân biệt *dữ liệu thật* với *bản tái tạo của R*, chứ không học phân biệt *bình thường* với *bất thường*. Khi D đã thắng hẳn, mọi window thật, kể cả window bất thường, đều được chấm là "thật". Vì vậy điểm −D(x) gần như ngẫu nhiên, AUROC quanh 0,5. Ở epoch đầu, khi D chưa bão hòa, điểm có thể tốt hơn, và nguyên nhân 1 sẽ cho thấy điều này nếu có điểm từng epoch.

Ghi chú: bài báo ALOCC gốc chấm điểm bằng D(R(x)). Repo MTGFlow lại dùng D(x). Đây là giới hạn của bản cài đặt tham chiếu, không phải lỗi của Khang.

### Nguyên nhân 3, DeepSVDD trên PSM (giả thuyết, chưa kiểm chứng): khác chế độ BatchNorm

Trong code gốc, `net.train()` chỉ được gọi **một lần** trước vòng lặp (`DeepSAD.py:101`). Mỗi epoch, `test()` lại đặt `net.eval()` (`DeepSAD.py:152`) mà không bật lại train mode. Vì vậy từ epoch 2 trở đi, code gốc thực chất **train với BatchNorm ở eval mode**: dùng running stats, cùng chế độ với lúc tính tâm `c`.

Khang chủ động bật `net.train()` ở mỗi epoch (`deepsvdd_colab_cells.py:218`, có ghi trong `adapter_notes`). Kết quả là tâm `c` được tính ở eval mode, việc train diễn ra ở train mode (thống kê theo batch), còn lúc chấm lại quay về eval mode. Không gian đặc trưng ở ba bước không nhất quán với nhau.

Log cũng phù hợp với giả thuyết này: loss cuối trên PSM là 0,87–1,59, cao hơn hẳn SMD (0,11–0,16), và PSM cũng là dataset duy nhất mà DeepSVDD hỏng.

## Đề xuất

1. **Chạy lại DeepSVDD và ALOCC theo pipeline chung** (`common.results.record_epoch`): chấm test và validation ở mỗi epoch, dùng seed 0, 1, 2. Khi đó `summarize.py` sẽ cho cả `best_test` và `last`, và kiểm chứng được nguyên nhân 1. Chi phí nhỏ: theo `runtime_s`, 18 lượt mất khoảng 15 phút GPU.
2. **Thêm biến thể DeepSVDD `upstream_eval_bn`**, giữ đúng hành vi BatchNorm của code gốc, để kiểm tra nguyên nhân 3. Ghi thành biến thể riêng, không thay dòng chính.
3. **ALOCC:** ghi D loss và R loss vào báo cáo. Có thể thêm phần phụ chấm bằng D(R(x)), nhưng không đưa vào bảng chính.
4. Ba model còn lại (MTGFlow, GANF, USAD) cũng chỉ có epoch cuối. Nếu nhóm báo cáo `best_test` cho DAMR thì cần chạy lại cả ba để so sánh công bằng.
5. Đối chiếu 15 số bài báo trong `configs/paper_results.csv` với Bảng 2.
