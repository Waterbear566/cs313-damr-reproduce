# Baseline (E3a, E3b)

Mỗi phương pháp một thư mục con `baselines/<Tên>/`. Bắt đầu bằng cách sao chép [`_template/`](_template/run.py).

## Bắt buộc dùng chung

| Thứ | Lấy từ | Không được |
|---|---|---|
| Dữ liệu, chuẩn hóa, cửa sổ 60 / bước 10, nhãn cửa sổ | `common.data.load_split` | tự đọc file hoặc tự chuẩn hóa |
| Validation 20% (cùng vị trí với DAMR theo seed) | `sp.val` | dùng tập test để chọn epoch hay siêu tham số |
| Metric, ngưỡng F1, lưu điểm, ghi CSV | `common.results.record_epoch` | tự tính AUROC/F1 |
| Seed | 0, 1, 2 | |

Mô hình, loss, optimizer, số epoch và siêu tham số giữ đúng **mặc định của repo gốc** baseline. Nếu repo gốc có
bước đặc thù (ví dụ point adjustment, ngưỡng riêng), bỏ bước đó và chỉ giữ điểm bất thường thô.

`sp.train[i]` trả về `(x, nhãn)`, `x` có dạng `(sensor, thời gian)`. Phần lớn baseline Transformer cần
`(thời gian, sensor)`: dùng `x.transpose(1, 2)` trong batch, hoặc `sp.to_numpy("train", layout="TC")`.

## Đầu ra

| Đường dẫn | Nội dung |
|---|---|
| `results/E3_<Tên>.csv` | mỗi epoch một dòng, đúng schema chung |
| `checkpoints/E3/<Tên>/<dataset>_seed_<k>/scores/` | `epoch_XXX.npy`, `val_epoch_XXX.npy`, `labels.npy` |
| `checkpoints/E3/<Tên>/<dataset>_seed_<k>/model_last.pt` | checkpoint epoch cuối, cho E5 |

## README riêng của mỗi baseline

Mỗi `baselines/<Tên>/README.md` ghi: link repo gốc và commit, phần nào chép nguyên, phần nào sửa để nối vào
pipeline, môi trường riêng (nếu khác `requirements.txt`), lệnh chạy.
