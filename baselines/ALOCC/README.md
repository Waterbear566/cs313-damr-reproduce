# ALOCC (E3a)

- **Repo gốc:** cài đặt ALOCC trong [zqhang/MTGFLOW](https://github.com/zqhang/MTGFLOW) @ `f8cb9ea`, `models/GAN.py` (`R_Net`, `D_Net`, `R_Loss`, `D_Loss`, `train_model`). Mạng chép ở [`../_mtgflow_ref/networks.py`](../_mtgflow_ref/networks.py).
- **Chép nguyên:** kiến trúc (width 16), loss, thứ tự cập nhật D trước R, Adam lr 1e-3, λ = 0,2, nhiễu std 0,2, 40 epoch, batch 512.
- **Sửa:**
  - `D_Net._compute_out_dim` dùng tensor 0 ở eval mode, thay cho `torch.Tensor` chưa khởi tạo (bản gốc làm bẩn thống kê BatchNorm). Kích thước đầu ra không đổi.
  - Dữ liệu và metric lấy từ `common/`.
- **Môi trường:** `requirements.txt` cộng thêm PyTorch.

## Biến thể (cùng một lần train, chỉ khác cách chấm điểm)

| `variant` trong CSV | Điểm bất thường |
|---|---|
| `full` | −D(x), như `test_single_epoch` của repo MTGFlow |
| `score_DRx` | −D(R(x)), như bài báo ALOCC gốc; R không thêm nhiễu khi chấm |

## Chạy

```bash
python baselines/ALOCC/run.py --dataset PSM --seed 0
```

Kết quả: `results/E3_ALOCC.csv`, điểm ở `checkpoints/E3/ALOCC/<variant>/<dataset>_seed_<k>/`.
