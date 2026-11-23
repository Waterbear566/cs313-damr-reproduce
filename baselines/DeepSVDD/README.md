# DeepSVDD (E3a)

- **Repo gốc:** cài đặt DeepSVDD trong [zqhang/MTGFLOW](https://github.com/zqhang/MTGFLOW) @ `f8cb9ea`, `models/DeepSAD.py` (lớp `DeepSVDDTrainer`) và encoder `CNNAE` trong `models/GAN.py`. Mạng chép ở [`../_mtgflow_ref/networks.py`](../_mtgflow_ref/networks.py).
- **Chép nguyên:** encoder CNN (width 16, kernel 5), cách tính tâm `c`, loss, điểm bất thường, optimizer, lịch learning rate, 20 epoch, batch 512, không pretrain (script gốc không gọi pretrain).
- **Sửa để nối pipeline:** dữ liệu, cửa sổ và validation lấy từ `common.data`. Điểm test và validation ghi mỗi epoch qua `common.results.record_epoch`. Script gốc chỉ chấm test, không chấm validation.
- **Môi trường:** `requirements.txt` cộng thêm PyTorch.

## Biến thể

| `--variant` | Khác gì |
|---|---|
| `full` | Giống code gốc: `net.train()` chỉ gọi một lần. Bước chấm điểm sau mỗi epoch đặt `net.eval()`, nên từ epoch 2 BatchNorm chạy ở eval mode |
| `bn_train_each_epoch` | Gọi `net.train()` đầu mỗi epoch, giống bản Hoàng Khang chạy cho E3a (seed 16–18) |

## Chạy

```bash
python baselines/DeepSVDD/run.py --dataset PSM --seed 0 --variant full
```

Kết quả: `results/E3_DeepSVDD.csv`, điểm và checkpoint ở `checkpoints/E3/DeepSVDD/<variant>/<dataset>_seed_<k>/`. Muốn chạy cả 3 dataset × 3 seed cho DeepSVDD và ALOCC thì dùng `scripts/run_e3a_rerun.ps1`.
