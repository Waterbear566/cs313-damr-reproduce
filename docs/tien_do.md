# Tiến độ và công việc – phần code (Nhóm 5)

Cập nhật: 28/11/2026. File này dành cho những người làm code trong repo. Ai xong việc hoặc đổi trạng thái thì sửa file này trong PR của mình.

Ký hiệu: ✅ xong · 🟡 đang làm · ⬜ chưa bắt đầu · ⏸ đang chờ việc khác

---

## Tóm tắt

- **Đã xong:** E1 (thống kê dataset), E2 (DAMR gốc, seed 16–25), E4 (ablation, seed 16–18), số liệu E3a (5 baseline × 3 dataset × seed 16–18).
- **Chưa có:** code E3a cho USAD, GANF, MTGFlow trong `baselines/`; toàn bộ E3b; E5, E6; script vẽ hình/bảng.
- **Việc gấp nhất:** E3b; bổ sung code USAD/GANF/MTGFlow; chạy lại ALOCC/DeepSVDD có ghi mọi epoch (AUROC hiện ≈ 0.5); đưa `e4_outputs.zip` lên Drive.

---

## Trạng thái theo thí nghiệm

| Mã | Thí nghiệm | Người làm | Số lần train | Trạng thái | Ghi chú |
|---|---|---|---|---|---|
| E1 | Thống kê dataset (Bảng 1) | Hoàng Khang | 0 | ✅ | `results/E1_dataset_stats.csv`, `E1_missing_values_by_feature.csv` |
| E2 | DAMR gốc, 3 dataset × 10 seed (16–25) | Mạnh Hùng, Huy Khang | 30 | ✅ | `results/E2_DAMR.csv` (16–25), bảng chính `E2_summary.csv` (16–18), `E2_summary_10seeds.csv`; seed 0, 1, 2 là thí nghiệm phụ. Checkpoint: Drive chung `checkpoints/E2/` (`e2_outputs_seeds16-25.zip`) |
| E3a | DeepSVDD, ALOCC, USAD, GANF, MTGFlow | Hoàng Khang | 45 | 🟡 | Đã có số liệu `results/E3_<Model>.csv` (seed 16–18, chỉ epoch cuối, chưa có F1). Code mới có ALOCC, DeepSVDD trong `baselines/`; thiếu USAD, GANF, MTGFlow. ALOCC/DeepSVDD AUROC ≈ 0.5, cần chạy lại ghi mọi epoch |
| E3b | Anomaly Transformer, DCdetector, TFMAE, MtsCID | Minh Hoàng | 36 | ⬜ | |
| E4 | Ablation (5 biến thể) | Mạnh Hùng, Huy Khang | 45 | ✅ | `results/E4_DAMR_ablation.csv`, `E4_summary.csv` (seed 16–18); nhận xét ở [`E4_ket_qua.md`](E4_ket_qua.md). Checkpoint: `e4_outputs.zip` (cần đưa lên Drive chung `checkpoints/E4/`) |
| E5 | Nhiễu Gaussian trên PSM | Huy Khang | 0 | ⏸ | Checkpoint E2 ở Drive `checkpoints/E2/`; chờ E3, E4 |
| E6 | t-SNE trên MSL, độ tương đồng giữa các head | Huy Khang | 0 | 🟡 | Có thể bắt đầu khi checkpoint E2 (seed 16–25) lên Drive `checkpoints/E2/` |

Tổng cộng khoảng 135 lần train. Cần ước lượng thời gian sau khi chạy notebook Kaggle để chia GPU.

---

## Việc của từng người

### Vũ Cao Mạnh Hùng – quản lý repo, E2, E4
- [x] Dựng repo, README, `.gitignore`, `requirements.txt`
- [x] Thêm cờ ghi log (`--log_extra`, `--save_dir`, `--variant`, `--results_csv`), `common/results.py`, `scripts/run_e2.sh`
- [x] Notebook `notebooks/stage1_kaggle_smoke.ipynb`
- [x] Merge `e2-damr-logging` và `docs-doi-vai-tro` vào `main`
- [x] Chạy E2 trên PSM, MSL, SMD với seed 0, 1, 2 (thí nghiệm phụ về độ nhạy theo seed)
- [x] Chạy lại E2 với seed 16–25 (seed chung 16, 17, 18; đủ 10 lần như bài báo)
- [x] Chạy E4 trên Kaggle (PSM, MSL, SMD; 5 biến thể × seed 16–18)
- [x] Thêm cờ cho MSASB(G), MSASB(L), τ = 0.25 cố định, rồi chạy E4
- [ ] Review PR của Hoàng Khang và Huy Khang

### Ngô Hoàng Khang – E1, E3a, hàm metric chung, demo
- [x] Chạy `python scripts/check_data.py --root DAMR/data` và xuất Bảng 1 vào `results/`
- [x] Hàm metric chung (`common/metrics.py`) và dữ liệu chung cho baseline (`common/data.py`): đã có, cần review
- [x] Chạy 5 baseline nhóm A (số liệu trong `results/E3_<Model>.csv`)
- [ ] Đưa code 5 baseline nhóm A vào `baselines/<tên>/` (đã có ALOCC, DeepSVDD; còn USAD, GANF, MTGFlow)
- [ ] Chuẩn bị demo
- [ ] Review PR của Mạnh Hùng và Minh Hoàng

### Lê Minh Hoàng – E3b
- [ ] Đưa 4 baseline nhóm B vào `baselines/<tên>/` theo `baselines/README.md` và mẫu `baselines/_template/`
- [ ] Ghi môi trường riêng (nếu có) vào `baselines/<tên>/README.md`

### Võ Huy Khang – chạy cùng E2/E4, E5, E6, đối chiếu số liệu
- [ ] Chạy cùng E2 và E4 để chia tải GPU
- [ ] Viết script E5: thêm nhiễu Gaussian vào test PSM, chấm bằng checkpoint có sẵn
- [ ] Viết script E6: t-SNE trên MSL, độ tương đồng giữa các head trên PSM
- [ ] Đối chiếu số của nhóm với số bài báo; chỗ lệch quá độ lệch chuẩn thì ghi nguyên nhân vào `docs/`

### Nguyễn Trung Hiếu – script vẽ hình và bảng
- [ ] Viết script vẽ hình/bảng (PDF hoặc SVG) vào `figures/`, đọc từ `results/summary.csv` do `scripts/summarize.py` sinh ra
- [ ] Bảng đặt số của nhóm cạnh số bài báo, ghi mean ± std trên 3 seed
- [ ] Có thể làm trước bằng CSV giả theo đúng các cột quy ước

---

## Phát hiện cần nắm khi code

Chi tiết ở [`code_vs_paper.md`](code_vs_paper.md). Những điểm ảnh hưởng tới cách chạy và cách đọc kết quả:

1. Code chọn checkpoint theo **AUROC trên tập test**. Vì vậy nhóm báo cáo cả epoch code gốc chọn và epoch cuối.
2. Tham số của adaptive temperature **không được đưa vào optimizer**, nên τ không thật sự được học. Khi viết phần E4, không được nói "τ học được so với τ cố định".
3. Bước chấm kNN lấy ngẫu nhiên 10% đặc trưng, nên chấm lại cùng một checkpoint sẽ ra điểm hơi khác. E5 và E6 so sánh với điểm đã lưu trong `scores/`.
4. Hàm `permutation` thay mọi sensor bằng sensor 0. MSASB làm FFT theo trục sensor. Cần đối chiếu với mục Method của bài báo trước khi kết luận.

---

## Thứ tự làm và phụ thuộc

```
Dữ liệu PSM/MSL/SMD ──► E1
        │
        ├──► Notebook Kaggle (ước lượng thời gian) ──► E2 ──► E5, E6
        │                                          └──► E4 ──► E5
        └──► Hàm metric chung ──► E3a, E3b ──► E5
                                              
results/*.csv ──► Script vẽ hình/bảng ──► (giao cho nhóm báo cáo)
```

---

## Việc chung còn treo

- [ ] Điền cột "Bài báo nói gì" trong `code_vs_paper.md` (đang ghi `<cần đối chiếu>`) sau khi nhóm báo cáo đọc bài báo
- [ ] Xin quyền dữ liệu SWaT, WADI từ iTrust (làm nếu được cấp kịp)
- [ ] Điền phiên bản CUDA và GPU vào README
- [x] Tạo thư mục Google Drive chung để lưu checkpoint, điền link vào README ([CS313_Nhom5](https://drive.google.com/drive/folders/1GFjHkxzl4lbacWTnEqqMfHJ2ybcXh9TP?usp=sharing))
- [ ] Chốt deadline cho từng giai đoạn: `<điền>`
