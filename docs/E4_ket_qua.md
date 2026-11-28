# Kết quả E4 – Ablation DAMR (Bảng 3, 4)

Chạy 26/11/2026 trên Kaggle (T4), 5 biến thể × 3 dataset × seed 16, 17, 18, 40 epoch, đúng siêu tham số E2
(`scripts/run_e4.sh`). Đủ 45 lần chạy × 40 epoch = 1800 dòng.

- Dữ liệu từng epoch: `results/E4_DAMR_ablation.csv`
- Tổng hợp: `results/E4_summary.csv` (`python scripts/summarize.py results/E4_DAMR_ablation.csv`, đã chạy lại và khớp)
- Bản `full` lấy từ E2 (`results/E2_summary.csv`, seed 16–18)
- Thời gian train: PSM ≈ 0.6 giờ, MSL ≈ 0.5 giờ, SMD ≈ 4.4 giờ (tổng 15 lần chạy mỗi dataset)

## AUROC (%), mean ± std trên 3 seed

Δ = biến thể − full. Số bài báo lấy từ Bảng 3, 4 (`configs/paper_results.csv`).

### Epoch cuối (`last`, cách chọn dùng cho bảng chính)

| Biến thể | PSM | Δ nhóm | Δ bài báo | MSL | Δ nhóm | Δ bài báo | SMD | Δ nhóm | Δ bài báo |
|---|---|---|---|---|---|---|---|---|---|
| full (E2) | 64.87 ± 3.81 | | | 67.60 ± 0.93 | | | 77.88 ± 2.11 | | |
| w/o MSASB | 64.24 ± 3.59 | −0.63 | −0.2 | 68.51 ± 0.42 | +0.91 | −0.9 | 77.89 ± 2.11 | +0.01 | −0.2 |
| w/o MSASB(G) | 65.09 ± 4.94 | +0.22 | −0.7 | 68.10 ± 1.38 | +0.50 | −2.1 | 77.89 ± 2.11 | +0.01 | −0.1 |
| w/o MSASB(L) | 64.40 ± 4.12 | −0.47 | −0.1 | 67.74 ± 0.56 | +0.14 | −1.4 | 77.88 ± 2.11 | 0.00 | −0.1 |
| w/o α(τ) (τ = 0.25) | 63.92 ± 3.25 | −0.95 | −1.3 | 66.15 ± 1.71 | −1.45 | −4.6 | 77.67 ± 2.62 | −0.21 | −0.2 |
| w/o α(τ) & MPs | 61.53 ± 4.09 | −3.34 | −1.0 | 65.91 ± 1.66 | −1.69 | −4.9 | 77.95 ± 2.44 | +0.07 | −0.2 |

### Epoch tốt nhất trên test (`best_test`, như code gốc, bảng phụ)

| Biến thể | PSM | Δ nhóm | MSL | Δ nhóm | SMD | Δ nhóm |
|---|---|---|---|---|---|---|
| full (E2) | 74.62 ± 5.35 | | 70.92 ± 2.09 | | 79.41 ± 1.17 | |
| w/o MSASB | 74.75 ± 5.31 | +0.13 | 71.04 ± 1.89 | +0.12 | 79.42 ± 1.17 | +0.01 |
| w/o MSASB(G) | 74.62 ± 5.34 | 0.00 | 70.59 ± 2.30 | −0.33 | 79.42 ± 1.17 | +0.01 |
| w/o MSASB(L) | 74.63 ± 5.34 | +0.01 | 70.19 ± 2.46 | −0.73 | 79.42 ± 1.17 | +0.01 |
| w/o α(τ) (τ = 0.25) | 75.97 ± 5.61 | +1.35 | 67.12 ± 1.00 | −3.80 | 79.58 ± 1.27 | +0.17 |
| w/o α(τ) & MPs | 75.92 ± 3.48 | +1.30 | 67.76 ± 0.90 | −3.16 | 79.66 ± 1.32 | +0.25 |

## Nhận xét

1. **MSASB gần như không có tác dụng trong lần tái hiện này.** Cả ba biến thể bỏ MSASB lệch full dưới 1 điểm,
   nhỏ hơn std giữa các seed, và dấu không ổn định (MSL `last` còn tăng khi bỏ MSASB). Bài báo cũng chỉ báo mức
   giảm 0.1–2.1 điểm, nên kết luận "MSASB giúp" không được xác nhận nhưng cũng không bị bác bỏ rõ.
2. **Trên SMD, mọi biến thể MSASB cho kết quả trùng full.** So từng epoch với E2, AUROC lệch tối đa 0.03 điểm,
   ngang mức nhiễu của GPU (≈ 0.02 điểm, xem `pipeline.md`). Lý do có thể là learning rate SMD là `1e-6`: mô hình
   gần như không rời điểm khởi tạo, và vì tham số MSASB vẫn được tạo trong mọi biến thể (mục 12
   `code_vs_paper.md`) nên khởi tạo giống nhau. Ablation trên SMD vì vậy không cho thông tin; số bài báo trên SMD
   (lệch 0.1–0.2) cũng khớp với điều này.
3. **Temperature cố định có tác động rõ nhất, chủ yếu trên MSL.** τ = 0.25 làm giảm 1.45 (`last`) và 3.80
   (`best_test`) điểm, cùng chiều với bài báo (−4.6). Bỏ thêm multi-head (P = 1) không giảm thêm trên MSL, khác
   bài báo. Lưu ý mục 6 `code_vs_paper.md`: τ thích nghi không được học, nên đây là so sánh "τ ngẫu nhiên theo
   mẫu" với "τ hằng số", không phải "τ học được" với "τ hằng số".
4. **PSM khó đọc.** `best_test` trên PSM rơi vào epoch 1–8 và std ≈ 5 điểm; với `best_test`, τ cố định còn cao
   hơn full 1.3 điểm. Với `last`, bỏ temperature và multi-head giảm 3.34 điểm, nhưng vẫn trong khoảng 1 std.
5. **Thứ tự bài báo chỉ khớp một phần.** Ở Bảng 4 (temperature) chiều giảm khớp trên MSL và PSM (`last`). Ở
   Bảng 3 (MSASB), chênh lệch của nhóm nằm trong nhiễu nên không tái hiện được thứ tự giữa G và L.

Với 3 seed, không chênh lệch nào ở trên vượt rõ std, nên báo cáo nên trình bày là "xu hướng", không phải kết luận
có ý nghĩa thống kê.
