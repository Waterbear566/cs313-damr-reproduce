# Khác biệt giữa bài báo và code công bố

Code gốc: `wyn404/DAMR` @ `faaea34d7468ede43a101858722377abca72d956`.

Mỗi mục ghi: chỗ trong code, code làm gì, bài báo nói gì, ảnh hưởng tới kết quả. Cột "Bài báo nói gì" cần đối chiếu với bản PDF.

| # | Chỗ trong code | Code làm gì | Bài báo nói gì | Ảnh hưởng |
|---|---|---|---|---|
| 1 | `train.py`, hàm `trainer` | Đánh giá trên tập test sau mỗi epoch, lưu checkpoint có AUROC test cao nhất và in `best Auc`. Phần đánh giá trên validation bị comment lại. | `<cần đối chiếu>` | Số báo cáo được chọn bằng tập test ("nhìn đáp án"). Nhóm báo cáo cả epoch tốt nhất theo test và epoch cuối. |
| 2 | `train.py`, hàm `model_evaluate` | Lấy ngẫu nhiên 10% đặc trưng train làm ngân hàng cho kNN (`sampling_ratio = 0.1`). | `<cần đối chiếu>` | Điểm bất thường dao động giữa các lần đánh giá, kể cả khi cùng một checkpoint. |
| 3 | `data_loader.py`, hàm `get_loader_segment` | Validation là một đoạn liên tục chiếm 20% train, vị trí bắt đầu ngẫu nhiên. | `<cần đối chiếu>` | Các baseline cần chia giống hệt để so sánh công bằng. |
| 4 | `run/run_*.sh` | Chỉ chạy seed 16, chạy nền, output đẩy vào `/dev/null`. | Bài báo báo cáo trung bình 10 lần chạy. | Không có log; nhóm chạy trực tiếp `main.py` với seed 0, 1, 2. |
| 5 | `main.py` (cờ dòng lệnh) | Có cờ `--MSASB`, `--adaptive_filter`, `--num_heads`; không có cờ cho MSASB(G), MSASB(L), τ cố định. | Bảng 3, 4 có các biến thể này. | Cần thêm cờ để chạy đủ E4; ghi rõ mọi chỗ sửa. |
