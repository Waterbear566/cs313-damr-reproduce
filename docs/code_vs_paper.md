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
| 6 | `train.py`, hàm `trainer` (dòng tạo `optimizer`) | Optimizer chỉ nhận `model.parameters()` và `A`; tham số của `AdaptiveTemperature` **không** được đưa vào. Đã kiểm tra: sau một epoch, trọng số `adaptive_temp` không đổi dù có gradient. Nhiệt độ τ do một MLP khởi tạo ngẫu nhiên sinh ra (chỉ BatchNorm cập nhật thống kê). | `<cần đối chiếu>`: bài báo mô tả τ "học được" cho từng head. | Ảnh hưởng trực tiếp E4 (biến thể τ = 0.25 cố định và bỏ temperature): so sánh là "τ ngẫu nhiên phụ thuộc mẫu" với "τ hằng số", không phải "τ học được" với "τ hằng số". |
| 7 | `model.py`, `DAMR.forward` | Nhận đối số `A` (ma trận n_sensor × n_sensor được tối ưu trong `train.py`) nhưng không dùng. | `<cần đối chiếu>` | Không ảnh hưởng kết quả; chỉ là code thừa. Ghi lại để khỏi nhầm khi đọc. |
| 8 | `augmentation.py`, hàm `permutation` | `ret[i] = pat[0, warp]`: với mẫu bị cắt đoạn, **mọi sensor bị thay bằng sensor 0 đã hoán vị** (đã kiểm tra bằng dữ liệu giả). Mẫu không bị cắt giữ nguyên. | `<cần đối chiếu>`: bài báo mô tả strong augmentation là permutation + jitter trên chuỗi đa biến. | View mạnh mất thông tin của các sensor khác; có thể giải thích một phần số lệch nếu bài báo dùng bản khác. Nhóm **không sửa**, chỉ ghi nhận. |
| 9 | `model.py`, `Multi_Scale_Spectral_Block.forward` | Đầu vào có dạng (batch, sensor, thời gian); `torch.fft.rfft(..., dim=1)` biến đổi theo **trục sensor**, còn trục thời gian (60) bị chia thành 6 head × 10 bước. | `<cần đối chiếu>`: bài báo nói MSASB học mẫu hình nhiều thang đo trong miền thời gian và tần số. | Nếu bài báo ý là FFT theo thời gian thì đây là khác biệt lớn về phương pháp; cần đọc kỹ mục Method trước khi kết luận. |
| 10 | `augmentation.py`, hàm `permutation` | **Nhóm sửa (tương thích):** numpy ≥ 1.24 báo lỗi khi `np.random.permutation` nhận list mảng dài khác nhau. Đổi sang hoán vị chỉ số. Đã kiểm tra trên 200 seed: cùng chuỗi số ngẫu nhiên nên cùng kết quả như numpy cũ. | — | Không đổi hành vi. Cần thiết vì Kaggle dùng Python/numpy mới hơn môi trường tác giả (Python 3.8). |
| 11 | `main.py`, `train.py` | **Nhóm thêm (ghi log):** cờ `--log_extra`, `--save_dir`, `--variant`, `--results_csv`. Khi bật: lưu điểm bất thường test mỗi epoch (`scores/epoch_XXX.npy`, `labels.npy`), checkpoint epoch cuối (`model_last.pt`), thêm khóa `epoch`, `test_auroc`, `adaptive_temp_state_dict`, `A`, `args` vào checkpoint, ghi CSV chung. Khi không truyền cờ, code chạy như bản gốc. | — | Không đổi mô hình, cách huấn luyện hay cách chọn checkpoint. `runtime_s` là thời gian của cả epoch (train + đánh giá trên test). |


## Ghi chú thêm khi chạy

- Điểm kNN dùng `np.random.choice` để lấy 10% ngân hàng đặc trưng (mục 2), nên nạp lại cùng checkpoint và chấm lại sẽ cho AUROC hơi khác. Điểm lưu trong `scores/` là điểm đúng lúc train, nên E5/E6 so sánh với điểm này thay vì chấm lại.
- Code không bật `torch.backends.cudnn.deterministic`, nên cùng seed trên GPU vẫn có thể lệch nhỏ giữa các lần chạy.
