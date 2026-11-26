# Tài nguyên cần tải

Cập nhật: 23/11/2026. Link đánh dấu **(chưa xác minh)** cần kiểm tra lại trước khi dùng.

## Code của tác giả

| Thứ | Link |
|---|---|
| DAMR (bản trong `DAMR/` là commit `faaea34`) | https://github.com/wyn404/DAMR |

Chưa tìm thấy bản arXiv của bài báo DAMR. Bản PDF lấy từ kỷ yếu KDD 2026 hoặc theo link trong README của tác giả.

## Dữ liệu

| Dataset | Cách lấy | Ghi chú |
|---|---|---|
| PSM, MSL, SMD | https://huggingface.co/datasets/thuml/Time-Series-Library | Bộ dữ liệu của Time-Series-Library, tác giả DAMR cũng lấy từ đây. Repo code: https://github.com/thuml/Time-Series-Library |
| PSM (nguồn gốc) | https://github.com/eBay/RANSynCoders (thư mục `data`) | Dự phòng |
| SMD (nguồn gốc) | https://github.com/NetManAIOps/OmniAnomaly (`ServerMachineDataset`) | Chia theo từng máy, phải ghép lại thành `.npy` |
| MSL (nguồn gốc) | https://github.com/khundman/telemanom | Phải tự ghép lại |
| SWaT, WADI | https://www.sutd.edu.sg/itrust/itrust-labs/datasets | Gửi đơn xin quyền, xử lý tới 3 ngày làm việc. Điều khoản: https://www.sutd.edu.sg/itrust/itrust-labs/datasets/terms-of-usage/ |

Lưu ý:
- Trên Hugging Face, `PSM/test_label.csv` đang hiện **0 byte**. Tải xong chạy `python scripts/check_data.py --root DAMR/data`. Nếu file rỗng thì lấy file này từ repo eBay.
- TSLib đặt tên thư mục chữ hoa (`PSM/`, `MSL/`, `SMD/`), còn code đọc chữ thường (`DAMR/data/psm`, `msl`, `smd`). Đổi tên sau khi tải.
- Theo điều khoản của iTrust, SWaT và WADI **không được chia sẻ lại**, kể cả trong nhóm qua Drive chung. Ai cần thì tự gửi đơn.
- Không commit `data/` lên GitHub (đã có trong `.gitignore`).

## Code của 9 baseline

| Nhóm | Phương pháp | Link |
|---|---|---|
| E3a | DeepSVDD | https://github.com/lukasruff/Deep-SVDD-PyTorch |
| E3a | ALOCC | https://github.com/khalooei/ALOCC-CVPR2018 **(chưa xác minh)** |
| E3a | USAD | https://github.com/manigalati/usad |
| E3a | GANF | https://github.com/EnyanDai/GANF |
| E3a | MTGFlow | https://github.com/zqhang/mtgflow |
| E3b | Anomaly Transformer | https://github.com/thuml/Anomaly-Transformer |
| E3b | DCdetector | https://github.com/DAMO-DI-ML/KDD2023-DCdetector |
| E3b | TFMAE | https://github.com/LMissher/TFMAE **(chưa xác minh)** |
| E3b | MtsCID | https://github.com/ilwoof/MtsCID |

Lưu ý:
- DeepSVDD và ALOCC vốn viết cho ảnh. Cần xem bài báo DAMR (hoặc repo DAMR-reproduce) đã chỉnh chúng thế nào cho chuỗi thời gian.
- MTGFlow yêu cầu Python 3.8.5 và PyTorch 1.7.1, có thể phải sửa cho tương thích.
- Trên Gitee có các bản sao TFMAE và MtsCID của tài khoản `wu-nil`. Đó không phải repo gốc, không dùng.
- Khi chép code baseline vào `baselines/<Tên>/`, ghi link repo gốc và commit vào `baselines/<Tên>/README.md` (xem [`baselines/README.md`](../baselines/README.md)).

## Môi trường

- Thư viện: [`requirements.txt`](../requirements.txt). Trên Kaggle cần cài thêm `faiss-cpu` và `thop`.
- PyTorch 2.1.0: https://pytorch.org/get-started/previous-versions/
