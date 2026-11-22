# CS313 – Tái hiện DAMR (Nhóm 5)

Repo của Nhóm 5, môn **CS313 – Khai thác dữ liệu**, Trường ĐH Công nghệ Thông tin (UIT).

Nhóm tái hiện các thí nghiệm của bài báo:

> **DAMR: Dual Adaptive Multi-Head Representation Learning for Multivariate Time Series Anomaly Detection** (KDD 2026)
> Code tác giả: [wyn404/DAMR](https://github.com/wyn404/DAMR)

Nhóm chạy lại **đúng code công bố**, không chỉnh sửa hay cải tiến mô hình. Mọi chỗ phải sửa code để chạy được hoặc để ghi log đều được liệt kê ở mục [Khác biệt giữa bài báo và code](#khác-biệt-giữa-bài-báo-và-code).

---

## Câu hỏi nghiên cứu

1. Chạy lại code của tác giả có ra kết quả như bài báo không? (AUROC bài báo: PSM 78.6, MSL 72.5, SMD 79.3)
   Code gốc chọn epoch có điểm test cao nhất, nên nhóm báo cáo cả hai con số: **epoch code gốc chọn** và **epoch cuối**.
2. Bảng ablation (Bảng 3, 4) có tái hiện được không?
3. DAMR có còn đứng đầu so với 9 phương pháp khác (Bảng 2) và chống nhiễu tốt (Hình 5) như bài báo nói không?
4. (Phụ) Nếu đo bằng thước đo khác thì kết luận có thay đổi không?

---

## Thành viên

| MSSV | Họ và tên | Phụ trách trong repo |
|---|---|---|
| 24520635 | Vũ Cao Mạnh Hùng | Quản lý repo; E2 (DAMR), E4 (ablation) |
| 24520748 | Ngô Hoàng Khang | E1 (thống kê dataset), E3a (nhóm A), hàm metric chung, demo |
| 24530556 | Nguyễn Huy Hoàng | E3b (nhóm B) |
| 24520772 | Võ Huy Khang | Chạy cùng E2/E4; E5 (nhiễu), E6 (t-SNE); đối chiếu số liệu |
| 24520508 | Nguyễn Trung Hiếu | Script vẽ hình và bảng từ `results/` |
| 24520411 | Nguyễn Hồng Gấm | Báo cáo, slide, sơ đồ kiến trúc |
| 24520543 | Lê Minh Hoàng | Báo cáo, slide, câu hỏi phản biện |

---

## Cấu trúc thư mục

```
.
├── DAMR/                 # code gốc của tác giả (ghi rõ commit hash bên dưới)
├── baselines/            # code 9 phương pháp so sánh, mỗi phương pháp một thư mục con
├── configs/              # file cấu hình cho từng thí nghiệm / biến thể
├── data/                 # dữ liệu thô và đã xử lý (KHÔNG commit)
├── scripts/              # script chạy thí nghiệm và tiền xử lý
├── common/               # hàm dùng chung: chia dữ liệu, cửa sổ, metric
├── results/              # file CSV kết quả (có commit)
├── logs/                 # log huấn luyện (KHÔNG commit file lớn)
├── checkpoints/          # checkpoint mô hình (KHÔNG commit)
├── figures/              # hình và bảng sinh từ results/ (PDF/SVG)
├── docs/                 # ghi chú: khác biệt code–paper, giải thích kết quả lệch
└── README.md
```

Phiên bản code gốc: `wyn404/DAMR` @ commit [`faaea34`](https://github.com/wyn404/DAMR/tree/faaea34d7468ede43a101858722377abca72d956) (15/08/2026). Thư mục `DAMR/` là bản sao nguyên trạng của commit này; mọi chỉnh sửa sau đó đều qua Pull Request và được ghi trong `docs/code_vs_paper.md`.

> Repo tác giả không kèm file LICENSE. Nhóm chỉ dùng code cho mục đích học tập trong môn học.

---

## Cài đặt môi trường

```bash
git clone https://github.com/<tên-tài-khoản>/<tên-repo>.git
cd <tên-repo>

conda create -n damr python=3.8 -y
conda activate damr
pip install -r requirements.txt
```

`requirements.txt` được lập từ các import trong `DAMR/` (repo tác giả không có file này). Nếu cài PyTorch bản CUDA, cài `torch` theo hướng dẫn trên pytorch.org trước rồi mới chạy lệnh trên. Các baseline có thể cần môi trường riêng; nếu có, ghi lại ở `baselines/<tên>/README.md`.

| Thành phần | Phiên bản |
|---|---|
| Python | 3.8 (theo README tác giả) |
| PyTorch | 2.1.0 (theo README tác giả) |
| CUDA | `<điền khi cài>` |
| GPU dùng để chạy | `<điền khi cài>` |

---

## Dữ liệu

| Dataset | Truy cập | Thứ tự chạy |
|---|---|---|
| PSM | Công khai | 1 – dùng để thử và chạy mọi thí nghiệm |
| MSL | Công khai | 2 |
| SMD | Công khai | 3 |
| SWaT | Xin quyền từ iTrust | 4 – nếu được cấp kịp |
| WADI | Xin quyền từ iTrust | 5 – nếu được cấp kịp |

Tác giả lấy dữ liệu từ [Time-Series-Library (TimesNet)](https://github.com/thuml/Time-Series-Library). Script gốc đọc dữ liệu ở `DAMR/data/psm`, `DAMR/data/msl`, `DAMR/data/smd` (chữ thường). Thư mục `data/` không được commit lên GitHub.

**Tiền xử lý thống nhất cho mọi phương pháp:**
- Cùng file dữ liệu và cùng cách chuẩn hóa.
- Cửa sổ dài **60**, bước **10** (mặc định `--win_size 60 --step 10` trong `DAMR/main.py`).
- Nhãn cửa sổ = 1 nếu cửa sổ có ít nhất một điểm bất thường.
- Validation = **20%** của tập train, chỉ chứa dữ liệu bình thường. Code gốc lấy một đoạn liên tục ở vị trí ngẫu nhiên (`DAMR/data_loader.py`, hàm `get_loader_segment`), các baseline cần chia theo đúng cách này.

---

## Danh sách thí nghiệm

| Mã | Thí nghiệm | Trong bài báo | Số lần train | Người làm |
|---|---|---|---|---|
| E1 | Thống kê dataset: số điểm, số chiều, tỉ lệ bất thường | Bảng 1 | 0 | Hoàng Khang |
| E2 | DAMR như code công bố, lưu điểm mỗi epoch và checkpoint | Bảng 2 (dòng DAMR) | 9 (3 dataset × 3 seed) | Mạnh Hùng |
| E3a | Nhóm A: DeepSVDD, ALOCC, USAD, GANF, MTGFlow | Bảng 2 | 45 | Hoàng Khang |
| E3b | Nhóm B: Anomaly Transformer, DCdetector, TFMAE, MtsCID | Bảng 2 | 36 | Huy Hoàng |
| E4 | Ablation: w/o MSASB, w/o MSASB(G), w/o MSASB(L), τ = 0.25 cố định, bỏ cả temperature và multi-head | Bảng 3, 4 | 45 (5 × 3 × 3) | Mạnh Hùng |
| E5 | Nhiễu Gaussian trên tập test PSM | Hình 5 | 0 (dùng checkpoint) | Huy Khang |
| E6 | t-SNE trên MSL; độ tương đồng giữa các head trên PSM | Hình 3, Bảng 5 | 0 (dùng checkpoint) | Huy Khang |

### Chạy thí nghiệm

Chạy DAMR bằng script gốc của tác giả (đối số là ID của GPU):

```bash
cd DAMR
bash run/run_PSM.sh 0
```

Lưu ý: script gốc chạy nền và đẩy toàn bộ output vào `/dev/null`, chỉ chạy seed 16. Để có log theo yêu cầu của nhóm, chạy trực tiếp:

```bash
cd DAMR
python3 main.py --dataset=PSM --data_path=./data/psm --seed=0 --device=cuda:0 --lr=1e-4 \
  2>&1 | tee ../logs/E2_PSM_seed0.log
```

Learning rate tác giả dùng cho từng dataset: PSM `1e-4`, MSL `5e-4`, SMD `1e-6`, SWaT `5e-5`, WADI `1e-4`. Về ablation (E4), `main.py` có sẵn cờ `--MSASB`, `--adaptive_filter` và `--num_heads`. Các biến thể w/o MSASB(G), w/o MSASB(L) và τ cố định chưa có cờ riêng, nên cần xem `model.py` để thêm.

> Script chạy hàng loạt cho E2/E4 (`scripts/`) sẽ được cập nhật sau.

Seed dùng chung cho mọi phương pháp: `0, 1, 2`. Seed mặc định của tác giả là 16.

---

## Quy ước kết quả

Mọi kết quả ghi vào CSV trong `results/` với **đúng các cột sau**:

```
model,variant,dataset,seed,noise,epoch,auroc,auprc,f1,flagged_rate,runtime_s
```

| Cột | Ý nghĩa |
|---|---|
| `model` | Tên phương pháp, ví dụ `DAMR`, `MTGFlow` |
| `variant` | `full` hoặc tên biến thể ablation |
| `dataset` | `PSM`, `MSL`, `SMD`, … |
| `seed` | 0, 1, 2 |
| `noise` | Độ lệch chuẩn nhiễu thêm vào test, `0` nếu không có |
| `epoch` | Epoch được đánh giá |
| `auroc`, `auprc`, `f1` | Metric tính bằng hàm chung trong `common/` |
| `flagged_rate` | Tỉ lệ cửa sổ bị gán bất thường |
| `runtime_s` | Thời gian chạy (giây) |

**Thước đo:** AUROC theo cửa sổ (như bài báo) và AUPRC là chính, F1 là phụ. Báo cáo mean ± std trên 3 seed.

**Quy tắc:**
- Mọi bảng và hình được sinh bằng script từ `results/`, xuất PDF hoặc SVG. **Không nhập số bằng tay.**
- Mọi lần chạy phải lưu log vào `logs/` và checkpoint vào `checkpoints/`, vì E5 và E6 dùng lại checkpoint của E2, E3, E4.

---

## Quy tắc so sánh công bằng

1. Cùng dữ liệu, cùng chuẩn hóa, cùng cửa sổ, cùng nhãn và cùng hàm metric cho mọi phương pháp.
2. Giữ siêu tham số mặc định của từng repo, cố định số epoch, **không tinh chỉnh dựa trên tập test**.
3. Mọi phương pháp, kể cả DAMR, chạy 3 seed (bài báo chạy 10 lần).
4. Bảng kết quả đặt số của nhóm cạnh số bài báo. Chỗ nào lệch hơn độ lệch chuẩn bài báo báo cáo thì ghi nguyên nhân có thể vào `docs/`.

---

## Khác biệt giữa bài báo và code

Danh sách đầy đủ ở [`docs/code_vs_paper.md`](docs/code_vs_paper.md). Các điểm đã thấy khi đọc code:

- Code gốc đánh giá trên tập **test** sau mỗi epoch và lưu checkpoint có AUROC test cao nhất; bước đánh giá trên validation bị comment lại.
- Khi chấm điểm kNN, code lấy ngẫu nhiên 10% đặc trưng của tập train làm ngân hàng so sánh, nên điểm có thêm một nguồn ngẫu nhiên.
- `<bổ sung sau khi đối chiếu với paper>`

Tham khảo thêm repo DAMR-reproduce, vốn đã nêu một số khác biệt trước nhóm.

---

## Quy trình làm việc với Git

- Không push thẳng lên `main`; mỗi người làm trên nhánh riêng, ví dụ `e2-damr`, `e3a-baselines`, `e3b-baselines`, `e5-noise`.
- Mở Pull Request để review chéo trước khi merge:
  - Mạnh Hùng review code của Hoàng Khang và Huy Khang.
  - Hoàng Khang review code của Mạnh Hùng và Huy Hoàng.
- Commit message ngắn gọn, ghi mã thí nghiệm, ví dụ: `E2: lưu điểm test mỗi epoch`.
- Không commit `data/`, `checkpoints/` và log lớn (đã có trong `.gitignore`). Checkpoint chia sẻ qua `<Google Drive / nơi lưu trữ chung>`.

---

## Trích dẫn

Nếu dùng lại repo này, vui lòng trích dẫn bài báo gốc DAMR (KDD 2026) và repo của tác giả [wyn404/DAMR](https://github.com/wyn404/DAMR). Code của từng baseline thuộc về tác giả tương ứng, xem trong `baselines/`.
