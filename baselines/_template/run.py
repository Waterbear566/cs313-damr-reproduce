"""Mẫu chạy một baseline theo pipeline chung. Sao chép thư mục này thành baselines/<TênPhươngPháp>/.

Chạy từ gốc repo:
    python baselines/<Tên>/run.py --dataset PSM --data_path DAMR/data/psm --seed 0 --device cuda:0

Chỉ cần sửa 3 chỗ đánh dấu TODO: dựng mô hình, train một epoch, chấm điểm.
Không đổi cửa sổ, chuẩn hóa, cách chia validation hay hàm metric: những thứ đó lấy từ common/.
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from common.data import load_split  # noqa: E402
from common.results import record_epoch  # noqa: E402

MODEL_NAME = "TEMPLATE"  # TODO: tên phương pháp, ví dụ "USAD" (cột model trong CSV)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def score(model, dataset, args, device):
    """Trả về mảng điểm bất thường (n_cửa_sổ,), điểm càng cao càng bất thường, theo đúng thứ tự dataset."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)              # (batch, sensor, thời gian); cần (batch, thời gian, sensor) thì x.transpose(1, 2)
            out.append(model(x).cpu())    # TODO: thay bằng cách chấm điểm của phương pháp, một điểm cho mỗi cửa sổ
    return torch.cat(out).numpy().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=10)       # giữ mặc định của repo gốc baseline
    ap.add_argument("--batch_size", type=int, default=128)  # giữ mặc định của repo gốc baseline
    ap.add_argument("--variant", default="full")
    ap.add_argument("--exp", default="E3")                  # E3a / E3b, quyết định đường dẫn lưu
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sp = load_split(args.dataset, args.data_path, seed=args.seed)
    run_dir = os.path.join(ROOT, "checkpoints", args.exp, MODEL_NAME, f"{args.dataset}_seed_{args.seed}")
    csv_path = os.path.join(ROOT, "results", f"{args.exp}_{MODEL_NAME}.csv")
    os.makedirs(run_dir, exist_ok=True)

    model = torch.nn.Identity().to(device)  # TODO: dựng mô hình với sp.n_sensor sensor, cửa sổ 60
    train_loader = torch.utils.data.DataLoader(sp.train, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        for x, _ in train_loader:
            pass  # TODO: một bước huấn luyện (chỉ dùng dữ liệu train, không dùng sp.test)
        m = record_epoch(run_dir, csv_path, model=MODEL_NAME, variant=args.variant, dataset=args.dataset,
                         seed=args.seed, epoch=epoch,
                         test_scores=score(model, sp.test, args, device), test_labels=sp.test_labels,
                         val_scores=score(model, sp.val, args, device), runtime_s=time.time() - t0)
        print(f"epoch {epoch}: AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  F1 {m['f1']:.4f}")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "args": vars(args)},
                   os.path.join(run_dir, "model_last.pt"))


if __name__ == "__main__":
    main()
