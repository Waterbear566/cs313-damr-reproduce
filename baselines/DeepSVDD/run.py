"""DeepSVDD theo cài đặt trong repo MTGFlow (zqhang/MTGFLOW@f8cb9ea, models/DeepSAD.py), nối vào pipeline chung.

Chạy từ gốc repo:
    python baselines/DeepSVDD/run.py --dataset PSM --seed 0
    python baselines/DeepSVDD/run.py --dataset PSM --seed 0 --variant bn_train_each_epoch

Giữ đúng code gốc: encoder CNN của CNNAE (width 16), không pretrain, tâm c tính một lần ở eval mode
(kẹp ±0.1, lấy trung bình theo thời gian), loss = ||mean_t(z) - c||², Adam lr 1e-4, weight decay 5e-4,
MultiStepLR milestone 20 (gọi scheduler.step() ĐẦU mỗi epoch như bản gốc), 20 epoch, batch 512.
Điểm bất thường = ||mean_t(z) - c||².

Biến thể:
  full                 Như bản gốc: net.train() chỉ gọi MỘT lần trước vòng lặp; chấm test sau mỗi epoch đặt
                       net.eval() và không bật lại, nên từ epoch 2 BatchNorm chạy ở eval mode (DeepSAD.py:101,152).
  bn_train_each_epoch  Gọi net.train() đầu mỗi epoch (cách bàn giao E3a của Hoàng Khang).
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from baselines._mtgflow_ref.common_run import already_recorded, batches, load_tensors, set_seed  # noqa: E402
from baselines._mtgflow_ref.networks import CNNAE  # noqa: E402
from common.results import record_epoch  # noqa: E402

MODEL_NAME = "DeepSVDD"
VARIANTS = ("full", "bn_train_each_epoch")


def init_center_c(net, x_train, batch_size, device, eps=0.1):
    """DeepSAD.py:176-198."""
    n_samples, c = 0, 0.0
    net.eval()
    with torch.no_grad():
        for x in batches(x_train, batch_size, shuffle=True, drop_last=True):
            outputs = net(x.to(device))          # (B, 64, 8)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
    c /= n_samples
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c.mean(dim=-1)


def score(net, c, x_all, device, batch_size=2048):
    net.eval()
    out = []
    with torch.no_grad():
        for x in batches(x_all, batch_size):
            z = net(x.to(device)).mean(dim=-1)
            out.append(torch.sum((z - c) ** 2, dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--variant", default="full", choices=VARIANTS)
    ap.add_argument("--exp", default="E3")
    args = ap.parse_args()

    csv_path = os.path.join(ROOT, "results", f"{args.exp}_{MODEL_NAME}.csv")
    if already_recorded(csv_path, MODEL_NAME, args.variant, args.dataset, args.seed):
        print(f"BỎ QUA: {MODEL_NAME}/{args.variant}/{args.dataset}/seed {args.seed} đã có trong {csv_path}")
        return
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sp, X = load_tensors(args.dataset, args.data_path, args.seed)
    run_dir = os.path.join(ROOT, "checkpoints", args.exp, MODEL_NAME, args.variant, f"{args.dataset}_seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    net = CNNAE(sp.n_sensor, 16).to(device).encoder
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
    c = init_center_c(net, X["train"], args.batch_size, device)
    net.train()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.variant == "bn_train_each_epoch":
            net.train()
        scheduler.step()  # đầu epoch, như DeepSAD.py:109
        total, n = 0.0, 0
        for x in batches(X["train"], args.batch_size, shuffle=True, drop_last=True):
            optimizer.zero_grad()
            z = net(x.to(device)).mean(dim=-1)
            loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
        m = record_epoch(run_dir, csv_path, model=MODEL_NAME, variant=args.variant, dataset=args.dataset,
                         seed=args.seed, epoch=epoch,
                         test_scores=score(net, c, X["test"], device), test_labels=sp.test_labels,
                         val_scores=score(net, c, X["val"], device), runtime_s=time.time() - t0)
        print(f"{args.dataset} seed {args.seed} {args.variant} epoch {epoch:02d}: loss {total / n:.4f} "
              f"lr {optimizer.param_groups[0]['lr']:.0e} AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f}", flush=True)
        torch.save({"model_state_dict": net.state_dict(), "c": c.cpu(), "epoch": epoch, "args": vars(args)},
                   os.path.join(run_dir, "model_last.pt"))


if __name__ == "__main__":
    main()
