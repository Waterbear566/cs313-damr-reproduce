"""ALOCC theo cài đặt trong repo MTGFlow (zqhang/MTGFLOW@f8cb9ea, models/GAN.py), nối vào pipeline chung.

Chạy từ gốc repo:
    python baselines/ALOCC/run.py --dataset PSM --seed 0

Giữ đúng code gốc (GAN.py train_model / train_single_epoch / test_single_epoch): R_Net và D_Net width 16,
nhiễu đầu vào std 0.2, Adam lr 1e-3 không weight decay, không scheduler, λ = 0.2, 40 epoch, batch 512,
mỗi batch cập nhật D trước rồi R. Mỗi epoch bật train mode cho cả hai mạng, chấm điểm ở eval mode.

Một lần train ghi HAI biến thể điểm (cùng mô hình, chỉ khác cách chấm):
  full       -D(x)     như test_single_epoch của repo MTGFlow (GAN.py:325-329)
  score_DRx  -D(R(x))  như bài báo ALOCC gốc (Sabokrou et al., 2018), R không thêm nhiễu khi chấm
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from baselines._mtgflow_ref.common_run import already_recorded, batches, load_tensors, set_seed  # noqa: E402
from baselines._mtgflow_ref.networks import D_Loss, D_Net, R_Loss, R_Net  # noqa: E402
from common.results import record_epoch  # noqa: E402

MODEL_NAME = "ALOCC"
VARIANTS = ("full", "score_DRx")


def score(r_net, d_net, x_all, device, through_r, batch_size=2048):
    r_net.eval()
    d_net.eval()
    out = []
    with torch.no_grad():
        for x in batches(x_all, batch_size):
            x = x.to(device)
            if through_r:
                x = r_net(x, noise=False)
            out.append(-d_net(x).reshape(-1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--exp", default="E3")
    args = ap.parse_args()

    csv_path = os.path.join(ROOT, "results", f"{args.exp}_{MODEL_NAME}.csv")
    if all(already_recorded(csv_path, MODEL_NAME, v, args.dataset, args.seed) for v in VARIANTS):
        print(f"BỎ QUA: {MODEL_NAME}/{args.dataset}/seed {args.seed} đã có trong {csv_path}")
        return
    if any(already_recorded(csv_path, MODEL_NAME, v, args.dataset, args.seed) for v in VARIANTS):
        raise SystemExit(f"CSV có một phần lượt {args.dataset}/seed {args.seed}: xóa các dòng đó rồi chạy lại")
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sp, X = load_tensors(args.dataset, args.data_path, args.seed)
    run_dirs = {v: os.path.join(ROOT, "checkpoints", args.exp, MODEL_NAME, v, f"{args.dataset}_seed_{args.seed}")
                for v in VARIANTS}
    for d in run_dirs.values():
        os.makedirs(d, exist_ok=True)

    r_net = R_Net(in_channels=sp.n_sensor, n_channels=16).to(device)
    d_net = D_Net(in_resolution=60, in_channels=sp.n_sensor, n_channels=16).to(device)
    optim_r = torch.optim.Adam(r_net.parameters(), lr=0.001)
    optim_d = torch.optim.Adam(d_net.parameters(), lr=0.001)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        r_net.train()
        d_net.train()
        sums, n = {"rec_loss": 0.0, "gen_loss": 0.0, "dis_loss": 0.0}, 0
        for x in batches(X["train"], args.batch_size, shuffle=True, drop_last=True):
            x_real = x.to(device)
            # train_single_epoch, GAN.py:255-268
            x_fake = r_net(x_real)
            d_net.zero_grad()
            dis_loss = D_Loss(d_net, x_real, x_fake)
            dis_loss.backward()
            optim_d.step()
            r_net.zero_grad()
            r_metrics = R_Loss(d_net, x_real, x_fake, 0.2)
            r_metrics["L_r"].backward()
            optim_r.step()
            for k in ("rec_loss", "gen_loss"):
                sums[k] += r_metrics[k].item() * len(x)
            sums["dis_loss"] += dis_loss.item() * len(x)
            n += len(x)
        train_s = time.time() - t0
        msg = [f"{args.dataset} seed {args.seed} epoch {epoch:02d}: D {sums['dis_loss'] / n:.4f} "
               f"G {sums['gen_loss'] / n:.4f} rec {sums['rec_loss'] / n:.4f}"]
        for v in VARIANTS:
            t1 = time.time()
            through_r = v == "score_DRx"
            test_s = score(r_net, d_net, X["test"], device, through_r)
            val_s = score(r_net, d_net, X["val"], device, through_r)
            m = record_epoch(run_dirs[v], csv_path, model=MODEL_NAME, variant=v, dataset=args.dataset,
                             seed=args.seed, epoch=epoch, test_scores=test_s, test_labels=sp.test_labels,
                             val_scores=val_s, runtime_s=train_s + time.time() - t1)
            msg.append(f"{v} AUROC {m['auroc']:.4f}")
        print(" | ".join(msg), flush=True)
        state = {"r_net": r_net.state_dict(), "d_net": d_net.state_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(state, os.path.join(run_dirs["full"], "model_last.pt"))


if __name__ == "__main__":
    main()
