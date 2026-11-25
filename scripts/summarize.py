"""Gộp CSV kết quả theo từng lần chạy thành bảng mean ± std trên các seed, đặt cạnh số bài báo.

Cách dùng (từ gốc repo):
    python scripts/summarize.py results/E2_DAMR.csv [results/E3_*.csv ...] --out results/summary.csv
    python scripts/summarize.py results/E2_DAMR.csv --seeds 16 17 18      # chỉ lấy các seed này

Mỗi lần chạy (model, variant, dataset, noise, seed) được rút gọn thành HAI dòng:
  best_test  epoch có AUROC test cao nhất, đúng cách code gốc DAMR chọn checkpoint (epoch ĐẦU TIÊN đạt max)
  last       epoch cuối cùng, không nhìn tập test
Sau đó lấy mean ± std (độ lệch chuẩn mẫu, ddof=1) trên các seed. Metric nhân 100 cho giống bài báo.
Số bài báo lấy từ configs/paper_results.csv.
"""
import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS = ["auroc", "auprc", "f1", "flagged_rate"]
KEY = ["model", "variant", "dataset", "noise"]


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def read_rows(paths):
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["epoch"] = int(r["epoch"])
                r["noise"] = r.get("noise") or "0"
                rows.append(r)
    return rows


def pick_epochs(rows):
    """{(model, variant, dataset, noise): {'best_test': {seed: row}, 'last': {seed: row}}}"""
    runs = defaultdict(list)
    for r in rows:
        runs[tuple(r[k] for k in KEY) + (r["seed"],)].append(r)
    out = defaultdict(lambda: {"best_test": {}, "last": {}})
    warnings = []
    for key, rs in runs.items():
        rs.sort(key=lambda r: r["epoch"])
        epochs = [r["epoch"] for r in rs]
        if len(set(epochs)) != len(epochs):
            warnings.append(f"{key}: trùng epoch {epochs} (chạy lại mà không xóa CSV cũ?)")
        best = None
        for r in rs:  # epoch đầu tiên có AUROC lớn nhất, như `if best_auc < test_auc` trong train.py
            if best is None or _num(r["auroc"]) > _num(best["auroc"]):
                best = r
        out[key[:-1]]["best_test"][key[-1]] = best
        out[key[:-1]]["last"][key[-1]] = rs[-1]
    return out, warnings


def read_paper(path):
    paper = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                paper[(r["model"], r["variant"], r["dataset"], r["metric"])] = float(r["value"])
    return paper


def summarize(picked, paper):
    table = []
    for key in sorted(picked):
        model, variant, dataset, noise = key
        for sel in ("best_test", "last"):
            by_seed = picked[key][sel]
            row = {"model": model, "variant": variant, "dataset": dataset, "noise": noise,
                   "selection": sel, "n_seeds": len(by_seed),
                   "seeds": " ".join(sorted(by_seed)),
                   "epochs": " ".join(str(by_seed[s]["epoch"]) for s in sorted(by_seed))}
            for m in METRICS:
                vals = [_num(r[m]) for r in by_seed.values()]
                vals = [100 * v for v in vals if v is not None]
                row[f"{m}_mean"] = round(statistics.mean(vals), 2) if vals else ""
                row[f"{m}_std"] = round(statistics.stdev(vals), 2) if len(vals) > 1 else ""
            p = paper.get((model, variant, dataset, "auroc")) if noise in ("0", "0.0") else None
            row["paper_auroc"] = "" if p is None else p
            row["diff_auroc"] = "" if p is None or row["auroc_mean"] == "" else round(row["auroc_mean"] - p, 2)
            table.append(row)
    return table


def print_markdown(table):
    print("| model | variant | dataset | noise | chọn epoch | seed | AUROC | AUPRC | F1 | bài báo AUROC | lệch |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in table:
        def ms(m):
            if r[f"{m}_mean"] == "":
                return "—"
            return f"{r[f'{m}_mean']:.2f}" + (f" ± {r[f'{m}_std']:.2f}" if r[f"{m}_std"] != "" else "")
        print(f"| {r['model']} | {r['variant']} | {r['dataset']} | {r['noise']} | {r['selection']} | "
              f"{r['n_seeds']} | {ms('auroc')} | {ms('auprc')} | {ms('f1')} | {r['paper_auroc'] or '—'} | "
              f"{r['diff_auroc'] if r['diff_auroc'] != '' else '—'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", help="CSV kết quả; bỏ trống = mọi results/E*.csv")
    ap.add_argument("--paper", default=os.path.join(ROOT, "configs", "paper_results.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "summary.csv"))
    ap.add_argument("--seeds", nargs="+", help="chỉ tính các seed này, vd. --seeds 16 17 18 (mặc định: mọi seed)")
    args = ap.parse_args()

    paths = args.csv or sorted(glob.glob(os.path.join(ROOT, "results", "E*.csv")))
    if not paths:
        raise SystemExit("Không có file kết quả nào trong results/")
    rows = read_rows(paths)
    if args.seeds:
        rows = [r for r in rows if r["seed"] in set(args.seeds)]
        if not rows:
            raise SystemExit(f"Không có dòng nào với seed {args.seeds}")
    picked, warnings = pick_epochs(rows)
    table = summarize(picked, read_paper(args.paper))
    for w in warnings:
        print("CẢNH BÁO:", w)
    for r in table:
        if r["n_seeds"] < 3:
            print(f"CẢNH BÁO: {r['model']}/{r['variant']}/{r['dataset']} mới có {r['n_seeds']} seed")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print_markdown(table)
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
