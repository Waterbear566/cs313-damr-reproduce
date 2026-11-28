# %% CELL 1: reference source and model definitions; no training
import ast
import gc
import hashlib
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Subset

SVDD_COMMIT = "f8cb9ea225fd7e64cfb4556dae3525accac285f5"
assert Path("/content/drive/MyDrive").is_dir(), "Mount Drive trước khi nạp source."
SVDD_CACHE = Path("/content/drive/MyDrive/CS313_Nhom5/source_cache") / ("DeepSVDD_" + SVDD_COMMIT)
SVDD_CACHE.mkdir(parents=True, exist_ok=True)
SVDD_HASHES = {
    "models/GAN.py": "2ba491d5e94e880660ae584d6cac045d6f85dea9a864f4f2fab8a71bc04b5e5c",
    "models/DeepSAD.py": "8dcfc3a40edfebb03c8ea3935caa4066358e283047f8696c1bcd08a481aac965",
    "train_other_model.py": "db2a383e95521c9ad69b2abf60848e1e90242c4088fc5ec68982783390f1ec11",
    "LICENSE": "23a15f9f18973b2376cb281cb3bba6fea79c80c1af79982a48920b40f3a25f4f",
}
for filename, expected in SVDD_HASHES.items():
    path = SVDD_CACHE / filename
    if not path.exists():
        response = requests.get(
            f"https://raw.githubusercontent.com/zqhang/MTGFLOW/{SVDD_COMMIT}/{filename}",
            timeout=(30, 120)
        )
        response.raise_for_status()
        assert hashlib.sha256(response.content).hexdigest() == expected, filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, filename

# Import exactly these network definitions, not the broken top-level training script.
source = (SVDD_CACHE / "models/GAN.py").read_text(encoding="utf-8")
names = {"ConvEncoder", "ConvDecoder", "CNNAE"}
nodes = [node for node in ast.parse(source).body
         if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
assert {node.name for node in nodes} == names
network_code = "import torch\nimport torch.nn as nn\n\n" + "\n\n".join(ast.unparse(node) for node in nodes) + "\n"
SVDD_NETWORK_NS = {}
exec(compile(network_code, "svdd_reference_network.py", "exec"), SVDD_NETWORK_NS)
SVDD_CNNAE = SVDD_NETWORK_NS["CNNAE"]
print("Đã nạp encoder CNN tham chiếu. Chưa train.")


# %% CELL 2: full single-run adapter
def run_deepsvdd(dataset_name, seed):
    assert dataset_name in ("PSM", "MSL", "SMD") and seed in (16, 17, 18)
    assert torch.cuda.is_available(), "Chưa có CUDA."
    assert "PSMWindowDataset" in globals(), "Chạy ô định nghĩa PSMWindowDataset trước."
    required_meta = "PSM_EXPECTED_SHA" if dataset_name == "PSM" else "EXTRA_DATA_REFERENCES"
    assert required_meta in globals(), f"Thiếu {required_meta}: chạy ô khai báo nguồn dữ liệu tương ứng trước."
    device = torch.device("cuda:0")
    root = Path("/content/drive/MyDrive/CS313_Nhom5/hoang_khang/E3a")
    data_dir = Path("/content/drive/MyDrive/CS313_Nhom5/data") / dataset_name
    required = ["completed.json", "run_config.json", "config.json", "metrics.csv",
                "scaler.joblib", "window_split.npz", "validation_scores.csv", "test_scores.csv"]
    # Latest completed compatible MTGFlow run; never select by its test score.
    reference = None
    for folder in sorted(root.glob(f"MTGFlow_{dataset_name}_seed{seed}_*"), reverse=True):
        if not all((folder / name).is_file() for name in required):
            continue
        cfg = json.loads((folder / "run_config.json").read_text())
        status = json.loads((folder / "completed.json").read_text())
        if (status.get("status") == "completed" and cfg.get("seed") == seed
                and cfg.get("dataset") == dataset_name and cfg.get("model") == "MTGFlow"
                and cfg.get("epochs") == 40
                and cfg.get("checkpoint_policy") == "last_epoch_fixed_in_advance"
                and cfg.get("protocol_id") == f"DAMR_public_code_{dataset_name}_v1"):
            reference = folder
            break
    assert reference is not None, f"Không tìm được MTGFlow hoàn tất cho {dataset_name}, seed{seed}."
    prep = json.loads((reference / "config.json").read_text())
    assert prep["dataset"] == dataset_name and prep["seed"] == seed
    assert prep["window"] == 60 and prep["stride"] == 10 and prep["val_ratio"] == 0.2
    assert prep["scaler_fit_on"] == "full_train_file_before_window_split"

    filenames = (["train.csv", "test.csv", "test_label.csv"] if dataset_name == "PSM"
                 else [f"{dataset_name}_{s}.npy" for s in ["train", "test", "test_label"]])
    data_manifest = {}
    for filename in filenames:
        path = data_dir / filename
        assert path.is_file(), path
        expected = (PSM_EXPECTED_SHA[filename] if dataset_name == "PSM"
                    else EXTRA_DATA_REFERENCES[f"{dataset_name}/{filename}"]["sha256"])
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected, path
        data_manifest[filename] = {"path": str(path), "sha256": expected}

    if dataset_name == "PSM":
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        label_df = pd.read_csv(data_dir / "test_label.csv")
        assert "timestamp" in str(train_df.columns[0]).lower()
        assert train_df.columns.tolist() == test_df.columns.tolist()
        assert label_df.shape[1] == 2
        np.testing.assert_array_equal(test_df.iloc[:, 0], label_df.iloc[:, 0])
        raw_train = np.nan_to_num(train_df.values[:, 1:])
        raw_test = np.nan_to_num(test_df.values[:, 1:])
        labels = label_df.iloc[:, 1].to_numpy()
    else:
        raw_train = np.load(data_dir / f"{dataset_name}_train.npy", allow_pickle=False)
        raw_test = np.load(data_dir / f"{dataset_name}_test.npy", allow_pickle=False)
        raw_y = np.load(data_dir / f"{dataset_name}_test_label.npy", allow_pickle=False)
        assert raw_y.ndim == 1 or (raw_y.ndim == 2 and raw_y.shape[1] == 1)
        labels = raw_y.reshape(-1)
    assert raw_train.shape == (prep["train_file_rows"], prep["n_features"])
    assert raw_test.shape == (prep["test_rows"], prep["n_features"])
    assert labels.shape == (len(raw_test),) and np.isin(labels, [0, 1]).all()
    assert np.isfinite(raw_train).all() and np.isfinite(raw_test).all()
    scaler = joblib.load(reference / "scaler.joblib")
    assert int(scaler.n_samples_seen_) == len(raw_train)
    train_z = scaler.transform(raw_train).astype(np.float32)
    test_z = scaler.transform(raw_test).astype(np.float32)
    full = PSMWindowDataset(train_z, window=60, stride=10)
    test_ds = PSMWindowDataset(test_z, labels=labels, window=60, stride=10)
    with np.load(reference / "window_split.npz", allow_pickle=False) as split:
        train_ids, val_ids = split["train_ids"].copy(), split["val_ids"].copy()
    all_ids = np.arange(len(full))
    start = random.Random(seed).randrange(int(len(full) * 0.8))
    expected_val = all_ids[start:start + int(len(full) * 0.2)]
    np.testing.assert_array_equal(val_ids, expected_val)
    np.testing.assert_array_equal(train_ids, np.r_[all_ids[:start], all_ids[start + len(val_ids):]])
    np.testing.assert_array_equal(np.sort(np.r_[train_ids, val_ids]), all_ids)
    truth_start = test_ds.starts
    cumulative = np.r_[0, np.cumsum(labels == 1)]
    truth = (cumulative[truth_start + 60] - cumulative[truth_start] > 0).astype(int)
    ref_test = pd.read_csv(reference / "test_scores.csv").sort_values("window_id")
    ref_val = pd.read_csv(reference / "validation_scores.csv").sort_values("window_id")
    np.testing.assert_array_equal(ref_test.window_id, np.arange(len(test_ds)))
    np.testing.assert_array_equal(ref_test.start, truth_start)
    np.testing.assert_array_equal(ref_test.label, truth)
    np.testing.assert_array_equal(ref_val.window_id, np.sort(val_ids))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = {"in_channels": prep["n_features"], "n_channels": 16, "kernel_size": 5}
    # Construct the complete source CNNAE before taking its encoder, as in DeepSVDD.__init__.
    ae = SVDD_CNNAE(**model_config).to(device)
    net = ae.encoder
    del ae
    train_loader = DataLoader(Subset(full, train_ids.tolist()), batch_size=512,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(Subset(full, val_ids.tolist()), batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    epochs = 20
    config = {
        "model": "DeepSVDD", "variant": "mtgflow_repo_cnn_no_pretrain",
        "dataset": dataset_name, "seed": seed, "epochs": epochs, "batch_size": 512,
        "lr": 0.0001, "weight_decay": 0.0005,
        "lr_policy": "epoch1_19_1e-4_epoch20_1e-5_reference_pre_step_milestone20",
        "model_config": model_config, "center_eps": 0.1,
        "center_policy": "mean_feature_map_then_eps_clip_then_temporal_mean",
        "score": "squared_l2_of_temporally_averaged_encoder_output_to_fixed_c",
        "pretraining_epochs": 0, "reference_commit": SVDD_COMMIT,
        "protocol_id": prep["protocol_id"], "preprocessing_reference": str(reference),
        "checkpoint_policy": "last_epoch_fixed_in_advance",
        "auprc_definition": "average_precision", "threshold_policy": "not_selected",
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / f"DeepSVDD_{dataset_name}_seed{seed}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    for name in ["config.json", "scaler.joblib", "window_split.npz"]:
        shutil.copy2(reference / name, out / name)
    shutil.copytree(SVDD_CACHE, out / "upstream_source")
    (out / "network.py").write_text(network_code, encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")
    changes = {
        "architecture": "Unchanged ConvEncoder from source GAN.py; CNNAE constructor default width16. Caller size is undefined in upstream script, so exact paper width cannot be verified.",
        "pretraining": "Not called by upstream DeepSVDD entry point; not enabled here.",
        "loss_center_score": "Reference squared-distance loss, fixed c, epsilon clipping before time pooling.",
        "adapter": "Input squeeze(-1) is equivalent to upstream transpose+reshape; preserve batch axis instead of bare squeeze().",
        "training_mode": "Train mode each epoch. Upstream test() sets eval() without restoring train in following epochs; no per-epoch test calls here.",
        "scheduler": "Explicit LR values matching current PyTorch pre-epoch MultiStepLR milestone20; no scheduler-before-optimizer warning.",
        "evaluation": "Last epoch only, no test-based selection. Shared DAMR data/scaler/splits, not original MTGFlow loaders.",
    }
    (out / "adapter_notes.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
    print(f"DeepSVDD {dataset_name} seed{seed}: train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ds)}")
    print("20 epoch; CNN width16; no AE pretraining; source split:", reference.name)
    print("Output:", out, flush=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-4)
    torch.cuda.synchronize()
    started = time.perf_counter()
    net.eval()
    center_sum, center_count = None, 0
    with torch.no_grad():
        for x, _, _ in train_loader:
            maps = net(x.to(device).squeeze(-1))
            center_sum = maps.sum(0) if center_sum is None else center_sum + maps.sum(0)
            center_count += len(x)
    center_map = center_sum / center_count
    center_map[(center_map.abs() < 0.1) & (center_map < 0)] = -0.1
    center_map[(center_map.abs() < 0.1) & (center_map > 0)] = 0.1
    c = center_map.mean(-1).detach()
    assert c.shape == (64,) and torch.isfinite(c).all()
    torch.save(c.cpu(), out / "center_initial.pt")

    history = []
    for epoch in range(1, epochs + 1):
        net.train()
        lr = 1e-4 if epoch < 20 else 1e-5
        for group in optimizer.param_groups:
            group["lr"] = lr
        total, count = 0.0, 0
        for x, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            z = net(x.to(device).squeeze(-1)).mean(-1)
            loss = ((z - c) ** 2).sum(1).mean()
            assert torch.isfinite(loss).item()
            loss.backward()
            assert all(torch.isfinite(p.grad).all().item() for p in net.parameters() if p.grad is not None)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        history.append({"epoch": epoch, "train_loss": total / count, "lr": lr, "trained_windows": count})
        tmp = out / "checkpoint_last.tmp"
        torch.save({"epoch": epoch, "model_state_dict": net.state_dict(), "c": c.cpu(),
                    "optimizer_state_dict": optimizer.state_dict(), "model_config": model_config,
                    "run_config": config}, tmp)
        tmp.replace(out / "checkpoint_last.pt")
        pd.DataFrame(history).to_csv(out / "epoch_history.csv", index=False)
        message = f"DeepSVDD {dataset_name} seed{seed}: epoch {epoch:02d}/{epochs}, loss={total/count:.6f}"
        print(message, flush=True)
        with (out / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started

    eval_started = time.perf_counter()
    saved = torch.load(out / "checkpoint_last.pt", map_location=device, weights_only=True)
    assert saved["epoch"] == epochs
    fresh = SVDD_CNNAE(**saved["model_config"]).to(device).encoder
    fresh.load_state_dict(saved["model_state_dict"], strict=True)
    saved_c = saved["c"].to(device)
    net.eval()
    fresh.eval()
    probe = next(iter(test_loader))[0][:8].to(device).squeeze(-1)
    with torch.no_grad():
        torch.testing.assert_close(((net(probe).mean(-1) - c) ** 2).sum(1),
                                   ((fresh(probe).mean(-1) - saved_c) ** 2).sum(1))
    def collect(loader, base, expected_ids, split_name):
        frames = []
        with torch.no_grad():
            for x, y, idx in loader:
                z = fresh(x.to(device).squeeze(-1)).mean(-1)
                scores = ((z - saved_c) ** 2).sum(1)
                assert scores.shape == (len(x),) and torch.isfinite(scores).all()
                ids = idx.numpy()
                frames.append(pd.DataFrame({"split": split_name, "window_id": ids,
                    "start": base.starts[ids], "end_exclusive": base.starts[ids] + 60,
                    "label": y.numpy(), "score": scores.cpu().numpy()}))
        frame = pd.concat(frames, ignore_index=True).sort_values("window_id").reset_index(drop=True)
        np.testing.assert_array_equal(frame.window_id, np.sort(expected_ids))
        assert not frame.window_id.duplicated().any()
        frame.to_csv(out / f"{split_name}_scores.csv", index=False)
        return frame
    validation = collect(val_loader, full, val_ids, "validation")
    test = collect(test_loader, test_ds, np.arange(len(test_ds)), "test")
    np.testing.assert_array_equal(test.start, truth_start)
    np.testing.assert_array_equal(test.label, truth)
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_started
    row = {"model": "DeepSVDD", "variant": config["variant"], "dataset": dataset_name,
        "seed": seed, "noise": 0, "epoch": epochs,
        "auroc": float(roc_auc_score(truth, test.score)),
        "auprc": float(average_precision_score(truth, test.score)),
        "f1": np.nan, "flagged_rate": np.nan, "runtime_s": train_seconds + eval_seconds}
    pd.DataFrame([row]).to_csv(out / "metrics.csv", index=False)
    (out / "completed.json").write_text(json.dumps({"status": "completed",
        "checkpoint_policy": "last_epoch", "validation_windows": len(validation),
        "test_windows": len(test), "train_runtime_s": train_seconds,
        "evaluation_runtime_s": eval_seconds, "checkpoint_reload_matches": True,
        "f1_status": "not_computed_threshold_not_selected"}, indent=2), encoding="utf-8")
    print(pd.DataFrame([row]).to_string(index=False))
    print("HOÀN THÀNH:", out)
    del net, fresh, optimizer, saved, probe, maps, z, loss
    gc.collect()
    torch.cuda.empty_cache()
    return out
