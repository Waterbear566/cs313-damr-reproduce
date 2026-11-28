# %% CELL 1: pinned source, original architecture/loss, shared preprocessing
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

assert Path("/content/drive/MyDrive").is_dir(), "Mount Drive trước."
USAD_COMMIT = "e25af45c8e1c32783aed94fc7e5ab85effef2b1a"
USAD_CACHE = Path("/content/drive/MyDrive/CS313_Nhom5/source_cache") / ("USAD_" + USAD_COMMIT)
USAD_CACHE.mkdir(parents=True, exist_ok=True)
USAD_HASHES = {
    "usad.py": "eea96f59ff0e3d4b2b7de31b8b7d4dbe4c02b104e76fdc04fb2cf5575ca1a444",
    "USAD.ipynb": "a656a963a1fae67402f43c39b07abd4cdda28a414394bae38b63dce9930d477f",
    "LICENSE": "bc835a2b34b2961d2bfc449e9307745d85ba2ce9d774f4bf129238c5d1aa7fa7",
}
for filename, expected in USAD_HASHES.items():
    path = USAD_CACHE / filename
    if not path.exists():
        response = requests.get(f"https://raw.githubusercontent.com/manigalati/usad/{USAD_COMMIT}/{filename}", timeout=(30, 120))
        response.raise_for_status()
        assert hashlib.sha256(response.content).hexdigest() == expected, filename
        path.write_bytes(response.content)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, filename

usad_source = (USAD_CACHE / "usad.py").read_text(encoding="utf-8")
usad_nodes = [node for node in ast.parse(usad_source).body
              if isinstance(node, ast.ClassDef) and node.name in {"Encoder", "Decoder", "UsadModel"}]
assert {node.name for node in usad_nodes} == {"Encoder", "Decoder", "UsadModel"}
USAD_NETWORK_CODE = "import torch\nimport torch.nn as nn\n\n" + "\n\n".join(ast.unparse(n) for n in usad_nodes) + "\n"
USAD_NS = {}
exec(compile(USAD_NETWORK_CODE, "usad_network.py", "exec"), USAD_NS)
USAD_CLASS = USAD_NS["UsadModel"]


def load_usad_shared_data(dataset_name, seed):
    assert dataset_name in ("PSM", "MSL", "SMD") and seed in (16, 17, 18)
    assert "PSMWindowDataset" in globals(), "Thiếu lớp Dataset ở mục 8."
    root = Path("/content/drive/MyDrive/CS313_Nhom5/hoang_khang/E3a")
    data_dir = Path("/content/drive/MyDrive/CS313_Nhom5/data") / dataset_name
    required = ["completed.json", "run_config.json", "config.json", "scaler.joblib",
                "window_split.npz", "validation_scores.csv", "test_scores.csv"]
    reference = None
    for folder in sorted(root.glob(f"MTGFlow_{dataset_name}_seed{seed}_*"), reverse=True):
        if not all((folder / name).is_file() for name in required):
            continue
        cfg = json.loads((folder / "run_config.json").read_text())
        status = json.loads((folder / "completed.json").read_text())
        if (status.get("status") == "completed" and cfg.get("model") == "MTGFlow"
                and cfg.get("dataset") == dataset_name and cfg.get("seed") == seed
                and cfg.get("epochs") == 40
                and cfg.get("checkpoint_policy") == "last_epoch_fixed_in_advance"
                and cfg.get("protocol_id") == f"DAMR_public_code_{dataset_name}_v1"):
            reference = folder
            break
    assert reference is not None, f"Thiếu MTGFlow hoàn tất: {dataset_name}, seed{seed}"
    prep = json.loads((reference / "config.json").read_text())
    assert prep["dataset"] == dataset_name and prep["seed"] == seed
    assert prep["window"] == 60 and prep["stride"] == 10 and prep["val_ratio"] == 0.2
    assert prep["scaler_fit_on"] == "full_train_file_before_window_split"
    filenames = (["train.csv", "test.csv", "test_label.csv"] if dataset_name == "PSM"
                 else [f"{dataset_name}_{s}.npy" for s in ["train", "test", "test_label"]])
    manifest = {}
    meta_name = "PSM_EXPECTED_SHA" if dataset_name == "PSM" else "EXTRA_DATA_REFERENCES"
    assert meta_name in globals(), f"Thiếu {meta_name}: chạy ô khai báo nguồn dữ liệu trước."
    for filename in filenames:
        path = data_dir / filename
        expected = (PSM_EXPECTED_SHA[filename] if dataset_name == "PSM"
                    else EXTRA_DATA_REFERENCES[f"{dataset_name}/{filename}"]["sha256"])
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected, path
        manifest[filename] = {"path": str(path), "sha256": expected}
    if dataset_name == "PSM":
        tr = pd.read_csv(data_dir / "train.csv")
        te = pd.read_csv(data_dir / "test.csv")
        lab = pd.read_csv(data_dir / "test_label.csv")
        assert "timestamp" in str(tr.columns[0]).lower() and lab.shape[1] == 2
        assert tr.columns.tolist() == te.columns.tolist()
        np.testing.assert_array_equal(te.iloc[:, 0], lab.iloc[:, 0])
        raw_train, raw_test = np.nan_to_num(tr.values[:, 1:]), np.nan_to_num(te.values[:, 1:])
        labels = lab.iloc[:, 1].to_numpy()
    else:
        raw_train = np.load(data_dir / f"{dataset_name}_train.npy", allow_pickle=False)
        raw_test = np.load(data_dir / f"{dataset_name}_test.npy", allow_pickle=False)
        raw_labels = np.load(data_dir / f"{dataset_name}_test_label.npy", allow_pickle=False)
        assert raw_labels.ndim == 1 or (raw_labels.ndim == 2 and raw_labels.shape[1] == 1)
        labels = raw_labels.reshape(-1)
    assert raw_train.shape == (prep["train_file_rows"], prep["n_features"])
    assert raw_test.shape == (prep["test_rows"], prep["n_features"])
    assert labels.shape == (len(raw_test),) and np.isin(labels, [0, 1]).all()
    assert np.isfinite(raw_train).all() and np.isfinite(raw_test).all()
    scaler = joblib.load(reference / "scaler.joblib")
    assert scaler.__class__.__name__ == "StandardScaler"
    assert int(scaler.n_samples_seen_) == len(raw_train)
    train_z, test_z = scaler.transform(raw_train).astype(np.float32), scaler.transform(raw_test).astype(np.float32)
    assert np.isfinite(train_z).all() and np.isfinite(test_z).all()
    full = PSMWindowDataset(train_z, window=60, stride=10)
    test_ds = PSMWindowDataset(test_z, labels=labels, window=60, stride=10)
    with np.load(reference / "window_split.npz", allow_pickle=False) as split:
        train_ids, val_ids = split["train_ids"].copy(), split["val_ids"].copy()
    ids = np.arange(len(full))
    start = random.Random(seed).randrange(int(len(full) * 0.8))
    np.testing.assert_array_equal(val_ids, ids[start:start + int(len(full) * .2)])
    np.testing.assert_array_equal(train_ids, np.r_[ids[:start], ids[start + len(val_ids):]])
    np.testing.assert_array_equal(np.sort(np.r_[train_ids, val_ids]), ids)
    starts = test_ds.starts
    cumulative = np.r_[0, np.cumsum(labels == 1)]
    truth = (cumulative[starts + 60] - cumulative[starts] > 0).astype(int)
    ref_test = pd.read_csv(reference / "test_scores.csv").sort_values("window_id")
    ref_val = pd.read_csv(reference / "validation_scores.csv").sort_values("window_id")
    np.testing.assert_array_equal(ref_test.window_id, np.arange(len(test_ds)))
    np.testing.assert_array_equal(ref_test.start, starts)
    np.testing.assert_array_equal(ref_test.label, truth)
    np.testing.assert_array_equal(ref_val.window_id, np.sort(val_ids))
    ranges = {name: {"below_zero_fraction": float((x < 0).mean()),
                     "above_one_fraction": float((x > 1).mean())}
              for name, x in [("full_train", train_z), ("test", test_z)]}
    return reference, prep, full, test_ds, train_ids, val_ids, truth, manifest, ranges


def usad_flatten(x):
    # Dataset gives (B,F,W,1); upstream notebook flattens (B,W,F).
    return x.squeeze(-1).transpose(1, 2).reshape(len(x), -1)


def usad_two_steps(model, batch, n, optimizer1, optimizer2):
    # Keep exactly the optimizer/zero_grad order of upstream training().
    # Do NOT add zeroing/freezing/detaching that changes its gradient behavior.
    loss1, _ = model.training_step(batch, n)
    assert torch.isfinite(loss1).item()
    loss1.backward()
    assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    optimizer1.step()
    optimizer1.zero_grad()
    _, loss2 = model.training_step(batch, n)  # recompute after AE1 update
    assert torch.isfinite(loss2).item()
    loss2.backward()
    assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    optimizer2.step()
    optimizer2.zero_grad()
    return loss1.item(), loss2.item()


def usad_window_score(model, batch):
    w1 = model.decoder1(model.encoder(batch))
    w3 = model.decoder2(model.encoder(w1))
    return .5 * ((batch - w1) ** 2).mean(1) + .5 * ((batch - w3) ** 2).mean(1)

print("Đã nạp USAD: giữ Sigmoid/loss gốc; dùng StandardScaler chung. Chưa train.")


# %% CELL 2: official single-run runner
def run_usad(dataset_name, seed):
    assert torch.cuda.is_available(), "Chưa có CUDA."
    device = torch.device("cuda:0")
    reference, prep, full, test_ds, train_ids, val_ids, truth, manifest, ranges = load_usad_shared_data(dataset_name, seed)
    # Reference notebook example (SWaT): epochs100, batch7919, hidden_size100.
    # Keep those values a priori; group window60/stride10/split/scaler are adaptations.
    epochs, batch_size, hidden_size = 100, 7919, 100
    model_config = {"w_size": 60 * prep["n_features"], "z_size": 60 * hidden_size}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = USAD_CLASS(**model_config).to(device)
    optimizer1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=.001, weight_decay=0)
    optimizer2 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=.001, weight_decay=0)
    train_loader = DataLoader(Subset(full, train_ids.tolist()), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    val_loader = DataLoader(Subset(full, val_ids.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    config = {
        "model": "USAD", "variant": "shared_standard_sigmoid",
        "dataset": dataset_name, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "hidden_size": hidden_size, "model_config": model_config,
        "lr": .001, "weight_decay": 0, "alpha": .5, "beta": .5,
        "shuffle": False, "drop_last": False,
        "protocol_id": prep["protocol_id"], "preprocessing_reference": str(reference),
        "scaler": "StandardScaler", "decoder_activation": "Sigmoid_unchanged",
        "checkpoint_policy": "last_epoch_fixed_in_advance", "save_every_epochs": 10, "also_save_epoch1": True,
        "checkpoint_contents": "inference_only_no_optimizer_state_not_training_resume",
        "auprc_definition": "average_precision", "threshold_policy": "not_selected",
        "source_commit": USAD_COMMIT, "source_sha256": USAD_HASHES,
        "config_source": "USAD.ipynb SWaT example; not verified USAD configuration from DAMR paper",
        "runtime_definition": "training_including_epoch_validation_and_checkpoint_IO_plus_final_evaluation_excluding_preparation",
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
    }
    root = Path("/content/drive/MyDrive/CS313_Nhom5/hoang_khang/E3a")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / f"USAD_{dataset_name}_seed{seed}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    for name in ["config.json", "scaler.joblib", "window_split.npz"]:
        shutil.copy2(reference / name, out / name)
    shutil.copytree(USAD_CACHE, out / "upstream_source")
    (out / "network.py").write_text(USAD_NETWORK_CODE, encoding="utf-8")
    notes = {
        "architecture_loss_score": "Original Encoder/Decoder/UsadModel definitions, original training_step, alpha=beta=.5 score. No decoder activation/loss changes.",
        "optimizer_order": "Preserve upstream two Adam steps and zero_grad after each respective step, including cross-decoder residual gradients. Recompute both losses after AE1 step.",
        "preprocessing_exception": "Intentionally StandardScaler to match group DAMR protocol, not MinMaxScaler of USAD notebook. No clipping to [0,1]. Sigmoid cannot reconstruct negative or >1 input values; scores may reflect this range mismatch.",
        "data": "Same pinned raw files, scaler and actual window split as MTGFlow. Group window60/stride10 rather than source window12/stride1. Merged-sequence boundary limitation unchanged.",
        "evaluation": "Validation losses use no labels and do not select checkpoint. Test scored after fixed final epoch; no test threshold optimization.",
        "checkpoint": "Model state only at epoch1, every10 epochs and final, to limit large Drive writes. No automatic resume; source and notebook needed for inference.",
    }
    for name, value in [("run_config.json", config), ("adapter_notes.json", notes),
                        ("data_manifest.json", manifest), ("range_diagnostics.json", ranges)]:
        (out / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(f"USAD {dataset_name} seed{seed}: train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ds)}")
    print(f"{epochs} epoch; batch={batch_size}; latent={model_config['z_size']}; StandardScaler + Sigmoid")
    print("Input range diagnostics (not used for tuning):", ranges)
    print("Output:", out, flush=True)
    history = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total1, total2, count = 0., 0., 0
        for x, _, _ in train_loader:
            batch = usad_flatten(x).to(device)
            loss1, loss2 = usad_two_steps(model, batch, epoch, optimizer1, optimizer2)
            total1 += loss1 * len(x)
            total2 += loss2 * len(x)
            count += len(x)
        model.eval()
        with torch.no_grad():
            outputs = [model.validation_step(usad_flatten(x).to(device), epoch) for x, _, _ in val_loader]
            validation_loss = model.validation_epoch_end(outputs)
        assert all(np.isfinite(v) for v in validation_loss.values())
        history.append({"epoch": epoch, "train_loss1": total1 / count,
                        "train_loss2": total2 / count, "trained_windows": count, **validation_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            tmp = out / "checkpoint_last.tmp"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "model_config": model_config, "run_config": config}, tmp)
            tmp.replace(out / "checkpoint_last.pt")
        pd.DataFrame(history).to_csv(out / "epoch_history.csv", index=False)
        message = (f"USAD {dataset_name} seed{seed}: epoch {epoch:03d}/{epochs} | "
                   f"L1={total1/count:.6f} L2={total2/count:.6f}")
        print(message, flush=True)
        with (out / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    del optimizer1, optimizer2
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    eval_started = time.perf_counter()
    saved = torch.load(out / "checkpoint_last.pt", map_location="cpu", weights_only=True)
    assert saved["epoch"] == epochs
    restored = USAD_CLASS(**saved["model_config"]).to(device)
    restored.load_state_dict(saved["model_state_dict"], strict=True)
    restored.eval()
    model.eval()
    probe = usad_flatten(next(iter(test_loader))[0][:8]).to(device)
    with torch.no_grad():
        torch.testing.assert_close(usad_window_score(model, probe), usad_window_score(restored, probe))
    def collect(loader, base, expected_ids, split_name):
        frames = []
        with torch.no_grad():
            for x, y, idx in loader:
                scores = usad_window_score(restored, usad_flatten(x).to(device))
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
    np.testing.assert_array_equal(test.start, test_ds.starts)
    np.testing.assert_array_equal(test.label, truth)
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_started
    row = {"model": "USAD", "variant": config["variant"], "dataset": dataset_name,
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
    del model, restored, saved, probe, batch, x
    gc.collect()
    torch.cuda.empty_cache()
    return out
