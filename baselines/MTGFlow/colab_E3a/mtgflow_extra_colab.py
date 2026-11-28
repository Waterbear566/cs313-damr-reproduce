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
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset


def run_mtgflow_extra(dataset_name, seed):
    assert dataset_name in ("MSL", "SMD")
    assert seed in (16, 17, 18)
    needed = ["MTGFLOW_CLASS", "PSMWindowDataset", "EXTRA_DATA_REFERENCES",
              "MODEL_LOCAL", "MODEL_CACHE", "MTG_COMMIT", "DEBUG_RUN"]
    missing = [k for k in needed if k not in globals()]
    assert not missing, f"Thiếu {missing}: chạy phần import model và tải MSL/SMD trước."
    assert torch.cuda.is_available(), "Chưa có CUDA; không bắt đầu train."
    device = torch.device("cuda:0")
    window, stride, val_ratio = 60, 10, 0.2
    epochs, batch_size, lr, weight_decay = 40, 512, 0.002, 0.0005
    data_root = Path("/content/drive/MyDrive/CS313_Nhom5/data")
    root = Path("/content/drive/MyDrive/CS313_Nhom5/hoang_khang/E3a")
    assert data_root.is_dir()
    assert Path(MODEL_LOCAL).is_dir()
    for name in ["compatibility.patch", "source_provenance.json"]:
        assert (Path(DEBUG_RUN) / name).is_file(), f"Thiếu nguồn tham chiếu: {name}"

    arrays, manifest = {}, {}
    for split in ["train", "test", "test_label"]:
        key = f"{dataset_name}/{dataset_name}_{split}.npy"
        path = data_root / key
        assert path.is_file(), f"Thiếu dữ liệu: {path}"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == EXTRA_DATA_REFERENCES[key]["sha256"], path
        manifest[key] = {**EXTRA_DATA_REFERENCES[key], "path": str(path)}
        arrays[split] = np.load(path, allow_pickle=False)

    raw_train, raw_test = arrays["train"], arrays["test"]
    raw_labels = arrays["test_label"]
    assert raw_labels.ndim == 1 or (raw_labels.ndim == 2 and raw_labels.shape[1] == 1)
    labels = raw_labels.reshape(-1)
    expected_shape = {"MSL": (58317, 73729, 55), "SMD": (708405, 708420, 38)}
    n_train, n_test, features = expected_shape[dataset_name]
    assert raw_train.shape == (n_train, features)
    assert raw_test.shape == (n_test, features)
    assert labels.shape == (n_test,) and np.isin(labels, [0, 1]).all()
    assert np.isfinite(raw_train).all() and np.isfinite(raw_test).all()

    # DAMR MSL/SMD: giữ mọi feature; không có bước điền NaN ở loader gốc.
    scaler = StandardScaler().fit(raw_train)
    train_z = scaler.transform(raw_train).astype(np.float32)
    test_z = scaler.transform(raw_test).astype(np.float32)
    assert np.isfinite(train_z).all() and np.isfinite(test_z).all()
    base = PSMWindowDataset(train_z, window=window, stride=stride)
    test_ds = PSMWindowDataset(test_z, labels, window, stride)
    n = len(base)
    val_size = int(n * val_ratio)
    val_start = random.Random(seed).randrange(int(n * (1 - val_ratio)))
    ids = np.arange(n, dtype=np.int64)
    val_ids = ids[val_start:val_start + val_size]
    train_ids = np.concatenate([ids[:val_start], ids[val_start + val_size:]])
    assert np.intersect1d(train_ids, val_ids).size == 0
    np.testing.assert_array_equal(np.sort(np.r_[train_ids, val_ids]), ids)
    train_ds, val_ds = Subset(base, train_ids.tolist()), Subset(base, val_ids.tolist())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = {
        "n_blocks": 1, "input_size": 1, "hidden_size": 32, "n_hidden": 1,
        "window_size": window, "n_sensor": features, "dropout": 0.0,
        "model": "MAF", "batch_norm": False,
    }
    model = MTGFLOW_CLASS(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    assert len(train_loader) > 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / f"MTGFlow_{dataset_name}_seed{seed}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    cfg = {
        "model": "MTGFlow", "dataset": dataset_name, "seed": seed,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "model_config": model_config,
        "protocol_id": f"DAMR_public_code_{dataset_name}_v1",
        "checkpoint_policy": "last_epoch_fixed_in_advance",
        "auprc_definition": "average_precision", "threshold_policy": "not_selected",
        "evaluation_mode": "model.eval", "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0), "model_commit": MTG_COMMIT,
    }
    prep = {
        "dataset": dataset_name, "seed": seed, "protocol_id": cfg["protocol_id"],
        "window": window, "stride": stride, "val_ratio": val_ratio,
        "scaler_fit_on": "full_train_file_before_window_split",
        "missing_policy": "no_imputation_assert_finite_as_DAMR_MSL_SMD",
        "split_policy": "random_start_contiguous_validation_window_block",
        "train_file_rows": n_train, "test_rows": n_test, "n_features": features,
        "full_train_windows": n, "train_windows": len(train_ds),
        "validation_windows": len(val_ds), "test_windows": len(test_ds),
        "val_start_window_id": val_start,
        "sequence_boundary_policy": "merged_arrays_no_boundary_metadata",
    }
    for name, value in [("run_config.json", cfg), ("config.json", prep),
                        ("data_manifest.json", manifest)]:
        (out / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    joblib.dump(scaler, out / "scaler.joblib")
    restored_scaler = joblib.load(out / "scaler.joblib")
    assert int(restored_scaler.n_samples_seen_) == n_train
    for raw, scaled in [(raw_train, train_z), (raw_test, test_z)]:
        np.testing.assert_allclose(restored_scaler.transform(raw).astype(np.float32), scaled)
    np.savez(out / "window_split.npz", train_ids=train_ids, val_ids=val_ids,
             val_start_index=val_start)
    shutil.copytree(MODEL_LOCAL, out / "source")
    shutil.copy2(Path(MODEL_CACHE) / "LICENSE", out / "source" / "LICENSE")
    for name in ["compatibility.patch", "source_provenance.json"]:
        shutil.copy2(Path(DEBUG_RUN) / name, out / name)
    (out / "README.md").write_text(
        "MTGFlow; last epoch; AP=average precision; F1 chưa chọn ngưỡng.\n"
        "Dữ liệu là mảng gộp, không có ranh giới chuỗi/máy; window có thể cắt qua ranh giới.\n"
        "Train/val có thể chồng lấn điểm gốc, giống split window của DAMR.\n"
        "Gửi kèm notebook chứa runner; source/ chỉ chứa code model.\n", encoding="utf-8")
    print(f"{dataset_name}, seed={seed}: train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print("Thư mục:", out, flush=True)

    history = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x, _, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = -model(x)
            assert loss.ndim == 0 and torch.isfinite(loss).item()
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all().item() for g in grads)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
            count += len(x)
        history.append({"epoch": epoch, "train_loss": total_loss / count, "trained_windows": count})
        temporary = out / "checkpoint_last.tmp"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config, "run_config": cfg}, temporary)
        temporary.replace(out / "checkpoint_last.pt")
        pd.DataFrame(history).to_csv(out / "epoch_history.csv", index=False)
        message = f"{dataset_name} seed{seed}: epoch {epoch:02d}/{epochs}, loss={total_loss/count:.6f}"
        print(message, flush=True)
        with (out / "train.log").open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    eval_started = time.perf_counter()
    ckpt = torch.load(out / "checkpoint_last.pt", map_location=device, weights_only=True)
    assert ckpt["epoch"] == epochs
    restored = MTGFLOW_CLASS(**ckpt["model_config"]).to(device)
    restored.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    restored.eval()
    probe = next(iter(test_loader))[0][:8].to(device)
    with torch.no_grad():
        torch.testing.assert_close(model.test(probe), restored.test(probe), rtol=1e-5, atol=1e-6)

    def score(loader, base_ds, expected_ids, split_name):
        frames = []
        with torch.no_grad():
            for x, y, window_ids in loader:
                values = -restored.test(x.to(device))
                assert values.shape == (len(x),) and torch.isfinite(values).all().item()
                idx = window_ids.numpy()
                frames.append(pd.DataFrame({
                    "split": split_name, "window_id": idx, "start": base_ds.starts[idx],
                    "end_exclusive": base_ds.starts[idx] + window,
                    "label": y.numpy(), "score": values.cpu().numpy(),
                }))
        frame = pd.concat(frames, ignore_index=True).sort_values("window_id").reset_index(drop=True)
        np.testing.assert_array_equal(frame.window_id.to_numpy(), np.sort(expected_ids))
        assert not frame.window_id.duplicated().any()
        frame.to_csv(out / f"{split_name}_scores.csv", index=False)
        return frame

    validation = score(val_loader, base, val_ids, "validation")
    test = score(test_loader, test_ds, np.arange(len(test_ds)), "test")
    # Tính đáp án độc lập bằng cumulative sum để kiểm tra Dataset/window alignment.
    cumulative = np.r_[0, np.cumsum(labels == 1)]
    starts = np.arange(0, len(labels) - window + 1, stride)
    truth = (cumulative[starts + window] - cumulative[starts] > 0).astype(int)
    np.testing.assert_array_equal(test.start.to_numpy(), starts)
    np.testing.assert_array_equal(test.label.to_numpy(), truth)
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - eval_started
    row = {
        "model": "MTGFlow", "variant": "base", "dataset": dataset_name, "seed": seed,
        "noise": 0, "epoch": epochs, "auroc": float(roc_auc_score(truth, test.score)),
        "auprc": float(average_precision_score(truth, test.score)),
        "f1": np.nan, "flagged_rate": np.nan, "runtime_s": train_seconds + evaluation_seconds,
    }
    pd.DataFrame([row]).to_csv(out / "metrics.csv", index=False)
    (out / "completed.json").write_text(json.dumps({
        "status": "completed", "checkpoint_policy": "last_epoch",
        "validation_windows": len(validation), "test_windows": len(test),
        "train_runtime_s": train_seconds, "evaluation_runtime_s": evaluation_seconds,
        "f1_status": "not_computed_threshold_not_selected",
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame([row]).to_string(index=False))
    print("HOÀN THÀNH:", out)
    del model, restored, optimizer, ckpt, probe, x, loss, grads
    gc.collect()
    torch.cuda.empty_cache()
    return out
