# %% CELL 1: pinned official GANF; no training on load
import ast
import gc
import hashlib
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

assert Path('/content/drive/MyDrive').is_dir(), 'Mount Drive trước.'
assert 'load_usad_shared_data' in globals(), 'Cần helper dữ liệu chung từ setup USAD, không cần train lại USAD.'
GANF_COMMIT = 'bce38333b109f325a403dae9aff4987ae5bd6e1f'
GANF_ROOT = Path('/content/drive/MyDrive/CS313_Nhom5/hoang_khang/E3a')
GANF_CACHE = Path('/content/drive/MyDrive/CS313_Nhom5/source_cache') / ('GANF_' + GANF_COMMIT)
GANF_HASHES = {
    'models/GANF.py': '2d891bdc707ebfdabe9561224f82bf9a739b4eed89d29469ffbd0e4d5de21cd7',
    'models/NF.py': 'f2d7513d5825e8ec4499b714c4dd188324de16bf9cc64924f2656ac575624a19',
    'train_water.py': '8a585211c50e9966b7cb0ccfbf1032aa1cca8ba1a3fbfbae80f9d5f8d399241a',
    'train_water.sh': 'f584a4929e65f59ed62770192cfb1efac3e9e9243a289f6091dcfd5d7cacfa5e',
    'dataset.py': '959e0a42cb6c7808075df9aaa0d915dfe01eb96516b20ef3c178616d77959883',
    'README.md': '5216bbd8f5a685ae65652d2e2bc4ef01e5e245e84f14b33ac9d28e8e643f1f98',
}
for name, expected in GANF_HASHES.items():
    path = GANF_CACHE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(f'https://raw.githubusercontent.com/EnyanDai/GANF/{GANF_COMMIT}/{name}', timeout=(30, 120))
        response.raise_for_status()
        assert hashlib.sha256(response.content).hexdigest() == expected, name
        path.write_bytes(response.content)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name

# Separate namespace avoids importing an existing notebook's unrelated models.NF.
ganf_tree = ast.parse((GANF_CACHE / 'models/GANF.py').read_text(encoding='utf-8'))
ganf_tree.body = [n for n in ganf_tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'models.NF')]
GANF_NETWORK_CODE = (GANF_CACHE / 'models/NF.py').read_text(encoding='utf-8') + '\n\n' + ast.unparse(ganf_tree) + '\n'
GANF_NS = {}
exec(compile(GANF_NETWORK_CODE, 'ganf_network.py', 'exec'), GANF_NS)
GANF_CLASS = GANF_NS['GANF']


def ganf_spec(dataset_name, seed):
    assert dataset_name in ('PSM', 'MSL', 'SMD') and seed in (16, 17, 18)
    return {
        'model': 'GANF', 'variant': 'official_water_augLag_shared_standard',
        'adapter_version': 'ganf_shared_v1', 'dataset': dataset_name, 'seed': seed,
        'model_config': {'n_blocks': 1, 'input_size': 1, 'hidden_size': 32, 'n_hidden': 1,
                         'dropout': 0.0, 'model': 'MAF', 'batch_norm': False},
        'batch_size': 512, 'lr': 0.002, 'weight_decay_model': 0.0005, 'weight_decay_A': 0,
        'adam_betas': [0.9, 0.999], 'adam_eps': 1e-8,
        'max_iter': 20, 'epochs_per_augmented_round': 1, 'final_epochs': 30,
        'rho_init': 1.0, 'alpha_init': 0.0, 'rho_max': 1e16, 'h_tol': 1e-4,
        'rho_multiplier': 10, 'h_improvement_ratio': 0.5,
        'graph_initialization': 'abs_xavier_uniform_zero_diagonal_before_model_init',
        'graph_constraint': 'trace_matrix_exp_A_squared_minus_n_sensors',
        'graph_projection': 'clamp_0_1_after_each_step_no_extra_diagonal_mask',
        'schedule_h': 'source_last_batch_preupdate_value',
        'clip_model_grad_value': 1.0, 'clip_A_gradient': False,
        'shuffle': True, 'drop_last': False, 'window': 60, 'stride': 10,
        'scaler': 'StandardScaler', 'protocol_id': f'DAMR_public_code_{dataset_name}_v1',
        'checkpoint_policy': 'final_scheduled_epoch_no_test_selection',
        'score': 'negative_mean_log_likelihood_over_features_and_time',
        'source_commit': GANF_COMMIT, 'source_sha256': GANF_HASHES,
        'network_sha256': hashlib.sha256(GANF_NETWORK_CODE.encode()).hexdigest(),
    }


def ganf_optimizer(model, A, cfg):
    # Upstream reinitializes Adam at EACH augmented round; one Adam for final phase.
    return torch.optim.Adam([{'params': model.parameters(), 'weight_decay': cfg['weight_decay_model']},
                             {'params': [A]}], lr=cfg['lr'], weight_decay=0.0)


def ganf_train_epoch(model, A, optimizer, loader, rho, alpha, device):
    model.train()
    total, count, h_value = 0.0, 0, None
    for x, _, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        nll = -model(x, A)
        h = torch.trace(torch.matrix_exp(A * A)) - len(A)
        loss = nll + 0.5 * rho * h * h + alpha * h
        assert loss.ndim == 0 and torch.isfinite(loss).item(), 'GANF loss NaN/Inf'
        loss.backward()
        assert all(torch.isfinite(p.grad).all().item() for p in list(model.parameters()) + [A] if p.grad is not None)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            A.clamp_(0, 1)
        total += nll.item() * len(x)
        count += len(x)
        h_value = h.item()  # EXACT source schedule uses last pre-update h, not recomputed h
    assert count == len(loader.dataset) and h_value is not None
    assert torch.isfinite(A).all().item()
    assert all(torch.isfinite(v).all().item() for v in model.state_dict().values())
    return {'train_nll': total / count, 'trained_windows': count, 'h_schedule': h_value}


def ganf_schedule(cfg, epoch_fn):
    rho, alpha, h_old = cfg['rho_init'], cfg['alpha_init'], float('inf')
    epoch = 0
    for outer in range(cfg['max_iter']):
        while rho < cfg['rho_max']:
            for inner_epoch in range(cfg['epochs_per_augmented_round']):
                epoch += 1
                h = epoch_fn(epoch, 'augmented', outer, rho, alpha, inner_epoch == 0)
            if h > cfg['h_improvement_ratio'] * h_old:
                rho *= cfg['rho_multiplier']
            else:
                break
        h_old = h
        alpha += rho * h
        if h_old <= cfg['h_tol'] or rho >= cfg['rho_max']:
            break
    for i in range(cfg['final_epochs']):
        epoch += 1
        epoch_fn(epoch, 'final', outer, rho, alpha, i == 0)
    return epoch, rho, alpha


def find_completed_ganf(dataset_name, seed):
    required = ['completed.json', 'run_config.json', 'config.json', 'checkpoint_last.pt',
                'scaler.joblib', 'window_split.npz', 'validation_scores.csv', 'test_scores.csv',
                'metrics.csv', 'network.py', 'graph_last.npy', 'adapter_notes.json']
    spec = ganf_spec(dataset_name, seed)
    for out in sorted(GANF_ROOT.glob(f'GANF_{dataset_name}_seed{seed}_*'), reverse=True):
        if not all((out / name).is_file() for name in required):
            continue
        status = json.loads((out / 'completed.json').read_text(encoding='utf-8'))
        cfg = json.loads((out / 'run_config.json').read_text(encoding='utf-8'))
        if status.get('status') != 'completed' or not status.get('checkpoint_reload_matches'):
            continue
        if not all(cfg.get(k) == v for k, v in spec.items()):
            continue
        assert hashlib.sha256((out / 'network.py').read_bytes()).hexdigest() == spec['network_sha256']
        metric = pd.read_csv(out / 'metrics.csv')
        assert len(metric) == 1
        row = metric.iloc[0]
        assert row['model'] == 'GANF' and row['variant'] == spec['variant']
        assert row['dataset'] == dataset_name and int(row['seed']) == seed
        assert int(row['epoch']) == status['epoch']
        assert np.isfinite(metric[['auroc', 'auprc', 'runtime_s']].to_numpy()).all()
        return out
    return None


def run_ganf(dataset_name, seed):
    assert torch.cuda.is_available(), 'Chưa có CUDA.'
    device = torch.device('cuda:0')
    reference, prep, full, test_ds, train_ids, val_ids, truth, manifest, _ = load_usad_shared_data(dataset_name, seed)
    cfg = ganf_spec(dataset_name, seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init = torch.nn.init.xavier_uniform_(torch.zeros(prep['n_features'], prep['n_features'])).abs()
    init.fill_diagonal_(0.0)
    A = init.to(device).detach().clone().requires_grad_(True)
    model = GANF_CLASS(**cfg['model_config']).to(device)
    train_loader = DataLoader(Subset(full, train_ids.tolist()), batch_size=cfg['batch_size'], shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(Subset(full, val_ids.tolist()), batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    cfg.update({'preprocessing_reference': str(reference), 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name(0), 'n_features': prep['n_features'],
        'auprc_definition': 'average_precision', 'threshold_policy': 'not_selected',
        'checkpoint_contents': 'model_and_A_inference_only_no_auto_resume',
        'runtime_definition': 'training_and_checkpoint_IO_plus_final_evaluation_excluding_preparation'})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = GANF_ROOT / f'GANF_{dataset_name}_seed{seed}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    for name in ('config.json', 'scaler.joblib', 'window_split.npz'):
        shutil.copy2(reference / name, out / name)
    shutil.copytree(GANF_CACHE, out / 'upstream_source')
    (out / 'network.py').write_bytes(GANF_NETWORK_CODE.encode())
    notes = {
        'reference': 'Official EnyanDai/GANF train_water.py defaults, not verified GANF hyperparameters in DAMR paper.',
        'architecture': 'Exact GANF.py and NF.py definitions; only models.NF import replaced by isolated shared namespace. No PyG needed for this GANF path.',
        'training': 'Source objective NLL + rho*h*h/2 + alpha*h; Adam reset every augmented round; model-only grad clip1; A clamped0..1 each step. No extra diagonal mask or lambda1 term (unused in source). Source last-batch pre-update h drives rho/alpha; final h recomputed separately for diagnostics.',
        'schedule': 'Up to20 augmented outer iterations with rho retries, then30 final epochs. Variable total epoch depends only on training constraint, never test/validation metrics. Not a forced40-epoch adaptation.',
        'evaluation': 'Remove source per-epoch validation/test scoring and best-validation checkpoint selection. Use final scheduled checkpoint and score=-model.test(x,A); no nan_to_num. Log sample-weighted detached NLL.',
        'data': 'Shared pinned raw files, StandardScaler, window60/stride10, actual split IDs and any-positive window label via common helper; not original contaminated SWaT preprocessing/split. Known scaler-before-split/window overlap/merged-boundary limitations remain.',
        'license': 'No LICENSE file found in pinned upstream repository tree; preserve upstream README and attribution. Do not invent a license.',
        'checkpoint': 'Saves model AND graph A every epoch for inference; automatic interrupted-run resume not implemented.',
    }
    for name, value in [('run_config.json', cfg), ('adapter_notes.json', notes), ('data_manifest.json', manifest)]:
        (out / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(f'GANF {dataset_name} seed{seed}: train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ds)}')
    print('Lịch train nguồn: augmented DAG rounds +30 final epochs; batch512. Output:', out, flush=True)
    history, optimizer = [], None
    torch.cuda.synchronize()
    started = time.perf_counter()

    def train_one(epoch, phase, outer, rho, alpha, reset_optimizer):
        nonlocal optimizer
        if reset_optimizer:
            optimizer = ganf_optimizer(model, A, cfg)
        values = ganf_train_epoch(model, A, optimizer, train_loader, rho, alpha, device)
        history.append({'epoch': epoch, 'phase': phase, 'outer': outer, 'rho': rho, 'alpha': alpha, **values})
        pd.DataFrame(history).to_csv(out / 'epoch_history.csv', index=False)
        tmp = out / 'checkpoint_last.tmp'
        torch.save({'epoch': epoch, 'phase': phase, 'model_state_dict': model.state_dict(),
            'A': A.detach(), 'model_config': cfg['model_config'], 'run_config': cfg}, tmp)
        tmp.replace(out / 'checkpoint_last.pt')
        message = f"GANF {dataset_name} seed{seed}: epoch {epoch:03d} [{phase}] NLL={values['train_nll']:.6f} h={values['h_schedule']:.6g} rho={rho:.3g}"
        print(message, flush=True)
        with (out / 'train.log').open('a', encoding='utf-8') as handle:
            handle.write(message + '\n')
        return values['h_schedule']

    epochs, rho, alpha = ganf_schedule(cfg, train_one)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    eval_started = time.perf_counter()
    saved = torch.load(out / 'checkpoint_last.pt', map_location='cpu', weights_only=True)
    assert saved['epoch'] == epochs and saved['phase'] == 'final'
    restored = GANF_CLASS(**saved['model_config']).to(device)
    restored.load_state_dict(saved['model_state_dict'], strict=True)
    restored_A = saved['A'].to(device)
    np.save(out / 'graph_last.npy', restored_A.cpu().numpy())
    model.eval()
    restored.eval()
    probe = next(iter(test_loader))[0][:8].to(device)
    with torch.no_grad():
        torch.testing.assert_close(A.detach(), restored_A, atol=0, rtol=0)
        torch.testing.assert_close(model.test(probe, A), restored.test(probe, restored_A))
        h_final = (torch.trace(torch.matrix_exp(restored_A * restored_A)) - len(A)).item()
    assert np.isfinite(h_final)

    def collect(loader, base, expected_ids, split):
        frames = []
        with torch.no_grad():
            for x, y, idx in loader:
                scores = -restored.test(x.to(device), restored_A)
                assert scores.shape == (len(x),) and torch.isfinite(scores).all().item()
                ids = idx.numpy()
                frames.append(pd.DataFrame({'split': split, 'window_id': ids, 'start': base.starts[ids],
                    'end_exclusive': base.starts[ids] + 60, 'label': y.numpy(), 'score': scores.cpu().numpy()}))
        frame = pd.concat(frames, ignore_index=True).sort_values('window_id').reset_index(drop=True)
        np.testing.assert_array_equal(frame.window_id, np.sort(expected_ids))
        assert not frame.window_id.duplicated().any()
        frame.to_csv(out / f'{split}_scores.csv', index=False)
        return frame

    validation = collect(val_loader, full, val_ids, 'validation')
    test = collect(test_loader, test_ds, np.arange(len(test_ds)), 'test')
    np.testing.assert_array_equal(test.label, truth)
    np.testing.assert_array_equal(test.start, test_ds.starts)
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_started
    row = {'model': 'GANF', 'variant': cfg['variant'], 'dataset': dataset_name, 'seed': seed,
        'noise': 0, 'epoch': epochs, 'auroc': float(roc_auc_score(truth, test.score)),
        'auprc': float(average_precision_score(truth, test.score)), 'f1': np.nan,
        'flagged_rate': np.nan, 'runtime_s': train_seconds + eval_seconds}
    pd.DataFrame([row]).to_csv(out / 'metrics.csv', index=False)
    (out / 'completed.json').write_text(json.dumps({'status': 'completed', 'epoch': epochs,
        'validation_windows': len(validation), 'test_windows': len(test), 'checkpoint_reload_matches': True,
        'rho_final': rho, 'alpha_final': alpha, 'h_final_recomputed': h_final,
        'train_runtime_s': train_seconds, 'evaluation_runtime_s': eval_seconds,
        'f1_status': 'not_computed_threshold_not_selected'}, indent=2), encoding='utf-8')
    print(pd.DataFrame([row]).to_string(index=False))
    print('HOÀN THÀNH:', out)
    del optimizer, model, A, restored, restored_A, saved, probe
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_ganf_if_missing(dataset_name, seed):
    out = find_completed_ganf(dataset_name, seed)
    if out is not None:
        print(f'BỎ QUA — đã hoàn tất: GANF {dataset_name} seed{seed}', flush=True)
        return out
    return run_ganf(dataset_name, seed)


def export_nhk_handover():
    """Audit 45 completed runs and export reports; never train/select by score."""
    import joblib
    variants = {'MTGFlow': 'base', 'DeepSVDD': 'mtgflow_repo_cnn_no_pretrain',
                'USAD': 'shared_standard_sigmoid', 'ALOCC': 'mtgflow_cnn_width16_safe_probe',
                'GANF': 'official_water_augLag_shared_standard'}
    fixed_epochs = {'MTGFlow': 40, 'DeepSVDD': 20, 'USAD': 100, 'ALOCC': 40}
    columns = ['model', 'variant', 'dataset', 'seed', 'noise', 'epoch', 'auroc',
               'auprc', 'f1', 'flagged_rate', 'runtime_s']
    required = ['completed.json', 'run_config.json', 'config.json', 'metrics.csv',
                'checkpoint_last.pt', 'scaler.joblib', 'window_split.npz',
                'validation_scores.csv', 'test_scores.csv']
    selected, rows, missing, duplicates = {}, [], [], []
    for model, variant in variants.items():
        for dataset in ('PSM', 'MSL', 'SMD'):
            for seed in (16, 17, 18):
                candidates = []
                for folder in sorted(GANF_ROOT.glob(f'{model}_{dataset}_seed{seed}_*'), reverse=True):
                    if not all((folder / name).is_file() for name in required):
                        continue
                    status = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
                    cfg = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
                    if status.get('status') != 'completed':
                        continue
                    if (cfg.get('model'), cfg.get('dataset'), cfg.get('seed'), cfg.get('protocol_id')) != (model, dataset, seed, f'DAMR_public_code_{dataset}_v1'):
                        continue
                    if model == 'GANF':
                        if not all(cfg.get(k) == v for k, v in ganf_spec(dataset, seed).items()):
                            continue
                        if not (folder / 'graph_last.npy').is_file():
                            continue
                    elif (cfg.get('epochs') != fixed_epochs[model] or
                          cfg.get('checkpoint_policy') != 'last_epoch_fixed_in_advance'):
                        continue
                    metric = pd.read_csv(folder / 'metrics.csv')
                    assert len(metric) == 1, folder
                    row = metric.iloc[0]
                    if row['variant'] != variant:
                        continue
                    assert row['model'] == model and row['dataset'] == dataset and int(row['seed']) == seed
                    assert float(row['noise']) == 0
                    assert int(row['epoch']) == (status['epoch'] if model == 'GANF' else fixed_epochs[model])
                    assert np.isfinite(metric[['auroc', 'auprc', 'runtime_s']].to_numpy()).all()
                    assert 0 <= row['auroc'] <= 1 and 0 <= row['auprc'] <= 1 and row['runtime_s'] >= 0
                    candidates.append((folder, row))
                key = (model, dataset, seed)
                if not candidates:
                    missing.append(key)
                    continue
                # Timestamp order only, NEVER best AUROC/AP. Keep all original runs.
                folder, row = candidates[0]
                selected[key] = folder
                rows.append(row[columns].to_dict())
                for old_folder, _ in candidates[1:]:
                    duplicates.append({'model': model, 'dataset': dataset, 'seed': seed,
                                       'selected': str(folder), 'other_completed': str(old_folder)})
    assert not missing, f'Chưa bàn giao đủ45 lượt; còn thiếu hoặc chưa tương thích: {missing}'
    results = pd.DataFrame(rows, columns=columns).sort_values(['model', 'dataset', 'seed']).reset_index(drop=True)
    assert len(results) == 45 and not results.duplicated(['model', 'dataset', 'seed']).any()
    inventory = []
    for dataset in ('PSM', 'MSL', 'SMD'):
        for seed in (16, 17, 18):
            ref = selected[('MTGFlow', dataset, seed)]
            with np.load(ref / 'window_split.npz', allow_pickle=False) as split:
                ref_train, ref_val = split['train_ids'].copy(), split['val_ids'].copy()
            ref_scaler = joblib.load(ref / 'scaler.joblib')
            ref_scores = {s: pd.read_csv(ref / f'{s}_scores.csv').sort_values('window_id').reset_index(drop=True)
                          for s in ('validation', 'test')}
            for model in variants:
                folder = selected[(model, dataset, seed)]
                prep = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
                assert prep['window'] == 60 and prep['stride'] == 10
                assert prep['dataset'] == dataset and prep['seed'] == seed
                assert prep['scaler_fit_on'] == 'full_train_file_before_window_split'
                with np.load(folder / 'window_split.npz', allow_pickle=False) as split:
                    np.testing.assert_array_equal(split['train_ids'], ref_train)
                    np.testing.assert_array_equal(split['val_ids'], ref_val)
                scaler = joblib.load(folder / 'scaler.joblib')
                assert scaler.__class__.__name__ == ref_scaler.__class__.__name__ == 'StandardScaler'
                for attribute in ('mean_', 'var_', 'scale_', 'n_samples_seen_', 'n_features_in_'):
                    np.testing.assert_array_equal(getattr(scaler, attribute), getattr(ref_scaler, attribute))
                for split_name in ('validation', 'test'):
                    score = pd.read_csv(folder / f'{split_name}_scores.csv').sort_values('window_id').reset_index(drop=True)
                    base = ref_scores[split_name]
                    assert len(score) == len(base) and not score.window_id.duplicated().any()
                    for col in ('window_id', 'start', 'end_exclusive', 'label'):
                        np.testing.assert_array_equal(score[col], base[col])
                    assert np.isfinite(score.score).all(), folder
                    if split_name == 'test':
                        row = results[(results.model == model) & (results.dataset == dataset) & (results.seed == seed)].iloc[0]
                        np.testing.assert_allclose(row.auroc, roc_auc_score(score.label, score.score), atol=1e-8, rtol=1e-8)
                        np.testing.assert_allclose(row.auprc, average_precision_score(score.label, score.score), atol=1e-8, rtol=1e-8)
                inventory.append({'model': model, 'dataset': dataset, 'seed': seed,
                    'run_dir': str(folder), 'checkpoint': str(folder / 'checkpoint_last.pt'),
                    'test_windows': len(ref_scores['test']), 'validation_windows': len(ref_scores['validation']),
                    'same_scaler_and_split': True, 'scores_and_metrics_checked': True})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = GANF_ROOT / f'HANDOVER_NHK_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    results.to_csv(out / 'results_45_runs.csv', index=False)
    pd.DataFrame(inventory).to_csv(out / 'run_inventory.csv', index=False)
    pd.DataFrame(duplicates, columns=['model', 'dataset', 'seed', 'selected', 'other_completed']).to_csv(out / 'other_completed_runs.csv', index=False)
    summary = results.groupby(['model', 'variant', 'dataset']).agg(
        n_seeds=('seed', 'nunique'), auroc_mean=('auroc', 'mean'), auroc_std=('auroc', 'std'),
        auprc_mean=('auprc', 'mean'), auprc_std=('auprc', 'std'),
        runtime_s_mean=('runtime_s', 'mean'), epoch_min=('epoch', 'min'), epoch_max=('epoch', 'max')).reset_index()
    assert len(summary) == 15 and (summary.n_seeds == 3).all()
    summary.to_csv(out / 'summary_15_groups.csv', index=False)
    readme = '''# Bàn giao baseline Hoàng Khang — CS313

## Các file
- results_45_runs.csv: 5 model × 3 dataset × 3 seed; đúng schema nhóm.
- summary_15_groups.csv: AUROC/AP mean và sample std (ddof=1); không chọn best seed. Giá trị nằm trên thang0..1, không phải phần trăm.
- run_inventory.csv: đường dẫn đủ45 thư mục/checkpoint; giữ nguyên thư mục gốc, data và source_cache trên Drive.
- other_completed_runs.csv: nếu có chạy lặp, lấy timestamp mới nhất tương thích, không chọn theo điểm test; không xóa lượt cũ.

## Đã kiểm tra
- Đủ file của45 lượt; StandardScaler và train/validation IDs khớp MTGFlow theo dataset/seed.
- IDs/start/end/label validation và test khớp; score hữu hạn; AUROC/AP tính lại từ CSV khớp metrics.
- Đây KHÔNG phải lần reload độc lập toàn bộ45 checkpoint. Các runner lưu kết quả reload-check của chính lượt chạy; xem completed.json và notebook.

## Protocol và giới hạn cần báo cáo
- PSM bỏ timestamp/nan_to_num; MSL/SMD giữ mọi feature, yêu cầu finite. StandardScaler fit toàn file train trước khi chia window.
- Window60, stride10; nhãn window=1 nếu có bất kỳ điểm bất thường. Split validation block liên tục với điểm đầu chọn theo seed; cùng IDs giữa các model.
- Train/val không trùng window ID nhưng có thể chồng lấn điểm gốc. Scaler có dùng phần eventual validation. MSL/SMD là mảng gộp, thiếu ranh giới chuỗi/máy.
- AUROC và cột auprc=average_precision_score, ở mức window; không point adjustment.
- F1/flagged_rate chưa tính vì chưa chốt threshold không dùng test. Đây là bàn giao kết quả train/AUROC/AP, KHÔNG phải mọi metric/E3/demo đã hoàn tất.
- noise=0 là không thêm corruption bên ngoài; ALOCC vẫn có noise nội bộ std.2 khi train R.
- USAD giữ Sigmoid nhưng dùng StandardScaler chung; không tái tạo được input ngoài[0,1]. Variant phải được giữ trong báo cáo.
- DeepSVDD: CNN từ repo MTGFlow, không pretrain, width16,20 epoch. ALOCC: CNN width16 và safe shape probe; score=-D(real).
- GANF: source chính thức EnyanDai/GANF; graph A được học và lưu cùng model. Lịch augmented-DAG rồi30 epoch cuối, số epoch có thể khác giữa các lượt. Score=âm log-likelihood.
- Không khẳng định đã tái lập chính xác cấu hình/điểm paper DAMR. Các thay đổi/hyperparameter/source commit nằm trong run_config.json và adapter_notes.json từng lượt.
- Không so runtime như cùng ngân sách: kiến trúc, số epoch và lịch train khác nhau.

## Cách nhận bàn giao
1. Khang Ctrl+S và tải notebook mới nhất, gửi kèm các CSV/README này.
2. Share Drive data/, source_cache/ và hoang_khang/E3a/ (không chỉ thư mục HANDOVER). Kiểm tra quyền truy cập bằng tài khoản người nhận.
3. File ipynb chứa code/output, không chứa checkpoint riêng. Không di chuyển/xóa artifact gốc; inventory là đường dẫn tham chiếu chứ không phải bản sao model.
4. Không Run all; lượt hoàn tất có thể đọc score/checkpoint mà không train lại. Các runner mới chưa có automatic interrupted-training resume.
5. Trước báo cáo cuối: nhóm chốt threshold không dùng test; tính F1/flagged_rate từ score đã lưu, hoàn thành review/demo và đối chiếu cấu hình paper còn thiếu.
'''
    (out / 'README_BAN_GIAO.md').write_text(readme, encoding='utf-8')
    if 'GANF_ADAPTER_PATH' in globals() and Path(GANF_ADAPTER_PATH).is_file():
        shutil.copy2(GANF_ADAPTER_PATH, out / 'ganf_colab_cells.py')
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out / 'report_checksums.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')
    print('ĐÃ KIỂM TRA ĐỦ45 LƯỢT VÀ XUẤT BÁO CÁO:', out)
    print('Còn bước người dùng: lưu notebook và share các thư mục artifact gốc trên Drive.')
    return out, results, summary


print('GANF đã nạp, CHƯA train. Dùng dữ liệu/scaler/split chung; checkpoint lưu cả model+A.')
