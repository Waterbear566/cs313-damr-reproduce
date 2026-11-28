# %% CELL 1: pinned reference definitions and narrowly scoped safe shape probe
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
assert 'load_usad_shared_data' in globals(), 'Nạp lại ô setup USAD; không chạy lại ô train.'
ALOCC_COMMIT = 'f8cb9ea225fd7e64cfb4556dae3525accac285f5'
ALOCC_CACHE = Path('/content/drive/MyDrive/CS313_Nhom5/source_cache') / ('ALOCC_' + ALOCC_COMMIT)
ALOCC_CACHE.mkdir(parents=True, exist_ok=True)
ALOCC_ROOT = Path('/content/drive/MyDrive/CS313_Nhom5/hoang_khang/E3a')
ALOCC_HASHES = {
    'models/GAN.py': '2ba491d5e94e880660ae584d6cac045d6f85dea9a864f4f2fab8a71bc04b5e5c',
    'train_other_model.py': 'db2a383e95521c9ad69b2abf60848e1e90242c4088fc5ec68982783390f1ec11',
    'LICENSE': '23a15f9f18973b2376cb281cb3bba6fea79c80c1af79982a48920b40f3a25f4f',
}
for name, digest in ALOCC_HASHES.items():
    path = ALOCC_CACHE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(f'https://raw.githubusercontent.com/zqhang/MTGFLOW/{ALOCC_COMMIT}/{name}', timeout=(30, 120))
        response.raise_for_status()
        assert hashlib.sha256(response.content).hexdigest() == digest, name
        path.write_bytes(response.content)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name

names = {'ConvEncoder', 'ConvDecoder', 'R_Net', 'D_Net', 'R_Loss', 'D_Loss'}
nodes = [n for n in ast.parse((ALOCC_CACHE / 'models/GAN.py').read_text(encoding='utf-8')).body
         if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
assert {n.name for n in nodes} == names
safe_probe = ast.parse('''
def _compute_out_dim(self):
    was_training = self.cnn.training
    self.cnn.eval()
    try:
        with torch.no_grad():
            output = self.cnn(torch.zeros(1, self.in_channels, self.in_resolution))
        return output[0].numel()
    finally:
        self.cnn.train(was_training)
''').body[0]
for node in nodes:
    if node.name == 'D_Net':
        node.body = [safe_probe if isinstance(n, ast.FunctionDef) and n.name == '_compute_out_dim' else n for n in node.body]
ALOCC_NETWORK_CODE = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n' + '\n\n'.join(ast.unparse(n) for n in nodes) + '\n'
ALOCC_NS = {}
exec(compile(ALOCC_NETWORK_CODE, 'alocc_network.py', 'exec'), ALOCC_NS)


def alocc_spec(dataset_name, seed):
    assert dataset_name in ('PSM', 'MSL', 'SMD') and seed in (16, 17, 18)
    return {
        'model': 'ALOCC', 'variant': 'mtgflow_cnn_width16_safe_probe',
        'adapter_version': 'alocc_shared_v1', 'dataset': dataset_name, 'seed': seed,
        'epochs': 40, 'batch_size': 512, 'lr': 0.001, 'weight_decay': 0,
        'adam_betas': [0.9, 0.999], 'adam_eps': 1e-8, 'lambda_rec': 0.2,
        'input_noise_std': 0.2, 'shuffle': True, 'drop_last': False,
        'model_config': {'in_channels': {'PSM': 25, 'MSL': 55, 'SMD': 38}[dataset_name], 'n_channels': 16, 'kernel_size': 5},
        'window': 60, 'stride': 10, 'scaler': 'StandardScaler',
        'protocol_id': f'DAMR_public_code_{dataset_name}_v1',
        'checkpoint_policy': 'last_epoch_fixed_in_advance',
        'score': 'negative_discriminator_logit_on_real_window',
        'source_commit': ALOCC_COMMIT, 'source_sha256': ALOCC_HASHES,
        'network_sha256': hashlib.sha256(ALOCC_NETWORK_CODE.encode()).hexdigest(),
    }


def alocc_two_steps(r_net, d_net, optimizer_r, optimizer_d, real):
    fake = r_net(real)  # one noisy R forward, shared by the two steps
    d_net.zero_grad()
    loss_d = ALOCC_NS['D_Loss'](d_net, real, fake)  # source detaches fake here
    assert torch.isfinite(loss_d).item(), 'D loss NaN/Inf'
    loss_d.backward()
    assert all(torch.isfinite(p.grad).all().item() for p in d_net.parameters() if p.grad is not None)
    optimizer_d.step()
    r_net.zero_grad()
    losses = ALOCC_NS['R_Loss'](d_net, real, fake, 0.2)
    assert all(torch.isfinite(v).item() for v in losses.values()), 'R loss NaN/Inf'
    losses['L_r'].backward()
    assert all(torch.isfinite(p.grad).all().item() for p in list(r_net.parameters()) + list(d_net.parameters()) if p.grad is not None)
    optimizer_r.step()
    return {**{k: v.item() for k, v in losses.items()}, 'dis_loss': loss_d.item()}


def alocc_score(d_net, real):
    return -d_net(real).reshape(-1)  # preserve batch dimension when batch=1

print('ALOCC đã nạp; chưa train. Dò shape an toàn; width16 là mặc định repo, chưa xác minh paper DAMR.')


# %% CELL 2: runner; use the common loader, NOT the USAD optimizer or score
def run_alocc(dataset_name, seed):
    assert torch.cuda.is_available(), 'Chưa có CUDA.'
    device = torch.device('cuda:0')
    reference, prep, full, test_ds, train_ids, val_ids, truth, manifest, _ = load_usad_shared_data(dataset_name, seed)
    cfg = alocc_spec(dataset_name, seed)
    assert prep['n_features'] == cfg['model_config']['in_channels']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    r_net = ALOCC_NS['R_Net'](**cfg['model_config'], std=cfg['input_noise_std']).to(device)
    d_net = ALOCC_NS['D_Net'](in_resolution=60, **cfg['model_config']).to(device)
    optimizer_r = torch.optim.Adam(r_net.parameters(), lr=cfg['lr'], weight_decay=0)
    optimizer_d = torch.optim.Adam(d_net.parameters(), lr=cfg['lr'], weight_decay=0)
    train_loader = DataLoader(Subset(full, train_ids.tolist()), batch_size=cfg['batch_size'], shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(Subset(full, val_ids.tolist()), batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    cfg.update({
        'preprocessing_reference': str(reference), 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name(0), 'auprc_definition': 'average_precision',
        'threshold_policy': 'not_selected', 'checkpoint_contents': 'R_and_D_inference_only_no_resume',
        'runtime_definition': 'training_and_checkpoint_IO_plus_final_evaluation_excluding_preparation',
    })
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = ALOCC_ROOT / f'ALOCC_{dataset_name}_seed{seed}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    for name in ('config.json', 'scaler.joblib', 'window_split.npz'):
        shutil.copy2(reference / name, out / name)
    shutil.copytree(ALOCC_CACHE, out / 'upstream_source')
    (out / 'network.py').write_bytes(ALOCC_NETWORK_CODE.encode('utf-8'))
    notes = {
        'reference': 'ALOCC implementation in MTGFLOW repo; not a verified exact reproduction of original ALOCC or DAMR paper settings.',
        'hyperparameters': 'GAN.py train_model defaults: epochs40/lr.001/lambda.2/no scheduler/Adam WD0. R noise std.2. Entry loader batch512; unused train_model batch_size1024 does not control loader.',
        'width': 'Entry script has undefined size/n_channels. Use n_features and constructor default width16, fixed before seeing test scores.',
        'shape_probe_fix': 'Replace D uninitialized torch.Tensor probe with zeros under eval/no_grad; restore training flag; no BatchNorm buffer mutation or parameter freezing. All other network/loss definitions unchanged.',
        'training': 'Source order D then R; fake computed once; do not freeze D during R step. Keep BN updates. Detached sample-weighted scalar logging avoids retaining source loss graphs. Train shuffle=True is an explicit adapter choice.',
        'evaluation': 'Do not test each epoch. Final fixed epoch only; no tuning or checkpoint selection on validation/test. Score=-D(real), not reconstruction loss. Reject NaN/Inf rather than source nan_to_num. reshape(-1) fixes batch1.',
        'data': 'Exact common StandardScaler and window IDs from MTGFlow via shared helper. Existing split overlap/scaler-before-split/merged-sequence limitations remain.',
        'noise': 'metrics noise=0 means no external benchmark corruption; internal denoising std=.2 remains enabled for R training.',
    }
    for name, value in [('run_config.json', cfg), ('adapter_notes.json', notes), ('data_manifest.json', manifest)]:
        (out / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(f'ALOCC {dataset_name} seed{seed}: train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ds)}')
    print(f"{cfg['epochs']} epoch; batch{cfg['batch_size']}; width16; score=-D(real). Output:", out, flush=True)
    history = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        r_net.train()
        d_net.train()
        totals = {k: 0. for k in ('rec_loss', 'gen_loss', 'L_r', 'dis_loss')}
        count = 0
        for x, _, _ in train_loader:
            real = x.squeeze(-1).to(device)  # (B,F,60), not USAD flatten
            losses = alocc_two_steps(r_net, d_net, optimizer_r, optimizer_d, real)
            for key in totals:
                totals[key] += losses[key] * len(x)
            count += len(x)
        assert count == len(train_ids)
        assert all(torch.isfinite(v).all().item() for net in (r_net, d_net) for v in net.state_dict().values())
        history.append({'epoch': epoch, 'trained_windows': count, **{k: v / count for k, v in totals.items()}})
        pd.DataFrame(history).to_csv(out / 'epoch_history.csv', index=False)
        tmp = out / 'checkpoint_last.tmp'
        torch.save({'epoch': epoch, 'r_net': r_net.state_dict(), 'd_net': d_net.state_dict(), 'run_config': cfg}, tmp)
        tmp.replace(out / 'checkpoint_last.pt')
        message = f"ALOCC {dataset_name} seed{seed}: epoch {epoch:02d}/{cfg['epochs']} | D={totals['dis_loss']/count:.6f} R={totals['L_r']/count:.6f}"
        print(message, flush=True)
        with (out / 'train.log').open('a', encoding='utf-8') as handle:
            handle.write(message + '\n')
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    eval_started = time.perf_counter()
    saved = torch.load(out / 'checkpoint_last.pt', map_location='cpu', weights_only=True)
    assert saved['epoch'] == cfg['epochs']
    restored_r = ALOCC_NS['R_Net'](**cfg['model_config'], std=cfg['input_noise_std']).to(device)
    restored_d = ALOCC_NS['D_Net'](in_resolution=60, **cfg['model_config']).to(device)
    restored_r.load_state_dict(saved['r_net'], strict=True)
    restored_d.load_state_dict(saved['d_net'], strict=True)
    for net in (r_net, d_net, restored_r, restored_d):
        net.eval()
    probe = next(iter(test_loader))[0][:8].squeeze(-1).to(device)
    with torch.no_grad():
        torch.testing.assert_close(alocc_score(d_net, probe), alocc_score(restored_d, probe))
        torch.testing.assert_close(r_net(probe, noise=False), restored_r(probe, noise=False))

    def collect(loader, base, expected_ids, split):
        frames = []
        with torch.no_grad():
            for x, y, idx in loader:
                scores = alocc_score(restored_d, x.squeeze(-1).to(device))
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
    row = {'model': 'ALOCC', 'variant': cfg['variant'], 'dataset': dataset_name, 'seed': seed,
        'noise': 0, 'epoch': cfg['epochs'], 'auroc': float(roc_auc_score(truth, test.score)),
        'auprc': float(average_precision_score(truth, test.score)), 'f1': np.nan,
        'flagged_rate': np.nan, 'runtime_s': train_seconds + eval_seconds}
    pd.DataFrame([row]).to_csv(out / 'metrics.csv', index=False)
    (out / 'completed.json').write_text(json.dumps({'status': 'completed', 'epoch': cfg['epochs'],
        'validation_windows': len(validation), 'test_windows': len(test), 'checkpoint_reload_matches': True,
        'train_runtime_s': train_seconds, 'evaluation_runtime_s': eval_seconds,
        'f1_status': 'not_computed_threshold_not_selected'}, indent=2), encoding='utf-8')
    print(pd.DataFrame([row]).to_string(index=False))
    print('HOÀN THÀNH:', out)
    del r_net, d_net, restored_r, restored_d, optimizer_r, optimizer_d, saved, probe
    gc.collect()
    torch.cuda.empty_cache()
    return out


# %% CELL 3: compatible-completion lookup; no training on definition
def find_completed_alocc(dataset_name, seed):
    required = ['completed.json', 'metrics.csv', 'run_config.json', 'config.json',
                'scaler.joblib', 'window_split.npz', 'checkpoint_last.pt',
                'validation_scores.csv', 'test_scores.csv', 'network.py', 'adapter_notes.json']
    spec = alocc_spec(dataset_name, seed)
    for folder in sorted(ALOCC_ROOT.glob(f'ALOCC_{dataset_name}_seed{seed}_*'), reverse=True):
        if not all((folder / name).is_file() for name in required):
            continue
        status = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
        cfg = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
        if status.get('status') != 'completed' or not status.get('checkpoint_reload_matches'):
            continue
        if not all(cfg.get(k) == v for k, v in spec.items()):
            continue
        assert hashlib.sha256((folder / 'network.py').read_bytes()).hexdigest() == spec['network_sha256']
        row = pd.read_csv(folder / 'metrics.csv')
        assert len(row) == 1 and row.iloc[0]['model'] == 'ALOCC'
        assert row.iloc[0]['variant'] == spec['variant'] and row.iloc[0]['dataset'] == dataset_name
        assert int(row.iloc[0]['seed']) == seed and int(row.iloc[0]['epoch']) == spec['epochs']
        assert np.isfinite(row[['auroc', 'auprc', 'runtime_s']].to_numpy()).all()
        return folder
    return None


def run_alocc_if_missing(dataset_name, seed):
    folder = find_completed_alocc(dataset_name, seed)
    if folder is not None:
        print(f'BỎ QUA — đã hoàn tất: ALOCC {dataset_name} seed{seed}', flush=True)
        return folder
    return run_alocc(dataset_name, seed)
