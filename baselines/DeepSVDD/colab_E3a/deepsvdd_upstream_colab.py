"""New, separate source-BN variant. Never overwrite prior DeepSVDD results.

Requires load_usad_shared_data + Dataset/checksum definitions; the companion
Colab notebook embeds those definitions, with no execution of old train cells.
"""
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

UP_COMMIT = 'f8cb9ea225fd7e64cfb4556dae3525accac285f5'
UP_HASHES = {
    'models/GAN.py': '2ba491d5e94e880660ae584d6cac045d6f85dea9a864f4f2fab8a71bc04b5e5c',
    'models/DeepSAD.py': '8dcfc3a40edfebb03c8ea3935caa4066358e283047f8696c1bcd08a481aac965',
    'train_other_model.py': 'db2a383e95521c9ad69b2abf60848e1e90242c4088fc5ec68982783390f1ec11',
    'LICENSE': '23a15f9f18973b2376cb281cb3bba6fea79c80c1af79982a48920b40f3a25f4f',
}


def setup_svdd_upstream(output_root='/content/drive/MyDrive/NHK_DeepSVDD_UPSTREAM/E3a',
                        shared_project='/content/drive/MyDrive/CS313_Nhom5'):
    global UP_ROOT, UP_CACHE, UP_NETWORK_CODE, UP_NS
    shared = Path(shared_project)
    assert (shared / 'data').is_dir() and (shared / 'hoang_khang/E3a').is_dir(), 'Chưa thấy data/kết quả cũ; kiểm tra quyền và lối tắt Drive.'
    UP_ROOT = Path(output_root)
    UP_ROOT.mkdir(parents=True, exist_ok=True)
    UP_CACHE = UP_ROOT.parent / 'source_cache' / ('DeepSVDD_' + UP_COMMIT)
    old_cache = shared / 'source_cache' / ('DeepSVDD_' + UP_COMMIT)
    for name, expected in UP_HASHES.items():
        path = UP_CACHE / name
        if not path.exists():
            if (old_cache / name).is_file():
                content = (old_cache / name).read_bytes()
            else:
                response = requests.get(f'https://raw.githubusercontent.com/zqhang/MTGFLOW/{UP_COMMIT}/{name}', timeout=(30, 120))
                response.raise_for_status()
                content = response.content
            assert hashlib.sha256(content).hexdigest() == expected, name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    source = ast.parse((UP_CACHE / 'models/GAN.py').read_text(encoding='utf-8'))
    names = {'ConvEncoder', 'ConvDecoder', 'CNNAE'}
    nodes = [n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    UP_NETWORK_CODE = 'import torch\nimport torch.nn as nn\n\n' + '\n\n'.join(ast.unparse(n) for n in nodes) + '\n'
    UP_NS = {}
    exec(compile(UP_NETWORK_CODE, 'svdd_upstream_network.py', 'exec'), UP_NS)
    # Check write access on OWN Drive, not on the shared shortcut.
    probe = UP_ROOT / ('WRITE_CHECK_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '.txt')
    probe.write_text('write check only; no training', encoding='utf-8')
    assert probe.read_text(encoding='utf-8') == 'write check only; no training'
    print('Đã nạp source và kiểm tra quyền ghi. CHƯA train.')
    print('Output riêng:', UP_ROOT)


def upstream_spec(dataset, seed):
    assert dataset in ('PSM', 'MSL', 'SMD') and seed in (16, 17, 18)
    return {
        'model': 'DeepSVDD', 'variant': 'upstream_eval_bn', 'adapter_version': 'svdd_upstream_bn_v1',
        'dataset': dataset, 'seed': seed, 'epochs': 20, 'batch_size': 512,
        'model_config': {'in_channels': {'PSM': 25, 'MSL': 55, 'SMD': 38}[dataset], 'n_channels': 16, 'kernel_size': 5},
        'lr': 1e-4, 'weight_decay': 5e-4, 'adam_betas': [0.9, 0.999], 'adam_eps': 1e-8,
        'lr_policy': 'epoch1_19_1e-4_epoch20_1e-5_reference_pre_step_milestone20',
        'center_eps': 0.1, 'center_policy': 'mean_feature_map_then_eps_clip_then_temporal_mean',
        'bn_policy': 'epoch1_train_epoch2_20_eval_gradients_enabled',
        'score': 'squared_l2_temporal_mean_encoder_to_fixed_c', 'pretraining_epochs': 0,
        'shuffle': True, 'drop_last': True, 'window': 60, 'stride': 10,
        'protocol_id': f'DAMR_public_code_{dataset}_v1', 'scaler': 'StandardScaler',
        'checkpoint_policy': 'last_epoch_fixed_in_advance',
        'source_commit': UP_COMMIT, 'source_sha256': UP_HASHES,
        'network_sha256': hashlib.sha256(UP_NETWORK_CODE.encode()).hexdigest(),
    }


def upstream_init_center(net, loader, device):
    net.eval()
    total, count = None, 0
    with torch.no_grad():
        for x, _, _ in loader:
            maps = net(x.squeeze(-1).to(device))
            total = maps.sum(0) if total is None else total + maps.sum(0)
            count += len(x)
    assert count > 0
    center = total / count
    center[(center.abs() < .1) & (center < 0)] = -.1
    center[(center.abs() < .1) & (center > 0)] = .1
    c = center.mean(-1).detach()
    assert c.shape == (64,) and torch.isfinite(c).all().item()
    return c


def upstream_scores(net, c, loader, device):
    # Like upstream test(): this intentionally leaves net in EVAL mode.
    net.eval()
    scores, labels, ids = [], [], []
    with torch.no_grad():
        for x, y, idx in loader:
            z = net(x.squeeze(-1).to(device)).mean(-1)
            s = ((z - c) ** 2).sum(1)
            assert s.shape == (len(x),) and torch.isfinite(s).all().item()
            scores.append(s.cpu().numpy())
            labels.append(y.numpy())
            ids.append(idx.numpy())
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(ids)


def find_completed_upstream(dataset, seed):
    expected = upstream_spec(dataset, seed)
    needed = ['completed.json', 'metrics.csv', 'run_config.json', 'checkpoint_last.pt',
              'scaler.joblib', 'window_split.npz', 'validation_scores.csv', 'test_scores.csv',
              'epoch_history.csv', 'epoch_metrics.csv', 'network.py', 'scores/epoch_020.npy']
    for out in sorted(UP_ROOT.glob(f'DeepSVDD_upstream_eval_bn_{dataset}_seed{seed}_*'), reverse=True):
        if not all((out / name).is_file() for name in needed):
            continue
        status = json.loads((out / 'completed.json').read_text())
        cfg = json.loads((out / 'run_config.json').read_text())
        if status.get('status') != 'completed' or status.get('epoch') != 20 or not status.get('checkpoint_reload_matches'):
            continue
        if not all(cfg.get(k) == v for k, v in expected.items()):
            continue
        metric = pd.read_csv(out / 'metrics.csv')
        assert len(metric) == 1 and metric.iloc[0]['variant'] == 'upstream_eval_bn'
        assert int(metric.iloc[0]['epoch']) == 20
        assert np.isfinite(metric[['auroc', 'auprc', 'runtime_s']].to_numpy()).all()
        return out
    return None


def run_svdd_upstream(dataset, seed):
    assert torch.cuda.is_available(), 'Chưa có GPU; không bắt đầu train.'
    assert 'load_usad_shared_data' in globals(), 'Nạp ô dữ liệu trong notebook mới trước.'
    device = torch.device('cuda:0')
    ref, prep, full, test_ds, train_ids, val_ids, truth, manifest, _ = load_usad_shared_data(dataset, seed)
    cfg = upstream_spec(dataset, seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ae = UP_NS['CNNAE'](**cfg['model_config']).to(device)
    net = ae.encoder
    del ae
    train_loader = DataLoader(Subset(full, train_ids.tolist()), batch_size=cfg['batch_size'], shuffle=True, drop_last=True, num_workers=0)
    # Default generator for test preserves the same DataLoader RNG consumption as source's per-epoch test.
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    val_loader = DataLoader(Subset(full, val_ids.tolist()), batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    cfg.update({'preprocessing_reference': str(ref), 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name(0), 'auprc_definition': 'average_precision',
        'threshold_policy': 'not_selected_not_computed_here', 'test_score_every_epoch': True,
        'validation_scored': 'final_only_to_avoid_extra_RNG_during_training',
        'checkpoint_contents': 'model_center_and_optimizer_without_rng_no_auto_resume',
        'runtime_definition': 'center_init_training_all_epoch_test_scoring_and_IO_plus_final_val_reload_eval_excluding_preparation'})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = UP_ROOT / f'DeepSVDD_upstream_eval_bn_{dataset}_seed{seed}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'scores').mkdir()
    for name in ['config.json', 'scaler.joblib', 'window_split.npz']:
        shutil.copy2(ref / name, out / name)
    shutil.copytree(UP_CACHE, out / 'upstream_source')
    (out / 'network.py').write_bytes(UP_NETWORK_CODE.encode())
    notes = {
        'purpose': 'Predefined code-behavior comparison, not proof of original paper hyperparameters. Keep old9 runs unchanged; never select main variant by test score.',
        'difference_from_old': 'ONLY intended training-behavior change is BN mode: net.train once before loop, source test() sets eval after epoch1; epochs2..20 use eval with autograd still enabled.',
        'source_match_scope': 'Exact source encoder definitions/center/loss/Adam/LR policy and mode transition. Shared DAMR preprocessing/splits and width16 remain documented adaptations; upstream entry width undefined.',
        'changes': 'Safe batch-axis handling; detached scalar logs; save model+c; finite checks; per-epoch test arrays/metrics for diagnostics; final epoch is main, no best-test checkpoint selection.',
        'rng': 'Source performs test each epoch, consuming a DataLoader iterator seed. This runner preserves that pass; validation/reload probes deferred until after all training. No claim bitwise equivalence across software/hardware.',
        'batchnorm': 'Eval mode does not disable parameter gradients. Running BN buffers should change only during first training epoch.',
        'center': 'Init eval over shuffled/drop_last train loader, not every train ID necessarily included. Epsilon clipping before temporal mean.',
        'threshold': 'No P95/P99/F1 selected here. Saved final validation/test scores support later agreed calibration by CPU.',
    }
    for name, value in [('run_config.json', cfg), ('adapter_notes.json', notes), ('data_manifest.json', manifest)]:
        (out / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(f'DeepSVDD upstream_eval_bn {dataset} seed{seed}: train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ds)}')
    print('Epoch1=train, epoch2..20=eval nhưng vẫn cập nhật gradient. Output:', out, flush=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    c = upstream_init_center(net, train_loader, device)
    torch.save(c.cpu(), out / 'center_initial.pt')
    net.train()  # EXACTLY ONCE, as source DeepSVDDTrainer.train line101
    history, epoch_metrics = [], []
    for epoch in range(1, 21):
        torch.cuda.synchronize()
        epoch_started = time.perf_counter()
        assert net.training == (epoch == 1)
        lr = 1e-4 if epoch < 20 else 1e-5
        for group in optimizer.param_groups:
            group['lr'] = lr
        total, count = 0.0, 0
        for x, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            z = net(x.squeeze(-1).to(device)).mean(-1)
            loss = ((z - c) ** 2).sum(1).mean()
            assert torch.isfinite(loss).item()
            loss.backward()
            assert all(torch.isfinite(p.grad).all().item() for p in net.parameters() if p.grad is not None)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        assert count > 0
        score, label, ids = upstream_scores(net, c, test_loader, device)
        np.testing.assert_array_equal(ids, np.arange(len(test_ds)))
        np.testing.assert_array_equal(label, truth)
        np.save(out / 'scores' / f'epoch_{epoch:03d}.npy', score)
        if epoch == 1:
            np.save(out / 'scores/labels.npy', truth.astype(np.int8))
        tmp = out / 'checkpoint_last.tmp'
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'c': c.cpu(),
            'optimizer_state_dict': optimizer.state_dict(), 'model_config': cfg['model_config'], 'run_config': cfg}, tmp)
        tmp.replace(out / 'checkpoint_last.pt')
        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_started
        bn_batches = [int(m.num_batches_tracked.item()) for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        history.append({'epoch': epoch, 'train_loss': total / count, 'lr': lr, 'trained_windows': count,
                        'training_mode_during_updates': 'train' if epoch == 1 else 'eval',
                        'bn_num_batches_tracked': json.dumps(bn_batches), 'epoch_train_test_checkpoint_s': epoch_seconds})
        epoch_metrics.append({'model': 'DeepSVDD', 'variant': cfg['variant'], 'dataset': dataset, 'seed': seed,
            'noise': 0, 'epoch': epoch, 'auroc': float(roc_auc_score(truth, score)),
            'auprc': float(average_precision_score(truth, score)), 'f1': np.nan, 'flagged_rate': np.nan, 'runtime_s': epoch_seconds})
        pd.DataFrame(history).to_csv(out / 'epoch_history.csv', index=False)
        pd.DataFrame(epoch_metrics).to_csv(out / 'epoch_metrics.csv', index=False)
        msg = f"{dataset} seed{seed} upstream_eval_bn epoch {epoch:02d}/20 | loss={total/count:.6f} AUROC={epoch_metrics[-1]['auroc']:.6f} AP={epoch_metrics[-1]['auprc']:.6f}"
        print(msg, flush=True)
        with (out / 'train.log').open('a', encoding='utf-8') as handle:
            handle.write(msg + '\n')
    torch.cuda.synchronize()
    training_s = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    saved = torch.load(out / 'checkpoint_last.pt', map_location='cpu', weights_only=True)
    assert saved['epoch'] == 20
    restored = UP_NS['CNNAE'](**saved['model_config']).encoder.to(device)
    restored.load_state_dict(saved['model_state_dict'], strict=True)
    restored.eval()
    saved_c = saved['c'].to(device)
    probe = next(iter(test_loader))[0][:8].squeeze(-1).to(device)
    with torch.no_grad():
        torch.testing.assert_close(c, saved_c)
        torch.testing.assert_close(((net(probe).mean(-1) - c) ** 2).sum(1),
                                  ((restored(probe).mean(-1) - saved_c) ** 2).sum(1))
    vs, vy, vi = upstream_scores(restored, saved_c, val_loader, device)
    np.testing.assert_array_equal(vi, val_ids)
    np.save(out / 'scores/val_epoch_020.npy', vs)
    def save_frame(values, labels, indices, base, split):
        frame = pd.DataFrame({'split': split, 'window_id': indices, 'start': base.starts[indices],
            'end_exclusive': base.starts[indices] + 60, 'label': labels, 'score': values}).sort_values('window_id')
        assert not frame.window_id.duplicated().any() and np.isfinite(frame.score).all()
        frame.to_csv(out / f'{split}_scores.csv', index=False)
    save_frame(vs, vy, vi, full, 'validation')
    save_frame(score, truth, ids, test_ds, 'test')
    torch.cuda.synchronize()
    evaluation_s = time.perf_counter() - evaluation_started
    final = dict(epoch_metrics[-1], runtime_s=training_s + evaluation_s)
    pd.DataFrame([final]).to_csv(out / 'metrics.csv', index=False)
    (out / 'completed.json').write_text(json.dumps({'status': 'completed', 'epoch': 20,
        'variant': cfg['variant'], 'checkpoint_reload_matches': True,
        'validation_windows': len(val_ids), 'test_windows': len(test_ds),
        'training_including_epoch_tests_s': training_s, 'final_eval_s': evaluation_s,
        'f1_status': 'not_selected_not_computed'}, indent=2), encoding='utf-8')
    print(pd.DataFrame([final]).to_string(index=False))
    print('HOÀN THÀNH:', out)
    del net, restored, optimizer, saved, probe, z, loss
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_svdd_upstream_if_missing(dataset, seed):
    out = find_completed_upstream(dataset, seed)
    if out is not None:
        print(f'BỎ QUA — đã hoàn tất upstream_eval_bn {dataset} seed{seed}', flush=True)
        return out
    return run_svdd_upstream(dataset, seed)


def summarize_svdd_upstream():
    records, inventory = [], []
    for dataset in ('PSM', 'MSL', 'SMD'):
        for seed in (16, 17, 18):
            folder = find_completed_upstream(dataset, seed)
            assert folder is not None, f'Còn thiếu upstream_eval_bn {dataset} seed{seed}'
            records.append(pd.read_csv(folder / 'metrics.csv'))
            inventory.append({'model': 'DeepSVDD', 'variant': 'upstream_eval_bn', 'dataset': dataset,
                              'seed': seed, 'run_dir': str(folder), 'checkpoint': str(folder / 'checkpoint_last.pt')})
    result = pd.concat(records, ignore_index=True).sort_values(['dataset', 'seed']).reset_index(drop=True)
    summary = result.groupby(['model', 'variant', 'dataset']).agg(n_seeds=('seed', 'nunique'),
        auroc_mean=('auroc', 'mean'), auroc_std=('auroc', 'std'),
        auprc_mean=('auprc', 'mean'), auprc_std=('auprc', 'std'), runtime_s_mean=('runtime_s', 'mean')).reset_index()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = UP_ROOT / f'SUMMARY_upstream_eval_bn_{stamp}'
    out.mkdir(exist_ok=False)
    result.to_csv(out / 'results_9_LAST.csv', index=False)
    summary.to_csv(out / 'summary_3_LAST.csv', index=False)
    pd.DataFrame(inventory).to_csv(out / 'run_inventory.csv', index=False)
    print('ĐỦ9 LƯỢT MỚI. Không thay bảng45 lượt cũ:', out)
    print(summary.to_string(index=False))
    return out
