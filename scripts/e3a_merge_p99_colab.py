"""Read-only input audit + new 45-run P99 report. No torch, training or GPU.

Upload this file, original results_45_runs.csv + run_inventory.csv, and the
9-run DeepSVDD ZIP to Colab. Call merge_nhk_p99(...) explicitly; importing or
executing this file only defines functions. Originals are never overwritten.
"""
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

KEYS = ['model', 'dataset', 'seed']
SCHEMA = ['model', 'variant', 'dataset', 'seed', 'noise', 'epoch',
          'auroc', 'auprc', 'f1', 'flagged_rate', 'runtime_s']
VARIANTS = {
    'MTGFlow': 'base', 'DeepSVDD': 'upstream_eval_bn',
    'USAD': 'shared_standard_sigmoid', 'ALOCC': 'mtgflow_cnn_width16_safe_probe',
    'GANF': 'official_water_augLag_shared_standard',
}
COUNTS = {'PSM': (8779, 2648), 'MSL': (7367, 1165), 'SMD': (70837, 14167)}
POLICY_ID = 'validation_p99_linear_strict_gt_window_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _csv(content):
    return pd.read_csv(io.BytesIO(content), float_precision='round_trip')


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _grid(frame, models):
    _require(set(KEYS).issubset(frame.columns), 'Inventory/results thiếu cột định danh.')
    seeds = pd.to_numeric(frame['seed'], errors='raise').to_numpy(dtype=float)
    _require(np.isfinite(seeds).all() and np.isin(seeds, [16, 17, 18]).all(), 'Seed không hợp lệ.')
    frame = frame.copy()
    frame['seed'] = seeds.astype(int)
    expected = {(m, d, s) for m in models for d in COUNTS for s in (16, 17, 18)}
    actual = set(frame[KEYS].itertuples(index=False, name=None))
    _require(len(frame) == len(expected) and not frame.duplicated(KEYS).any()
             and actual == expected, 'Thiếu/trùng/sai tổ hợp model-dataset-seed.')
    return frame


def _scores(content, dataset, split):
    frame = _csv(content)
    columns = ['window_id', 'start', 'end_exclusive', 'label', 'score']
    _require(set(columns).issubset(frame.columns), f'{dataset}/{split}: thiếu cột score CSV.')
    for name in columns:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
        values = frame[name].to_numpy(dtype=np.float64)
        _require(np.isfinite(values).all(), f'{dataset}/{split}/{name}: chứa NaN/Inf.')
        if name != 'score':
            _require((values == np.floor(values)).all(), f'{name}: không được là số thập phân.')
    expected_n = COUNTS[dataset][0 if split == 'test' else 1]
    _require(len(frame) == expected_n, f'{dataset}/{split}: sai số window.')
    _require(not frame.window_id.duplicated().any(), 'Trùng window_id.')
    frame = frame.sort_values('window_id').reset_index(drop=True)
    _require((frame.window_id >= 0).all(), 'window_id âm.')
    np.testing.assert_array_equal(frame.start, frame.window_id * 10)
    np.testing.assert_array_equal(frame.end_exclusive, frame.start + 60)
    if 'split' in frame:
        _require((frame['split'] == split).all(), 'Cột split không khớp tên file.')
    if split == 'test':
        _require(np.isin(frame.label, [0, 1]).all(), 'Test label phải là 0/1, không cast nhãn sai thành int.')
        _require(frame.label.nunique() == 2, 'Test phải có cả hai lớp để đối chiếu AUROC.')
        np.testing.assert_array_equal(frame.window_id, np.arange(len(frame)))
    else:
        _require(np.isin(frame.label, [-1, 0, 1]).all(), 'Validation label ngoài -1/0/1.')
    return frame


def _metric_match(frame, expected):
    _require(len(frame) == 1 and set(SCHEMA).issubset(frame.columns), 'metrics.csv sai schema/số dòng.')
    row = frame.iloc[0]
    for col in ['model', 'variant', 'dataset', 'seed', 'noise', 'epoch']:
        _require(row[col] == expected[col], f'metrics.csv khác summary ở {col}.')
    values = frame[['auroc', 'auprc', 'runtime_s']].to_numpy(dtype=float)
    _require(np.isfinite(values).all(), 'Metric/runtime chứa NaN/Inf.')
    _require(0 <= row.auroc <= 1 and 0 <= row.auprc <= 1 and row.runtime_s >= 0, 'Metric ngoài miền hợp lệ.')
    np.testing.assert_allclose(values[0], [expected[c] for c in ['auroc', 'auprc', 'runtime_s']], rtol=1e-10, atol=1e-10)


def merge_nhk_p99(old_results_path, old_inventory_path, svdd_zip_path,
                  output_parent='/content/drive/MyDrive/NHK_BASELINES_UPDATED',
                  path_map=None):
    """Use exact inventories, never discover/select a run by test performance.

    path_map optionally maps old Drive path prefixes to explicitly chosen new
    prefixes if a shortcut was named differently. Default paths stay unchanged.
    """
    started = time.perf_counter()
    originals = {}
    for name, path in [('old_results', old_results_path), ('old_inventory', old_inventory_path), ('svdd_zip', svdd_zip_path)]:
        p = Path(path)
        _require(p.is_file(), f'Thiếu input: {p}')
        originals[name] = p.read_bytes()
    old = _grid(_csv(originals['old_results']), VARIANTS)
    old_inventory = _grid(_csv(originals['old_inventory']), VARIANTS)
    _require(set(SCHEMA).issubset(old.columns), 'CSV cũ thiếu schema.')
    _require(old.f1.isna().all() and old.flagged_rate.isna().all(), 'CSV cũ đã có F1: cần kiểm tra policy trước.')
    for model, variant in VARIANTS.items():
        expected = 'mtgflow_repo_cnn_no_pretrain' if model == 'DeepSVDD' else variant
        _require((old.loc[old.model == model, 'variant'] == expected).all(), f'Sai variant cũ: {model}')
    old_index = old.set_index(KEYS)
    folder_index = old_inventory.set_index(KEYS)

    def remap(path):
        value = str(path)
        for prefix, replacement in sorted((path_map or {}).items(), key=lambda x: -len(str(x[0]))):
            prefix = str(prefix).rstrip('/')
            if value == prefix or value.startswith(prefix + '/'):
                return Path(str(replacement).rstrip('/') + value[len(prefix):])
        return Path(value)

    # Preflight all 36 exact old folders before doing work or writing output.
    old_files = ['metrics.csv', 'run_config.json', 'config.json', 'completed.json',
                 'validation_scores.csv', 'test_scores.csv']
    missing = []
    for row in old_inventory.itertuples(index=False):
        if row.model == 'DeepSVDD':
            continue
        folder = remap(row.run_dir)
        missing.extend(str(folder / name) for name in old_files if not (folder / name).is_file())
    _require(not missing, 'Chưa đọc được các file sau (kiểm tra share + lối tắt TK1/TK2):\n' + '\n'.join(missing))

    rows, thresholds, audit, inventory, runtime_notes = [], [], [], [], []
    with ZipFile(io.BytesIO(originals['svdd_zip'])) as z:
        _require(len(z.namelist()) == len(set(z.namelist())), 'ZIP chứa tên file trùng.')
        summaries = [n.rsplit('/', 1)[0] for n in z.namelist() if n.endswith('/results_9_LAST.csv')]
        _require(len(summaries) == 1, 'ZIP phải có đúng một summary 9 lượt; không tự chọn summary.')
        summary_name = summaries[0]
        new = _grid(_csv(z.read(summary_name + '/results_9_LAST.csv')), ['DeepSVDD'])
        new_inv = _grid(_csv(z.read(summary_name + '/run_inventory.csv')), ['DeepSVDD'])
        _require((new.variant == 'upstream_eval_bn').all() and (new.epoch == 20).all(), 'Sai variant/epoch SVDD mới.')
        new_index = new.set_index(KEYS)
        new_inv_index = new_inv.set_index(KEYS)

        # MTGFlow provides the existing window identity/label reference, not a
        # model or seed selected by test performance.
        references = {}
        for dataset in COUNTS:
            for seed in (16, 17, 18):
                folder = remap(folder_index.loc[('MTGFlow', dataset, seed), 'run_dir'])
                references[(dataset, seed)] = {
                    split: _scores((folder / f'{split}_scores.csv').read_bytes(), dataset, split)
                    for split in ['validation', 'test']
                }

        for model in VARIANTS:
            for dataset in COUNTS:
                for seed in (16, 17, 18):
                    key = (model, dataset, seed)
                    if model == 'DeepSVDD':
                        selected = new_index.loc[key]
                        inv_row = new_inv_index.loc[key]
                        run_dir = str(inv_row['run_dir'])
                        prefix = PurePosixPath(run_dir).name

                        def get(name):
                            return z.read(prefix + '/' + name)

                        source_kind = 'svdd_zip'
                        score_source = str(svdd_zip_path) + '!' + prefix
                    else:
                        selected = old_index.loc[key]
                        inv_row = folder_index.loc[key]
                        folder = remap(inv_row['run_dir'])
                        run_dir = str(folder)

                        def get(name):
                            return (folder / name).read_bytes()

                        source_kind = 'drive_original'
                        score_source = run_dir
                    expected = dict(zip(KEYS, key), **selected.to_dict())
                    metric_bytes = get('metrics.csv')
                    _metric_match(_csv(metric_bytes), expected)
                    cfg = json.loads(get('run_config.json'))
                    prep = json.loads(get('config.json'))
                    status = json.loads(get('completed.json'))
                    _require(status.get('status') == 'completed', f'{key}: chưa hoàn tất.')
                    _require((cfg.get('model'), cfg.get('dataset'), cfg.get('seed')) == key, f'{key}: config sai định danh.')
                    _require(cfg.get('variant', VARIANTS[model]) == VARIANTS[model], f'{key}: config sai variant.')
                    _require(expected['noise'] == 0, 'Chỉ ghép protocol noise=0.')
                    _require(cfg.get('protocol_id') == f'DAMR_public_code_{dataset}_v1', 'Sai protocol_id.')
                    _require(prep.get('dataset') == dataset and prep.get('seed') == seed
                             and prep.get('window') == 60 and prep.get('stride') == 10
                             and prep.get('scaler_fit_on') == 'full_train_file_before_window_split', 'Sai preprocessing config.')
                    if model == 'GANF':
                        _require(cfg.get('checkpoint_policy') == 'final_scheduled_epoch_no_test_selection'
                                 and status.get('epoch') == expected['epoch'], 'GANF không phải final scheduled epoch.')
                    else:
                        epoch = {'MTGFlow': 40, 'DeepSVDD': 20, 'USAD': 100, 'ALOCC': 40}[model]
                        _require(cfg.get('checkpoint_policy') == 'last_epoch_fixed_in_advance'
                                 and cfg.get('epochs') == expected['epoch'] == epoch, 'Sai fixed-last policy.')
                    if model == 'DeepSVDD':
                        _require(cfg.get('variant') == 'upstream_eval_bn', 'Thiếu variant SVDD mới.')
                        _require(status.get('epoch') == 20 and status.get('checkpoint_reload_matches') is True, 'SVDD thiếu trạng thái reload/final 20.')
                        _require(cfg.get('bn_policy') == 'epoch1_train_epoch2_20_eval_gradients_enabled', 'Sai BN policy.')
                        _require(cfg.get('batch_size') == 512 and cfg.get('lr') == 1e-4
                                 and cfg.get('weight_decay') == 5e-4 and cfg.get('pretraining_epochs') == 0,
                                 'SVDD khác hyperparameter đã chốt.')
                        _require(cfg.get('model_config') == {'in_channels': {'PSM': 25, 'MSL': 55, 'SMD': 38}[dataset],
                                                           'n_channels': 16, 'kernel_size': 5}, 'SVDD sai kiến trúc.')
                        history = _csv(get('epoch_history.csv'))
                        np.testing.assert_array_equal(history.epoch, np.arange(1, 21))
                        _require(history.training_mode_during_updates.tolist() == ['train'] + ['eval'] * 19, 'Mode không khớp source.')
                        counters = [json.loads(value) for value in history.bn_num_batches_tracked]
                        expected_batches = {'PSM': 10595, 'MSL': 4661, 'SMD': 56668}[dataset] // 512
                        _require(all(value == [expected_batches] * 3 for value in counters), 'BN counters không khớp epoch1-only.')
                        np.testing.assert_allclose(history.lr, [1e-4] * 19 + [1e-5], rtol=1e-10, atol=0)
                        _require(np.isfinite(history.train_loss.to_numpy(dtype=float)).all(), 'History loss không hữu hạn.')
                        per_epoch = _csv(get('epoch_metrics.csv'))
                        np.testing.assert_array_equal(per_epoch.epoch, np.arange(1, 21))
                        _metric_match(per_epoch.iloc[[-1]], expected | {'runtime_s': per_epoch.iloc[-1].runtime_s})

                    val_bytes, test_bytes = get('validation_scores.csv'), get('test_scores.csv')
                    val = _scores(val_bytes, dataset, 'validation')
                    test = _scores(test_bytes, dataset, 'test')
                    for split, frame in [('validation', val), ('test', test)]:
                        ref = references[(dataset, seed)][split]
                        for col in ['window_id', 'start', 'end_exclusive', 'label']:
                            np.testing.assert_array_equal(frame[col], ref[col], err_msg=f'{key}/{split}/{col}')
                    if model == 'DeepSVDD':
                        ref_name = PurePosixPath(str(folder_index.loc[('MTGFlow', dataset, seed), 'run_dir'])).name
                        _require(PurePosixPath(cfg['preprocessing_reference']).name == ref_name, 'SVDD không dùng reference MTGFlow đã chọn.')
                    y = test.label.to_numpy(dtype=np.int64)  # only AFTER exact binary-label validation
                    scores = test.score.to_numpy(dtype=np.float64)
                    vs = val.score.to_numpy(dtype=np.float64)
                    auc, ap = float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))
                    np.testing.assert_allclose([auc, ap], [expected['auroc'], expected['auprc']], rtol=1e-8, atol=1e-8,
                                               err_msg=f'{key}: AUROC/AP từ score khác CSV gốc; dừng, không tự sửa.')
                    threshold = float(np.quantile(vs, .99, method='linear'))
                    _require(np.isfinite(threshold), 'Threshold NaN/Inf.')
                    pred = scores > threshold
                    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
                    calculated = {
                        'f1': float(f1_score(y, pred, zero_division=0)),
                        'flagged_rate': float(pred.mean()),
                        'precision': float(precision_score(y, pred, zero_division=0)),
                        'recall': float(recall_score(y, pred, zero_division=0)),
                        'threshold': threshold, 'threshold_policy': POLICY_ID,
                        'n_test': len(test), 'n_validation': len(val),
                        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                    }
                    rows.append({c: expected[c] for c in SCHEMA} | calculated)
                    thresholds.append(dict(zip(KEYS, key), variant=VARIANTS[model], **{
                        'policy_id': POLICY_ID, 'q': .99, 'quantile_method': 'linear', 'comparison': 'strict_gt',
                        'threshold': threshold, 'n_validation': len(val), 'validation_sha256': _sha(val_bytes),
                        'test_sha256': _sha(test_bytes), 'validation_flagged_rate': float((vs > threshold).mean()),
                        'validation_unique_scores': int(np.unique(vs).size), 'constant_validation': bool(np.ptp(vs) == 0),
                        'constant_test': bool(np.ptp(scores) == 0),
                    }))
                    audit.append(dict(zip(KEYS, key), variant=VARIANTS[model], auroc_saved=expected['auroc'],
                                      auroc_recomputed=auc, auprc_saved=expected['auprc'], auprc_recomputed=ap,
                                      window_identity_labels_match_mtgflow=True, metrics_sha256=_sha(metric_bytes)))
                    inventory.append(dict(zip(KEYS, key), variant=VARIANTS[model], run_dir=run_dir,
                                          checkpoint=str(inv_row['checkpoint']), score_source_kind=source_kind,
                                          score_source=score_source, n_test=len(test), n_validation=len(val)))
                    runtime_notes.append(dict(zip(KEYS, key), runtime_s=expected['runtime_s'],
                                              runtime_definition=cfg.get('runtime_definition', 'not_recorded')))
                    print(f'OK {model:8s} {dataset} seed{seed}: score/label/metric hợp lệ; đã tính P99.', flush=True)

        # No files are written until all 45 runs have passed the above checks.
        result = _grid(pd.DataFrame(rows), VARIANTS).sort_values(KEYS).reset_index(drop=True)
        _require(np.isfinite(result[['f1', 'flagged_rate', 'threshold', 'precision', 'recall']]).all().all(), 'Output chưa hữu hạn.')
        aggregations = {'n_seeds': ('seed', 'nunique')}
        for col in ['auroc', 'auprc', 'f1', 'flagged_rate', 'precision', 'recall', 'runtime_s']:
            aggregations[col + '_mean'] = (col, 'mean')
            aggregations[col + '_std'] = (col, 'std')  # pandas sample std ddof=1
        summary = result.groupby(['model', 'variant', 'dataset']).agg(**aggregations).reset_index()
        _require(len(summary) == 15 and (summary.n_seeds == 3).all(), 'Summary không đủ 15 nhóm x 3 seed.')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        out = Path(output_parent) / ('MERGED_45_LAST_P99_' + stamp)
        out.mkdir(parents=True, exist_ok=False)
        result.to_csv(out / 'results_45_LAST_P99.csv', index=False)
        summary.to_csv(out / 'summary_15_LAST_P99.csv', index=False)
        pd.DataFrame(thresholds).to_csv(out / 'thresholds_45_P99.csv', index=False)
        pd.DataFrame(audit).to_csv(out / 'score_audit_45.csv', index=False)
        pd.DataFrame(inventory).to_csv(out / 'run_inventory_45_UPDATED.csv', index=False)
        pd.DataFrame(runtime_notes).to_csv(out / 'runtime_definitions_45.csv', index=False)
        old.loc[old.model == 'DeepSVDD'].to_csv(out / 'SVDD_OLD_9_REFERENCE_ONLY.csv', index=False)
        (out / 'results_45_runs_ORIGINAL_COPY.csv').write_bytes(originals['old_results'])
        (out / 'run_inventory_ORIGINAL_COPY.csv').write_bytes(originals['old_inventory'])
        for name in ['results_9_LAST.csv', 'summary_3_LAST.csv', 'run_inventory.csv']:
            (out / ('SVDD_NEW_' + name)).write_bytes(z.read(summary_name + '/' + name))

    policy = {
        'policy_id': POLICY_ID, 'q': .99, 'quantile_method': 'linear', 'comparison': 'score > threshold',
        'calibration': 'All saved final validation scores, without filtering label==-1 or label==0.',
        'evaluation': 'Binary window labels; sklearn F1/precision/recall zero_division=0; no point adjustment.',
        'selection': 'Exact old inventory for 36 non-SVDD runs + exact uploaded ZIP inventory for 9 upstream_eval_bn runs. No best-test selection.',
        'timing': 'P99 postprocessing decided after observing AUROC/AP; not preregistered. No test-F1 sweep performed by this script.',
        'limitations': 'Validation windows overlap and scaler fit uses full train. Normality not independently proven; P99 does not guarantee 1% test alerts/FPR.',
        'runtime': 'Original runtime_s preserved; postprocessing separate. New SVDD includes 20 test passes unlike old final-only runners; hardware also differs. Do not infer pure training speed from runtime comparison.',
        'audit_scope': 'Scores/identities/labels/configs/completion metadata/metrics/BN history. No independent checkpoint inference, no scaler pickle loading, no raw-data or checkpoint hash verification.',
        'inputs': {name: {'path': str(path), 'sha256': _sha(originals[name])}
                   for name, path in [('old_results', old_results_path), ('old_inventory', old_inventory_path), ('svdd_zip', svdd_zip_path)]},
        'path_map': {str(k): str(v) for k, v in (path_map or {}).items()},
        'postprocess_runtime_s': time.perf_counter() - started,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    (out / 'threshold_policy.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'README.md').write_text(
        '# 45 baseline runs: last + validation P99\n\n'
        '36 non-SVDD original runs + 9 DeepSVDD upstream_eval_bn. Original 45 and all run artifacts unchanged.\n\n'
        'F1/precision/recall are window-level, NOT point-adjusted. Threshold is validation P99 linear, test score strictly greater. All metric/rate values in [0,1]. Summary std uses ddof=1 across 3 seeds.\n\n'
        'See threshold_policy.json for post-hoc policy disclosure, audit limits and timing. Runtime_s is preserved, NOT harmonized across hardware/evaluation schedules.\n\n'
        'The report contains CSV/provenance, NOT the 45 model checkpoints. Preserve/share all source Drive folders and the SVDD ZIP.\n', encoding='utf-8')
    checksums = {p.name: _sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}
    (out / 'report_checksums.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')
    print('\nĐỦ 45 LƯỢT. Không sửa file gốc. Output:', out)
    print(summary.to_string(index=False))
    return out, result, summary


print('Đã nạp script ghép/P99. CHƯA tính và KHÔNG train; gọi merge_nhk_p99(...) để bắt đầu.')
