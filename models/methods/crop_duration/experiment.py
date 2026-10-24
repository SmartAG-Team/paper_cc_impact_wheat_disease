"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_duration_20261010')
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from analysis.methods.wheat_phenology.wheat_transfer import HORIZON, STAGES, WheatPhenology, digest, predict_batches, quantiles, validate_splits
HERE = evidence_dir('analysis/paper_study/phenology_duration_20261010')
ROOT = PROJECT_ROOT
SOURCE = HERE.parent / 'phenology_transfer'
SEEDS = [17, 29, 43]
CANDIDATES = {'duration_0p25': 0.25, 'duration_1': 1.0}
TOLERANCE = 1e-06

def json_write(path, value):

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')

def file_record(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), sha256=digest(path))

def verify_records(records):
    for record in records:
        path = Path(record['path'])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or digest(path) != record['sha256']:
            raise ValueError('Frozen input changed: ' + str(path))

def crps_values(cdf, targets):
    cdf, targets = (np.asarray(cdf), np.asarray(targets))
    truth = np.arange(cdf.shape[2])[None, None, :] >= targets[:, :, None]
    values = np.square(cdf.astype(np.float64) - truth).sum(2)
    return np.where(targets >= 0, values, np.nan)

def restricted_duration(cdf):
    """Expected within-horizon heading-to-yellow-ripeness occupancy, in days."""
    return (cdf[:, 2, :] - cdf[:, 3, :]).sum(1)

def duration_regularizer(cdf, targets):
    """Empirical absolute-error penalty on restricted expected duration.

    This is not a proper joint probabilistic score. Only two observed endpoint
    dates inside the supplied horizon supervise it; no targets enter features.
    """
    known = (targets[:, 2] >= 0) & (targets[:, 3] >= 0)
    if torch.any(known & (targets[:, 3] < targets[:, 2])):
        raise ValueError('Conflicting heading and terminal observation order')
    valid = known & (targets[:, 2] < cdf.shape[2]) & (targets[:, 3] < cdf.shape[2])
    if not torch.any(valid):
        return cdf.sum() * 0.0
    observed = (targets[:, 3] - targets[:, 2]).to(cdf.dtype)
    return (restricted_duration(cdf)[valid] - observed[valid]).abs().mean()

def fitting_loss(cdf, targets, stage_weights, duration_weight):
    truth = torch.arange(cdf.shape[2])[None, None, :] >= targets[:, :, None]
    valid = targets >= 0
    scores = ((cdf - truth.double()) ** 2).sum(2)
    denominator = (valid * stage_weights[None]).sum()
    if denominator <= 0:
        raise ValueError('Training batch contains no observed stages')
    crps = (scores * valid * stage_weights[None]).sum() / denominator
    return crps + duration_weight * duration_regularizer(cdf, targets)

def selection_metrics(cdf, targets, baseline_cdf):
    """Score on baseline-resolved support without hiding lost crossings."""
    targets = np.asarray(targets)
    prediction = quantiles(cdf, levels=(0.5,))[:, :, 0]
    reference = quantiles(baseline_cdf, levels=(0.5,))[:, :, 0]
    observed = targets >= 0
    support = observed & np.isfinite(reference)
    missing = support & ~np.isfinite(prediction)
    errors = prediction - targets
    counts = support.sum(0)
    mae, bias = ([], [])
    for j in range(4):
        values = errors[support[:, j], j]
        mae.append(float(np.mean(np.abs(values))) if len(values) else np.nan)
        bias.append(float(np.mean(values)) if len(values) else np.nan)
    pairs = support[:, 2] & support[:, 3]
    duration_error = (prediction[:, 3] - prediction[:, 2] - (targets[:, 3] - targets[:, 2]))[pairs]
    crps = np.nanmean(crps_values(cdf, targets), axis=0)
    return dict(crps=float(np.mean(crps)), crps_by_stage=crps.tolist(), stage_equal_mae=float(np.mean(mae)), terminal_mae=mae[3], mae_by_stage=mae, bias_by_stage=bias, duration_mae=float(np.mean(np.abs(duration_error))) if pairs.any() else np.nan, duration_bias=float(np.mean(duration_error)) if pairs.any() else np.nan, unresolved_observed_events=int((observed & ~np.isfinite(prediction)).sum()), lost_baseline_resolved_events=int(missing.sum()), lost_baseline_duration_pairs=int((pairs & (~np.isfinite(prediction[:, 2]) | ~np.isfinite(prediction[:, 3]))).sum()), stage_event_counts=counts.tolist(), duration_pairs=int(pairs.sum()))

def choose_candidate(metrics, tolerance=TOLERANCE):
    base = metrics['baseline']
    checks = {}
    passing = []
    for name in sorted(set(metrics) - {'baseline'}):
        candidate = metrics[name]
        numeric = all((candidate.get(k) is not None and np.isfinite(candidate[k]) and (base.get(k) is not None) and np.isfinite(base[k]) for k in ['crps', 'stage_equal_mae', 'terminal_mae', 'duration_mae']))
        checks[name] = dict(finite_metrics=numeric, complete_scoring_support=all((n > 0 for n in candidate['stage_event_counts'])) and candidate['duration_pairs'] > 0, no_lost_baseline_events=candidate['lost_baseline_resolved_events'] == 0, no_lost_baseline_pairs=candidate['lost_baseline_duration_pairs'] == 0, no_more_unresolved=candidate['unresolved_observed_events'] <= base['unresolved_observed_events'], crps_not_worse=numeric and candidate['crps'] <= base['crps'] + tolerance, stage_equal_mae_not_worse=numeric and candidate['stage_equal_mae'] <= base['stage_equal_mae'] + tolerance, terminal_mae_lower=numeric and candidate['terminal_mae'] < base['terminal_mae'] - tolerance, duration_mae_lower=numeric and candidate['duration_mae'] < base['duration_mae'] - tolerance)
        if all(checks[name].values()):
            passing.append(name)
    if passing:
        best = min((metrics[name]['crps'] for name in passing))
        selected = min((name for name in passing if metrics[name]['crps'] <= best + tolerance), key=lambda name: (CANDIDATES[name], name))
    else:
        selected = 'baseline'
    return dict(selected=selected, adopted=bool(passing), checks=checks, tolerance_days=tolerance)

def load_historical():
    """Return read-only cached arrays; callers restrict targets to train/val."""
    manifest = json.loads((SOURCE / 'data_manifest.json').read_text())
    for name, expected in manifest['outputs'].items():
        if digest(SOURCE / 'arrays' / name) != expected:
            raise ValueError('Historical feature/target array changed: ' + name)
    arrays = {name: np.load(SOURCE / 'arrays' / (file + '.npy'), mmap_mode='r') for name, file in [('x', 'features'), ('p', 'progress'), ('inc', 'increments'), ('y', 'targets')]}
    records = pd.read_csv(SOURCE / 'cycles.csv')
    validate_splits(records)
    train = np.flatnonzero(records.partition.eq('train'))
    validation = np.flatnonzero(records.partition.eq('val'))
    if not len(train) or not len(validation):
        raise ValueError('Empty historical partition')
    return (arrays, records, json.loads((SOURCE / 'model_metadata.json').read_text()), train, validation)

def predict_ensemble(package, arrays, indices):
    predictions = []
    for member in package['members']:
        model = WheatPhenology(member['metadata'], member['mode'])
        model.load_state_dict(member['state_dict'])
        predictions.append(predict_batches(model, arrays['x'], arrays['p'], arrays['inc'], indices))
    return np.mean(predictions, axis=0, dtype=np.float64).astype(np.float32)

def freeze():
    torch.set_num_threads(2)
    frozen = HERE / 'protocol.json'
    if frozen.exists():
        protocol = json.loads(frozen.read_text())
        verify_records(protocol['inputs'] + protocol['code'])
        verify_records(protocol['baseline_evidence'])
        return protocol
    arrays, records, metadata, train, validation = load_historical()
    baseline = torch.load(SOURCE / 'selected_wheat_ensemble.pt', map_location='cpu', weights_only=False)
    if any((m['metadata'] != metadata for m in baseline['members'])):
        raise ValueError('Baseline and new-fit preprocessing differ')
    cdf = predict_ensemble(baseline, arrays, validation)
    targets = np.asarray(arrays['y'][validation])
    medians = quantiles(cdf, levels=(0.5,))[:, :, 0]
    support = (targets >= 0) & np.isfinite(medians)
    folder = HERE / 'validation'
    folder.mkdir(exist_ok=True)
    output = folder / 'baseline.npz'
    if output.exists():
        raise FileExistsError('Partial baseline evidence already exists')
    np.savez_compressed(output, indices=validation, cdf=cdf, targets=targets, event_support=support, pair_support=support[:, 2] & support[:, 3])
    records.iloc[validation].to_csv(folder / 'cycles.csv', index=False)
    metrics = selection_metrics(cdf, targets, cdf)
    json_write(folder / 'baseline_metrics.json', metrics)
    observed_pair = (targets[:, 2] >= 0) & (targets[:, 3] >= 0)
    valid_pair = observed_pair & (targets[:, 2] < HORIZON) & (targets[:, 3] < HORIZON)
    inputs = [SOURCE / name for name in ['data_manifest.json', 'model_metadata.json', 'cycles.csv', 'selected_wheat_ensemble.pt']]
    inputs.extend((SOURCE / 'arrays' / name for name in ['features.npy', 'progress.npy', 'increments.npy', 'targets.npy']))
    code = [Path(__file__), HERE / 'test_duration.py', SOURCE / 'wheat_transfer.py']
    protocol = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(), scope='Historical-only crop-model development; no later-year scoring or disease changes', candidates=CANDIDATES, seeds=SEEDS, maximum_epochs=24, patience=5, batch_size=128, threads=2, device='cpu', learning_rate=0.001, weight_decay=0.0001, gradient_clip=1.0, horizon=HORIZON, targets=STAGES, train_cycles=len(train), validation_cycles=len(validation), train_stations=int(records.iloc[train].PEP_ID.nunique()), validation_stations=int(records.iloc[validation].PEP_ID.nunique()), latest_fitting_harvest=int(records.iloc[np.r_[train, validation]].harvest_year.max()), train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(), validation_indices_sha256=hashlib.sha256(validation.tobytes()).hexdigest(), objective='Inverse-observed-stage-frequency restricted CRPS plus lambda times complete-pair absolute error of restricted mean duration', auxiliary_duration='sum(F51-F87) over days0..329; equals E[min(T87,330)-min(T51,330)]', auxiliary_limit='Empirical regularizer, not a proper joint probabilistic score; complete-case and within-horizon supervision', checkpoint_selection='Minimum historical all-observed-event stage-equal restricted CRPS, improvement tolerance0.00001', candidate_selection='On fixed baseline-resolved events and median-duration pairs, all guards in choose_candidate; otherwise baseline', selection_tolerance_days=TOLERANCE, tie_break='smaller lambda within CRPS tolerance', validation_pair_counts=dict(observed=int(observed_pair.sum()), within_horizon=int(valid_pair.sum()), missing=int((~observed_pair).sum()), beyond_horizon=int((observed_pair & ~valid_pair).sum())), no_2025_targets_accessed=True, later_2017_2024_outcomes_previously_inspected=True, inputs=[file_record(p) for p in inputs], code=[file_record(p) for p in code], baseline_evidence=[file_record(p) for p in [output, folder / 'cycles.csv', folder / 'baseline_metrics.json']])
    json_write(frozen, protocol)
    return protocol

def train():
    protocol = freeze()
    selection_path = HERE / 'selection.json'
    if selection_path.exists():
        selected = json.loads(selection_path.read_text())
        if selected.get('status') != 'selected_before_2025_access':
            raise ValueError('Unrecognized existing historical selection status')
        verify_records([selected['protocol'], selected['selected_checkpoint'], selected['baseline_checkpoint']] + selected['checkpoints'] + selected['validation_evidence'])
        return selected
    torch.set_num_threads(protocol['threads'])
    arrays, records, metadata, training, validation = load_historical()
    if hashlib.sha256(training.tobytes()).hexdigest() != protocol['train_indices_sha256']:
        raise ValueError('Training indices changed')
    baseline_data = np.load(HERE / 'validation/baseline.npz')
    baseline_cdf, validation_targets = (baseline_data['cdf'], baseline_data['targets'])
    counts = (arrays['y'][training] >= 0).sum(0)
    if np.any(counts == 0):
        raise ValueError('A training stage is unobserved')
    weights = torch.tensor(counts.max() / counts, dtype=torch.float64)
    checkpoints = HERE / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    metrics = {'baseline': selection_metrics(baseline_cdf, validation_targets, baseline_cdf)}
    checkpoint_records = []
    for candidate, coefficient in protocol['candidates'].items():
        members = []
        for seed in protocol['seeds']:
            stem = f'{candidate}_s{seed}'
            path = checkpoints / f'{stem}.pt'
            report_path = checkpoints / f'{stem}.json'
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if digest(path) != report['checkpoint_sha256'] or report['protocol_sha256'] != digest(HERE / 'protocol.json'):
                    raise ValueError('Existing fit differs from its protocol or receipt')
                members.append(torch.load(path, map_location='cpu', weights_only=False))
                checkpoint_records.append(file_record(path))
                continue
            if path.exists():
                raise FileExistsError('Partial checkpoint without receipt: ' + str(path))
            torch.manual_seed(seed)
            generator = np.random.default_rng(seed)
            model = WheatPhenology(metadata, 'ordered_lstm')
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
            best = np.inf
            best_state = None
            best_epoch = 0
            stale = 0
            history = []
            started = time.time()
            for epoch in range(1, protocol['maximum_epochs'] + 1):
                model.train()
                order = generator.permutation(training)
                total = 0.0
                tick = time.time()
                for offset in range(0, len(order), protocol['batch_size']):
                    ix = order[offset:offset + protocol['batch_size']]
                    x, p, inc, y = [torch.as_tensor(np.asarray(arrays[k][ix])) for k in ['x', 'p', 'inc', 'y']]
                    optimizer.zero_grad(set_to_none=True)
                    loss = fitting_loss(model(x, p, inc), y, weights, coefficient)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite fitting objective: {stem}/{epoch}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
                    optimizer.step()
                    total += float(loss.detach()) * len(ix)
                predicted = predict_batches(model, arrays['x'], arrays['p'], arrays['inc'], validation)
                score = float(np.nanmean(crps_values(predicted, validation_targets), axis=0).mean())
                if not np.isfinite(score):
                    raise FloatingPointError(f'Nonfinite historical validation score: {stem}/{epoch}')
                history.append(dict(epoch=epoch, training_objective=total / len(training), validation_crps=score, seconds=time.time() - tick))
                if score < best - 1e-05:
                    best = score
                    best_epoch = epoch
                    stale = 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    stale += 1
                print(f'{stem} epoch={epoch} validation_CRPS={score:.6f} best={best:.6f}', flush=True)
                if stale >= protocol['patience']:
                    break
            package = dict(schema_version=1, mode='ordered_lstm', metadata=metadata, state_dict=best_state, seed=seed, best_epoch=best_epoch, duration_weight=coefficient, stage_codes=STAGES, maize_weights_used=False, fit_protocol_sha256=digest(HERE / 'protocol.json'))
            torch.save(package, path)
            json_write(report_path, dict(status='complete', seed=seed, candidate=candidate, best_epoch=best_epoch, validation_crps=best, history=history, seconds=time.time() - started, protocol_sha256=digest(HERE / 'protocol.json'), checkpoint_sha256=digest(path), post2015_target_use=False))
            members.append(package)
            checkpoint_records.append(file_record(path))
        ensemble = dict(schema_version=1, model_family='ordered_lstm', members=members, stages=STAGES, maize_weights_used=False, weight_policy='equal mean of three stage CDFs', horizon_days=HORIZON)
        ensemble_path = HERE / f'{candidate}_ensemble.pt'
        torch.save(ensemble, ensemble_path)
        cdf = predict_ensemble(ensemble, arrays, validation)
        np.savez_compressed(HERE / 'validation' / f'{candidate}.npz', indices=validation, cdf=cdf, targets=validation_targets)
        metrics[candidate] = selection_metrics(cdf, validation_targets, baseline_cdf)
        checkpoint_records.append(file_record(ensemble_path))
    decision = choose_candidate(metrics)
    selected_path = SOURCE / 'selected_wheat_ensemble.pt' if decision['selected'] == 'baseline' else HERE / f"{decision['selected']}_ensemble.pt"
    decision.update(status='selected_before_2025_access', created_at_utc=datetime.now(timezone.utc).isoformat(), selected_checkpoint=file_record(selected_path), baseline_checkpoint=file_record(SOURCE / 'selected_wheat_ensemble.pt'), protocol=file_record(HERE / 'protocol.json'), checkpoints=checkpoint_records, validation_metrics=metrics, selection_is_independent_validation=False, post2015_targets_used_for_selection=False)
    pd.DataFrame([dict(model=name, **value) for name, value in metrics.items()]).to_csv(HERE / 'validation_metrics.csv', index=False)
    evidence = [HERE / 'validation_metrics.csv', *sorted((HERE / 'validation').glob('*')), *sorted(checkpoints.glob('*.json'))]
    decision['validation_evidence'] = [file_record(path) for path in evidence if path.is_file()]
    json_write(selection_path, decision)
    return decision
