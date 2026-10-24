"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_residual_ml_20261012')
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analysis.methods.crop_duration.experiment import crps_values

def daily_features(x, progress, increments, base_cdf):
    x = np.asarray(x, dtype=np.float32)
    progress, increments, base = map(np.asarray, (progress, increments, base_cdf))
    if x.ndim != 3 or base.shape != (len(x), 4, x.shape[1]) or progress.shape != x.shape[:2] or (increments.shape != x.shape[:2]):
        raise ValueError('Incompatible daily predictor shapes')
    state = base.transpose(0, 2, 1).astype(np.float32)
    previous = np.concatenate([np.zeros_like(state[:, :1]), state[:, :-1]], axis=1)
    result = np.concatenate([x, state, state - previous, (progress / 1000)[..., None], (increments / 20)[..., None]], axis=2).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Daily predictors must be finite')
    return result

def forward_project(proposal, increments):
    proposal = np.asarray(proposal)
    increments = np.asarray(increments)
    if proposal.ndim != 3 or proposal.shape[1] != 4 or increments.shape != (len(proposal), proposal.shape[2]):
        raise ValueError('Incompatible probability/increment shapes')
    if not np.isfinite(proposal).all() or not np.isfinite(increments).all() or np.any(increments < 0):
        raise ValueError('Finite proposals and nonnegative thermal increments are required')
    ordered = np.minimum.accumulate(np.clip(proposal, 0, 1), axis=1)
    output = np.zeros_like(ordered, dtype=np.float32)
    state = np.zeros(proposal.shape[:2], dtype=np.float32)
    for day in range(proposal.shape[2]):
        active = increments[:, day] > 0
        state[active] = np.maximum(state[active], ordered[active, :, day])
        output[:, :, day] = state
    return output

def sample_rows(features, base_cdf, targets, stage, stride=30, levels=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Label-free day sampling with Voronoi day-count quadrature weights.

    Each observed cycle-stage contributes exactly ``horizon`` total day weight.
    A missing stage contributes no row. An event after the horizon remains an
    unreached label throughout; it is never silently clipped to the last day.
    """
    features, base, targets = map(np.asarray, (features, base_cdf, targets))
    horizon = features.shape[1]
    if stride < 1 or stage not in range(4) or targets.shape != (len(features), 4) or (base.shape != (len(features), 4, horizon)):
        raise ValueError('Invalid residual sampling shapes/settings')
    cycle_rows, day_rows, weights = ([], [], [])
    grid = np.unique(np.r_[np.arange(0, horizon, stride), horizon - 1])
    for cycle in np.flatnonzero(targets[:, stage] >= 0):
        series = base[cycle, stage]
        quantile_days = [int(np.argmax(series >= level)) for level in levels if series[-1] >= level]
        days = np.unique(np.r_[grid, quantile_days])
        edges = np.r_[0, (days[:-1] + days[1:]) // 2 + 1, horizon]
        cycle_rows.extend([cycle] * len(days))
        day_rows.extend(days)
        weights.extend(np.diff(edges))
    cycles = np.asarray(cycle_rows, dtype=np.int64)
    days = np.asarray(day_rows, dtype=np.int64)
    response = (days >= targets[cycles, stage]).astype(np.float32) - base[cycles, stage, days]
    return dict(cycle=cycles, day=days, features=features[cycles, days], target=response, weight=np.asarray(weights, dtype=np.float64))

def station_membership(records, folds=3, seed=20261012):
    train = records.loc[records.partition.eq('train')]
    val = records.loc[records.partition.eq('val')].copy()
    reserved = records.loc[records.historical_station_group.eq('test')]
    if set(val.PEP_ID) & set(train.PEP_ID):
        raise ValueError('Residual station overlaps base fitting station')
    if set(val.PEP_ID) & set(reserved.PEP_ID):
        raise ValueError('Residual station overlaps reserved station')
    if not val.harvest_year.le(2015).all():
        raise ValueError('Residual outcomes must be historical through 2015')
    stations = np.sort(val.PEP_ID.unique())
    if folds < 2 or len(stations) < folds:
        raise ValueError('Insufficient stations for residual folds')
    order = np.random.default_rng(seed).permutation(stations)
    mapping = {station: i % folds for i, station in enumerate(order)}
    val['source_index'] = val.index
    val['fold'] = val.PEP_ID.map(mapping).astype(int)
    return val.reset_index(drop=True)

def select_candidate(metrics, tolerance=1e-06):
    """Original duration-study promotion guards, generalized candidate names."""
    base = metrics['baseline']
    checks = {}
    passing = []
    for name in sorted(set(metrics) - {'baseline'}):
        candidate = metrics[name]
        finite = all((candidate.get(k) is not None and np.isfinite(candidate[k]) and (base.get(k) is not None) and np.isfinite(base[k]) for k in ('crps', 'stage_equal_mae', 'terminal_mae', 'duration_mae')))
        check = dict(finite_metrics=finite, complete_scoring_support=all((n > 0 for n in candidate['stage_event_counts'])) and candidate['duration_pairs'] > 0, no_lost_baseline_events=candidate['lost_baseline_resolved_events'] == 0, no_lost_baseline_pairs=candidate['lost_baseline_duration_pairs'] == 0, no_more_unresolved=candidate['unresolved_observed_events'] <= base['unresolved_observed_events'], crps_not_worse=finite and candidate['crps'] <= base['crps'] + tolerance, stage_equal_mae_not_worse=finite and candidate['stage_equal_mae'] <= base['stage_equal_mae'] + tolerance, terminal_mae_lower=finite and candidate['terminal_mae'] < base['terminal_mae'] - tolerance, duration_mae_lower=finite and candidate['duration_mae'] < base['duration_mae'] - tolerance)
        checks[name] = check
        if all(check.values()):
            passing.append(name)
    selected = min(passing, key=lambda name: (metrics[name]['crps'], name)) if passing else 'baseline'
    return dict(selected=selected, adopted=bool(passing), checks=checks, tolerance_days=tolerance)

def basis_features(features, stage):
    f = np.asarray(features)
    probability = f[..., 18 + stage]
    return np.stack([probability, probability ** 2, probability ** 3, f[..., 4], probability * f[..., 4]], axis=-1)

def fit_heads(features, base_cdf, targets, indices, family, parameters, seed=20261012):
    indices = np.asarray(indices, dtype=int)
    local_features = np.asarray(features)[indices]
    local_base = np.asarray(base_cdf)[indices]
    local_targets = np.asarray(targets)[indices]
    heads, row_counts = ([], [])
    for stage in range(4):
        rows = sample_rows(local_features, local_base, local_targets, stage)
        if len(rows['target']) == 0:
            raise ValueError('Residual stage has no observed training events')
        if family == 'hgb':
            head = HistGradientBoostingRegressor(random_state=seed + stage, early_stopping=False, **parameters)
        elif family == 'rf':
            head = RandomForestRegressor(random_state=seed + stage, n_jobs=2, **parameters)
        elif family == 'basis':
            head = make_pipeline(StandardScaler(), Ridge(**parameters))
        else:
            raise ValueError('Unknown residual family: ' + family)
        xx = basis_features(rows['features'], stage) if family == 'basis' else rows['features']
        weight = rows['weight'] / rows['weight'].mean()
        if family == 'basis':
            head.fit(xx, rows['target'], ridge__sample_weight=weight)
        else:
            head.fit(xx, rows['target'], sample_weight=weight)
        heads.append(head)
        row_counts.append(len(xx))
    return dict(heads=heads, family=family, parameters=dict(parameters), seed=seed, training_indices=indices.tolist(), stage_row_counts=row_counts)

def predict_residual(package, features, batch=256):
    features = np.asarray(features)
    output = np.empty((len(features), 4, features.shape[1]), dtype=np.float32)
    for start in range(0, len(features), batch):
        xx = features[start:start + batch]
        for stage, head in enumerate(package['heads']):
            inputs = basis_features(xx, stage) if package['family'] == 'basis' else xx
            output[start:start + len(xx), stage] = head.predict(inputs.reshape(-1, inputs.shape[-1])).reshape(len(xx), xx.shape[1])
    return output

def correct_cdf(base_cdf, residual, increments, strength, mask=(1, 1, 1, 1)):
    base_cdf, residual = map(np.asarray, (base_cdf, residual))
    if base_cdf.shape != residual.shape or not np.isfinite(strength) or strength < 0 or (len(mask) != 4):
        raise ValueError('Invalid residual correction settings')
    if strength == 0:
        return base_cdf.copy()
    proposal = base_cdf + strength * np.asarray(mask)[None, :, None] * residual
    return forward_project(proposal, increments)
