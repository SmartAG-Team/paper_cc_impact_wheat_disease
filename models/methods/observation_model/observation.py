"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from __future__ import annotations
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_observation_revision_20261010')
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
RIDGE_GRID = (0.0001, 0.001, 0.01, 0.1)
EVALUATION_PARTITIONS = ('reused_BASF2019', 'reused_strict_Corteva')
GROUP_KEYS = ['coordinate_year', 'field_id', 'leaf_index']
SERIES_KEYS = ['partition', 'field_id', 'coordinate_year', 'source', 'leaf_index', 'endpoint_series']
SEED = 20261010

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def validate_core(frame, *, outcome=True):
    keys = GROUP_KEYS + ['site_id', 'source', 'date', 'endpoint_series']
    values = ['frozen_damage_percent'] + (['observed_percent'] if outcome else [])
    if frame.empty or frame[keys].isna().any().any():
        raise ValueError('Nonempty data and complete grouping required')
    if frame.groupby('coordinate_year').site_id.nunique().gt(1).any():
        raise ValueError('Coordinate-year maps to multiple sites')
    if frame.groupby('field_id').coordinate_year.nunique().gt(1).any():
        raise ValueError('Field maps to multiple coordinate-years')
    array = frame[values].to_numpy(float)
    if not np.isfinite(array).all() or ((array < 0) | (array > 100)).any():
        raise ValueError('Finite percentages inside [0, 100] required')
    leaf = frame.leaf_index.to_numpy(float)
    if not np.isfinite(leaf).all() or (leaf < 0).any() or (leaf != np.floor(leaf)).any():
        raise ValueError('Nonnegative integer source leaf indices required')
    if pd.to_datetime(frame.date, errors='coerce').isna().any():
        raise ValueError('Valid dates required')

def hierarchical_weights(frame):
    """Unit mass per coordinate-year, split over fields/leaves/records."""
    if frame.empty or frame[GROUP_KEYS].isna().any().any():
        raise ValueError('Nonempty complete hierarchy required')
    fields = frame.groupby(GROUP_KEYS[0]).field_id.transform('nunique').to_numpy(float)
    leaves = frame.groupby(GROUP_KEYS[:2]).leaf_index.transform('nunique').to_numpy(float)
    records = frame.groupby(GROUP_KEYS).leaf_index.transform('size').to_numpy(float)
    return 1 / (fields * leaves * records)

def assert_disjoint(training, validation):
    for key in ['site_id', 'coordinate_year', 'field_id']:
        if set(training[key]) & set(validation[key]):
            raise ValueError('Training/validation overlap in ' + key)

def clean_development(frame):
    evaluation = frame[frame.partition.isin(EVALUATION_PARTITIONS)]
    training = frame[frame.partition.eq('calibration') & ~frame.site_id.isin(evaluation.site_id)].copy()
    assert_disjoint(training, evaluation)
    return training

def site_folds(frame, *, n_splits=5, seed=SEED):
    """Outcome-blind site folds stratified by source and balanced on years."""
    if frame.site_id.nunique() < n_splits or n_splits < 2:
        raise ValueError('At least two folds and one site per fold required')
    sites = frame.groupby('site_id', sort=True).agg(source=('source', 'first'), sources=('source', 'nunique'), mass=('coordinate_year', 'nunique'))
    if sites.sources.gt(1).any():
        raise ValueError('Cross-source sites require a common source stratum')
    rng = np.random.default_rng(seed)
    assignment = {}
    total_mass = np.zeros(n_splits)
    for _, group in sites.groupby('source', sort=True):
        shuffled = group.iloc[rng.permutation(len(group))].sort_values('mass', ascending=False, kind='stable')
        mass = np.zeros(n_splits)
        for site, row in shuffled.iterrows():
            fold = min(range(n_splits), key=lambda i: (mass[i], total_mass[i], i))
            assignment[site] = fold
            mass[fold] += row['mass']
            total_mass[fold] += row['mass']
    result = frame.site_id.map(assignment).astype(int)
    if result.nunique() != n_splits:
        raise ValueError('Empty validation fold')
    return result

def design(frame, kind):
    validate_core(frame, outcome=False)
    leaf = frame.leaf_index.to_numpy(int)
    cols = [np.ones(len(frame)), (leaf == 1).astype(float), (leaf == 2).astype(float), (leaf >= 3).astype(float)]
    names = ['intercept', 'leaf_1', 'leaf_2', 'leaf_ge3']
    if kind == 'bounded_leaf_raw':
        cols.append(np.log1p(frame.frozen_damage_percent.to_numpy(float)) / np.log(101))
        names.append('log1p_raw_percent_scaled')
    elif kind == 'leaf_stage_time':
        stage = frame[['stage_from_audit', 'stage_to_audit']].mean(axis=1).to_numpy(float)
        if np.isinf(stage).any() or ((stage[np.isfinite(stage)] < 0) | (stage[np.isfinite(stage)] > 99)).any():
            raise ValueError('Observed stage outside supported coding')
        elapsed = frame.elapsed_field_assessment_days.to_numpy(float)
        if not np.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError('Finite nonnegative elapsed assessment days required')
        cols.extend([(np.isfinite(stage) & (stage >= 39) & (stage < 60)).astype(float), (np.isfinite(stage) & (stage >= 60) & (stage < 70)).astype(float), (np.isfinite(stage) & (stage >= 70)).astype(float), (~np.isfinite(stage)).astype(float), np.log1p(elapsed) / np.log(101)])
        names.extend(['stage_39_59', 'stage_60_69', 'stage_ge70', 'stage_missing', 'log1p_elapsed_days_scaled'])
    else:
        raise ValueError('Unknown regression kind: ' + kind)
    return (np.column_stack(cols), names)

def fit_model(training, kind, *, ridge=0.001, multiplier=None):
    validate_core(training)
    weights = hierarchical_weights(training)
    if multiplier is not None:
        multiplier = np.asarray(multiplier, float)
        if multiplier.shape != weights.shape or not np.isfinite(multiplier).all() or (multiplier < 0).any():
            raise ValueError('Nonnegative finite row multiplicities required')
        weights *= multiplier
    if weights.sum() <= 0:
        raise ValueError('Positive training mass required')
    weights /= weights.sum()
    y = training.observed_percent.to_numpy(float) / 100
    shared = dict(kind=kind, training_records=len(training), training_sites=sorted(training.site_id.unique()), training_coordinate_years=sorted(training.coordinate_year.unique()), training_global_target_indices=sorted(training.global_target_index.astype(int).tolist()))
    if kind in {'clean_leaf_mean', 'pooled_leaf_mean'}:
        labels = training.leaf_index.to_numpy(int)
        if kind == 'pooled_leaf_mean':
            labels = np.minimum(labels, 3)
        means = {str(int(k)): float(np.average(y[labels == k], weights=weights[labels == k]) * 100) for k in np.unique(labels) if weights[labels == k].sum() > 0}
        return dict(**shared, leaf_means=means, fallback=float(np.average(y, weights=weights) * 100), weighting='Original coordinate-year/field/leaf/assessment weights within training; conditional leaf means')
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError('Positive finite ridge required')
    X, names = design(training, kind)
    penalty = np.r_[0.0, np.full(X.shape[1] - 1, ridge)]

    def objective(beta):
        z = X @ beta
        risk = np.sum(weights * (np.logaddexp(0, z) - y * z)) + 0.5 * np.sum(penalty * beta ** 2)
        gradient = X.T @ (weights * (expit(z) - y)) + penalty * beta
        return (risk, gradient)
    initial = np.zeros(X.shape[1])
    initial[0] = logit(np.clip(np.average(y, weights=weights), 1e-06, 1 - 1e-06))
    bounds = [(None, None)] * X.shape[1]
    if kind == 'bounded_leaf_raw':
        bounds[-1] = (0, None)
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=4000, ftol=1e-13, gtol=1e-09))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Observation fit did not converge: ' + str(result.message))
    prob = expit(X @ result.x)
    hessian = X.T @ ((weights * prob * (1 - prob))[:, None] * X)
    eigen = np.linalg.eigvalsh(hessian)
    penalty_hessian = hessian + np.diag(penalty)
    return dict(**shared, ridge=float(ridge), coefficients=result.x.tolist(), coefficient_names=names, convergence=str(result.message), objective=float(result.fun), iterations=int(result.nit), design_rank=int(np.linalg.matrix_rank(X)), design_columns=X.shape[1], information_eigenvalues=eigen.tolist(), penalized_information_condition=float(np.linalg.cond(penalty_hessian)), raw_slope_at_boundary=bool(kind == 'bounded_leaf_raw' and result.x[-1] < 1e-08), raw_training_min=float(training.frozen_damage_percent.min()), raw_training_max=float(training.frozen_damage_percent.max()), interpretation='Fractional quasi-logistic measurement score; no biological binomial likelihood or new mechanism')

def predict_model(model, frame):
    validate_core(frame, outcome=False)
    kind = model['kind']
    if kind in {'clean_leaf_mean', 'pooled_leaf_mean'}:
        labels = frame.leaf_index.to_numpy(int)
        if kind == 'pooled_leaf_mean':
            labels = np.minimum(labels, 3)
        result = np.array([model['leaf_means'].get(str(int(k)), model['fallback']) for k in labels])
    else:
        X, names = design(frame, kind)
        if names != model['coefficient_names']:
            raise ValueError('Feature order mismatch')
        result = 100 * expit(X @ np.asarray(model['coefficients']))
    if not np.isfinite(result).all() or ((result < 0) | (result > 100)).any():
        raise ValueError('Invalid bounded prediction')
    return result

def windows(frame, prediction_columns):
    """Trapezoids only between assessment dates; no endpoint extrapolation."""
    if frame.duplicated(SERIES_KEYS + ['date']).any():
        raise ValueError('Duplicate date within source leaf series')
    result = []
    for key, part in frame.groupby(SERIES_KEYS, sort=True):
        part = part.sort_values('date')
        days = (pd.to_datetime(part.date) - pd.to_datetime(part.date.iloc[0])).dt.total_seconds().to_numpy() / 86400
        if len(days) < 2 or days[-1] <= 0:
            continue
        record = dict(zip(SERIES_KEYS, key))
        for meta in ['site_id', 'country', 'season_year', 'outer_fold']:
            if meta in part:
                if part[meta].nunique() != 1:
                    raise ValueError('Inconsistent window metadata: ' + meta)
                record[meta] = part[meta].iloc[0]
        record.update(span_days=float(days[-1]), n_assessments=len(part), window_start=part.date.iloc[0], window_end=part.date.iloc[-1])
        for col in ['observed_percent'] + list(prediction_columns):
            values = part[col].to_numpy(float)
            if col == 'observed_percent' and (not np.isfinite(values).all()):
                raise ValueError('Nonfinite observed window score')
            integral = float(np.sum(np.diff(days) * (values[:-1] + values[1:]) / 2)) if np.isfinite(values).all() else np.nan
            record[col] = integral / days[-1]
            record['observed_audpc' if col == 'observed_percent' else col + '_audpc'] = integral
        result.append(record)
    return pd.DataFrame(result)

def scores(frame, prediction):
    weights = hierarchical_weights(frame)
    observed = frame.observed_percent.to_numpy(float)
    predicted = frame[prediction].to_numpy(float)
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError('Finite paired scores required')
    error = predicted - observed
    mean = np.average(observed, weights=weights)
    variance = np.average((observed - mean) ** 2, weights=weights)
    mse = np.average(error ** 2, weights=weights)
    return dict(rmse=float(np.sqrt(mse)), mae=float(np.average(abs(error), weights=weights)), bias=float(np.average(error, weights=weights)), r2=float(1 - mse / variance) if variance > 0 else np.nan, observed_mean=float(mean), prediction_mean=float(np.average(predicted, weights=weights)))

def bootstrap_pair(frame, model, benchmark, *, replicates=20000, seed=SEED):
    """Paired whole-geographic-site bootstrap preserving every retained year."""
    weight = hierarchical_weights(frame)
    cols = {'mass': weight}
    observed = frame.observed_percent.to_numpy(float)
    for name, col in [('model', model), ('benchmark', benchmark)]:
        values = frame[col].to_numpy(float)
        if not np.isfinite(values).all() or not np.isfinite(observed).all():
            raise ValueError('Complete finite pairing required')
        err = values - observed
        cols.update({name + '_sse': weight * err ** 2, name + '_ae': weight * abs(err), name + '_error': weight * err})
    group = pd.DataFrame(cols).assign(site=frame.site_id.to_numpy()).groupby('site', sort=True).sum()
    array = group.to_numpy(float)
    positions = {key: i for i, key in enumerate(group.columns)}

    def reduce(values):
        mass = values[:, positions['mass']]
        out = {}
        for name in ['model', 'benchmark']:
            out[name + '_rmse'] = np.sqrt(values[:, positions[name + '_sse']] / mass)
            out[name + '_mae'] = values[:, positions[name + '_ae']] / mass
            out[name + '_bias'] = values[:, positions[name + '_error']] / mass
        for metric in ['rmse', 'mae', 'bias']:
            out[metric + '_difference'] = out['model_' + metric] - out['benchmark_' + metric]
        return out
    point = {k: float(v[0]) for k, v in reduce(array.sum(axis=0, keepdims=True)).items()}
    rng = np.random.default_rng(seed)
    sample = np.empty((replicates, array.shape[1]))
    for start in range(0, replicates, 500):
        draws = rng.integers(0, len(array), size=(min(500, replicates - start), len(array)))
        sample[start:start + len(draws)] = array[draws].sum(axis=1)
    return (point, reduce(sample))

def tune_model(training, kind, *, n_splits=5, seed=SEED):
    """Select ridge using only whole-site training-fold window predictions."""
    fold_ids = site_folds(training, n_splits=n_splits, seed=seed)
    results = []
    predictions = {}
    for ridge in RIDGE_GRID:
        prediction = pd.Series(np.nan, index=training.index)
        for fold in range(n_splits):
            inner_train = training[fold_ids.ne(fold)]
            inner_test = training[fold_ids.eq(fold)]
            assert_disjoint(inner_train, inner_test)
            fit = fit_model(inner_train, kind, ridge=ridge)
            prediction.loc[inner_test.index] = predict_model(fit, inner_test)
        shared = training.assign(candidate=prediction)
        integral = windows(shared, ['candidate'])
        if integral.empty:
            raise ValueError('Training folds lack observed windows')
        result = scores(integral, 'candidate')
        results.append(dict(kind=kind, ridge=ridge, window_rmse=result['rmse'], window_mae=result['mae'], assessments=len(shared), windows=len(integral), sites=training.site_id.nunique(), folds=n_splits))
        predictions[ridge] = prediction
    best = min(results, key=lambda row: (row['window_rmse'], -row['ridge']))
    selected = fit_model(training, kind, ridge=best['ridge'])
    selected['selection'] = dict(endpoint='observed_window_mean_percentage', folds=n_splits, seed=seed, rule='Minimum pooled out-of-fold window RMSE; exact ties choose stronger ridge', ridge_grid=list(RIDGE_GRID), selected_window_cv_rmse=best['window_rmse'])
    return (selected, pd.DataFrame(results), fold_ids, predictions[best['ridge']])
