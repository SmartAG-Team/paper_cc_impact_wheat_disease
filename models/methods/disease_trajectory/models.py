"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_trajectory_restructure_20261011')
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit, logit
from analysis.methods.severity_probability.data import weights
from analysis.methods.disease_trajectory import basis
N_LEAVES = 7
INTEGRAL_SCALE_DAYS = 30.0
OPTIMIZER = dict(maxiter=1500, ftol=1e-12, gtol=1e-07, maxls=50)

def window_operator(frame):
    """Linear trapezoid operator on the frame's original row order."""
    keys = ['field_id', 'endpoint_series']
    if frame.empty or frame.duplicated(keys + ['date']).any():
        raise ValueError('Duplicate or empty source-series assessments')
    dated = frame.reset_index(drop=True).assign(_position=np.arange(len(frame)))
    dated['_date'] = pd.to_datetime(dated.date, errors='coerce')
    if dated._date.isna().any():
        raise ValueError('Valid assessment dates required')
    rows, cols, values, metadata = ([], [], [], [])
    meta = [k for k in ['field_id', 'field_index', 'site_id', 'coordinate_year', 'source', 'partition', 'leaf_index', 'endpoint_series', 'season_year'] if k in frame]
    for _, group in dated.groupby(keys, sort=True):
        group = group.sort_values('_date')
        if len(group) < 2:
            continue
        if any((group[k].nunique(dropna=False) != 1 for k in meta)):
            raise ValueError('Inconsistent source-series metadata')
        day = (group._date - group._date.iloc[0]).dt.total_seconds().to_numpy() / 86400
        if np.any(np.diff(day) <= 0):
            raise ValueError('Strictly increasing assessment dates required')
        q = np.zeros(len(group))
        q[:-1] += np.diff(day) / 2
        q[1:] += np.diff(day) / 2
        rows.extend([len(metadata)] * len(group))
        cols.extend(group._position.to_list())
        values.extend(q)
        item = {k: group[k].iloc[0] for k in meta}
        item.update(span_days=float(day[-1]), n_assessments=len(group), window_start=str(group.date.iloc[0]), window_end=str(group.date.iloc[-1]))
        metadata.append(item)
    return (csr_matrix((values, (rows, cols)), shape=(len(metadata), len(frame))), pd.DataFrame(metadata))

def _hazards(theta, x, valid):
    p = x.shape[-1]
    linear = theta[0] + np.einsum('ftp,p->ft', x, theta[1:p + 1])[..., None] + theta[p + 1:][None, None, :]
    rate = (np.logaddexp(0, linear) + 1e-12) * valid[..., None]
    hazard = np.concatenate([np.zeros_like(rate[:, :1]), np.cumsum(rate, axis=1)], axis=1)
    return (linear, hazard)

def _bounds(frame, valid):
    fi = frame.field_index.to_numpy(int)
    leaf = frame.leaf_index.to_numpy(int)
    lower_raw = frame.lower_day.to_numpy(float)
    upper = frame.upper_day.to_numpy(float)
    if not np.isfinite(lower_raw).all() or not np.equal(lower_raw, np.floor(lower_raw)).all() or np.any(fi < 0) or np.any(fi >= len(valid)) or np.any(leaf < 0) or np.any(leaf >= N_LEAVES):
        raise ValueError('Valid integer field, leaf and lower-bound indices required')
    lower = lower_raw.astype(int)
    selected = np.flatnonzero(np.isfinite(upper))
    horizon = valid.sum(axis=1)[fi]
    if np.any(lower < 0) or np.any(lower > horizon) or np.isinf(upper).any() or np.any(upper[selected] < lower[selected] + 1) or np.any(upper[selected] > horizon[selected]) or (not np.equal(upper[selected], np.floor(upper[selected])).all()):
        raise ValueError('Censoring bounds must lie within contiguous valid horizons')
    return (fi, leaf, lower, upper, selected)

def onset_losses(theta, x, valid, intervals):
    linear, h = _hazards(theta, x, valid)
    fi, leaf, lower, upper, selected = _bounds(intervals, valid)
    loss = h[fi, lower, leaf].copy()
    _, delta = interval_increments(linear, valid, fi, leaf, lower, upper, selected)
    loss[selected] -= np.log(-np.expm1(-delta))
    return loss

def interval_increments(linear, valid, fi, leaf, lower, upper, selected):
    """Sum interval rates directly; subtraction of large hazards is unsafe."""
    days = np.arange(valid.shape[1])[None, :]
    mask = (days >= lower[selected, None]) & (days < upper[selected, None])
    rate = (np.logaddexp(0, linear) + 1e-12) * valid[..., None]
    delta = np.sum(rate[fi[selected], :, leaf[selected]] * mask, axis=1)
    return (mask, delta)

def onset_objective(theta, x, valid, intervals, *, ridge):
    linear, h = _hazards(theta, x, valid)
    fi, leaf, lower, upper, selected = _bounds(intervals, valid)
    w = weights(intervals)
    w /= w.sum()
    loss = h[fi, lower, leaf].copy()
    interval_mask, delta = interval_increments(linear, valid, fi, leaf, lower, upper, selected)
    denominator = -np.expm1(-delta)
    loss[selected] -= np.log(denominator)
    ratio = np.exp(-delta) / denominator
    adj = np.zeros_like(h)
    np.add.at(adj, (fi, lower, leaf), w)
    rate_grad = np.cumsum(adj[:, :0:-1, :], axis=1)[:, ::-1, :]
    rows, days = np.nonzero(interval_mask)
    locations = (fi[selected][rows] * valid.shape[1] + days) * N_LEAVES + leaf[selected][rows]
    negative = np.bincount(locations, weights=-w[selected][rows] * ratio[rows], minlength=rate_grad.size)
    rate_grad += negative.reshape(rate_grad.shape)
    g = rate_grad * expit(linear) * valid[..., None]
    gradient = np.concatenate([[g.sum()], np.einsum('ftp,ft->p', x, g.sum(axis=2)), g.sum(axis=(0, 1))])
    value = float(w @ loss + 0.5 * ridge * np.sum(theta[1:] ** 2))
    gradient[1:] += ridge * theta[1:]
    return (value, gradient)

def _fit_objective(function, initial):
    result = minimize(function, initial, jac=True, method='L-BFGS-B', options=OPTIMIZER)
    if not result.success or not np.isfinite(result.x).all() or (not np.isfinite(result.fun)):
        raise RuntimeError('Trajectory fit failed: ' + str(result.message))
    return dict(coefficients=result.x.tolist(), objective=float(result.fun), iterations=int(result.nit), evaluations=int(result.nfev), max_abs_gradient=float(np.max(np.abs(result.jac))), converged=bool(result.success), convergence=str(result.message), optimizer=OPTIMIZER)

def _identity(observations):
    return dict(training_fields=sorted(observations.field_index.unique().astype(int).tolist()), training_sites=sorted(observations.site_id.unique().tolist()), training_targets=sorted(observations.global_target_index.astype(int).tolist()))

def _validate_valid(valid):
    valid = np.asarray(valid, bool)
    if valid.ndim != 2 or not np.array_equal(valid, np.arange(valid.shape[1])[None, :] < valid.sum(axis=1)[:, None]):
        raise ValueError('Contiguous daily feature masks required')
    return valid

def fit_onset(features, valid, names, observations, intervals, *, family, ridge):
    valid = _validate_valid(valid)
    fields = np.unique(observations.field_index.to_numpy(int))
    if set(intervals.field_index) != set(fields):
        raise ValueError('Onset bounds must match the training fields')
    fitted_basis = basis.fit_basis(features, valid, fields, names, family)
    lookup = {int(f): i for i, f in enumerate(fields)}
    bounds = intervals.assign(field_index=intervals.field_index.map(lookup))
    x = basis.transform(fitted_basis, features[fields])
    mask = valid[fields]
    initial = np.zeros(1 + x.shape[-1] + N_LEAVES)
    initial[0] = -7
    fitted = _fit_objective(lambda theta: onset_objective(theta, x, mask, bounds, ridge=ridge), initial)
    fitted.update(kind='separate_interval_onset', basis=fitted_basis, ridge=float(ridge), **_identity(observations))
    return fitted

def predict_onset(fitted, features, valid):
    valid = _validate_valid(valid)
    _, hazard = _hazards(np.asarray(fitted['coefficients']), basis.transform(fitted['basis'], features), valid)
    return -np.expm1(-hazard[:, 1:])

def onset_log_score(fitted, features, valid, intervals):
    return onset_losses(np.asarray(fitted['coefficients']), basis.transform(fitted['basis'], features), _validate_valid(valid), intervals)

def median_day(probability, valid):
    hit = (probability >= 0.5) & np.asarray(valid, bool)[..., None]
    crossed = hit.any(axis=1)
    return (np.where(crossed, hit.argmax(axis=1) + 1, np.asarray(valid).sum(axis=1)[:, None] + 1), crossed)

def severity_objective(theta, design, occurrence, y, weight, operator, window_weight, *, ridge, integral_weight):
    conditional = expit(design @ theta)
    residual = occurrence * conditional - y
    value = float(np.dot(weight, residual ** 2))
    derivative = 2 * weight * residual
    if operator.shape[0] and integral_weight:
        error = operator @ residual
        value += integral_weight * float(np.dot(window_weight, error ** 2)) / INTEGRAL_SCALE_DAYS ** 2
        derivative += 2 * integral_weight * (operator.T @ (window_weight * error)) / INTEGRAL_SCALE_DAYS ** 2
    gradient = design.T @ (derivative * occurrence * conditional * (1 - conditional))
    value += 0.5 * ridge * float(theta[1:] @ theta[1:])
    gradient[1:] += ridge * theta[1:]
    return (value, gradient)

def severity_design(x, leaves, columns):
    leaves = np.asarray(leaves, int)
    if np.any(leaves < 0) or np.any(leaves >= N_LEAVES):
        raise ValueError('Source leaf category outside 0–6')
    indicator = np.eye(N_LEAVES)[leaves]
    interaction = x[:, [columns.index(k) for k in basis.CALENDAR]]
    return np.column_stack([np.ones(len(x)), x, indicator, (indicator[:, :, None] * interaction[:, None, :]).reshape(len(x), -1)])

def fit_severity(features, valid, names, observations, occurrence, *, family, ridge, integral_weight):
    fields = np.unique(observations.field_index.to_numpy(int))
    fitted_basis = basis.fit_basis(features, _validate_valid(valid), fields, names, family)
    x = basis.transform(fitted_basis, features)
    fi = observations.field_index.to_numpy(int)
    day = observations.day_index.to_numpy(int) - 1
    leaf = observations.leaf_index.to_numpy(int)
    if not valid[fi, day].all():
        raise ValueError('Assessment outside valid daily support')
    design = severity_design(x[fi, day], leaf, fitted_basis['columns'])
    y = observations.observed_percent.to_numpy(float) / 100
    prob = occurrence[fi, day, leaf]
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)) or (not np.isfinite(prob).all()) or np.any((prob < 0) | (prob > 1)):
        raise ValueError('Finite bounded targets and occurrence probabilities required')
    w = weights(observations)
    w /= w.sum()
    operator, windows = window_operator(observations)
    ww = weights(windows) if len(windows) else np.array([])
    if len(ww):
        ww /= ww.sum()
    initial = np.zeros(design.shape[1])
    initial[0] = logit(np.clip(np.average(y, weights=w) / max(np.average(prob, weights=w), 1e-08), 0.001, 0.999))
    objective = lambda theta: severity_objective(theta, design, prob, y, w, operator, ww, ridge=ridge, integral_weight=integral_weight)
    fitted = _fit_objective(objective, initial)
    fitted.update(kind='bounded_conditional_severity', basis=fitted_basis, ridge=float(ridge), integral_weight=float(integral_weight), training_windows=len(windows), **_identity(observations))
    return fitted

def predict_severity(fitted, features, occurrence):
    x = basis.transform(fitted['basis'], features)
    p = x.shape[-1]
    theta = np.asarray(fitted['coefficients'])
    common = theta[0] + np.einsum('ftp,p->ft', x, theta[1:p + 1])
    interaction = x[..., [fitted['basis']['columns'].index(k) for k in basis.CALENDAR]]
    linear = common[..., None] + theta[p + 1:p + 1 + N_LEAVES][None, None, :]
    linear += np.einsum('ftq,lq->ftl', interaction, theta[p + 1 + N_LEAVES:].reshape(N_LEAVES, 3))
    if occurrence.shape != linear.shape or not np.isfinite(occurrence).all() or np.any((occurrence < 0) | (occurrence > 1)):
        raise ValueError('Bounded daily occurrence with matching support required')
    return 100 * occurrence * expit(linear)

def severity_selection_loss(frame, prediction, integral_weight):
    y = frame.observed_percent.to_numpy(float) / 100
    residual = np.asarray(prediction, float) / 100 - y
    w = weights(frame)
    value = float(np.average(residual ** 2, weights=w))
    operator, windows = window_operator(frame)
    if len(windows) and integral_weight:
        value += integral_weight * float(np.average((operator @ residual / INTEGRAL_SCALE_DAYS) ** 2, weights=weights(windows)))
    return value
