"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_joint_forecast_20261010')
import numpy as np
import torch
from scipy.optimize import minimize
from analysis.methods.observation_model.observation import hierarchical_weights
N_LEAVES = 7
OPTIMIZER = dict(maxiter=1000, ftol=1e-11, gtol=1e-06, maxls=50)

def fit_scaler(features, valid, field_indices):
    fields = np.unique(np.asarray(field_indices, dtype=int))
    if not len(fields):
        raise ValueError('Nonempty training fields required')
    selected = features[fields][valid[fields]]
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError('Finite training features required')
    return (selected.mean(axis=0), np.maximum(selected.std(axis=0), 1e-08))

def cumulative_hazards(theta, x, valid):
    """Two arrays of F x (T+1) x source leaf; index zero is pre-sowing."""
    p = x.shape[-1]
    linear = torch.einsum('ftp,kp->kft', x, theta[:, :p])[:, :, :, None] + theta[:, None, None, p:]
    rate = torch.nn.functional.softplus(linear) * valid[None, :, :, None]
    cumulative = torch.cumsum(rate, dim=2)
    return torch.cat([torch.zeros_like(cumulative[:, :, :1]), cumulative], dim=2)

def interval_log_probability(hazard, field, leaf, lower, upper):
    field = torch.as_tensor(field, dtype=torch.long)
    leaf = torch.as_tensor(leaf, dtype=torch.long)
    lower = torch.as_tensor(lower, dtype=torch.long)
    upper = np.asarray(upper, float)
    low = hazard[field, lower, leaf]
    result = -low
    selected = np.flatnonzero(np.isfinite(upper))
    if len(selected):
        upper_day = torch.as_tensor(upper[selected], dtype=torch.long)
        delta = hazard[field[selected], upper_day, leaf[selected]] - low[selected]
        result = result.clone()
        result[selected] = -low[selected] + torch.log(-torch.expm1(-delta.clamp_min(1e-15)))
    return result

def make_objective(features, valid, observations, intervals, *, ridge, onset_weight):
    x = torch.as_tensor(features, dtype=torch.float64)
    mask = torch.as_tensor(valid, dtype=torch.float64)
    p = x.shape[-1]
    fi = torch.as_tensor(observations.field_index.to_numpy(int))
    day = torch.as_tensor(observations.day_index.to_numpy(int))
    leaf = torch.as_tensor(observations.leaf_index.to_numpy(int))
    y = torch.as_tensor(observations.observed_percent.to_numpy(float) / 100)
    weight = hierarchical_weights(observations)
    weight = torch.as_tensor(weight / weight.sum())
    iw = hierarchical_weights(intervals)
    iw = torch.as_tensor(iw / iw.sum())

    def objective(coefficients):
        theta = torch.tensor(np.asarray(coefficients).reshape(2, p + N_LEAVES), dtype=torch.float64, requires_grad=True)
        hazards = cumulative_hazards(theta, x, mask)
        expected = -torch.expm1(-hazards[0, fi, day, leaf]) * -torch.expm1(-hazards[1, fi, day, leaf])
        logp = interval_log_probability(hazards[0], intervals.field_index.to_numpy(int), intervals.leaf_index.to_numpy(int), intervals.lower_day.to_numpy(int), intervals.upper_day.to_numpy(float))
        centered = theta[:, p:] - theta[:, p:].mean(dim=1, keepdim=True)
        penalty = 0.5 * ridge * (theta[:, :p].square().sum() + centered.square().sum())
        loss = (weight * (expected - y).square()).sum() - onset_weight * (iw * logp).sum() + penalty
        loss.backward()
        return (float(loss.detach()), theta.grad.detach().numpy().ravel().copy())
    return objective

def fit(features, valid, observations, intervals, *, ridge, onset_weight):
    """Every disease coefficient and scale is learned from these training rows."""
    torch.set_num_threads(1)
    fields = np.unique(observations.field_index.to_numpy(int))
    index = {int(f): i for i, f in enumerate(fields)}
    if not set(intervals.field_index) <= set(fields):
        raise ValueError('Onset fields outside training population')
    mean, scale = fit_scaler(features, valid, fields)
    train = observations.assign(field_index=observations.field_index.map(index))
    bounds = intervals.assign(field_index=intervals.field_index.map(index))
    objective = make_objective((features[fields] - mean) / scale, valid[fields], train, bounds, ridge=ridge, onset_weight=onset_weight)
    p = features.shape[-1]
    theta = np.zeros((2, p + N_LEAVES))
    theta[:, p:] = -7.0
    trace = []

    def callback(coefficients):
        value, grad = objective(coefficients)
        trace.append(dict(iteration=len(trace) + 1, objective=value, max_abs_gradient=float(abs(grad).max())))
    result = minimize(objective, theta.ravel(), method='L-BFGS-B', jac=True, callback=callback, options=OPTIMIZER)
    if not result.success or not np.isfinite(result.fun) or (not np.isfinite(result.x).all()):
        raise RuntimeError('Joint model failed convergence: ' + str(result.message))
    return dict(kind='joint_daily_score', ridge=float(ridge), onset_weight=float(onset_weight), mean=mean.tolist(), scale=scale.tolist(), coefficients=result.x.reshape(2, p + N_LEAVES).tolist(), training_fields=fields.tolist(), training_sites=sorted(observations.site_id.unique()), training_targets=sorted(observations.global_target_index.astype(int).tolist()), objective=float(result.fun), iterations=int(result.nit), evaluations=int(result.nfev), max_abs_gradient=float(abs(result.jac).max()), converged=bool(result.success), convergence=str(result.message), optimizer=OPTIMIZER, trace=trace)

def predict(fitted, features, valid):
    with torch.no_grad():
        x = torch.as_tensor((features - np.asarray(fitted['mean'])) / np.asarray(fitted['scale']), dtype=torch.float64)
        theta = torch.as_tensor(fitted['coefficients'], dtype=torch.float64)
        h = cumulative_hazards(theta, x, torch.as_tensor(valid, dtype=torch.float64))[:, :, 1:]
        p = -torch.expm1(-h)
        return dict(severity=(100 * p[0] * p[1]).numpy(), occurrence=p[0].numpy())

def median_day(probability, horizons):
    """A horizon+1 sentinel is a minimum delay bound, never an observed crossing."""
    valid = np.arange(probability.shape[1])[None, :] < np.asarray(horizons)[:, None]
    exceed = (probability >= 0.5) & valid[:, :, None]
    crossed = exceed.any(axis=1)
    day = np.where(crossed, exceed.argmax(axis=1) + 1, np.asarray(horizons)[:, None] + 1).astype(float)
    return (day, crossed)
