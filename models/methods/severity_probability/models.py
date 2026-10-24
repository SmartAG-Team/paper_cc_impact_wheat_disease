"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/severity_probability_20261011')
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from analysis.methods.severity_probability.data import weights
THRESHOLDS = np.array([10.0, 25.0, 50.0])
MIDPOINTS = np.array([5.0, 17.5, 37.5, 75.0])
OPTIMIZER = dict(maxiter=1000, ftol=1e-11, gtol=1e-06, maxls=50)
CUT_PENALTY = 1e-06
MIN_GAP = 0.0001

def classes(values):
    y = np.asarray(values, float)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise ValueError('Severity must be finite in [0,100]')
    return np.searchsorted(THRESHOLDS, y, side='right')

def cutpoints(raw):
    raw = np.asarray(raw, float)
    return raw[0] + np.r_[0.0, np.cumsum(np.logaddexp(0.0, raw[1:]) + MIN_GAP)]

def probabilities(eta, cuts):
    eta = np.asarray(eta, float)
    cuts = np.asarray(cuts, float)
    if cuts.shape != (3,) or not np.isfinite(cuts).all() or np.any(np.diff(cuts) <= 0) or (not np.isfinite(eta).all()):
        raise ValueError('Finite predictor and ordered cuts required')
    cdf = expit(cuts[None, :] - eta[:, None])
    return np.column_stack([cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1]])

def exceedance(probability):
    return np.cumsum(np.asarray(probability)[:, ::-1], axis=1)[:, ::-1][:, 1:]

def objective(design, labels, weight, ridge):
    x = torch.as_tensor(design, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    w = torch.as_tensor(np.asarray(weight) / np.sum(weight), dtype=torch.float64)

    def loss(parameters):
        theta = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
        raw = theta[-3:]
        gaps = torch.nn.functional.softplus(raw[1:]) + MIN_GAP
        cuts = raw[0] + torch.cat([torch.zeros(1, dtype=torch.float64), gaps.cumsum(0)])
        a = cuts[None, :] - (x @ theta[:-3])[:, None]
        sp = torch.nn.functional.softplus
        middle = -sp(-a[:, 1:]) - sp(a[:, :-1]) + torch.log(-torch.expm1(-gaps))[None, :]
        logp = torch.column_stack([-sp(-a[:, 0]), middle, -sp(a[:, -1])])
        value = -(w * logp[torch.arange(len(y)), y]).sum() + 0.5 * ridge * theta[:-3].square().sum() + 0.5 * CUT_PENALTY * raw.square().sum()
        value.backward()
        return (float(value.detach()), theta.grad.detach().numpy().copy())
    return loss

def summaries(frame):
    y = frame.observed_percent.to_numpy(float)
    label = classes(y)
    w = weights(frame)
    means = MIDPOINTS.copy()
    missing = []
    for k in range(4):
        mask = label == k
        if mask.any():
            means[k] = np.average(y[mask], weights=w[mask])
        else:
            missing.append(k)
    return (label, w, means, missing)

def fit(x, frame, ridge):
    torch.set_num_threads(1)
    x = np.asarray(x, float)
    raw_leaf = frame.leaf_index.to_numpy(float)
    if x.ndim != 2 or len(x) != len(frame) or (not np.isfinite(x).all()) or (not np.isfinite(raw_leaf).all()) or (not np.equal(raw_leaf, np.floor(raw_leaf)).all()) or np.any((raw_leaf < 0) | (raw_leaf > 6)) or (ridge <= 0):
        raise ValueError('Finite aligned training features, leaves and positive ridge required')
    leaf = raw_leaf.astype(int)
    label, w, means, missing = summaries(frame)
    mean = np.average(x, axis=0, weights=w)
    scale = np.maximum(np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w)), 1e-08)
    design = np.column_stack([(x - mean) / scale, np.eye(7)[leaf]])
    mass = np.bincount(label, weights=w * len(w) / w.sum(), minlength=4) + 0.5
    cuts = logit(np.cumsum(mass / mass.sum())[:-1])
    theta = np.r_[np.zeros(design.shape[1]), cuts[0], np.log(np.expm1(np.diff(cuts) - MIN_GAP))]
    fun = objective(design, label, w, ridge)
    trace = []

    def callback(t):
        loss, grad = fun(t)
        trace.append(dict(iteration=len(trace) + 1, objective=loss, max_abs_gradient=float(abs(grad).max())))
    result = minimize(fun, theta, jac=True, method='L-BFGS-B', callback=callback, options=OPTIMIZER)
    if not result.success or not np.isfinite(result.fun) or (not np.isfinite(result.x).all()):
        raise RuntimeError('Ordinal model convergence failure: ' + str(result.message))
    return dict(kind='proportional_odds', ridge=float(ridge), mean=mean.tolist(), scale=scale.tolist(), coefficients=result.x[:-3].tolist(), raw_cuts=result.x[-3:].tolist(), cuts=cutpoints(result.x[-3:]).tolist(), class_means=means.tolist(), missing_classes=missing, training_targets=sorted(frame.global_target_index.astype(int).tolist()), training_sites=sorted(frame.site_id.unique()), training_fields=sorted(frame.field_id.unique()), seen_leaves=sorted(np.unique(leaf).tolist()), converged=True, objective=float(result.fun), iterations=int(result.nit), max_abs_gradient=float(abs(result.jac).max()), message=str(result.message), trace=trace)

def predict(fitted, x, leaves):
    x = np.asarray(x, float)
    raw = np.asarray(leaves, float)
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all() or np.any((raw < 0) | (raw > 6)) or (len(raw) != len(x)):
        raise ValueError('Invalid prediction leaves')
    leaf = raw.astype(int)
    design = np.column_stack([(x - np.asarray(fitted['mean'])) / np.asarray(fitted['scale']), np.eye(7)[leaf]])
    p = probabilities(design @ np.asarray(fitted['coefficients']), np.asarray(fitted['cuts']))
    return (p, p @ np.asarray(fitted['class_means']))

def fit_reference(frame):
    label, w, means, missing = summaries(frame)
    w = w * len(w) / w.sum()
    leaf = frame.leaf_index.to_numpy(int)

    def frequency(mask):
        mass = np.bincount(label[mask], weights=w[mask], minlength=4) + 0.5
        return (mass / mass.sum()).tolist()
    return dict(kind='leaf_frequency', global_probabilities=frequency(np.ones(len(frame), bool)), leaf_probabilities={str(k): frequency(leaf == k) for k in sorted(np.unique(leaf))}, class_means=means.tolist(), missing_classes=missing, training_targets=sorted(frame.global_target_index.astype(int).tolist()), smoothing_per_class=0.5, training_weight_sum=len(frame))

def predict_reference(fitted, leaves):
    p = np.array([fitted['leaf_probabilities'].get(str(int(k)), fitted['global_probabilities']) for k in leaves])
    return (p, p @ np.asarray(fitted['class_means']))
