"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from analysis.methods.severity_probability.data import weights
from analysis.methods.residual_disease import config

def fit(kind, setting, x, target, weight, names):
    x = np.asarray(x, float)
    target = np.asarray(target, float)
    weight = np.asarray(weight, float)
    if x.ndim != 2 or x.shape != (len(target), len(names)) or weight.shape != target.shape or (not np.isfinite(x).all()) or (not np.isfinite(target).all()) or (not np.isfinite(weight).all()) or np.any(weight < 0) or (weight.sum() <= 0):
        raise ValueError('Finite aligned weighted ML training arrays required')
    if kind == 'rf':
        parameters = {**config.RF_COMMON, **config.RF_GRID[setting]}
        estimator = RandomForestRegressor(**parameters)
    elif kind == 'hgb':
        parameters = {**config.HGB_COMMON, **config.HGB_GRID[setting]}
        estimator = HistGradientBoostingRegressor(**parameters)
    else:
        raise ValueError('Unknown ML head')
    estimator.fit(x, target, sample_weight=weight * len(weight) / weight.sum())
    return dict(estimator=estimator, kind=kind, setting=int(setting), parameters=parameters, feature_names=list(names), target_units='percentage points')

def predict(fitted, x, names):
    if list(names) != fitted['feature_names']:
        raise ValueError('ML feature order mismatch')
    value = fitted['estimator'].predict(np.asarray(x, float))
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite ML prediction')
    return value

def correct(base, residual, multiplier):
    base = np.asarray(base, float)
    residual = np.asarray(residual, float)
    if base.shape != residual.shape or not np.isfinite(base).all() or (not np.isfinite(residual).all()) or (not np.isfinite(multiplier)) or (not 0 <= multiplier <= 1):
        raise ValueError('Aligned finite base/residual and bounded correction strength required')
    return np.clip(base + float(multiplier) * residual, 0, 100)

def fit_bias(frame, residual):
    residual = np.asarray(residual, float)
    if residual.shape != (len(frame),) or not np.isfinite(residual).all():
        raise ValueError('Aligned residuals required')
    w = weights(frame)
    leaf = frame.leaf_index.to_numpy(int)
    return dict(means={str(k): float(np.average(residual[leaf == k], weights=w[leaf == k])) for k in np.unique(leaf)}, fallback=float(np.average(residual, weights=w)))

def predict_bias(fitted, leaf):
    return np.array([fitted['means'].get(str(int(k)), fitted['fallback']) for k in leaf])
