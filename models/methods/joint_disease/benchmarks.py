"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_joint_forecast_20261010')
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from analysis.methods.observation_model.observation import hierarchical_weights
BOOSTING = dict(loss='squared_error', max_iter=150, max_leaf_nodes=7, max_depth=3, min_samples_leaf=25, learning_rate=0.05, l2_regularization=1.0, early_stopping=False, random_state=20261010)

def fit_leaf_mean(frame):
    w = hierarchical_weights(frame)
    y = frame.observed_percent.to_numpy(float)
    leaf = frame.leaf_index.to_numpy(int)
    return dict(means={str(k): float(np.average(y[leaf == k], weights=w[leaf == k])) for k in np.unique(leaf)}, fallback=float(np.average(y, weights=w)))

def predict_leaf_mean(fitted, leaves):
    return np.array([fitted['means'].get(str(int(k)), fitted['fallback']) for k in leaves])

def design(features, frame):
    daily = features[frame.field_index.to_numpy(int), frame.day_index.to_numpy(int) - 1]
    return np.column_stack([daily, np.eye(7)[frame.leaf_index.to_numpy(int)]])

def fit_boosting(features, frame):
    model = HistGradientBoostingRegressor(**BOOSTING)
    weights = hierarchical_weights(frame)
    weights *= len(weights) / weights.sum()
    model.fit(design(features, frame), frame.observed_percent.to_numpy(float), sample_weight=weights)
    return model

def predict_boosting(fitted, features, frame):
    return np.clip(fitted.predict(design(features, frame)), 0, 100)

def fit_calendar(intervals):
    frame = intervals.loc[intervals.genuine].copy()
    if frame.empty:
        raise ValueError('No training genuine onset bounds for calendar reference')
    frame['observed_percent'] = (frame.lower_day + frame.upper_day) / 2
    fitted = fit_leaf_mean(frame)
    fitted['interpretation'] = 'Training hierarchy-weighted mean bracket midpoint in days from scenario sowing'
    return fitted

def predict_calendar(fitted, leaves):
    return predict_leaf_mean(fitted, leaves)
