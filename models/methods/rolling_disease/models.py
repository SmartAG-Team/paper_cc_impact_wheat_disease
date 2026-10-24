"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_coupled_ml_20261012')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from analysis.methods.disease_estimators.models import leaf_means, metadata
from analysis.methods.severity_probability.data import weights
from analysis.methods.rolling_disease import config

def state(fitted, h, leaf):
    prior = fitted['baseline'][np.asarray(leaf, int)]
    return np.where(h[:, 0] > 0, h[:, 1], prior) if fitted['history_conditioned'] else prior

def fit(x, h, frame, spec):
    family = spec['family']
    conditioned = family in ['state_baseline', 'history_rf', 'history_hgb']
    result = dict(kind='baseline', family=family, setting=int(spec['setting']), baseline=leaf_means(frame), history_conditioned=conditioned, **metadata(frame))
    if family in ['fold_leaf_mean', 'state_baseline']:
        return result
    xx = np.column_stack([x, h]) if conditioned else x
    y = frame.observed_percent.to_numpy() / 100 - state(result, h, frame.leaf_index)
    if family.endswith('_rf'):
        estimator = RandomForestRegressor(n_estimators=300, max_features=0.8, max_samples=0.8, n_jobs=1, random_state=config.SEED, **config.RF_GRID[spec['setting']])
        kind = 'rf'
    elif family == 'history_hgb':
        estimator = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, early_stopping=False, max_depth=3, l2_regularization=0.1, random_state=config.SEED, **config.HGB_GRID[spec['setting']])
        kind = 'hgb'
    else:
        raise ValueError('Unknown residual family')
    w = weights(frame)
    estimator.fit(xx, y, sample_weight=w / w.mean())
    result.update(kind=kind, estimator=estimator)
    return result

def predict(fitted, x, h, leaf, strength):
    base = state(fitted, h, leaf)
    if fitted['kind'] == 'baseline':
        value = base
    else:
        xx = np.column_stack([x, h]) if fitted['history_conditioned'] else x
        value = base + strength * fitted['estimator'].predict(xx)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite rolling prediction')
    return 100 * np.clip(value, 0, 1)
