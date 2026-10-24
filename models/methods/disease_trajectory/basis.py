"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_trajectory_restructure_20261011')
import numpy as np
KNOT_DAYS = (60, 120, 180, 210, 240, 270, 300, 330)
CALENDAR = ('elapsed_days_scaled', 'calendar_sin', 'calendar_cos')

def raw_basis(features, names, family):
    features = np.asarray(features, float)
    if features.ndim != 3 or features.shape[-1] != len(names) or len(set(names)) != len(names):
        raise ValueError('Unique named field/day predictors required')
    if family not in {'full', 'calendar'} or not set(CALENDAR) <= set(names):
        raise ValueError('Known family and calendar predictors required')
    chosen = list(names) if family == 'full' else list(CALENDAR)
    out = features[..., [names.index(k) for k in chosen]].copy()
    elapsed = np.clip(features[..., names.index('elapsed_days_scaled')], 0, 390 / 330)
    out[..., chosen.index('elapsed_days_scaled')] = elapsed
    hinges = np.maximum(elapsed[..., None] - np.array(KNOT_DAYS) / 330, 0)
    return (np.concatenate([out, hinges], axis=-1), chosen + [f'elapsed_hinge_{k}' for k in KNOT_DAYS])

def fit_basis(features, valid, fields, names, family):
    raw, columns = raw_basis(features, names, family)
    valid = np.asarray(valid, bool)
    fields = np.unique(np.asarray(fields, int))
    if valid.shape != raw.shape[:2] or not len(fields) or np.any(fields < 0) or np.any(fields >= len(raw)):
        raise ValueError('Valid training field indices and daily mask required')
    values = raw[fields][valid[fields]]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty training predictors required')
    scale = values.std(axis=0)
    scale[scale < 1e-08] = 1.0
    return dict(family=family, input_names=list(names), columns=columns, mean=values.mean(axis=0).tolist(), scale=scale.tolist(), training_fields=fields.tolist())

def transform(fitted, features):
    raw, columns = raw_basis(features, fitted['input_names'], fitted['family'])
    if columns != fitted['columns'] or not np.isfinite(raw).all():
        raise ValueError('Finite predictors with the fitted column order required')
    return (raw - np.asarray(fitted['mean'])) / np.asarray(fitted['scale'])
