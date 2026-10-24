"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/yield_transfer_audit_20261009')
import json
from pathlib import Path
import numpy as np
HERE = evidence_dir('analysis/paper_study/yield_transfer_audit_20261009')
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())

def fit_predict(train, test, features):
    allowed = set(sum(PROTOCOL['models'].values(), []))
    if not set(features) <= allowed:
        raise ValueError('Forbidden predictor: ' + str(set(features) - allowed))
    y = train.yield_t_ha.to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite training yields')
    weights = 1.0 / train.groupby('site_year').site_year.transform('size').to_numpy(float)
    weights *= len(train) / weights.sum()
    ymean = np.average(y, weights=weights)
    if not features:
        return (np.full(len(test), ymean), {})
    train_columns, test_columns, unseen = ([], [], {})
    for field in features:
        if train[field].isna().any() or test[field].isna().any():
            raise ValueError('Missing predictor: ' + field)
        if field in PROTOCOL['categorical']:
            a, b = (train[field].astype(str).str.strip(), test[field].astype(str).str.strip())
            levels = sorted(a.unique())
            unseen[field] = int((~b.isin(levels)).sum())
            for level in levels:
                train_columns.append(a.eq(level).to_numpy(float))
                test_columns.append(b.eq(level).to_numpy(float))
        else:
            a, b = (train[field].to_numpy(float), test[field].to_numpy(float))
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite predictor: ' + field)
            mean = np.average(a, weights=weights)
            sd = np.sqrt(np.average((a - mean) ** 2, weights=weights)) or 1.0
            train_columns.append((a - mean) / sd)
            test_columns.append((b - mean) / sd)
    x, xt = (np.column_stack(train_columns), np.column_stack(test_columns))
    mean = np.average(x, axis=0, weights=weights)
    centered = x - mean
    lhs = centered.T @ (centered * weights[:, None]) + PROTOCOL['ridge_alpha'] * np.eye(x.shape[1])
    beta = np.linalg.solve(lhs, centered.T @ (weights * (y - ymean)))
    return (ymean + (xt - mean) @ beta, unseen)
