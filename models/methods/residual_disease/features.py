"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
import numpy as np

def design(z, fields, frame, base_predictions=None, *, geography=False):
    fi = frame.field_index.to_numpy(int)
    day_raw = frame.day_index.to_numpy(float)
    leaf_raw = frame.leaf_index.to_numpy(float)
    if not np.isfinite(day_raw).all() or not np.equal(day_raw, np.floor(day_raw)).all() or np.any(day_raw < 1) or np.any(day_raw > z['features'].shape[1]):
        raise ValueError('Valid integer assessment days required')
    if not np.isfinite(leaf_raw).all() or not np.equal(leaf_raw, np.floor(leaf_raw)).all() or np.any(leaf_raw < 0) or np.any(leaf_raw >= 7):
        raise ValueError('Source leaf indices must lie in 0–6')
    day = day_raw.astype(int) - 1
    leaf = leaf_raw.astype(int)
    if np.any(fi < 0) or np.any(fi >= len(z['features'])) or (not z['valid'][fi, day].all()):
        raise ValueError('Assessment outside valid field/day support')
    arrays = [z['features'][fi, day], np.eye(7)[leaf]]
    names = list(z['names']) + [f'leaf_{k}' for k in range(7)]
    if geography:
        if fields.field_index.duplicated().any():
            raise ValueError('Duplicate field registry')
        coordinates = fields.set_index('field_index').loc[fi, ['latitude', 'longitude']].to_numpy(float)
        if not np.isfinite(coordinates).all() or np.any(abs(coordinates[:, 0]) > 90) or np.any(abs(coordinates[:, 1]) > 180):
            raise ValueError('Valid geographic coordinates required')
        arrays.append(coordinates / np.array([90.0, 180.0]))
        names += ['latitude_scaled', 'longitude_scaled']
    if base_predictions is not None:
        if not np.array_equal(frame.global_target_index.to_numpy(), base_predictions.global_target_index.to_numpy()):
            raise ValueError('Base prediction target identity/order mismatch')
        values = base_predictions[['base_percent', 'base_occurrence']].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values[:, 0] < 0) | (values[:, 0] > 100)) or np.any((values[:, 1] < 0) | (values[:, 1] > 1)):
            raise ValueError('Bounded base output required')
        arrays.append(values / np.array([100.0, 1.0]))
        names += ['base_severity_fraction', 'base_occurrence']
    x = np.column_stack(arrays)
    if not np.isfinite(x).all():
        raise ValueError('Finite ML predictors required')
    return (x, names)

def base_columns(frame, name):
    return frame[['global_target_index', name + '_base', name + '_occurrence']].rename(columns={name + '_base': 'base_percent', name + '_occurrence': 'base_occurrence'})
