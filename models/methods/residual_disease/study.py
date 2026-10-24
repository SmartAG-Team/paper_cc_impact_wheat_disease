"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.methods.disease_trajectory import study as parent
from analysis.methods.joint_disease.data import site_assignments
from analysis.methods.severity_probability.data import weights
from analysis.methods.observation_model import observation as metrics
from analysis.methods.residual_disease import config
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
ROOT = PROJECT_ROOT
FILES = {**parent.FILES, 'trajectory_predictions': 'data/final_study/evidence/disease_trajectory_restructure/results/assessment_predictions.csv', 'trajectory_protocol': 'data/final_study/evidence/disease_trajectory_restructure/protocol.json'}

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def load():
    parent.check_protocol()
    obs, fields, bounds, z, reference, _ = parent.load()
    newer = pd.read_csv(ROOT / FILES['trajectory_predictions']).set_index('global_target_index')
    aligned = newer.loc[reference.global_target_index]
    for key in ['field_id', 'site_id', 'source', 'leaf_index', 'observed_percent', 'day_index']:
        if not np.array_equal(reference[key].to_numpy(), aligned[key].to_numpy()):
            raise ValueError('Prior trajectory target identity mismatch: ' + key)
    reference = reference.copy()
    for col in ['full_window', 'full_point', 'calendar_window']:
        reference[col] = aligned[col].to_numpy()
    return (obs, fields, bounds, z, reference)

def temporal_partition(obs, bounds, year):
    return parent.partition(obs, bounds, year)

def source_partition(obs, bounds, source):
    train = obs.loc[obs.source.ne(source)].copy()
    test = obs.loc[obs.source.eq(source)].copy()
    if train.empty or test.empty or set(train.site_id) & set(test.site_id):
        raise ValueError('Nonempty disjoint source populations required')
    test['held_source'] = source
    return dict(training=train, validation=test, train_bounds=bounds.loc[bounds.field_id.isin(train.field_id)].copy(), test_bounds=bounds.loc[bounds.field_id.isin(test.field_id)].copy(), assignment=site_assignments(train, 3, config.SEED))

def loss(frame, prediction, *, point_only=False):
    residual = np.asarray(prediction, float) - frame.observed_percent.to_numpy(float)
    if residual.shape != (len(frame),) or not np.isfinite(residual).all():
        raise ValueError('Complete finite held-site predictions required')
    assessment = float(np.average(residual ** 2, weights=weights(frame)))
    windows = metrics.windows(frame.assign(prediction=prediction), ['prediction'])
    integral = float(np.average((windows.prediction_audpc - windows.observed_audpc) ** 2, weights=weights(windows))) if len(windows) else 0.0
    return dict(assessment_mse=assessment, integral_mse=integral, score=assessment / 10000 + (0 if point_only else integral / 9000000))

def choose(rows, score='score'):
    if not rows or not np.isfinite([row[score] for row in rows]).all():
        raise ValueError('Complete finite selection scores required')
    minimum = min((row[score] for row in rows))
    eligible = [row for row in rows if row[score] <= minimum + 1e-12]
    return min(eligible, key=lambda row: (config.FAMILY_ORDER.index(row['family']), row['multiplier'], row['setting']))

def candidate_specs():
    rows = []
    for family in ['fold_leaf_mean', 'joint_base', 'trajectory_base']:
        rows.append(dict(family=family, setting=-1, multiplier=0.0))
    for strength in config.MULTIPLIERS:
        rows.append(dict(family='leaf_bias', setting=-1, multiplier=strength))
    for family, (_, kind, _) in config.ML_FAMILIES.items():
        grid = config.RF_GRID if kind == 'rf' else config.HGB_GRID
        for setting in range(len(grid)):
            for strength in [1.0] if family == 'direct_rf' else config.MULTIPLIERS:
                rows.append(dict(family=family, setting=setting, multiplier=strength))
    for i, row in enumerate(rows):
        row['candidate_id'] = i
    return rows

def check_protocol():
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')
