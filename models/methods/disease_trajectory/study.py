"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_trajectory_restructure_20261011')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.methods.severity_probability import data as source
from analysis.methods.joint_disease.data import onset_bounds
HERE = evidence_dir('analysis/paper_study/disease_trajectory_restructure_20261011')
ROOT = PROJECT_ROOT
FILES = dict(source.FILES)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def attach_archived(test, reference):
    if reference.global_target_index.duplicated().any() or test.global_target_index.duplicated().any():
        raise ValueError('Duplicate archived target identity')
    registry = reference.set_index('global_target_index')
    if not set(test.global_target_index) <= set(registry.index):
        raise ValueError('Missing archived target')
    aligned = registry.loc[test.global_target_index]
    for key in ['field_id', 'site_id', 'source', 'leaf_index', 'observed_percent', 'day_index']:
        if not np.array_equal(test[key].to_numpy(), aligned[key].to_numpy()):
            raise ValueError('Changed archived target: ' + key)
    out = test.copy()
    for key in ['joint', 'boosting', 'fold_leaf_mean', 'fold_occurrence', 'event_label', 'occurrence']:
        if key in aligned:
            out[key] = aligned[key].to_numpy()
    return out

def partition(observations, bounds, year):
    train, test, assignment = source.partition(observations, year)
    train_bounds = bounds.loc[bounds.field_id.isin(train.field_id)].copy()
    test_bounds = bounds.loc[bounds.field_id.isin(test.field_id)].copy()
    if set(train_bounds.field_id) != set(train.field_id) or set(test_bounds.field_id) != set(test.field_id):
        raise ValueError('Incomplete field-specific onset histories')
    for col in ['season_year', 'new_site', 'test_year', 'unseen_leaf']:
        if col in test:
            test_bounds[col] = test_bounds.field_id.map(test.groupby('field_id')[col].first())
    return dict(training=train, validation=test, train_bounds=train_bounds, test_bounds=test_bounds, assignment=assignment)

def select_ridge(scores):
    if not scores or not np.isfinite(list(scores.values())).all():
        raise ValueError('Complete finite inner scores required')
    minimum = min(scores.values())
    return max((ridge for ridge, score in scores.items() if score <= minimum + 1e-12))

def load(root=ROOT):
    root = Path(root)
    obs, fields, features, names, reference = source.load(root)
    with np.load(root / FILES['features'], allow_pickle=False) as data:
        z = dict(features=features, valid=data['valid'].copy(), horizons=data['horizons'].copy(), names=names)
    bounds = pd.read_csv(root / FILES['bounds'])
    expected = onset_bounds(obs)
    pd.testing.assert_frame_equal(bounds.reset_index(drop=True), expected[bounds.columns].reset_index(drop=True), check_dtype=False)
    bounds['season_year'] = bounds.field_id.map(fields.set_index('field_id').season_year).astype(int)
    archived_bounds = pd.read_csv(root / FILES['archived_bounds'])
    return (obs, fields, bounds, z, reference, archived_bounds)

def check_protocol():
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')
