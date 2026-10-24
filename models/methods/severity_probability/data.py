"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/severity_probability_20261011')
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = evidence_dir('analysis/paper_study/severity_probability_20261011')
ROOT = PROJECT_ROOT
YEARS = tuple(range(2015, 2020))
SEED = 20261011
FILES = {'observations': 'data/final_study/evidence/disease_joint_forecast/inputs/observations.csv', 'fields': 'data/final_study/evidence/disease_joint_forecast/inputs/fields.csv', 'features': 'data/final_study/evidence/disease_joint_forecast/inputs/features.npz', 'preparation': 'data/final_study/evidence/disease_joint_forecast/inputs/preparation.json', 'archived_predictions': 'data/final_study/evidence/disease_temporal_validation/results/assessment_predictions.csv', 'archived_onset': 'data/final_study/evidence/disease_temporal_validation/results/onset_summary.csv', 'archived_magnitude': 'data/final_study/evidence/disease_temporal_validation/results/summary.csv'}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def weights(frame):
    keys = ['coordinate_year', 'field_id', 'leaf_index']
    if frame.empty or frame[keys].isna().any().any():
        raise ValueError('Complete nonempty hierarchy required')
    fields = frame.groupby(keys[0]).field_id.transform('nunique').to_numpy(float)
    leaves = frame.groupby(keys[:2]).leaf_index.transform('nunique').to_numpy(float)
    repeats = frame.groupby(keys).leaf_index.transform('size').to_numpy(float)
    return 1 / (fields * leaves * repeats)

def columns(family, names):
    if family == 'full':
        return list(range(len(names)))
    if family == 'weather':
        return [i for i, n in enumerate(names) if not n.startswith('crop_cdf_')]
    if family == 'calendar':
        return [i for i, n in enumerate(names) if n.startswith('calendar_') or n == 'elapsed_days_scaled']
    raise ValueError('Unknown feature family')

def design(features, frame, indices):
    return features[frame.field_index.to_numpy(int), frame.day_index.to_numpy(int) - 1][:, indices]

def partition(frame, year):
    dates = pd.to_datetime(frame.date, errors='coerce')
    if frame.global_target_index.duplicated().any() or dates.isna().any() or (not dates.dt.year.eq(frame.season_year).all()):
        raise ValueError('Invalid calendar/target identity')
    train = frame.loc[frame.season_year.lt(year)].copy()
    test = frame.loc[frame.season_year.eq(year)].copy()
    if train.empty or test.empty:
        raise ValueError('Nonempty earlier training and current test required')
    for key in ['global_target_index', 'field_id', 'coordinate_year']:
        if set(train[key]) & set(test[key]):
            raise ValueError('Temporal identity overlap')
    sites = train.groupby('site_id', sort=True).agg(source=('source', 'first'), sources=('source', 'nunique'), mass=('coordinate_year', 'nunique'))
    if sites.sources.gt(1).any() or len(sites) < 3:
        raise ValueError('Insufficient source-specific sites')
    rng = np.random.default_rng(SEED + year)
    folds = {}
    total = np.zeros(3)
    for _, part in sites.groupby('source', sort=True):
        part = part.iloc[rng.permutation(len(part))].sort_values('mass', ascending=False, kind='stable')
        local = np.zeros(3)
        for site, row in part.iterrows():
            fold = min(range(3), key=lambda j: (local[j], total[j], j))
            folds[str(site)] = int(fold)
            local[fold] += row['mass']
            total[fold] += row['mass']
    if set(folds.values()) != {0, 1, 2}:
        raise ValueError('Empty inner fold')
    test['new_site'] = ~test.site_id.isin(train.site_id)
    test['unseen_leaf'] = ~test.leaf_index.isin(train.leaf_index)
    test['test_year'] = year
    return (train, test, folds)

def check_lock(root=ROOT):
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def load(root=ROOT):
    root = Path(root)
    obs = pd.read_csv(root / FILES['observations'])
    fields = pd.read_csv(root / FILES['fields'])
    archived = pd.read_csv(root / FILES['archived_predictions'])
    with np.load(root / FILES['features'], allow_pickle=False) as z:
        features = z['features'].copy()
        valid = z['valid']
        horizons = z['horizons']
        names = z['feature_names'].astype(str).tolist()
        if not np.array_equal(z['field_ids'].astype(str), fields.field_id.astype(str)):
            raise ValueError('Field matrix identity mismatch')
    if not np.array_equal(fields.field_index, np.arange(len(fields))):
        raise ValueError('Invalid field indexing')
    if not np.array_equal(valid, np.arange(features.shape[1])[None, :] < horizons[:, None]):
        raise ValueError('Invalid feature support')
    if not np.isfinite(features[valid]).all():
        raise ValueError('Nonfinite predictor')
    registry = fields.set_index('field_id')
    obs['season_year'] = obs.field_id.map(registry.season_year).astype(int)
    for key in ['field_index', 'site_id', 'source']:
        if not obs[key].eq(obs.field_id.map(registry[key])).all():
            raise ValueError('Field lookup mismatch: ' + key)
    dates = pd.to_datetime(obs.date)
    sowing = pd.to_datetime(obs.field_id.map(registry.sowing_date))
    if not np.array_equal((dates - sowing).dt.days + 1, obs.day_index):
        raise ValueError('Assessment day alignment')
    if not obs.coordinate_year.eq(obs.site_id + '|' + obs.season_year.astype(str)).all():
        raise ValueError('Coordinate/year identity')
    if obs.duplicated(['field_id', 'endpoint_series', 'date']).any():
        raise ValueError('Duplicate assessment')
    if not valid[obs.field_index.to_numpy(int), obs.day_index.to_numpy(int) - 1].all():
        raise ValueError('Uncovered assessment')
    if len(obs) != 2287 or len(fields) != 216 or obs.site_id.nunique() != 127:
        raise ValueError('Changed frozen population')
    old = archived.set_index('global_target_index')
    future = obs.loc[obs.season_year.isin(YEARS)]
    if old.index.duplicated().any() or set(old.index) != set(future.global_target_index):
        raise ValueError('Archived evaluation membership')
    for key in ['field_id', 'site_id', 'source', 'leaf_index', 'observed_percent', 'day_index']:
        a = future.set_index('global_target_index')[key]
        b = old.loc[a.index, key]
        if not np.array_equal(a, b):
            raise ValueError('Archived target mismatch: ' + key)
    return (obs, fields, features, names, archived)
