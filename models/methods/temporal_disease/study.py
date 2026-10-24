"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_temporal_validation_20261010')
import numpy as np
import pandas as pd
from analysis.methods.joint_disease import data as original
from analysis.methods.joint_disease import run as joint
SEED = 20261010

def augment(observations, fields, bounds, z):
    observations, fields, bounds = (observations.copy(), fields.copy(), bounds.copy())
    if observations.global_target_index.duplicated().any() or fields.field_id.duplicated().any():
        raise ValueError('Duplicate target or field identity')
    if observations.duplicated(['field_id', 'endpoint_series', 'date']).any():
        raise ValueError('Duplicate assessment in source series')
    if not np.array_equal(fields.field_index, np.arange(len(fields))):
        raise ValueError('Field indices do not match feature rows')
    if not np.array_equal(fields.field_id.to_numpy(str), z['field_ids'].astype(str)):
        raise ValueError('Field identities do not match feature rows')
    mask = np.asarray(z['valid'])
    expected = np.arange(mask.shape[1])[None, :] < z['horizons'][:, None]
    if mask.shape != z['features'].shape[:2] or not np.array_equal(mask, expected) or (not np.isfinite(z['features'][mask]).all()):
        raise ValueError('Invalid feature horizon or noncontiguous feature mask')
    registry = fields.set_index('field_id')
    if not set(observations.field_id) <= set(registry.index):
        raise ValueError('Unknown observation field')
    for name in ['field_index', 'site_id', 'source']:
        if not observations[name].eq(observations.field_id.map(registry[name])).all():
            raise ValueError('Observation/field mismatch: ' + name)
    observations['season_year'] = observations.field_id.map(registry.season_year).astype(int)
    if not observations.coordinate_year.eq(observations.site_id + '|' + observations.season_year.astype(str)).all():
        raise ValueError('Coordinate-year mismatch')
    dates = pd.to_datetime(observations.date, errors='coerce')
    if dates.isna().any() or not dates.dt.year.eq(observations.season_year).all():
        raise ValueError('Assessment calendar year mismatch')
    original.check_day_alignment(observations, fields)
    leaves = observations.leaf_index.to_numpy(float)
    values = observations.observed_percent.to_numpy(float)
    days = observations.day_index.to_numpy(float)
    if not np.isfinite(leaves).all() or not np.equal(leaves, np.floor(leaves)).all() or np.any((leaves < 0) | (leaves >= 7)):
        raise ValueError('Invalid source leaf category')
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ValueError('Invalid observed severity')
    if np.any(days < 1) or np.any(days > z['horizons'][observations.field_index.to_numpy(int)]):
        raise ValueError('Assessment outside feature horizon')
    expected_bounds = original.onset_bounds(observations)
    keys = ['field_id', 'endpoint_series', 'leaf_index']
    try:
        pd.testing.assert_frame_equal(bounds.sort_values(keys).reset_index(drop=True), expected_bounds[bounds.columns].sort_values(keys).reset_index(drop=True), check_dtype=False)
    except (AssertionError, KeyError) as exc:
        raise ValueError('Onset bounds do not match frozen observations') from exc
    bounds['season_year'] = bounds.field_id.map(registry.season_year).astype(int)
    fields['last_feature_date'] = (pd.to_datetime(fields.sowing_date) + pd.to_timedelta(z['horizons'] - 1, unit='D')).dt.strftime('%Y-%m-%d')
    return (observations, fields, bounds)

def load_inputs():
    joint.check_inputs()
    inputs = original.HERE / 'inputs'
    with np.load(inputs / 'features.npz', allow_pickle=False) as data:
        z = {k: data[k] for k in data.files}
    observations, fields, bounds = augment(pd.read_csv(inputs / 'observations.csv'), pd.read_csv(inputs / 'fields.csv'), pd.read_csv(inputs / 'onset_intervals.csv'), z)
    if len(observations) != 2287 or len(fields) != 216 or observations.site_id.nunique() != 127:
        raise ValueError('Frozen registry population changed')
    return (observations, fields, bounds, z)

def partition(observations, fields, bounds, year):
    training = observations.loc[observations.season_year.lt(year)].copy()
    validation = observations.loc[observations.season_year.eq(year)].copy()
    if training.empty or validation.empty:
        raise ValueError('Nonempty earlier-year training and current-year testing required')
    for key in ['field_id', 'coordinate_year', 'global_target_index']:
        if set(training[key]) & set(validation[key]):
            raise ValueError('Temporal identity overlap: ' + key)
    cutoff = pd.Timestamp(year=year, month=1, day=1)
    training_fields = fields.loc[fields.field_id.isin(training.field_id)]
    if pd.to_datetime(training.date).ge(cutoff).any() or pd.to_datetime(training_fields.last_feature_date).ge(cutoff).any() or pd.to_datetime(validation.date).lt(cutoff).any():
        raise ValueError('Observation or training feature exceeds calendar cutoff')
    train_bounds = bounds.loc[bounds.field_id.isin(training.field_id)].copy()
    test_bounds = bounds.loc[bounds.field_id.isin(validation.field_id)].copy()
    if not set(train_bounds.field_id) == set(training.field_id) or not set(test_bounds.field_id) == set(validation.field_id):
        raise ValueError('Incomplete field-specific onset bounds')
    for frame in [validation, test_bounds]:
        frame['new_site'] = ~frame.site_id.isin(training.site_id)
        frame['unseen_leaf'] = ~frame.leaf_index.isin(training.leaf_index)
        frame['test_year'] = year
    assignment = original.site_assignments(training, 3, SEED + year)
    return dict(training=training, validation=validation, train_bounds=train_bounds, test_bounds=test_bounds, assignment=assignment)

def check_protocol():
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')
