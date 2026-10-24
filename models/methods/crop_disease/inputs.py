"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_disease_coupling_20261010')
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pandas as pd
HERE = evidence_dir('analysis/paper_study/phenology_disease_coupling_20261010')
ROOT = PROJECT_ROOT
BASE = Path('data/final_study/evidence')
STAGES = BASE / 'field_stage_extension/results'
DISEASE = BASE / 'overwinter_leaf_model/disease'
OBSERVATION = BASE / 'disease_observation_revision/results'
MEMBERSHIP = BASE / 'nature_food_fix/validation/source_stage_and_prediction_membership.csv'
DATA_FILES = [BASE / 'seasonal_calibration_v1/basf_source_assessments_snapshot.csv', Path('data/final_study/inputs/paper_study/observations/corteva_external_assessments.csv'), Path('data/final_study/inputs/paper_study/field_weather_completed/daily_weather.parquet'), Path('data/final_study/inputs/paper_study/wheat_area/trial_point_calendar_scenarios.csv'), Path('process_model/parameters/calibration.json'), BASE / 'overwinter_leaf_model/phenology/calibrated_stage_thresholds.json', DISEASE / 'overwinter_source_model/frozen_selected_fit.json', DISEASE / 'original_sign_only_target_membership.parquet', OBSERVATION / 'frozen_clean_development_models.json', OBSERVATION / 'reused_assessment_predictions.csv', MEMBERSHIP, STAGES / 'daily_arrays.npz', STAGES / 'field_registry.csv', STAGES / 'fitted_parameters.json', STAGES / 'protocol.json', STAGES / 'receipt.json']
KEYS = ['field_id', 'endpoint_series', 'date']

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def align_membership(current, frozen):
    """Preserve frozen row identity while recovering current simulation indices."""
    c, f = (current.copy(), frozen.copy())
    for frame in (c, f):
        frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
        if frame.empty or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError('Unique complete field/leaf/date identities required')
    checks = ['leaf_index', 'day_index', 'site_id', 'coordinate_year', 'season_year']
    wanted = KEYS + checks + ['value', 'field_index', 'global_target_index']
    joined = f[KEYS].merge(c[wanted], on=KEYS, how='left', validate='one_to_one', indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Frozen observations are missing from current source data')
    for key in checks:
        if not np.array_equal(joined[key].to_numpy(), f[key].to_numpy()):
            raise ValueError('Frozen observation metadata differs: ' + key)
    if not np.allclose(joined.value, f.observed_percent, rtol=0, atol=1e-12):
        raise ValueError('Original observed magnitudes differ')
    f['field_index'] = joined.field_index.to_numpy(int)
    f['current_global_target_index'] = joined.global_target_index.to_numpy(int)
    return f

def load_inputs(root=ROOT):
    from analysis.methods.field_inputs.run import load_inputs as original_load, subset_fields
    root = Path(root)
    paths = dict(zip(['basf', 'corteva', 'weather', 'calendars', 'phenology'], [root / p for p in DATA_FILES[:5]]))
    data, _, accumulation, thresholds = original_load(paths)
    frozen = pd.read_csv(root / MEMBERSHIP).sort_values('global_target_index').reset_index(drop=True)
    aligned = align_membership(data.targets, frozen)
    positions = data.targets.reset_index().set_index('global_target_index')['index']
    indices = positions.loc[aligned.current_global_target_index].to_numpy(int)
    selected_fields = np.sort(data.targets.iloc[indices].field_index.unique())
    redacted = subset_fields(data, indices, redact_values=True)
    remap = dict(zip(selected_fields, range(len(selected_fields))))
    aligned['field_index'] = aligned.field_index.map(remap).astype(int)
    redacted.targets = redacted.targets[['global_target_index', 'field_id', 'field_index', 'leaf_index', 'date', 'day_index', 'value']].copy()
    registry = pd.read_csv(root / STAGES / 'field_registry.csv')
    field_ids = redacted.metadata.sort_values('field_index').field_id.to_numpy(str)
    if len(field_ids) != 216 or set(field_ids) != set(registry.field_id) or len(aligned) != 2287:
        raise ValueError('Original field/assessment membership changed')
    registry = registry.set_index('field_id').loc[field_ids].reset_index()
    metadata = redacted.metadata.sort_values('field_index').reset_index(drop=True)
    for key in ['site_id', 'sowing_date', 'forcing_days']:
        if not np.array_equal(metadata[key].astype(str), registry[key].astype(str)):
            raise ValueError('Stage and disease field forcing disagree: ' + key)
    signs = pd.read_parquet(root / DISEASE / 'original_sign_only_target_membership.parquet')
    original = signs.set_index('global_target_index').loc[aligned.global_target_index]
    if not (np.array_equal(original.field_id, aligned.field_id) and np.array_equal(original.partition, aligned.partition) and np.array_equal(original.value.gt(0), aligned.observed_percent.gt(0))):
        raise ValueError('Original sign-only population changed')
    thresholds.update({int(k): float(v) for k, v in json.loads((root / DATA_FILES[5]).read_text())['thresholds'].items()})
    with np.load(root / STAGES / 'daily_arrays.npz', allow_pickle=False) as arrays:
        lookup = {str(name): i for i, name in enumerate(arrays['field_ids'])}
        if len(lookup) != 216 or set(lookup) != set(field_ids):
            raise ValueError('Daily CDF field identities differ')
        order = [lookup[name] for name in field_ids]
        parent = arrays['parent_cdf'][order].copy()
        extended = arrays['extended_cdf'][order].copy()
    fit = json.loads((root / DATA_FILES[6]).read_text())['fitted']
    observation_fit = json.loads((root / DATA_FILES[8]).read_text())['bounded_leaf_raw']
    return SimpleNamespace(data=redacted, observations=aligned, registry=registry, retained_coordinate=accumulation[selected_fields].copy(), thresholds=thresholds, fitted=fit, observation_fit=observation_fit, parent_cdf=parent, extended_cdf=extended)

def source_files(root=ROOT):
    """Portable Python sources; numerical dependencies are listed separately."""
    root = Path(root)
    files = []
    for folder in ['analysis', 'model', 'calibration', 'process_model']:
        for path in (root / folder).rglob('*.py'):
            if '__pycache__' in path.parts or 'source_before' in path.parts:
                continue
            files.append(path.relative_to(root))
    return sorted(files)

def dependency_hashes(root=ROOT):
    files = [p for p in source_files(root) if not p.name.startswith('test_') and (not ('phenology_disease_coupling_20261010' in p.parts and p.name in {'verify.py', 'report.py', 'package.py'}))]
    return {str(p): sha(Path(root) / p) for p in sorted(set(DATA_FILES + files))}

def check_hashes(hashes, root=ROOT):
    for relative, expected in hashes.items():
        if sha(Path(root) / relative) != expected:
            raise ValueError('Frozen dependency changed: ' + relative)
