"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_joint_forecast_20261010')
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.methods.joint_disease.features import build
HERE = evidence_dir('analysis/paper_study/disease_joint_forecast_20261010')
ROOT = PROJECT_ROOT
SOURCE_PATHS = {'assessments': 'data/final_study/evidence/phenology_disease_coupling/results/assessment_predictions.csv', 'fields': 'data/final_study/evidence/field_stage_extension/results/field_registry.csv', 'crop': 'data/final_study/evidence/field_stage_extension/results/daily_arrays.npz', 'weather': 'data/final_study/inputs/paper_study/field_weather_completed/daily_weather.parquet', 'additional': 'data/final_study/inputs/stb_collection_20261009/analysis_ready/combined_severity_yield_assessments.csv'}
OBSERVATION_COLUMNS = ['global_target_index', 'field_id', 'site_id', 'coordinate_year', 'source', 'partition', 'leaf_index', 'endpoint_series', 'date', 'day_index', 'observed_percent']
INTERVAL_KEYS = ['field_id', 'field_index', 'site_id', 'coordinate_year', 'source', 'partition', 'endpoint_series', 'leaf_index']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_day_alignment(observations, fields):
    sowing = pd.to_datetime(observations.field_id.map(fields.set_index('field_id').sowing_date))
    expected = (pd.to_datetime(observations.date) - sowing).dt.days + 1
    if expected.isna().any() or not np.array_equal(expected, observations.day_index):
        raise ValueError('Observation day does not match the sowing-day update')

def onset_bounds(observations):
    rows = []
    for identity, series in observations.groupby(INTERVAL_KEYS, sort=True):
        series = series.sort_values('day_index')
        positive = series.loc[series.observed_percent.gt(0), 'day_index']
        upper = float(positive.min()) if len(positive) else np.nan
        negative = series.loc[series.observed_percent.eq(0), 'day_index']
        previous = negative[negative.lt(upper)] if len(positive) else negative
        lower = int(previous.max()) if len(previous) else 0
        rows.append(dict(zip(INTERVAL_KEYS, identity), lower_day=lower, upper_day=upper, genuine=bool(lower > 0 and np.isfinite(upper)), assessments=len(series)))
    return pd.DataFrame(rows)

def site_assignments(observations, n_splits, seed):
    """Balance source and coordinate-year support using identity columns only."""
    sites = observations.groupby('site_id', sort=True).agg(source=('source', 'first'), sources=('source', 'nunique'), mass=('coordinate_year', 'nunique'))
    if sites.sources.gt(1).any() or len(sites) < n_splits:
        raise ValueError('Enough uniquely stratified sites required')
    rng = np.random.default_rng(seed)
    assignment = {}
    total = np.zeros(n_splits)
    for _, group in sites.groupby('source', sort=True):
        group = group.iloc[rng.permutation(len(group))].sort_values('mass', ascending=False, kind='stable')
        local = np.zeros(n_splits)
        for site, row in group.iterrows():
            fold = min(range(n_splits), key=lambda j: (local[j], total[j], j))
            assignment[str(site)] = int(fold)
            local[fold] += row['mass']
            total[fold] += row['mass']
    if set(assignment.values()) != set(range(n_splits)):
        raise ValueError('Empty site fold')
    return assignment

def prepare(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('A new input directory is required')
    observations = pd.read_csv(ROOT / SOURCE_PATHS['assessments'], usecols=OBSERVATION_COLUMNS)
    fields = pd.read_csv(ROOT / SOURCE_PATHS['fields']).reset_index(drop=True)
    fields['field_index'] = np.arange(len(fields))
    observations['field_index'] = observations.field_id.map(fields.set_index('field_id').field_index)
    if len(fields) != 216 or len(observations) != 2287 or observations.field_index.isna().any() or observations.duplicated(['field_id', 'endpoint_series', 'date']).any():
        raise ValueError('Original field or assessment population differs')
    check_day_alignment(observations, fields)
    weather = pd.read_parquet(ROOT / SOURCE_PATHS['weather'])
    weather['date'] = pd.to_datetime(weather.date)
    if weather.duplicated(['location_id', 'date']).any():
        raise ValueError('Duplicate daily weather')
    groups = {str(k): g.set_index('date') for k, g in weather.groupby('location_id')}
    horizons = np.maximum(330, fields.forcing_days.to_numpy(int))
    forcing = np.zeros((len(fields), int(horizons.max()), 3), dtype=float)
    valid = np.arange(forcing.shape[1])[None, :] < horizons[:, None]
    for i, row in enumerate(fields.itertuples(index=False)):
        dates = pd.date_range(row.sowing_date, periods=horizons[i])
        values = groups[str(row.site_id)].reindex(dates)[['tmean_c', 'rh_mean_pct', 'precipitation_mm']].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError('Incomplete weather for ' + row.field_id)
        forcing[i, :horizons[i]] = values
    with np.load(ROOT / SOURCE_PATHS['crop'], allow_pickle=False) as crop:
        lookup = {str(k): i for i, k in enumerate(crop['field_ids'])}
        parents = crop['parent_cdf'][[lookup[k] for k in fields.field_id]].copy()
    features, names = build(forcing, parents, fields.sowing_date.tolist())
    if not valid[observations.field_index.to_numpy(int), observations.day_index.to_numpy(int) - 1].all():
        raise ValueError('An assessment lacks predictor support')
    intervals = onset_bounds(observations)
    outer = site_assignments(observations, 5, 20261010)
    nested = {}
    for fold in range(5):
        training = observations.loc[observations.site_id.map(outer).ne(fold)]
        nested[str(fold)] = site_assignments(training, 3, 20261020 + fold)
        if set(nested[str(fold)]) & {s for s, f in outer.items() if f == fold}:
            raise ValueError('Inner fold contains an outer test site')
    transport = {}
    for source in sorted(observations.source.unique()):
        training = observations.loc[observations.source.ne(source)]
        transport[source] = site_assignments(training, 3, 20261030)
    output.mkdir(parents=True)
    observations.to_csv(output / 'observations.csv', index=False)
    fields.to_csv(output / 'fields.csv', index=False)
    intervals.to_csv(output / 'onset_intervals.csv', index=False)
    np.savez_compressed(output / 'features.npz', features=features, valid=valid, horizons=horizons, field_ids=fields.field_id.to_numpy(str), feature_names=np.array(names))
    folds = dict(outer=outer, inner=nested, leave_source_out_inner=transport)
    (output / 'folds.json').write_text(json.dumps(folds, indent=2) + '\n')
    additional = pd.read_csv(ROOT / SOURCE_PATHS['additional'])
    nordic = additional.loc[additional.dataset_id.eq('nfts2023_2025') & additional.treatment.eq(1) & additional.independent_unit_eligible & ~additional.source_duplicate_control]
    nordic.to_csv(output / 'additional_nordic_candidates.csv', index=False)
    readiness = pd.DataFrame([dict(dataset='BASF/Corteva original', records=2287, sites=127, status='qualified original population', limitation='retrospective; source disease denominator and leaf tracking remain uncertain'), dict(dataset='Nordic untreated candidates', records=len(nordic), sites=nordic.environment_id.nunique(), status='requires leaf and weather qualification', limitation='all current leaf ranks missing; distinct environments need coordinate/site identity confirmation'), dict(dataset='Karisto natural field leaf necrosis', records=21420, sites=1, status='separate measurement target', limitation='one environment; source leaf ranks unspecified; leaf records are nested'), dict(dataset='Tunisia natural field leaf necrosis', records=3004, sites=1, status='separate measurement target', limitation='two years at one site; treatment qualification and incidence weighting required')])
    readiness.to_csv(output / 'additional_source_readiness.csv', index=False)
    receipt = dict(status='prepared_before_fitting', fields=len(fields), sites=observations.site_id.nunique(), assessments=len(observations), evaluation_assessments=int(observations.partition.ne('calibration').sum()), onset_histories=len(intervals), genuine_evaluation_histories=int((intervals.genuine & intervals.partition.ne('calibration')).sum()), predictors=names, feature_shape=list(features.shape), horizon_counts=pd.Series(horizons).value_counts().to_dict(), source_sha256={name: sha(ROOT / path) for name, path in SOURCE_PATHS.items()}, source_paths=SOURCE_PATHS, output_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}, model_fitted=False)
    (output / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
