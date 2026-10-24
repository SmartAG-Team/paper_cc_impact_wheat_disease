"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/structural_evaluation')
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from analysis.methods.structural_inputs import development_inputs
from calibration.seasonal_septoria.calibrate import fit as fit_seir, predict as predict_seir
from model.seasonal_septoria.core import Parameters
from calibration.seasonal_septoria.field_data import prepare_fields
from calibration.seasonal_septoria.refinement import calibration_weights
from calibration.seasonal_septoria.structural import fit_canopy, predict_canopy
from model.seasonal_septoria.wetness import duration_exposure
VALIDATION_LABEL = 'retrospective_development_validation'
KEYS = ['field_id', 'endpoint_series', 'date']

def write_frozen_json(path, value):
    """Install a complete record atomically, refusing to overwrite an archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def read_completed_fit(path, identity):
    path = Path(path)
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    if cached.get('status') != 'fit_complete':
        raise ValueError('A cached fit record is incomplete.')
    if cached.get('identity') != identity:
        raise ValueError('Cached fit identity hashes differ; use a new output directory.')
    return cached['fitted']

def subset_fields(data, target_indices, redact_values=False):
    """Physically remove withheld fields and outcomes from optimizer inputs."""
    indices = np.asarray(target_indices, int)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(data.targets)):
        raise ValueError('Unique nonempty field-season target indices are required.')
    target = data.targets.iloc[indices].copy()
    fields = np.sort(target.field_index.unique())
    complete = np.flatnonzero(data.targets.field_index.isin(fields))
    if not np.array_equal(np.sort(indices), complete):
        raise ValueError('A subset requires complete field-season assessment histories.')
    mapping = {old: new for new, old in enumerate(fields)}
    target['global_target_index'] = target.global_target_index.to_numpy() if 'global_target_index' in target else indices
    target['field_index'] = target.field_index.map(mapping).astype(int)
    if redact_values:
        target['value'] = np.nan
    metadata = data.metadata.set_index('field_index').loc[fields].reset_index().copy()
    metadata['field_index'] = metadata.field_index.map(mapping).astype(int)
    selected = replace(data, temperature=data.temperature[fields].copy(), humidity=data.humidity[fields].copy(), rain=data.rain[fields].copy(), host_active=data.host_active[fields].copy(), host_renewal=data.host_renewal[fields].copy(), metadata=metadata, targets=target.reset_index(drop=True), excluded=pd.DataFrame())
    if hasattr(data, 'maximum_temperature'):
        selected.maximum_temperature = data.maximum_temperature[fields].copy()
    return selected

def estimate_training_rain_rate(metadata, weather):
    """Rain/hour ratio from unique, positive-rain, warm training weather days."""
    forcing = weather.copy()
    forcing['date'] = pd.to_datetime(forcing.date)
    if forcing.duplicated(['location_id', 'date']).any():
        raise ValueError('Duplicate weather location/date records.')
    keys = []
    for meta in metadata.itertuples():
        keys.append(pd.DataFrame(dict(location_id=meta.site_id, date=pd.date_range(meta.first_forcing_date, meta.last_forcing_date))))
    if not keys:
        raise ValueError('Training field-season meteorology is empty.')
    keys = pd.concat(keys, ignore_index=True).drop_duplicates(['location_id', 'date'])
    matched = keys.merge(forcing, on=['location_id', 'date'], how='left', validate='one_to_one')
    required = ['tmean_c', 'precipitation_mm', 'rain_hours_gt_0_1mm']
    if not np.isfinite(matched[required].to_numpy(float)).all():
        raise ValueError('Complete finite training rainfall-duration meteorology is required.')
    eligible = matched.loc[matched.rain_hours_gt_0_1mm.gt(0) & matched.tmean_c.gt(2) & matched.precipitation_mm.gt(0)]
    if eligible.empty:
        raise ValueError('No positive-rain warm training days identify effective rainfall intensity.')
    rate = float(np.median(eligible.precipitation_mm / eligible.rain_hours_gt_0_1mm))
    return dict(rain_rate_mm_hour=rate, unique_training_location_dates=len(matched), eligible_positive_rain_days=len(eligible), estimation='median daily precipitation / observed positive rain hours; tmean > 2 C', meteorology_only=True)

def _exposure(data, preprocessing):
    if preprocessing['weather_operator'] == 'daily_or':
        return None
    return duration_exposure(data.temperature, data.maximum_temperature, data.humidity, data.rain, rain_rate_mm_hour=preprocessing['rain_rate_mm_hour'])['exposure']

def fit_record(destination, family, candidate, training, accumulation, thresholds, weather, weather_operator, run_identity, membership):
    preprocessing = dict(weather_operator=weather_operator if family == 'structural_canopy' else 'daily_or')
    if family == 'structural_canopy' and weather_operator == 'duration_proxy':
        preprocessing.update(estimate_training_rain_rate(training.metadata, weather))
    identity = dict(run_identity=run_identity, family=family, candidate=candidate, global_training_target_indices=training.targets.global_target_index.tolist(), membership=membership, preprocessing=preprocessing)
    fitted = read_completed_fit(destination, identity)
    if fitted is None:
        rows = np.arange(len(training.targets))
        if family == 'structural_canopy':
            fitted = fit_canopy(training, accumulation, thresholds, rows, candidate['host_parameters'], amplification_free=False, latent_days=candidate['latent_days'], final_weight=0.5, initiation_max=1.0 if weather_operator == 'duration_proxy' else 0.1, environmental_response=_exposure(training, preprocessing))
        else:
            weighted = replace(training, targets=training.targets.copy())
            weighted.targets['weight'] = calibration_weights(weighted.targets, final_weight=0.5)
            fitted = fit_seir(weighted, rows, latent_days=candidate['latent_days'], beta_free=True)
        fitted['weather_preprocessing'] = preprocessing
        fitted['validation_label'] = VALIDATION_LABEL
        write_frozen_json(destination, dict(status='fit_complete', identity=identity, fitted=fitted, frozen_utc=datetime.now(timezone.utc).isoformat(), evaluation_observations_used=False, prior_outcome_exposure=True))
    return fitted

def predict_record(family, data, accumulation, thresholds, fitted):
    if data.targets.value.notna().any():
        raise ValueError('Prediction inputs must have all disease values redacted.')
    if family == 'structural_canopy':
        forecast, _ = predict_canopy(data, accumulation, thresholds, fitted, environmental_response=_exposure(data, fitted['weather_preprocessing']))
    else:
        forecast, _ = predict_seir(data, Parameters(**fitted['parameters']))
    if forecast.shape != (len(data.targets),) or not np.isfinite(forecast).all() or np.any(forecast < -1e-08) or np.any(forecast > 100 + 1e-08):
        raise ValueError('A prediction is missing or outside the supported percentage range.')
    return forecast

def save_frozen_csv(path, frame):
    path = Path(path)
    content = frame.to_csv(index=False)
    if path.exists():
        if path.read_text() != content:
            raise ValueError('Recovered CSV results differ from their frozen values.')
    else:
        with path.open('x') as stream:
            stream.write(content)

def load_inputs(paths):
    basf, corteva = (pd.read_csv(paths['basf']), pd.read_csv(paths['corteva']))
    basf['source'] = 'BASF'
    corteva = corteva.loc[corteva.assessment_eligible.eq(True)].copy()
    corteva['source'] = 'Corteva'
    corteva['endpoint_series'] = corteva.physical_unit.astype(str) + '|' + corteva.organ.astype(str)
    source = pd.concat([basf, corteva], ignore_index=True, sort=False)
    source['physical_unit'] = source.physical_unit.astype(str)
    weather, calendars = (pd.read_parquet(paths['weather']), pd.read_csv(paths['calendars']))
    data = prepare_fields(source, calendars, weather)
    if data.targets.duplicated(KEYS).any():
        raise ValueError('Source field/leaf/date endpoints are not unique.')
    if not data.targets.metric.eq('infection_percent_unspecified_basis').all():
        raise ValueError('The canopy score requires source INFECT percentages.')
    data.targets['global_target_index'] = np.arange(len(data.targets))
    data.targets['validation_label'] = VALIDATION_LABEL
    phenology = json.loads(Path(paths['phenology']).read_text())
    accumulation, thresholds = development_inputs(data, weather, phenology)
    return (data, weather, accumulation, thresholds)
