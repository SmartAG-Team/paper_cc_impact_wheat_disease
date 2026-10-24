"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/overwinter_leaf_model_20261006/disease')
from dataclasses import replace
import numpy as np
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.wetness import duration_exposure
from model.seasonal_septoria.overwinter import OverwinterParameters, simulate_overwinter
FRACTION = 0.001

def _forcing(data, accumulation, thresholds, fitted):
    if data.targets.value.notna().any():
        raise ValueError('Forecast targets must be redacted.')
    lengths = data.metadata.set_index('field_index').forcing_days.reindex(range(len(data.temperature))).to_numpy(int)
    mask = np.arange(data.temperature.shape[1])[None, :] < lengths[:, None]
    host = leaf_host(accumulation, data.temperature, thresholds, rank_spacing_units=fitted['rank_spacing_units'], forcing_mask=mask, juvenile_policy=fitted.get('juvenile_policy', 'handover_31_39'))
    exposure = duration_exposure(data.temperature, data.maximum_temperature, data.humidity, data.rain, fitted['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    return (host, np.where(mask, exposure, 0.0), mask)

def predict_record(data, accumulation, thresholds, fitted, *, ablation='baseline', detection_fraction=FRACTION):
    """Signs and observed BBCH are absent from the prediction boundary."""
    setting = dict(fitted)
    setting['juvenile_policy'] = 'persistent' if ablation == 'persistent_juvenile' else 'handover_31_39'
    if ablation.startswith('spacing'):
        setting['rank_spacing_units'] = float(ablation[7:])
    host, exposure, mask = _forcing(data, accumulation, thresholds, setting)
    p = OverwinterParameters(**setting['parameters'])
    source = 1.0
    pressure = setting['constant_imported_pressure']
    if ablation in ('no_local_source', 'no_sources'):
        source = 0.0
    if ablation in ('imported_off', 'no_sources'):
        pressure = 0.0
    if ablation == 'secondary_off':
        p = replace(p, secondary_scale=0.0)
    if ablation == 'initial_ready_0':
        p = replace(p, initial_ready_fraction=0.0)
    if ablation == 'initial_ready_1':
        p = replace(p, initial_ready_fraction=1.0)
    if ablation == 'residue_lifetime_x0.5':
        p = replace(p, residue_decay_reference_days=p.residue_decay_reference_days * 0.5)
    if ablation == 'residue_lifetime_x2':
        p = replace(p, residue_decay_reference_days=p.residue_decay_reference_days * 2.0)
    if ablation == 'contact_off':
        p = replace(p, contact_fraction=0.0)
    if ablation in ('juvenile_removed_after31', 'lower_ranks_removed_after39'):
        removed = accumulation >= thresholds[31 if ablation == 'juvenile_removed_after31' else 39]
        slots = slice(7, 8) if ablation == 'juvenile_removed_after31' else slice(3, 8)
        for array in (host.active, host.area, host.renewal):
            view = array[:, :, slots]
            view[removed] = 0
    trajectory = simulate_overwinter(data.temperature, exposure, data.rain, host.active, host.renewal, host.area, mask, p, initial_local_source=source, imported_pressure=np.full_like(data.temperature, pressure), detection_fraction=detection_fraction)
    if np.max(abs(trajectory.state.sum(axis=-1) - 1.0)) > 1e-10:
        raise AssertionError('Tissue mass accounting failed.')
    return (trajectory, host)
