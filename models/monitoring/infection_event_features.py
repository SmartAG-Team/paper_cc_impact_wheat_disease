"""Compact, strictly earlier weather timing features for the field-history model."""
import numpy as np
import pandas as pd

from .features import build_features

EVENT_NAMES = ['rain_max_7d_mm', 'rain_days_7d', 'rain_run_7d',
    'rain_max_21d_mm', 'rain_days_21d', 'rain_run_21d',
    'humid_hours_7d', 'humid_hours_21d', 'rain_hours_7d', 'rain_hours_21d',
    'gap_wet_dose', 'gap_thermal_units', 'rain_age_days',
    'pending_infection_dose', 'matured_recent_dose']


def build_event_features(weather, fields, queries, history_provenance):
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld from weather features')
    if queries.global_target_index.duplicated().any() or history_provenance.global_target_index.duplicated().any():
        raise ValueError('Unique query and history identities required')
    provenance = history_provenance.set_index('global_target_index').loc[queries.global_target_index]
    same = pd.to_datetime(provenance.latest_same_date, errors='raise', format='mixed')
    any_date = pd.to_datetime(provenance.latest_any_date, errors='raise', format='mixed')
    selected = same.fillna(any_date)
    dates = pd.to_datetime(queries.date, errors='raise', format='mixed').dt.normalize().to_numpy()
    if not ((selected.to_numpy() < dates)|selected.isna().to_numpy()).all():
        raise ValueError('Selected history must be strictly earlier than the assessment')
    prior = {int(key): (date if pd.notna(date) else None)
             for key, date in zip(queries.global_target_index, selected)}
    features, names, ledger = build_features(weather, fields, queries,
        dict(weather_cutoff_days=1, rain_variable='rain_mm', prior_dates=prior, include_process=False))
    columns = [names.index(name) for name in EVENT_NAMES]
    matrix = features[:, columns]
    assert matrix.shape == (len(queries), len(EVENT_NAMES)) and np.isfinite(matrix).all()
    ledger['anchor_date'] = [str(date.date()) if pd.notna(date) else None for date in selected]
    ledger['anchor_rule'] = 'Latest same native leaf if available; otherwise latest earlier other-leaf field date; no anatomical equivalence'
    ledger['infection_features'] = 'Fixed wetness/thermal proxies, not measured leaf wetness or validated latent infection'
    return matrix, EVENT_NAMES.copy(), ledger


def insert_events(matrix, events):
    first, extra = np.asarray(matrix, float), np.asarray(events, float)
    if first.ndim != 2 or first.shape[1] < 3 or extra.shape != (len(first), len(EVENT_NAMES)):
        raise ValueError('Aligned baseline and registered event schema required')
    return np.column_stack([first[:, :-3], extra, first[:, -3:]])
