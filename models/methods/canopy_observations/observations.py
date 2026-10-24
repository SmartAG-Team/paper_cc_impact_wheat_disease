"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/canopy_observations_20261010')
import numpy as np
import pandas as pd
PAIR_KEYS = ['year', 'location', 'cultivar_id', 'block', 'nitrogen', 'water_regime']

def numeric(values):
    result = pd.to_numeric(values, errors='coerce').astype(float)
    if (values.notna() & result.isna()).any() or np.isinf(result).any():
        raise ValueError('Unrecognized or nonfinite numeric measurement')
    return result

def unique(frame, keys):
    if frame[keys].isna().any().any():
        raise ValueError(f'Missing identity in {keys}')
    if frame.duplicated(keys).any():
        raise ValueError(f'duplicate identity in {keys}')

def dry_yield_t_ha(values, unit):
    if unit not in ('dt_ha', 't_ha'):
        raise ValueError('Unrecognized dry-mass yield unit')
    result = numeric(values) / (10 if unit == 'dt_ha' else 1)
    if result.dropna().lt(0).any():
        raise ValueError('Negative dry-mass harvest yield')
    return result

def green_records(frame, *, value_col, date_col, unit, source_name):
    if unit not in ('fraction', 'percent'):
        raise ValueError('Explicit fraction or percent unit required')
    result = frame.copy()
    value = numeric(result[value_col]) / (100 if unit == 'percent' else 1)
    if not value.dropna().between(0, 1).all():
        raise ValueError('Green fraction outside [0, 1]')
    date = pd.to_datetime(result[date_col], errors='raise')
    if date.isna().any():
        raise ValueError('Missing observation date')
    result['date'] = date.dt.strftime('%Y-%m-%d')
    unique(result, ['plot_id', 'date'])
    result['green_fraction'] = value
    result['source_value_present'] = value.notna()
    result['measurement_eligible'] = True
    result['record_kind'] = 'source_dated_record'
    result['source_file'] = source_name
    result['observation_id'] = source_name + '|' + result.source_row.astype(str) + '|' + result.date
    result['measurement'] = 'whole_canopy_green_fraction'
    result['unit'] = 'fraction'
    return result[['observation_id', 'plot_id', 'date', 'green_fraction', 'source_value_present', 'measurement_eligible', 'record_kind', 'source_file', 'source_row', 'measurement', 'unit']]

def boundary_qualification(source):
    """Flag the source's unverified fitted boundaries without changing values."""
    unique(source, ['plot_id', 'source_row'])
    d = source[['plot_id', 'source_row']].copy()
    time = numeric(source.Tsum)
    start = numeric(source.start)
    end = numeric(source.end)
    if time.isna().any() or start.isna().any() or end.isna().any() or (start >= end).any():
        raise ValueError('Invalid source boundary metadata')
    a = np.isclose(time, start, atol=1e-08, rtol=0)
    b = np.isclose(time, end, atol=1e-08, rtol=0)
    counts = d.assign(start=a, end=b).groupby('plot_id')[['start', 'end']].sum()
    if not counts.eq(1).all().all():
        raise ValueError('Each plot needs one start and end boundary')
    values = numeric(source.perc_green)
    if not values[a].eq(0.9999).all() or not values[b].eq(0.0001).all():
        raise ValueError('Changed source boundary value signature')
    d['record_kind'] = np.select([a, b], ['unverified_start_boundary', 'unverified_end_boundary'], default='source_dated_record')
    d['measurement_eligible'] = ~(a | b)
    return d

def join_plot_observations(observations, plots):
    unique(plots, ['plot_id'])
    result = observations.merge(plots, on='plot_id', how='left', validate='many_to_one', indicator=True)
    if result['_merge'].ne('both').any():
        raise ValueError('unmatched canopy plot identity')
    assert len(result) == len(observations)
    return result.drop(columns='_merge')

def observed_windows(observations):
    unique(observations, ['plot_id', 'date'])
    if 'measurement_eligible' in observations and (not observations.measurement_eligible.all()):
        raise ValueError('Exclude unqualified source records before integration')
    rows = []
    for plot, original in observations.groupby('plot_id', sort=True):
        d = original.dropna(subset=['green_fraction']).sort_values('date')
        row = dict(plot_id=plot, records=len(original), observations=len(d), missing_records=int(original.green_fraction.isna().sum()), window_start=d.date.iloc[0] if len(d) else None, window_end=d.date.iloc[-1] if len(d) else None, span_days=np.nan, green_fraction_days=np.nan, window_mean_green_fraction=np.nan, maximum_gap_days=np.nan, status='insufficient_observations')
        if len(d) >= 2:
            t = (pd.to_datetime(d.date) - pd.to_datetime(d.date.iloc[0])).dt.total_seconds().to_numpy() / 86400
            g = d.green_fraction.to_numpy(float)
            dt = np.diff(t)
            integral = float(np.sum(dt * (g[:-1] + g[1:]) / 2))
            row.update(span_days=float(t[-1]), green_fraction_days=integral, window_mean_green_fraction=integral / t[-1], maximum_gap_days=float(dt.max()), status='observed_window')
        rows.append(row)
    return pd.DataFrame(rows)

def protection_contrasts(plots):
    unique(plots, PAIR_KEYS + ['protection'])
    d = plots[plots.primary_complete].copy()
    fields = PAIR_KEYS + ['plot_id', 'primary_green_fraction_days', 'yield_dry_t_ha', 'stb_percent']
    p = d[d.protection.eq('WF')][fields].merge(d[d.protection.eq('NF')][fields], on=PAIR_KEYS, how='inner', suffixes=('_WF', '_NF'), validate='one_to_one')
    p['green_fraction_days_difference'] = p.primary_green_fraction_days_WF - p.primary_green_fraction_days_NF
    p['yield_difference_t_ha'] = p.yield_dry_t_ha_WF - p.yield_dry_t_ha_NF
    p['stb_percent_difference'] = p.stb_percent_WF - p.stb_percent_NF
    p['interpretation'] = 'Matched protection-package contrast; not STB-specific causal loss'
    return p
