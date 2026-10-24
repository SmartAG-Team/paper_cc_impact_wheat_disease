"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/canopy_senescence_20261010')
import numpy as np
import pandas as pd

def build_features(green, plots, weather):
    if plots.plot_id.duplicated().any() or green.observation_id.duplicated().any():
        raise ValueError('duplicate plot or observation identity')
    if green.duplicated(['plot_id', 'date']).any():
        raise ValueError('duplicate plot/date')
    d = green[['observation_id', 'plot_id', 'date', 'green_fraction', 'measurement_eligible']].merge(plots[['plot_id', 'year', 'cultivar_id', 'heading_doy', 'treatment']], on='plot_id', how='left', validate='many_to_one', indicator=True)
    if d['_merge'].ne('both').any():
        raise ValueError('Unmatched plot')
    d = d.drop(columns='_merge')
    d['date'] = pd.to_datetime(d.date, errors='raise')
    w = weather[['date', 'tmean_c']].copy()
    w['date'] = pd.to_datetime(w.date, errors='raise')
    if w.date.duplicated().any():
        raise ValueError('duplicate weather date')
    if w.date.isna().any() or d.date.isna().any():
        raise ValueError('Missing date')
    w['tmean_c'] = pd.to_numeric(w.tmean_c, errors='raise')
    if np.isinf(w.tmean_c).any():
        raise ValueError('Nonfinite weather value')
    w = w.set_index('date').tmean_c
    heading = pd.to_numeric(d.heading_doy, errors='raise')
    origin = pd.to_datetime(d.year.astype(int).astype(str) + '-01-01')
    maximum = np.where(origin.dt.is_leap_year, 366, 365)
    present = heading.notna()
    if ((heading[present] % 1 != 0) | (heading[present] < 1) | (heading[present] > maximum[present])).any():
        raise ValueError('Invalid source heading DOY')
    d['heading_date'] = origin + pd.to_timedelta(heading - 1, unit='D')
    d['elapsed_days'] = (d.date - d.heading_date).dt.days.astype(float)
    d['degree_days'] = np.nan
    d['reason'] = np.select([~d.measurement_eligible, d.green_fraction.isna(), heading.isna(), d.elapsed_days.lt(0)], ['unverified_boundary', 'missing_green_fraction', 'missing_heading', 'before_heading'], default='eligible')
    for start, part in d[d.reason.eq('eligible')].groupby('heading_date', sort=True):
        dates = pd.date_range(start + pd.Timedelta(days=1), part.date.max())
        increments = np.maximum(w.reindex(dates).to_numpy(float), 0.0)
        cumulative = np.r_[0.0, np.cumsum(increments)]
        values = cumulative[part.elapsed_days.to_numpy(int)]
        d.loc[part.index, 'degree_days'] = values
        d.loc[part.index[~np.isfinite(values)], 'reason'] = 'incomplete_weather'
    d['eligible'] = d.reason.eq('eligible')
    d['date'] = d.date.dt.strftime('%Y-%m-%d')
    d['heading_date'] = d.heading_date.dt.strftime('%Y-%m-%d')
    return d
