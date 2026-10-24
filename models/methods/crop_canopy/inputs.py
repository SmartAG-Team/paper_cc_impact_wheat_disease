"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_canopy_coupling_20261011')
from pathlib import Path
import calendar
import importlib.util
import numpy as np
import pandas as pd
HERE = evidence_dir('analysis/paper_study/phenology_canopy_coupling_20261011')
POLICIES = ('trial_doy', 'das_when_available', 'source_origin')
HORIZON = 330

def day_date(year, value):
    if not np.isfinite(value) or value != int(value) or (not 1 <= value <= 365 + calendar.isleap(int(year))):
        raise ValueError('Invalid day of year')
    return pd.Timestamp(int(year), 1, 1) + pd.Timedelta(days=int(value) - 1)

def sowing_registry(plots, raw_green, latitude, longitude):
    rows = []
    for year, part in plots.groupby('year', sort=True):
        source = part.sowing_source_doy.dropna().unique()
        if len(source) != 1 or part.sowing_source_doy.isna().any():
            raise ValueError('One complete trial sowing day per year required')
        trial = day_date(year - 1, source[0])
        raw = raw_green.loc[raw_green.year.eq(year), ['Datum', 'days_after_sawing']].dropna()
        derived = pd.to_datetime(raw.Datum) - pd.to_timedelta(raw.days_after_sawing, unit='D')
        anchors = derived.unique()
        if len(anchors) > 1:
            raise ValueError('inconsistent derived sowing anchors')
        if len(anchors) and (pd.Timestamp(anchors[0]).year != year - 1 or not derived.eq(derived.dt.normalize()).all()):
            raise ValueError('Derived sowing must be a calendar date in the preceding year')
        for policy in POLICIES:
            date, basis = (trial, 'one_based_trial_DOY')
            if policy == 'source_origin':
                date, basis = (trial + pd.Timedelta(days=1), 'source_R_January1_plus_DOY')
            elif policy == 'das_when_available':
                date = pd.Timestamp(anchors[0]) if len(anchors) else trial
                basis = 'source_date_minus_DAS' if len(anchors) else 'trial_fallback_no_DAS'
            rows.append(dict(policy=policy, year=int(year), sowing_date=date.strftime('%Y-%m-%d'), sowing_basis=basis, latitude=latitude, longitude=longitude, horizon_days=HORIZON))
    return pd.DataFrame(rows)

def measurement_cohort(green, plots, old_features):
    for frame, key in ((green, 'observation_id'), (plots, 'plot_id'), (old_features, 'observation_id')):
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError('Missing or duplicate source identity')
    if not green.plot_id.isin(plots.plot_id).all():
        raise ValueError('unmatched measurement plot')
    cols = ['plot_id', 'year', 'treatment', 'cultivar_id', 'heading_doy']
    merged = green.merge(plots[cols], on='plot_id', validate='many_to_one')
    use = merged.measurement_eligible.eq(True) & merged.green_fraction.notna() & merged.treatment.eq('HN_WF') & merged.year.between(2015, 2018)
    result = merged.loc[use, ['observation_id', *cols, 'date', 'green_fraction']].copy()
    if not np.isfinite(result.green_fraction).all() or not result.green_fraction.between(0, 1).all():
        raise ValueError('Eligible measurements must be finite green fractions')
    if result.duplicated(['plot_id', 'date']).any():
        raise ValueError('duplicate plot assessment date')
    if not pd.to_datetime(result.date).dt.year.eq(result.year).all():
        raise ValueError('Observation dates must agree with harvest year')
    old = old_features.set_index('observation_id').eligible
    if not result.observation_id.isin(old.index).all():
        raise ValueError('unmatched original comparison identity')
    result['old_comparison_eligible'] = result.observation_id.map(old).astype(bool)
    return result.sort_values(['year', 'plot_id', 'date']).reset_index(drop=True)

def heading_labels(plots):
    rows = []
    for row in plots.loc[plots.treatment.eq('HN_WF') & plots.heading_doy.notna()].itertuples():
        rows.append(dict(plot_id=row.plot_id, year=int(row.year), cultivar_id=row.cultivar_id, source_stage=59, heading_date=day_date(row.year, row.heading_doy).strftime('%Y-%m-%d')))
    return pd.DataFrame(rows)

def weather_prefix(weather, sowing, length=HORIZON):
    dates = pd.date_range(sowing, periods=length)
    frame = weather.copy()
    frame['date'] = pd.to_datetime(frame.date)
    if frame.date.duplicated().any():
        raise ValueError('Complete unique daily weather required')
    values = frame.set_index('date').reindex(dates)[['tmean_c', 'tmax_c', 'tmin_c', 'precip_mm']].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('complete daily weather prefix required')
    if np.any(values[:, 2] > values[:, 0]) or np.any(values[:, 0] > values[:, 1]):
        raise ValueError('Invalid temperature order')
    if np.any(values[:, 3] < 0):
        raise ValueError('Negative precipitation')
    return values

def load_sources(root=HERE):
    base = Path(root) / 'inputs'
    weather = pd.read_csv(base / 'weather.csv', sep=';', decimal=',').rename(columns={'Date': 'date', 'Temperature.Air.Mean_C': 'tmean_c', 'Temperature.Air.Max_C': 'tmax_c', 'Temperature.Air.Min_C': 'tmin_c', 'Precipitation_mm_d': 'precip_mm'})
    return dict(plots=pd.read_csv(base / 'plots.csv'), green=pd.read_csv(base / 'canopy_observations.csv'), raw_green=pd.read_csv(base / 'gb_long.csv'), old_features=pd.read_csv(base / 'old_features.csv'), weather=weather[['date', 'tmean_c', 'tmax_c', 'tmin_c', 'precip_mm']], coordinates=pd.read_csv(base / 'coordinates.csv'), units=pd.read_excel(base / 'Unit.xlsx', header=None))

def prepare(root=HERE):
    source = load_sources(root)
    plots = source['plots'].loc[lambda x: x.treatment.eq('HN_WF') & x.year.between(2015, 2018)]
    coordinates = source['coordinates'].loc[lambda x: x.Location.eq('HAN')].iloc[0]
    registry = sowing_registry(plots, source['raw_green'], float(coordinates.Lat), float(coordinates.Long))
    frame = measurement_cohort(source['green'], source['plots'], source['old_features'])
    for row in registry.itertuples():
        weather_prefix(source['weather'], row.sowing_date)
        offsets = (pd.to_datetime(frame.loc[frame.year.eq(row.year), 'date']) - pd.Timestamp(row.sowing_date)).dt.days
        if not offsets.between(0, HORIZON - 1).all():
            raise ValueError('Assessment outside fixed sowing horizon')
    unit = source['units'].loc[source['units'][1].eq('BBCH59'), 0]
    if len(unit) != 1:
        raise ValueError('Unique literal source stage definition required')
    return dict(measurements=frame, labels=heading_labels(plots), registry=registry, weather=source['weather'], source59_definition=str(unit.iloc[0]))

def parent_predictions(registry, weather, checkpoint=HERE / 'inputs/parent.pt'):
    """Inputs deliberately exclude heading, canopy and terminal-date outcomes."""
    import torch
    spec = importlib.util.spec_from_file_location('frozen_wheat_parent', HERE / 'inputs/frozen_parent.py')
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    package = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if package['stages'] != [10, 31, 51, 87] or package['horizon_days'] != HORIZON or len(package['members']) != 3:
        raise ValueError('Unexpected frozen ensemble contract')
    accum = []
    for member in package['members']:
        features, progress, increments = ([], [], [])
        for row in registry.itertuples():
            dates = pd.date_range(row.sowing_date, periods=int(row.horizon_days))
            x, p, i = parent.build_features(weather_prefix(weather, row.sowing_date, len(dates)), dates, row.latitude, member['metadata']['gates'])
            features.append(x)
            progress.append(p)
            increments.append(i)
        model = parent.WheatPhenology(member['metadata'], member['mode'])
        model.load_state_dict(member['state_dict'])
        cdf = parent.predict_batches(model, np.asarray(features), np.asarray(progress), np.asarray(increments), np.arange(len(registry)))
        accum.append(cdf.astype(float))
    return np.mean(accum, axis=0)
