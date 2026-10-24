"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/fip_canopy_transfer_20261011')
from pathlib import Path
import datetime as dt
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
HERE = evidence_dir('analysis/paper_study/fip_canopy_transfer_20261011')

def iso(value):
    return None if value is None or pd.isna(value) else pd.Timestamp(value).strftime('%Y-%m-%d')

def explode_records(records):
    plots = []
    visits = []
    seen = set()
    counts = dict(source_plots=len(records), source_visits=0, plots_without_series=0, missing_scores=0, exact_duplicate_visits=0, qualified_visits=0)
    for row in records:
        ident = row['plot_uid']
        if not ident or ident in seen:
            raise ValueError('Missing or duplicate plot identity')
        seen.add(ident)
        plot = {key: row.get(key) for key in ['plot_uid', 'yearsite_uid', 'harvest_year', 'genotype_id', 'source_split', 'source_fragment', 'heading_value']}
        plot['source_spatial_x'] = row.get('latitude')
        plot['source_spatial_y'] = row.get('longitude')
        for key in ['sowing_date', 'harvest_date', 'heading_date']:
            plot[key] = iso(row.get(key))
        plots.append(plot)
        dates = row.get('senescence_dates')
        values = row.get('senescence_values')
        if dates is None and values is None:
            counts['plots_without_series'] += 1
            continue
        if dates is None or values is None or len(dates) != len(values):
            raise ValueError('Senescence parallel-array length mismatch')
        if not dates:
            counts['plots_without_series'] += 1
        local = {}
        for index, (date, value) in enumerate(zip(dates, values, strict=True)):
            counts['source_visits'] += 1
            if value is None or pd.isna(value):
                counts['missing_scores'] += 1
                continue
            date = iso(date)
            if date is None:
                raise ValueError('Missing date for observed score')
            value = float(value)
            if not np.isfinite(value) or not 0 <= value <= 10:
                raise ValueError('Score outside documented 0-10 range')
            if date in local:
                if value != local[date]['raw_score']:
                    raise ValueError('Conflicting plot/date score')
                counts['exact_duplicate_visits'] += 1
                continue
            local[date] = dict(observation_id=ident + ':' + date, plot_uid=ident, year=int(row['harvest_year']), date=date, raw_score=value, source_array_index=index, source_split=row.get('source_split'), source_fragment=row.get('source_fragment'), method=row.get('senescence_method_name'), trait=row.get('senescence_trait_name'), unit=row.get('senescence_si_unit'))
        visits.extend(local.values())
    counts['qualified_visits'] = len(visits)
    assert counts['source_visits'] == counts['qualified_visits'] + counts['missing_scores'] + counts['exact_duplicate_visits']
    return (pd.DataFrame(plots), pd.DataFrame(visits).sort_values(['year', 'plot_uid', 'date']).reset_index(drop=True), counts)

def green_fraction_from_score(values, *, direction):
    if direction != 'increasing_senescence':
        raise ValueError('Unresolved score direction')
    raw = np.asarray(values, float)
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 10):
        raise ValueError('Finite 0-10 scores required')
    return 1 - raw / 10.0

def source_registry(plots, location):
    lat = float(location['latitude_degrees'])
    lon = float(location['longitude_degrees'])
    if not np.isfinite([lat, lon]).all() or not -90 <= lat <= 90 or (not -180 <= lon <= 180):
        raise ValueError('Geographic coordinates in degrees required')
    rows = []
    for year, part in plots.groupby('harvest_year', sort=True):
        dates = part.sowing_date.dropna().unique()
        if len(dates) != 1 or part.sowing_date.isna().any():
            raise ValueError('One complete source sowing date per year required')
        rows.append({'year': int(year), 'policy': 'actual_source_sowing', 'sowing_date': dates[0], 'latitude': lat, 'longitude': lon, 'horizon_days': 330})
    return pd.DataFrame(rows)

def daily_weather(row):
    outputs = []
    intervals = {}
    details = {}
    for variable in ['temperature_air_200cm', 'precipitation_200cm']:
        dates = row[variable + '_dates']
        times = row[variable + '_times']
        values = row[variable + '_values']
        if not len(dates) == len(times) == len(values):
            raise ValueError('Weather parallel-array length mismatch')
        timestamps = [pd.Timestamp(dt.datetime.combine(day, time), tz='UTC') for day, time in zip(dates, times, strict=True)]
        frame = pd.DataFrame({'timestamp': timestamps, 'value': values}).sort_values('timestamp')
        if frame.groupby('timestamp').value.nunique(dropna=False).gt(1).any():
            raise ValueError('Conflicting weather timestamp')
        frame = frame.drop_duplicates('timestamp')
        differences = frame.timestamp.diff().dt.total_seconds().dropna()
        if differences.empty or differences.le(0).any():
            raise ValueError('Distinct weather times required')
        interval = int(differences.mode().min())
        if interval <= 0 or interval > 86400 or 86400 % interval:
            raise ValueError('Unresolved daily weather sampling interval')
        intervals[variable] = interval
        numeric = pd.to_numeric(frame.value, errors='raise').to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError('Infinite weather value')
        if variable.startswith('precipitation') and np.any(numeric < 0):
            raise ValueError('Negative precipitation')
        frame['value'] = numeric
        frame['date'] = frame.timestamp.dt.strftime('%Y-%m-%d')
        grouped = frame.groupby('date').value
        daily = grouped.agg(['count', 'mean', 'min', 'max', 'sum'])
        complete = daily['count'] >= 0.9 * (86400 / interval)
        if variable.startswith('temperature'):
            result = daily[['mean', 'min', 'max']].rename(columns={'mean': 'tmean_c', 'min': 'tmin_c', 'max': 'tmax_c'})
            result.loc[~complete, :] = np.nan
            result['temperature_complete'] = complete
            result['temperature_slots'] = daily['count']
        else:
            result = daily[['sum']].rename(columns={'sum': 'precip_mm'})
            result.loc[~complete, 'precip_mm'] = np.nan
            result['precipitation_complete'] = complete
            result['precipitation_slots'] = daily['count']
        details[variable] = {'unique_timestamps': len(frame), 'daily_rows': len(daily), 'complete_days': int(complete.sum())}
        outputs.append(result)
    combined = outputs[0].join(outputs[1], how='outer').sort_index().reset_index()
    combined.insert(0, 'year', int(row['harvest_year']))
    return (combined, {'year': int(row['harvest_year']), 'modal_interval_seconds': intervals, 'source_counts': details})

def prepare():
    records = []
    for path in sorted((HERE / 'inputs').glob('*_traits.parquet')):
        records.extend(pq.read_table(path).to_pylist())
    plots, visits, counts = explode_records(records)
    if plots.plot_uid.duplicated().any():
        raise ValueError('Duplicated plot registry')
    plots.to_csv(HERE / 'results/plots.csv', index=False)
    visits.to_csv(HERE / 'results/raw_visits.csv', index=False)
    weather = []
    weather_receipts = []
    for path in sorted((HERE / 'inputs').glob('weather_source_*.parquet')):
        data = pq.read_table(path).to_pylist()
        if len(data) != 1:
            raise ValueError('One source weather row expected')
        daily, receipt = daily_weather(data[0])
        weather.append(daily)
        weather_receipts.append(receipt)
    pd.concat(weather, ignore_index=True).to_csv(HERE / 'results/weather_daily.csv', index=False)
    (HERE / 'results/weather_qualification.json').write_text(json.dumps(weather_receipts, indent=2) + '\n')
    merged = visits.merge(plots[['plot_uid', 'heading_date', 'harvest_date', 'genotype_id']], on='plot_uid', validate='many_to_one')
    profiles = []
    for year, part in plots.groupby('harvest_year'):
        observations = merged[merged.year.eq(year)]
        differences = observations.groupby('plot_uid').raw_score.agg(['first', 'last'])
        profiles.append({'year': int(year), 'plots': len(part), 'genotypes': part.genotype_id.nunique(), 'plots_with_heading': int(part.heading_date.notna().sum()), 'raw_visits': len(observations), 'plots_with_visits': observations.plot_uid.nunique(), 'scores_min': None if observations.empty else float(observations.raw_score.min()), 'scores_max': None if observations.empty else float(observations.raw_score.max()), 'series_increasing': int(differences['last'].gt(differences['first']).sum()), 'series_decreasing': int(differences['last'].lt(differences['first']).sum()), 'visits_with_heading': int(observations.heading_date.notna().sum()), 'visits_after_harvest': int((pd.to_datetime(observations.date) > pd.to_datetime(observations.harvest_date)).sum())})
    pd.DataFrame(profiles).to_csv(HERE / 'results/source_profile.csv', index=False)
    result = counts | {'years': sorted((int(v) for v in plots.harvest_year.unique())), 'sites_documented': 1, 'source_profiles': profiles, 'score_direction_confirmed': False, 'model_scored': False}
    (HERE / 'results/qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
