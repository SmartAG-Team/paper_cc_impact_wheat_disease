"""DWD historical hourly station consistency checks for German field forcing.

Station matching uses coordinates and record dates, not disease outcomes.
R1 is total precipitation; it is compared with total precipitation, not liquid
rain. Temperature/RH summaries require all 24 valid UTC hourly records.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import requests

from .evaluation import distances_km
from .inputs import WEATHER, INPUT, sha

ROOT = Path(__file__).resolve().parents[2]
BASE = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/'
DIRECTORIES = {'TU': 'air_temperature', 'RR': 'precipitation'}


def parse_inventory(text):
    rows = []
    for line in text.splitlines()[2:]:
        a = line.split(maxsplit=6)
        if len(a) == 7:
            rows.append(dict(station_id=a[0].zfill(5), start=a[1], end=a[2], elevation_m=float(a[3]),
                latitude=float(a[4]), longitude=float(a[5]), station_name=a[6].strip()))
    return pd.DataFrame(rows)


def match_stations(sites, inventory, start='20130101', end='20191231', max_km=50):
    candidates = inventory.loc[inventory.start.le(start) & inventory.end.ge(end)].reset_index(drop=True)
    if candidates.empty:
        raise ValueError('No stations with declared temporal coverage')
    distance = distances_km(sites[['latitude', 'longitude']], candidates[['latitude', 'longitude']])
    rows = []
    for index, site in enumerate(sites.itertuples(index=False)):
        position = int(distance[index].argmin())
        station = candidates.iloc[position]
        rows.append(dict(site_id=site.site_id, station_id=station.station_id,
            distance_km=float(distance[index, position]), within_distance=bool(distance[index, position] <= max_km),
            field_latitude=site.latitude, field_longitude=site.longitude,
            station_latitude=station.latitude, station_longitude=station.longitude,
            station_name=station.station_name))
    return pd.DataFrame(rows)


def parse_hourly(content, variable):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.startswith('produkt') and name.endswith('.txt')]
        if len(names) != 1:
            raise ValueError('One DWD product table required')
        frame = pd.read_csv(archive.open(names[0]), sep=';', skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    frame['time'] = pd.to_datetime(frame.MESS_DATUM.astype(str), format='%Y%m%d%H', errors='coerce')
    if frame.time.isna().any() or frame.time.duplicated().any():
        raise ValueError('Invalid or duplicated UTC hourly timestamps')
    columns = ['TT_TU', 'RF_TU'] if variable == 'TU' else ['R1']
    for name in columns:
        values = pd.to_numeric(frame[name], errors='coerce')
        valid = values.between(-80, 60) if name == 'TT_TU' else values.between(0, 100) if name == 'RF_TU' else values.between(0, 500)
        frame[name] = values.where(valid)
    # R1 is the preceding-hour accumulation ending at MESS_DATUM (SYNOP).
    # The 00UTC report therefore completes the previous UTC calendar day.
    frame['date'] = (frame.time-pd.Timedelta(hours=1) if variable == 'RR' else frame.time).dt.normalize()
    return frame


def daily_aggregate(hourly, variable):
    groups = hourly.groupby('date')
    if variable == 'TU':
        result = groups.agg(tmean_c=('TT_TU', 'mean'), tmax_c=('TT_TU', 'max'),
            tmin_c=('TT_TU', 'min'), rh_mean_pct=('RF_TU', 'mean'),
            temperature_hours=('TT_TU', 'count'), humidity_hours=('RF_TU', 'count'))
        humid = hourly.assign(humid=hourly.RF_TU.ge(90).where(hourly.RF_TU.notna()))
        result['rh_hours_ge_90pct'] = pd.to_numeric(humid.groupby('date').humid.sum(min_count=24))
        result.loc[result.temperature_hours.ne(24), ['tmean_c', 'tmax_c', 'tmin_c']] = np.nan
        result.loc[result.humidity_hours.ne(24), ['rh_mean_pct', 'rh_hours_ge_90pct']] = np.nan
    else:
        result = groups.agg(precipitation_mm=('R1', lambda x: x.sum(min_count=24)),
                            rain_hours_available=('R1', 'count'))
        wet = hourly.assign(wet=hourly.R1.gt(.1).where(hourly.R1.notna()))
        result['station_precip_hours_gt_0_1mm'] = pd.to_numeric(wet.groupby('date').wet.sum(min_count=24))
        if 'WRTR' in hourly:
            result['solid_precipitation_observed'] = hourly.assign(
                solid=hourly.WRTR.isin([7,8])).groupby('date').solid.max()
        result.loc[result.rain_hours_available.ne(24), 'precipitation_mm'] = np.nan
        result.loc[result.rain_hours_available.ne(24), 'station_precip_hours_gt_0_1mm'] = np.nan
    return result.reset_index()


def fetch(url, path):
    path = Path(path)
    if not path.exists():
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        temporary = path.with_suffix(path.suffix+'.partial')
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return dict(url=url, path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def pair_metrics(pairs):
    rows = []
    for site, part in pairs.groupby('site_id'):
        for name in ['tmean_c', 'tmax_c', 'tmin_c', 'rh_mean_pct', 'rh_hours_ge_90pct', 'precipitation_mm']:
            station, model = name+'_station', name+'_forcing'
            if station not in part:
                continue
            numeric = part[[station, model]].apply(pd.to_numeric, errors='raise')
            use = numeric.loc[np.isfinite(numeric[station]) & np.isfinite(numeric[model])]
            if not len(use):
                continue
            e = use[model]-use[station]
            correlation = use[[station, model]].corr().iloc[0, 1] if len(use) > 2 and use[station].std() > 0 else np.nan
            row = dict(site_id=site, variable=name, n=len(use), bias=float(e.mean()),
                rmse=float(np.sqrt(np.mean(e**2))), mae=float(abs(e).mean()), correlation=correlation)
            if name == 'precipitation_mm':
                row.update(wet_day_agreement=float((use[station].ge(1) == use[model].ge(1)).mean()),
                           station_total=float(use[station].sum()), forcing_total=float(use[model].sum()))
            rows.append(row)
    return pd.DataFrame(rows)


def join_station_daily(forcing, station):
    names = ['tmean_c', 'tmax_c', 'tmin_c', 'rh_mean_pct', 'rh_hours_ge_90pct', 'precipitation_mm']
    first = forcing.rename(columns={name:name+'_forcing' for name in names if name in forcing})
    second = station.rename(columns={name:name+'_station' for name in names if name in station})
    return first.merge(second, on='date', how='left', validate='one_to_one')


def run(root=ROOT, output=None, cache=None, reuse_daily=False):
    root = Path(root)
    output = Path(output or root/'outputs/weather_validation_20261022/dwd')
    cache = Path(cache or root/'data/external_weather_validation_20261022/dwd')
    output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    fields = pd.read_csv(root/INPUT/'fields.csv')
    sites = fields.loc[fields.country.eq('GERMANY'), ['site_id', 'latitude', 'longitude']].drop_duplicates()
    forcing = pd.read_parquet(root/WEATHER); forcing['date'] = pd.to_datetime(forcing.date)
    receipts, all_matches, daily_tables, failures = [], [], [], []
    for variable, directory in DIRECTORIES.items():
        daily_path = output/f'{variable}_station_daily.parquet'
        cached_daily = pd.read_parquet(daily_path) if reuse_daily and daily_path.exists() else pd.DataFrame()
        base = BASE+directory+'/historical/'
        index_path, station_path = cache/f'{variable}_index.html', cache/f'{variable}_stations.txt'
        receipts += [fetch(base, index_path), fetch(base+f'{variable}_Stundenwerte_Beschreibung_Stationen.txt', station_path)]
        inventory = parse_inventory(station_path.read_bytes().decode('latin1'))
        links = re.findall(r'href="([^"]+\.zip)"', index_path.read_text())
        available = {name.split('_')[-4] for name in links
                     if name.split('_')[-3] <= '20130101' and name.split('_')[-2] >= '20191231'}
        inventory.loc[~inventory.station_id.isin(available)].assign(
            reason='no downloadable full-period historical archive').to_csv(
                output/f'{variable}_catalog_without_archive.csv', index=False)
        inventory = inventory.loc[inventory.station_id.isin(available)]
        matches = match_stations(sites, inventory)
        matches['variable_group'] = variable; all_matches.append(matches)
        tasks = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            for station in matches.loc[matches.within_distance, 'station_id'].unique():
                files = [name for name in links if f'_{station}_' in name
                         and name.split('_')[-3] <= '20130101' and name.split('_')[-2] >= '20191231']
                if len(files) != 1:
                    failures.append(dict(variable=variable, station=station, reason='No unique full-period historical archive'))
                    continue
                name = files[0]; path = cache/name
                tasks[executor.submit(fetch, base+name, path)] = (station, path)
            for future in as_completed(tasks):
                station, path = tasks[future]
                try:
                    receipts.append(future.result())
                    table = cached_daily.loc[cached_daily.station_id.eq(station)].copy() if len(cached_daily) else pd.DataFrame()
                    if not len(table) or (variable == 'RR' and 'solid_precipitation_observed' not in table):
                        hourly = parse_hourly(path.read_bytes(), variable)
                        hourly = hourly.loc[hourly.date.between('2013-01-01', '2019-12-31')]
                        table = daily_aggregate(hourly, variable); table['station_id'] = station
                    daily_tables.append((variable, table))
                    print(variable, station, len(table), flush=True)
                except Exception as error:
                    failures.append(dict(variable=variable, station=station, reason=str(error)))
    matches = pd.concat(all_matches, ignore_index=True)
    matches.to_csv(output/'station_matches.csv', index=False)
    paired = None
    for variable in DIRECTORIES:
        tables = [table for kind, table in daily_tables if kind == variable]
        if not tables:
            continue
        daily = pd.concat(tables, ignore_index=True)
        daily.to_parquet(output/f'{variable}_station_daily.parquet', index=False)
        mapping = matches.loc[matches.variable_group.eq(variable) & matches.within_distance]
        rows = []
        for m in mapping.itertuples(index=False):
            field_weather = forcing.loc[forcing.location_id.eq(m.site_id)]
            station_weather = daily.loc[daily.station_id.eq(m.station_id)].drop(columns='station_id')
            joined = join_station_daily(field_weather, station_weather)
            joined['site_id'] = m.site_id; joined['station_id_'+variable] = m.station_id
            joined['distance_km_'+variable] = m.distance_km
            rows.append(joined)
        block = pd.concat(rows, ignore_index=True)
        if variable == 'TU':
            paired = block
        elif paired is not None:
            use = ['site_id', 'date', 'precipitation_mm_station', 'station_id_RR', 'distance_km_RR', 'rain_hours_available', 'station_precip_hours_gt_0_1mm','solid_precipitation_observed']
            paired = paired.merge(block[use], on=['site_id','date'], how='outer', validate='one_to_one')
        else:
            paired = block
    if paired is None:
        raise ValueError('No DWD station pairs retrieved')
    paired.to_parquet(output/'paired_daily.parquet', index=False)
    metrics = pair_metrics(paired); metrics.to_csv(output/'station_metrics.csv', index=False)
    (output/'receipt.json').write_text(json.dumps(dict(
        retrieved_utc=datetime.now(timezone.utc).isoformat(), source='DWD CDC historical hourly TU and RR',
        matching='nearest catalog station covering 2013–2019, at most50km; no disease outcomes used',
        daily_policy='all24 valid UTC hourly values per variable; R1 ending01..next00UTC belongs to previous00..24UTC day; missing stays missing',
        sites_requested=len(sites), sites_paired=paired.site_id.nunique(),
        observations_may_overlap_reanalysis_assimilation=True,
        rainfall='R1 total precipitation, compared to forcing precipitation_mm; never substituted for liquid rain',
        station_relocation_metadata='retained in each downloaded archive; nearest matching uses catalog coordinates',
        receipts=receipts, failures=failures), indent=2)+'\n')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reuse-daily', action='store_true', help='Reuse previously parsed local daily tables')
    args = parser.parse_args()
    print(run(args.root, reuse_daily=args.reuse_daily).groupby('variable')[['n','bias','rmse']].mean().to_string())
