"""Acquire ERA5 point forcing in a new archive without changing earlier results."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import gather_era5_weather as provider


def main():
    destination = ROOT/'data/paper_study/field_weather'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'raw').mkdir(exist_ok=True)
    provider.DEST = destination
    assessment = pd.read_csv(ROOT/'data/paper_study/observations/assessments.csv')
    basf = assessment.loc[assessment.dataset_id.eq('basf-wheat-diseases'),
        ['site_id', 'season_year', 'latitude', 'longitude', 'country']].drop_duplicates()
    basf = basf.rename(columns={'site_id': 'location_id'})
    external = pd.read_csv(ROOT/'data/paper_study/observations/corteva_weather_requests.csv')
    external = external[['location_id','season_year','latitude','longitude','country']].drop_duplicates()
    registry = pd.concat([basf, external], ignore_index=True).drop_duplicates(
        ['location_id', 'season_year']).sort_values(['location_id','season_year'])
    registry['requested_latitude'] = registry.latitude
    registry['requested_longitude'] = registry.longitude
    registry['window_start'] = (registry.season_year.astype(int)-1).astype(str)+'-08-01'
    registry['window_end'] = registry.season_year.astype(int).astype(str)+'-09-30'
    registry.to_csv(destination/'requests.csv', index=False)
    frames, inventory = [], []
    session = requests.Session()
    session.headers['User-Agent'] = 'NWAFU-public-meteorology-research/1.0'
    for i, row in enumerate(registry.to_dict('records'), 1):
        cache = destination/f"{row['location_id']}_{int(row['season_year'])}.parquet"
        try:
            if cache.exists():
                daily = pd.read_parquet(cache)
            else:
                payload, provenance, hourly = provider.retrieve(session, row, 3.)
                # Precipitation at time t is the total over the preceding hour.
                # D01...D+1 00 aligns accumulation with day D00...D23 UTC.
                hourly['precipitation_mm'] = hourly.precipitation_mm.shift(-1)
                hourly['rain_mm'] = hourly.rain_mm.shift(-1)
                final_date = pd.Timestamp(row['window_end'])
                hourly = hourly.loc[hourly.time_utc.dt.tz_localize(None) < final_date]
                daily = provider.daily_from_hourly(hourly)
                if not daily.hour_count.eq(24).all() or daily.isna().any().any():
                    raise ValueError('Daily weather coverage or values are incomplete.')
                daily.to_parquet(cache, index=False, compression='zstd')
            frames.append(daily)
            inventory.append(dict(location_id=row['location_id'],season_year=int(row['season_year']),
                status='success',rows=len(daily),first_date=daily.date.min(),last_date=daily.date.max(),
                sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
        except Exception as exc:
            inventory.append(dict(location_id=row['location_id'],season_year=int(row['season_year']),
                status='failed',error=str(exc)))
        pd.DataFrame(inventory).to_csv(destination/'retrieval_inventory.csv', index=False)
        print(f"{i}/{len(registry)} {row['location_id']} {row['season_year']} {inventory[-1]['status']}",flush=True)
    all_daily = pd.concat(frames, ignore_index=True)
    conflicts = all_daily.groupby(['location_id','date']).tmean_c.nunique().gt(1)
    if conflicts.any():
        raise ValueError('Overlapping weather requests disagree.')
    all_daily = all_daily.drop_duplicates(['location_id','date']).sort_values(['location_id','date'])
    all_daily.to_parquet(destination/'daily_weather.parquet',index=False,compression='zstd')
    receipt = dict(requests=len(registry),successful=sum(x['status']=='success' for x in inventory),
        failed=sum(x['status']=='failed' for x in inventory),unique_daily_rows=len(all_daily),
        precipitation_day='Hourly totals ending D01UTC through D+1 00UTC',
        temperature_humidity_day='24 hourly instantaneous values D00UTC through D23UTC',
        earlier_data_modified=False,observed_sowing_inferred=False,
        source_model='ERA5',source_provider=provider.API)
    (destination/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
