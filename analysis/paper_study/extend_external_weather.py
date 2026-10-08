"""Preserve a late source assessment and acquire its missing ERA5 forcing."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import gather_era5_weather as provider


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dest = ROOT / 'data/paper_study/field_weather_extension'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'raw').mkdir(exist_ok=True)
    source = ROOT / 'data/paper_study/field_weather/daily_weather.parquet'
    original_hash = sha(source)
    provider.DEST = dest
    request = dict(location_id='era5loc_147e0282030a', season_year=2015,
                   requested_latitude=58.3, requested_longitude=14.8,
                   window_start='2015-09-29', window_end='2015-11-01')
    session = requests.Session()
    session.headers['User-Agent'] = 'NWAFU-public-meteorology-research/1.0'
    _, provenance, hourly = provider.retrieve(session, request, 3.)
    hourly['precipitation_mm'] = hourly.precipitation_mm.shift(-1)
    hourly['rain_mm'] = hourly.rain_mm.shift(-1)
    hourly = hourly.loc[hourly.time_utc.dt.tz_localize(None) < pd.Timestamp(request['window_end'])]
    extension = provider.daily_from_hourly(hourly)
    if not extension.hour_count.eq(24).all() or extension.isna().any().any():
        raise ValueError('Incomplete extension forcing.')
    original = pd.read_parquet(source)
    overlap = original.merge(extension, on=['location_id', 'date'], suffixes=('_original', '_extension'))
    for column in extension.select_dtypes(include='number').columns:
        if not np.allclose(overlap[column+'_original'], overlap[column+'_extension'], rtol=0, atol=1e-10):
            raise ValueError(f'Overlapping forcing changed: {column}')
    merged = pd.concat([original, extension], ignore_index=True).drop_duplicates(['location_id', 'date'])
    merged = merged.sort_values(['location_id', 'date'])
    extension.to_parquet(dest / 'extension_daily.parquet', index=False, compression='zstd')
    output = dest / 'daily_weather_with_extension.parquet'
    merged.to_parquet(output, index=False, compression='zstd')
    if sha(source) != original_hash:
        raise ValueError('Original weather archive changed.')
    receipt = dict(original_weather_sha256=original_hash, merged_weather_sha256=sha(output),
                   extension_rows=len(extension), overlap_rows=len(overlap), new_rows=len(merged)-len(original),
                   original_weather_modified=False, original_assessment_date_modified=False,
                   assessment=dict(source_unit='2015-58', original_date='2015-10-30',
                                   unusual_late_assessment=True),
                   request=request, response_sha256=provenance['response_sha256'],
                   precipitation_day='D01UTC through D+1 00UTC',
                   temperature_humidity_day='D00UTC through D23UTC')
    (dest / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
