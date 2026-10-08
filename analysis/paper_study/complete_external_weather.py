"""Acquire the additional assessment-coordinate season found by input QA."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import gather_era5_weather as provider
from analysis.paper_study.extend_external_weather import sha


def main():
    dest = ROOT / 'data/paper_study/field_weather_completed'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'raw').mkdir(exist_ok=True)
    source = ROOT / 'data/paper_study/field_weather_extension/daily_weather_with_extension.parquet'
    source_hash = sha(source)
    provider.DEST = dest
    request = dict(location_id='era5loc_9ec703701638', season_year=2018,
                   requested_latitude=55.3, requested_longitude=11.4,
                   window_start='2017-08-01', window_end='2018-09-30')
    _, provenance, hourly = provider.retrieve(requests.Session(), request, 3.)
    for column in ['precipitation_mm', 'rain_mm']:
        hourly[column] = hourly[column].shift(-1)
    hourly = hourly.loc[hourly.time_utc.dt.tz_localize(None) < pd.Timestamp(request['window_end'])]
    additional = provider.daily_from_hourly(hourly)
    if not additional.hour_count.eq(24).all() or additional.isna().any().any():
        raise ValueError('Additional forcing is incomplete.')
    original = pd.read_parquet(source)
    overlap = original.merge(additional, on=['location_id', 'date'], suffixes=('_original', '_additional'))
    for column in additional.select_dtypes(include='number').columns:
        if not np.allclose(overlap[column+'_original'], overlap[column+'_additional'], rtol=0, atol=1e-10):
            raise ValueError(f'Overlapping forcing changed: {column}')
    merged = pd.concat([original, additional], ignore_index=True).drop_duplicates(['location_id', 'date'])
    merged = merged.sort_values(['location_id', 'date'])
    additional.to_parquet(dest / 'additional_season.parquet', index=False, compression='zstd')
    output = dest / 'daily_weather.parquet'
    merged.to_parquet(output, index=False, compression='zstd')
    if sha(source) != source_hash:
        raise ValueError('Earlier archive changed.')
    receipt = dict(source_weather_sha256=source_hash, merged_weather_sha256=sha(output),
                   request=request, response_sha256=provenance['response_sha256'],
                   additional_rows=len(additional), overlap_rows=len(overlap),
                   new_rows=len(merged)-len(original), previous_weather_modified=False,
                   reason='assessment coordinate season absent from original acquisition registry',
                   affected_source_unit='2018-176', source_coordinate_modified=False)
    (dest / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
