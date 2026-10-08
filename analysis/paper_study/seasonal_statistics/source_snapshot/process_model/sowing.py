"""Explicit management records available on or before each sowing date."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_records(records):
    required = ['PEP_ID', 'SOWING_DATE', 'SOWING_KNOWN_AT']
    if not set(required) <= set(records):
        raise ValueError('Sowing records require PEP_ID, SOWING_DATE, SOWING_KNOWN_AT')
    records = records[required].copy()
    records['SOWING_DATE'] = pd.to_datetime(records.SOWING_DATE)
    records['SOWING_KNOWN_AT'] = pd.to_datetime(records.SOWING_KNOWN_AT)
    if records.isna().any().any() or records.duplicated(['PEP_ID', 'SOWING_DATE']).any():
        raise ValueError('Sowing records must be complete and unique by station/date')
    if (records.SOWING_KNOWN_AT > records.SOWING_DATE).any():
        raise ValueError('This sowing protocol requires management records known by the sowing date')
    return records.sort_values(['PEP_ID', 'SOWING_DATE']).reset_index(drop=True)


def attach_sowing(frame, records):
    """As-of assignment: a later sowing record never changes earlier daily inputs."""
    records = validate_records(records)
    frame = frame.copy().reset_index(drop=True)
    frame['DATE'] = pd.to_datetime(frame.DATE)
    sowing = np.full(len(frame), np.datetime64('NaT'), dtype='datetime64[ns]')
    known = sowing.copy()
    by_station = {site: group for site, group in records.groupby('PEP_ID')}
    for site, positions in frame.groupby('PEP_ID').indices.items():
        if site not in by_station:
            continue
        events = by_station[site]
        dates = events.SOWING_DATE.to_numpy()
        match = np.searchsorted(dates, frame.DATE.iloc[positions].to_numpy(), side='right') - 1
        valid = match >= 0
        sowing[positions[valid]] = dates[match[valid]]
        known[positions[valid]] = events.SOWING_KNOWN_AT.to_numpy()[match[valid]]
    frame['SOWING_DATE'] = sowing
    frame['SOWING_KNOWN_AT'] = known
    return frame


def extract_known_sowing(source, output):
    """Export historical sowing observations under the user-confirmed availability contract.

This is an explicit one-time migration, not a stage-label lookup during inference.
"""
    source = Path(source).resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    events = []
    for chunk in pd.read_csv(source, usecols=['PEP_ID', 'DATE', 'CODE'], chunksize=200000):
        part = chunk.loc[chunk.CODE == 0, ['PEP_ID', 'DATE']].rename(columns={'DATE': 'SOWING_DATE'})
        part['SOWING_KNOWN_AT'] = part.SOWING_DATE
        events.append(part)
    records = validate_records(pd.concat(events, ignore_index=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output, index=False)
    metadata = dict(source=str(source), source_event='CODE == 0 sowing observation',
        availability_contract='User confirmed sowing date is known when sowing occurs',
        records=len(records), stations=int(records.PEP_ID.nunique()),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract_known_sowing(args.source, args.output)
