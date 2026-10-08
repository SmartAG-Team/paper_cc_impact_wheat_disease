"""Audit actual climate files against required dates, fields and registry hash."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('climate_retrieval', HERE/'retrieve_climate.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def audit(path, expected_dates, registry_hash, expected_cell_ids, fields, deep):
    provenance = path.with_suffix('.json')
    if not path.exists() or not provenance.exists():
        return {'path': str(path.relative_to(C.ROOT)), 'status': 'missing'}
    meta = json.loads(provenance.read_text())
    result = {'path': str(path.relative_to(C.ROOT)), 'status': 'valid',
              'bytes': path.stat().st_size, 'days': len(expected_dates)}
    errors = []
    if C.sha256(path) != meta.get('parquet_sha256'):
        errors.append('checksum')
    if meta.get('registry_sha256') != registry_hash:
        errors.append('registry hash')
    quality = meta.get('quality', {})
    if quality.get('cells') != len(expected_cell_ids):
        errors.append('registry cell count')
    if quality.get('rows') != len(expected_dates) * len(expected_cell_ids):
        errors.append('daily cell row count')
    if quality.get('days') != len(expected_dates):
        errors.append('date count')
    if quality.get('date_min') != expected_dates[0] or quality.get('date_max') != expected_dates[-1]:
        errors.append('date endpoints')
    if not set(fields).issubset(pq.read_schema(path).names):
        errors.append('required fields')
    if any(quality.get('missing_values', {}).get(f, -1) != 0 for f in fields):
        errors.append('missing values')
    if deep and not errors:
        frame = pd.read_parquet(path)
        if list(sorted(frame.date.unique())) != expected_dates:
            errors.append('actual dates')
        if set(frame.cell_id.unique()) != expected_cell_ids:
            errors.append('actual registry cell IDs')
        C.quality(frame, fields)
    if errors:
        result['status'] = 'invalid'
        result['errors'] = errors
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep', action='store_true', help='Read full values and dates, beyond checksum/schema/manifest checks')
    parser.add_argument('--source', choices=['all','era5','nasa'], default='all')
    args = parser.parse_args()
    registry_hash = C.sha256(C.REGISTRY)
    expected_cell_ids = set(pd.read_parquet(C.REGISTRY, columns=['cell_id']).cell_id)
    results = {}
    if args.source in ['all','era5']:
        records = []
        for start in pd.date_range('1990-01-01','2020-12-01',freq='MS'):
            end = start + pd.offsets.MonthBegin()
            dates = pd.date_range(start, end-pd.Timedelta(days=1)).strftime('%Y-%m-%d').tolist()
            path = C.DATA/'era5_daily'/f'{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet'
            records.append(audit(path, dates, registry_hash, expected_cell_ids, C.FIELDS, args.deep))
        results['era5'] = records
    if args.source in ['all','nasa']:
        records = []
        for model, scenario, year in C.year_plan():
            dates = pd.date_range(f'{year}-01-01',f'{year}-12-31').strftime('%Y-%m-%d').tolist()
            path = C.DATA/'nasa_v2/daily'/model/scenario/f'{year}.parquet'
            records.append(audit(path, dates, registry_hash, expected_cell_ids, C.FIELDS, args.deep))
        results['nasa'] = records
    summary = {name: {'expected': len(records), 'valid': sum(r['status']=='valid' for r in records),
                      'invalid': sum(r['status']=='invalid' for r in records),
                      'missing': sum(r['status']=='missing' for r in records),
                      'bytes': sum(r.get('bytes',0) for r in records)} for name,records in results.items()}
    output = {'checked_utc': C.utc(), 'registry_sha256': registry_hash,
              'registry_cells': len(expected_cell_ids),
              'coverage_status': 'Complete' if all(not s['invalid'] and not s['missing'] for s in summary.values()) else 'Incomplete',
              'fields': C.FIELDS, 'deep': args.deep, 'summary': summary, 'files': results}
    filename = 'actual_coverage_validation.json' if args.source == 'all' else f'actual_coverage_validation_{args.source}.json'
    C.write_json(C.OUT/filename, output)
    print(json.dumps(summary), flush=True)
    if any(s['invalid'] or s['missing'] for s in summary.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
