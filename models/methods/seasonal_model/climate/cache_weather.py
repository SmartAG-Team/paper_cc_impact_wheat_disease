"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/overwinter_leaf_model_20261006/climate')
from pathlib import Path
import hashlib, json, os, uuid
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
HERE = evidence_dir('analysis/paper_study/overwinter_leaf_model_20261006/climate')
ROOT = PROJECT_ROOT
MODELS = ['ACCESS-CM2', 'MPI-ESM1-2-HR', 'MRI-ESM2-0']
SCENARIOS = ['ssp126', 'ssp245', 'ssp585']
YEARS = list(range(1991, 2021)) + list(range(2031, 2061)) + list(range(2071, 2101))
FIELDS = ['tmean_c', 'tmax_c', 'rh_mean_pct', 'precipitation_mm']

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def frozen_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def source_path(model, scenario, year):
    return ROOT / 'data/paper_study/climate/nasa_v2/daily' / model / ('historical' if year <= 2014 else scenario) / f'{year}.parquet'

def registered_sources():
    paths = sorted({source_path(model, scenario, y) for model in MODELS for scenario in SCENARIOS for year in YEARS for y in [year - 1, year]})
    result = []
    for path in paths:
        receipt = path.with_suffix('.json')
        if not path.exists() or not receipt.exists():
            raise FileNotFoundError(path)
        metadata = json.loads(receipt.read_text())
        result.append(dict(path=str(path.relative_to(ROOT)), parquet_sha256=metadata['parquet_sha256'], source_receipt_sha256=sha(receipt), parquet_bytes=path.stat().st_size))
    return result

def cache_one(record, ids):
    path = ROOT / record['path']
    relative = path.relative_to(ROOT / 'data/paper_study/climate/nasa_v2/daily')
    destination = HERE / 'annual_weather_cache' / relative.with_suffix('.npz')
    receipt = destination.with_suffix('.json')
    if destination.exists() or receipt.exists():
        if not destination.exists() or not receipt.exists():
            raise ValueError('Partial raw weather cache.')
        prior = json.loads(receipt.read_text())
        assert prior['source'] == record
        assert sha(destination) == prior['cache_sha256']
        return prior
    assert sha(path.with_suffix('.json')) == record['source_receipt_sha256']
    if sha(path) != record['parquet_sha256']:
        raise ValueError(f'Annual source checksum changed: {path}')
    table = pq.read_table(path, columns=['date', 'cell_id', *FIELDS])
    table = table.filter(pc.is_in(table['cell_id'], value_set=pa.array(ids)))
    frame = table.to_pandas()
    frame['date'] = pd.to_datetime(frame.date)
    year = int(path.stem)
    dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    if frame.duplicated(['cell_id', 'date']).any():
        raise ValueError('Duplicate source cell/date weather.')
    if not set(frame.cell_id) == set(ids):
        raise ValueError('A sampled cell is missing from the annual weather source.')
    lookup = {cell: index for index, cell in enumerate(ids)}
    cells = frame.cell_id.map(lookup).to_numpy(int)
    days = dates.get_indexer(frame.date)
    if (days < 0).any():
        raise ValueError('A weather row lies outside the declared source year.')
    arrays = {field: np.full((len(ids), len(dates)), np.nan, np.float32) for field in FIELDS}
    for field in FIELDS:
        arrays[field][cells, days] = frame[field].to_numpy(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, dates=dates.to_numpy(dtype='datetime64[D]'), cell_ids=np.asarray(ids), **arrays)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report = dict(source=record, cache_sha256=sha(destination), cells=len(ids), days=len(dates), source_rows_selected=len(frame), expected_rows=len(ids) * len(dates), missing_values_by_field={field: int((~np.isfinite(array)).sum()) for field, array in arrays.items()}, data_are_raw_unaligned=True, model_predictions_computed=False)
    frozen_json(receipt, report)
    return report

def cache_model(model, records, ids):
    reports = []
    for number, record in enumerate(records):
        reports.append(cache_one(record, ids))
        if (number + 1) % 25 == 0:
            print(f'raw cache {model}: {number + 1}/{len(records)} annual sources', flush=True)
    return reports
