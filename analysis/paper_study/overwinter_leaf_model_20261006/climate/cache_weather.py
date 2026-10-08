"""Immutable raw annual 62-cell weather extraction; no model/calibration imports."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,uuid
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
MODELS=['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']
SCENARIOS=['ssp126','ssp245','ssp585']
YEARS=list(range(1991,2021))+list(range(2031,2061))+list(range(2071,2101))
FIELDS=['tmean_c','tmax_c','rh_mean_pct','precipitation_mm']
DRAW_PATH=ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def frozen_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def source_path(model,scenario,year):
    return ROOT/'data/paper_study/climate/nasa_v2/daily'/model/('historical' if year<=2014 else scenario)/f'{year}.parquet'


def registered_sources():
    paths=sorted({source_path(model,scenario,y) for model in MODELS for scenario in SCENARIOS for year in YEARS for y in [year-1,year]})
    result=[]
    for path in paths:
        receipt=path.with_suffix('.json')
        if not path.exists() or not receipt.exists():raise FileNotFoundError(path)
        metadata=json.loads(receipt.read_text())
        result.append(dict(path=str(path.relative_to(ROOT)),parquet_sha256=metadata['parquet_sha256'],
            source_receipt_sha256=sha(receipt),parquet_bytes=path.stat().st_size))
    return result


def cache_one(record,ids):
    path=ROOT/record['path'];relative=path.relative_to(ROOT/'data/paper_study/climate/nasa_v2/daily')
    destination=HERE/'annual_weather_cache'/relative.with_suffix('.npz');receipt=destination.with_suffix('.json')
    if destination.exists() or receipt.exists():
        if not destination.exists() or not receipt.exists():raise ValueError('Partial raw weather cache.')
        prior=json.loads(receipt.read_text());assert prior['source']==record
        assert sha(destination)==prior['cache_sha256']
        return prior
    assert sha(path.with_suffix('.json'))==record['source_receipt_sha256']
    if sha(path)!=record['parquet_sha256']:raise ValueError(f'Annual source checksum changed: {path}')
    table=pq.read_table(path,columns=['date','cell_id',*FIELDS])
    table=table.filter(pc.is_in(table['cell_id'],value_set=pa.array(ids)))
    frame=table.to_pandas();frame['date']=pd.to_datetime(frame.date)
    year=int(path.stem);dates=pd.date_range(f'{year}-01-01',f'{year}-12-31')
    if frame.duplicated(['cell_id','date']).any():raise ValueError('Duplicate source cell/date weather.')
    if not set(frame.cell_id)==set(ids):raise ValueError('A sampled cell is missing from the annual weather source.')
    lookup={cell:index for index,cell in enumerate(ids)}
    cells=frame.cell_id.map(lookup).to_numpy(int);days=dates.get_indexer(frame.date)
    if (days<0).any():raise ValueError('A weather row lies outside the declared source year.')
    arrays={field:np.full((len(ids),len(dates)),np.nan,np.float32) for field in FIELDS}
    for field in FIELDS:arrays[field][cells,days]=frame[field].to_numpy(np.float32)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('wb') as stream:np.savez_compressed(stream,dates=dates.to_numpy(dtype='datetime64[D]'),cell_ids=np.asarray(ids),**arrays)
        os.link(temporary,destination)
    finally:temporary.unlink(missing_ok=True)
    report=dict(source=record,cache_sha256=sha(destination),cells=len(ids),days=len(dates),
        source_rows_selected=len(frame),expected_rows=len(ids)*len(dates),
        missing_values_by_field={field:int((~np.isfinite(array)).sum()) for field,array in arrays.items()},
        data_are_raw_unaligned=True,model_predictions_computed=False)
    frozen_json(receipt,report)
    return report


def cache_model(model,records,ids):
    reports=[]
    for number,record in enumerate(records):
        reports.append(cache_one(record,ids))
        if (number+1)%25==0:print(f'raw cache {model}: {number+1}/{len(records)} annual sources',flush=True)
    return reports


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    draws=pd.read_csv(DRAW_PATH);ids=draws.cell_id.drop_duplicates().tolist()
    sample=ROOT/'data/paper_study/regional_parameter_uncertainty/sampling_receipt.json'
    sampling=json.loads(sample.read_text());assert len(draws)==64 and len(ids)==62 and draws.stratum.nunique()==16
    assert draws.groupby('stratum').size().eq(4).all()
    assert sha(DRAW_PATH)==sampling['draw_sha256']
    records=registered_sources()
    configuration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),purpose='raw annual sample weather cache only; no numerical model results',
        spatial_draw_sha256=sha(DRAW_PATH),sampling_receipt_sha256=sha(sample),cells=len(ids),draws=64,
        eligible_harvested_ha=sampling['eligible_harvested_ha'],annual_source_count=len(records),sources=records,
        fields=FIELDS,worker_count=3,source_code_sha256=sha(Path(__file__)))
    contract=HERE/'weather_cache_configuration_before_extraction.json'
    if contract.exists():
        previous=json.loads(contract.read_text());configuration['registered_utc']=previous['registered_utc'];assert configuration==previous
    else:frozen_json(contract,configuration)
    reports=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(cache_model,model,[record for record in records if f'/{model}/' in record['path']],ids) for model in MODELS]
        for future in as_completed(futures):reports.extend(future.result())
    receipt=dict(status='complete',raw_annual_cache_count=len(reports),spatial_cells=62,spatial_draws=64,
        numerical_model_results_computed=False,cache_configuration_sha256=sha(contract),
        missing_values_total=sum(sum(record['missing_values_by_field'].values()) for record in reports),
        output_sha256={str(path.relative_to(HERE)):sha(path) for path in (HERE/'annual_weather_cache').rglob('*') if path.is_file()})
    target=HERE/'weather_cache_receipt.json'
    if target.exists():assert json.loads(target.read_text())==receipt
    else:frozen_json(target,receipt)
    print(json.dumps({key:value for key,value in receipt.items() if key!='output_sha256'}),flush=True)


if __name__=='__main__':main()
