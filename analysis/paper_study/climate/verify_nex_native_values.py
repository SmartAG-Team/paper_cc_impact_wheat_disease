"""Compare native annual NCSS samples with committed quarterly-derived data."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from io import BytesIO
from pathlib import Path
import argparse,hashlib,json,time,xml.etree.ElementTree as ET
import numpy as np,pandas as pd,requests,xarray as xr

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'analysis/paper_study/climate'
DATA=ROOT/'data/paper_study/climate'
MODELS=['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']
FIELDS={'tas':'tmean_c','tasmax':'tmax_c','pr':'precipitation_mm','hurs':'rh_mean_pct'}
BASE='https://ds.nccs.nasa.gov/thredds/ncss/grid/'

def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def jsonable(value):
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value

def check(job,cells):
    model,scenario,year,variable=job;field=FIELDS[variable]
    catalog=DATA/'catalogs'/f'{model}_{scenario}_r1i1p1f1_{variable}.xml'
    name=f'{variable}_day_{model}_{scenario}_r1i1p1f1_gn_{year}_v2.0.nc'
    matches=[r.attrib['urlPath'] for r in ET.fromstring(catalog.read_bytes()).iter() if r.attrib.get('urlPath','').endswith('/'+name)]
    assert len(matches)==1;source=matches[0]
    query={'var':variable,'north':51.125,'south':50.125,'west':10.125,'east':11.125,
           'time_start':f'{year}-01-01T12:00:00Z','time_end':f'{year}-12-31T12:00:00Z',
           'horizStride':1,'accept':'netcdf4','addLatLon':'true'}
    path=DATA/'independent_native_checks'/model/scenario/f'{variable}_{year}.nc';receipt_path=path.with_suffix('.json')
    cached=False
    if path.exists() and receipt_path.exists():
        previous=json.loads(receipt_path.read_text());cached=previous['query']==query and previous['sha256']==sha_bytes(path.read_bytes())
    retries=[]
    if not cached:
        for attempt in range(3):
            try:
                response=requests.get(BASE+source,params=query,timeout=(30,180));response.raise_for_status();content=response.content
                with xr.open_dataset(BytesIO(content),engine='h5netcdf') as test:assert variable in test
                path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp.nc');temporary.write_bytes(content);temporary.replace(path)
                receipt={'native_source_path':source,'query':query,'bytes':len(content),'sha256':sha_bytes(content),'retrieved_utc':utc()}
                receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');break
            except (requests.RequestException,OSError) as exc:
                retries.append({'attempt':attempt+1,'error_type':type(exc).__name__})
                if attempt==2:raise
                time.sleep(3*(attempt+1))
    expected=pd.date_range(f'{year}-01-01',f'{year}-12-31').strftime('%Y-%m-%d').tolist()
    with xr.open_dataset(path,engine='h5netcdf') as ds:
        assert ds.attrs['version']=='2.0' and ds.attrs['cmip6_source_id']==model and ds.attrs['scenario']==scenario
        assert ds.attrs['variant_label']=='r1i1p1f1'
        assert pd.DatetimeIndex(ds.time.values).strftime('%Y-%m-%d').tolist()==expected
        assert ds.time.encoding.get('calendar')=='proleptic_gregorian'
        lat=ds.lat.to_numpy();lon=(ds.lon.to_numpy()+180)%360-180
        assert np.array_equal(np.sort(lat),np.arange(50.125,51.126,.25))
        assert np.array_equal(np.sort(lon),np.arange(10.125,11.126,.25))
        row={round(float(v),6):i for i,v in enumerate(lat)};col={round(float(v),6):i for i,v in enumerate(lon)}
        values=ds[variable].to_numpy()[:,[row[round(v,6)] for v in cells.latitude],[col[round(v,6)] for v in cells.longitude]]
        assert np.isfinite(values).all();native=values.astype('float64')
        units=ds[variable].attrs['units']
        if variable.startswith('tas'):assert units=='K';converted=native-273.15
        elif variable=='pr':assert units in ['kg m-2 s-1','kg/m2/s','kg m**-2 s**-1'];converted=np.maximum(native*86400,0)
        else:assert units=='%';converted=np.clip(native,0,100)
        native_attrs=jsonable(ds.attrs)
    annual=DATA/'nasa_v2/variables'/model/scenario/variable/f'{year}.parquet'
    columns=['date','cell_id',field]+(['rh_mean_pct_raw'] if variable=='hurs' else [])
    frame=pd.read_parquet(annual,columns=columns,filters=[('cell_id','in',cells.cell_id.tolist())])
    assert len(frame)==len(expected)*len(cells) and not frame.duplicated(['date','cell_id']).any()
    actual=frame.pivot(index='date',columns='cell_id',values=field).reindex(index=expected,columns=cells.cell_id).to_numpy()
    difference=np.abs(actual-converted);assert np.allclose(actual,converted,rtol=1e-6,atol=.0001)
    raw_difference=None
    if variable=='hurs':
        raw=frame.pivot(index='date',columns='cell_id',values='rh_mean_pct_raw').reindex(index=expected,columns=cells.cell_id).to_numpy()
        raw_difference=float(np.abs(raw-native).max());assert raw_difference<.0001
    adjacent_days=['03-31','04-01','06-30','07-01','09-30','10-01']+(['02-29'] if len(expected)==366 else [])
    boundary_rows=sum(d.endswith(tuple(adjacent_days)) for d in expected)*len(cells)
    source_meta=json.loads(annual.with_suffix('.json').read_text())
    assert native_attrs['cmip6_license']==source_meta['native_file_license']
    receipt=json.loads(receipt_path.read_text());receipt['native_global_attributes']=native_attrs;receipt['native_units']=units;receipt['native_calendar']='proleptic_gregorian'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return {'model':model,'scenario':scenario,'year':year,'variable':variable,'field':field,'native_source_path':source,
            'cells':len(cells),'days':len(expected),'compared_cell_days':len(frame),'quarter_boundary_and_leap_cell_days':boundary_rows,
            'max_abs_difference':float(difference.max()),'raw_humidity_max_abs_difference':raw_difference,
            'absolute_tolerance':.0001,'relative_tolerance':1e-6,'source_license':native_attrs['cmip6_license'],
            'independent_response':str(path.relative_to(ROOT)),'independent_response_sha256':receipt['sha256'],'retry_events':retries,'passed':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    registry=DATA.parent/'wheat_area/europe_wheat_cells_025.parquet';cells=pd.read_parquet(registry)
    cells=cells.loc[cells.latitude.between(50.125,51.125)&cells.longitude.between(10.125,11.125)].sort_values(['row','col']).reset_index(drop=True)
    assert len(cells)>0
    jobs=[(m,s,y,v) for m in MODELS for s,y in [('historical',1992),('ssp126',2080),('ssp245',2080),('ssp585',2080)] for v in FIELDS]
    results=[];errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(check,j,cells):j for j in jobs}
        for future in as_completed(futures):
            try:result=future.result();results.append(result);print(json.dumps({'completed':len(results),'total':len(jobs),**result}),flush=True)
            except Exception as exc:errors.append({'job':futures[future],'error_type':type(exc).__name__,'error':str(exc)});print(json.dumps(errors[-1]),flush=True)
    receipt={'checked_utc':utc(),'method':'Independent native annual NCSS response compared with archived full-Europe quarterly-derived variables; separate coordinate lookup and float64 unit conversion',
             'expected_checks':48,'passed_checks':len(results),'failed_checks':len(errors),'target_cells':len(cells),
             'compared_cell_days':sum(r['compared_cell_days'] for r in results),'quarter_boundary_and_leap_cell_days':sum(r['quarter_boundary_and_leap_cell_days'] for r in results),
             'results':results,'errors':errors,'status':'passed' if len(results)==48 and not errors else 'failed'}
    (OUT/'nex_independent_native_value_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if errors or len(results)!=48:raise SystemExit(1)

if __name__=='__main__':main()
