"""Separate, resumable native-CMIP6 meteorology candidate; never overwrites NEX.

Native Google Cloud Zarr chunks are cached by exact version/key, then daily
fields are bilinearly remapped to the frozen positive-wheat registry. Source
roles remain explicit: this archive is a candidate until selected downstream.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,threading,time
import cftime,gcsfs,numpy as np,pandas as pd
from numcodecs import get_codec

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('climate_common',HERE/'retrieve_climate.py')
C=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(C)
DATA=C.DATA/'cmip6_native_candidate';OUT=C.OUT
CACHE_LOCK=threading.Lock();CHUNK_LOCKS={};MERGE_LOCK=threading.Lock();SOURCE_LOCK=threading.Lock()
FS=gcsfs.GCSFileSystem(token='anon',access='read_only')
FIELD_MAP={'tas':'tmean_c','tasmax':'tmax_c','pr':'precipitation_mm','hurs':'rh_mean_pct'}

def key_lock(path):
    with CACHE_LOCK:return CHUNK_LOCKS.setdefault(str(path),threading.Lock())

def cached(source,key):
    version=source['zstore'].rstrip('/').rsplit('/',1)[-1]
    path=DATA/'zarr_cache'/source['model']/source['scenario']/source['variable']/version/key
    meta=path.with_suffix(path.suffix+'.receipt.json')
    with key_lock(path):
        if path.exists() and meta.exists():
            receipt=json.loads(meta.read_text())
            if receipt['sha256']==C.sha256(path):return path.read_bytes(),receipt
        if shutil.disk_usage(C.ROOT).free<15_000_000_000:raise OSError('Less than 15GB free before native chunk download')
        uri=source['zstore'].rstrip('/')+'/'+key;started=time.monotonic()
        content=FS.cat_file(uri)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+f'.{threading.get_ident()}.tmp');temp.write_bytes(content);temp.replace(path)
        receipt={'source_uri':uri,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'retrieved_utc':C.utc(),'seconds':time.monotonic()-started}
        C.write_json(meta,receipt)
        return content,receipt

def decode(content,array):
    assert not array.get('filters'), 'Unexpected native Zarr filters'
    raw=get_codec(array['compressor']).decode(content) if array.get('compressor') else content
    return np.frombuffer(raw,dtype=array['dtype'])

def coordinate(source,metadata,name):
    array=metadata[name+'/.zarray'];assert len(array['shape'])==1
    values=[]
    for index in range((array['shape'][0]+array['chunks'][0]-1)//array['chunks'][0]):
        content,_=cached(source,f'{name}/{index}');values.append(decode(content,array))
    return np.concatenate(values)[:array['shape'][0]]

SOURCES={};GRIDS={}
def prepare_source(source,cells):
    identifier=(source['model'],source['scenario'],source['variable'])
    with SOURCE_LOCK:
        if identifier in SOURCES:return SOURCES[identifier]
        meta=json.loads((C.ROOT/source['metadata_local_path']).read_text())['metadata']
        assert meta['.zattrs']['variant_label']=='r1i1p1f1'
        assert meta['time/.zattrs']['calendar']=='proleptic_gregorian'
        array=meta[source['variable']+'/.zarray']
        assert array['chunks'][1:]==array['shape'][1:], 'Unexpected spatial chunking requires a separate verified reader'
        lat=coordinate(source,meta,'lat');lon=(coordinate(source,meta,'lon')+180)%360-180
        if source['model'] in GRIDS:
            reference_lat,reference_lon=GRIDS[source['model']]
            if not np.array_equal(lat,reference_lat) or not np.array_equal(lon,reference_lon):
                raise ValueError('Native coordinates differ across model meteorological fields/scenarios')
        else:GRIDS[source['model']]=(lat.copy(),lon.copy())
        lat_order=np.argsort(lat);lon_order=np.argsort(lon);slat=lat[lat_order];slon=lon[lon_order]
        iy=np.searchsorted(slat,cells.latitude.to_numpy())-1;ix=np.searchsorted(slon,cells.longitude.to_numpy())-1
        assert (iy>=0).all() and (iy+1<len(slat)).all() and (ix>=0).all() and (ix+1<len(slon)).all()
        wy=(cells.latitude.to_numpy()-slat[iy])/(slat[iy+1]-slat[iy]);wx=(cells.longitude.to_numpy()-slon[ix])/(slon[ix+1]-slon[ix])
        native_time=coordinate(source,meta,'time');assert np.all(np.diff(native_time)==1)
        result={'source':source,'metadata':meta,'array':array,'native_time':native_time,
                'i0':lat_order[iy],'i1':lat_order[iy+1],'j0':lon_order[ix],'j1':lon_order[ix+1],
                'weights':[(1-wy)*(1-wx),wy*(1-wx),(1-wy)*wx,wy*wx]}
        SOURCES[identifier]=result;return result

def merged_year(model,scenario,year):
    destination=DATA/'daily'/model/scenario/f'{year}.parquet'
    if C.done(destination,C.FIELDS):return True
    paths=[DATA/'variables'/model/scenario/v/f'{year}.parquet' for v in FIELD_MAP]
    if not all(C.done(path,[FIELD_MAP[v]]) for path,v in zip(paths,FIELD_MAP)):return False
    frames=[pd.read_parquet(p) for p in paths];frame=frames[0]
    for other in frames[1:]:frame=frame.merge(other,on=['date','cell_id'],how='outer',validate='one_to_one')
    C.commit_frame(destination,frame,{'source_product':'Native CMIP6, Google Cloud public Zarr','source_role':'Separate candidate; not silently substituted for NEX v2','model':model,'scenario':scenario,'year':year,'member_id':'r1i1p1f1','registry_sha256':C.sha256(C.REGISTRY),'target_transform':C.TRANSFORM,'spatial_alignment':'Explicit bilinear four-corner interpolation from native atmospheric gn grid','source_variable_provenance':[str(p.with_suffix('.json').relative_to(C.ROOT)) for p in paths]},C.FIELDS)
    return True

def retrieve(job,cells,lookup):
    model,scenario,variable,year=job;field=FIELD_MAP[variable]
    path=DATA/'variables'/model/scenario/variable/f'{year}.parquet'
    if C.done(path,[field]):
        with MERGE_LOCK:merged_year(model,scenario,year)
        return {'status':'existing','job':job}
    prepared=prepare_source(lookup[(model,scenario,variable)],cells)
    source=prepared['source'];meta=prepared['metadata'];array=prepared['array'];attrs=meta['time/.zattrs']
    start=cftime.date2num(datetime(year,1,1,12),attrs['units'],attrs['calendar'])
    stop=cftime.date2num(datetime(year+1,1,1,12),attrs['units'],attrs['calendar'])
    indices=np.flatnonzero((prepared['native_time']>=start)&(prepared['native_time']<stop))
    dates=pd.date_range(f'{year}-01-01',f'{year}-12-31').strftime('%Y-%m-%d')
    assert len(indices)==len(dates)
    values=np.empty((len(indices),len(cells)),dtype='float32');receipts=[];chunk_days=array['chunks'][0]
    for chunk in np.unique(indices//chunk_days):
        source_key=f'{variable}/{chunk}.0.0';content,receipt=cached(source,source_key);receipts.append(receipt)
        decoded=decode(content,array).reshape(array['chunks'],order=array['order'])
        positions=np.flatnonzero(indices//chunk_days==chunk);local=(indices[positions]-chunk*chunk_days)[:,None]
        mapped=np.zeros((len(positions),len(cells)),dtype='float64')
        for row,col,weight in zip([prepared['i0'],prepared['i1'],prepared['i0'],prepared['i1']],[prepared['j0'],prepared['j0'],prepared['j1'],prepared['j1']],prepared['weights']):
            corner=decoded[local,row[None,:],col[None,:]]
            if not np.isfinite(corner).all():raise ValueError('Nonfinite native interpolation input; no automatic fill')
            missing_markers=[array.get('fill_value'),meta[variable+'/.zattrs'].get('missing_value'),meta[variable+'/.zattrs'].get('_FillValue')]
            for marker in missing_markers:
                if marker is not None and np.any(corner==np.asarray(marker,dtype=corner.dtype)):
                    raise ValueError('Native missing-value marker in interpolation input; no automatic fill')
            mapped+=corner*weight[None,:]
        values[positions]=mapped.astype('float32')
    if not np.isfinite(values).all():raise ValueError('Missing native-CMIP6 interpolation input; no automatic fill')
    frame=pd.DataFrame({'date':np.repeat(dates,len(cells)),'cell_id':np.tile(cells.cell_id.values,len(dates))})
    raw=values.ravel();native_attrs=meta[variable+'/.zattrs'];units=native_attrs['units'];adjustments={}
    if variable in ['tas','tasmax']:
        assert units=='K';frame[field]=raw-np.float32(273.15)
    elif variable=='pr':
        assert units=='kg m-2 s-1';converted=raw*np.float32(86400)
        if converted.min() < -.01:raise ValueError('Native precipitation below −0.01 mm/day')
        frame[field]=np.clip(converted,0,None)
        if (converted<0).any():frame['precipitation_mm_raw']=converted;adjustments['negative_precipitation_values']=int((converted<0).sum())
    else:
        assert units=='%';frame['rh_mean_pct_raw']=raw;frame[field]=np.clip(raw,0,100)
        adjustments['relative_humidity_outside_0_100_values']=int(((raw<0)|(raw>100)).sum())
    frame[field+'_native_missing']=False
    provenance={'source_product':'Native CMIP6, Google Cloud public Zarr','source_role':'Separate candidate; not silently substituted for NEX v2','model':model,'scenario':scenario,'year':year,'member_id':'r1i1p1f1','zstore':source['zstore'],'source_metadata_sha256':source['metadata_sha256'],'native_global_attributes':meta['.zattrs'],'native_variable_attributes':native_attrs,'native_time_attributes':attrs,'source_chunks':receipts,'registry_path':str(C.REGISTRY.relative_to(C.ROOT)),'registry_sha256':C.sha256(C.REGISTRY),'target_transform':C.TRANSFORM,'spatial_alignment':'Explicit bilinear four-corner interpolation from native atmospheric gn grid, same coordinates/weights across four meteorological fields','native_grid_shape':array['shape'][1:],'numeric_adjustments':adjustments,'license':meta['.zattrs'].get('license'),'unit_conversion':'K−273.15; precipitation kg m−2 s−1×86400; relative humidity percent'}
    result=C.commit_frame(path,frame,provenance,[field])
    with MERGE_LOCK:merged_year(model,scenario,year)
    return {'status':'downloaded','job':job,'bytes':result['parquet_bytes'],'quality':result['quality']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=6);parser.add_argument('--models',nargs='+',default=C.MODELS);parser.add_argument('--scenarios',nargs='+',default=C.SCENARIOS+['historical']);parser.add_argument('--years',nargs='+',type=int)
    args=parser.parse_args();cells=C.registry(C.REGISTRY)
    proof=json.loads((OUT/'pangeo_ensemble_source_verification.json').read_text());assert not proof['errors'] and proof['verified_stores']==48
    lookup={(r['model'],r['scenario'],r['variable']):r for r in proof['stores']}
    years=[j for j in C.year_plan() if j[0] in args.models and j[1] in args.scenarios and (args.years is None or j[2] in args.years)]
    years.sort(key=lambda j:(j[2]>2020,j[2],j[0],j[1]));jobs=[(m,s,v,y) for m,s,y in years for v in FIELD_MAP]
    errors=[];completed=0
    def retry(job):
        for attempt in range(3):
            try:return retrieve(job,cells,lookup)
            except (OSError,TimeoutError,ConnectionError) as exc:
                if attempt==2:raise
                C.report({'source':'native_cmip6_candidate','status':'retry','job':job,'attempt':attempt+1,'error':C.safe_error(exc)});time.sleep(3*(attempt+1))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(retry,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:result=future.result();completed+=1;C.report({'source':'native_cmip6_candidate','completed':completed,'total':len(jobs),**result})
            except Exception as exc:errors.append({'job':job,'error':C.safe_error(exc),'utc':C.utc()});C.report({'source':'native_cmip6_candidate','status':'failed',**errors[-1]})
    C.write_json(OUT/'native_cmip6_candidate_errors.json',errors)
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
