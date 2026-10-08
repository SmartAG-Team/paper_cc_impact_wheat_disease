"""Verify official public CMIP6 Zarr availability and native metadata, without substitution."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import requests
import cftime
import numpy as np
from numcodecs import get_codec

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'analysis/paper_study/climate'
DATA=ROOT/'data/paper_study/climate/pangeo_metadata'
API='https://storage.googleapis.com/storage/v1/b/cmip6/o'
VARIABLES=['tas','tasmax','hurs','pr']

def time_coverage(store, metadata):
    array=metadata['time/.zarray'];attrs=metadata['time/.zattrs']
    assert not array.get('filters'), 'Unexpected time-coordinate filters'
    chunks=[];downloads=[];length=array['shape'][0];chunk_size=array['chunks'][0]
    for index in range((length+chunk_size-1)//chunk_size):
        url='https://storage.googleapis.com/cmip6/'+store+f'time/{index}'
        response=requests.get(url,timeout=(20,90));response.raise_for_status()
        raw=get_codec(array['compressor']).decode(response.content) if array.get('compressor') else response.content
        chunk=np.frombuffer(raw,dtype=array['dtype']);chunks.append(chunk)
        downloads.append({'url':url,'bytes':len(response.content),'sha256':hashlib.sha256(response.content).hexdigest()})
    values=np.concatenate(chunks)[:length]
    assert len(values)==length and np.all(np.diff(values)==1), 'Native time coordinate is not complete daily sequence'
    assert attrs['units'].startswith('days since ')
    endpoints=cftime.num2date(values[[0,-1]],attrs['units'],attrs.get('calendar','standard'))
    return {'first_date':str(endpoints[0])[:10],'last_date':str(endpoints[1])[:10],
            'time_count':length,'strict_daily_sequence_verified':True,'coordinate_downloads':downloads}

def folders(prefix):
    response=requests.get(API,params={'prefix':prefix,'delimiter':'/','maxResults':1000},timeout=(20,90))
    response.raise_for_status()
    result=response.json()
    assert not result.get('nextPageToken'), 'Incomplete bucket folder listing'
    return result.get('prefixes',[])

def inspect(job):
    model,scenario,variable,prefix=job
    grids=folders(prefix)
    preferred=[p for p in grids if p.rstrip('/').endswith('/gn')]
    if len(preferred)!=1:raise ValueError(f'No unique native gn grid: {job}')
    versions=folders(preferred[0]); assert versions
    required_start='1990-01-01' if scenario=='historical' else '2015-01-01'
    required_end='2014-12-31' if scenario=='historical' else '2100-12-31'
    examined=[]
    for store in sorted(versions,reverse=True):
        url='https://storage.googleapis.com/cmip6/'+store+'.zmetadata'
        response=requests.get(url,timeout=(20,90));response.raise_for_status()
        parsed=response.json();metadata=parsed['metadata'];attributes=metadata['.zattrs']
        period=time_coverage(store,metadata)
        covers=period['first_date']<=required_start and period['last_date']>=required_end
        examined.append({'zstore':'gs://cmip6/'+store,'period':period,'covers_required_dates':covers})
        DATA.mkdir(exist_ok=True,parents=True)
        version_name=store.rstrip('/').rsplit('/',1)[1]
        (DATA/f'{model}_{scenario}_{variable}_{version_name}.zmetadata.json').write_bytes(response.content)
        if covers:break
    else:raise ValueError(f'No native store covers required dates: {job}')
    assert attributes['source_id']==model and attributes['experiment_id']==scenario
    assert attributes['variant_label']=='r1i1p1f1'
    assert attributes['table_id']=='day'
    array=metadata[variable+'/.zarray'];varattrs=metadata[variable+'/.zattrs']
    name=f'{model}_{scenario}_{variable}'
    DATA.mkdir(exist_ok=True,parents=True);path=DATA/(name+'.zmetadata.json');path.write_bytes(response.content)
    result={'model':model,'scenario':scenario,'variable':variable,'member_id':attributes['variant_label'],
            'zstore':'gs://cmip6/'+store,'source_url':url,'metadata_sha256':hashlib.sha256(response.content).hexdigest(),
            'metadata_local_path':str(path.relative_to(ROOT)),'available_versions':versions,
            'actual_time_coverage':period,'examined_versions':examined,
            'native_global_attributes':attributes,'native_variable_attributes':varattrs,
            'native_shape':array['shape'],'native_chunks':array['chunks'],'compressor':array.get('compressor'),
            'native_time_attributes':metadata['time/.zattrs'],'native_time_shape':metadata['time/.zarray']['shape'],
            'native_latitude_shape':metadata['lat/.zarray']['shape'],'native_longitude_shape':metadata['lon/.zarray']['shape']}
    return result

def main():
    availability=json.loads((OUT/'pangeo_live_bucket_availability.json').read_text())
    jobs=[]
    for row in availability:
        mapping={p.rstrip('/').rsplit('/',1)[1]:p for p in row['variable_prefixes']}
        jobs.extend((row['model'],row['scenario'],v,mapping[v]) for v in VARIABLES)
    rows=[];errors=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        lookup={pool.submit(inspect,j):j for j in jobs}
        for future in as_completed(lookup):
            job=lookup[future]
            try:rows.append(future.result())
            except Exception as exc:errors.append({'job':job,'error':str(exc)})
    rows.sort(key=lambda r:(r['model'],r['scenario'],r['variable']))
    receipt={'checked_utc':datetime.now(timezone.utc).isoformat(),'catalog':'https://storage.googleapis.com/cmip6/pangeo-cmip6.csv',
             'verification':'Live anonymous public Google Cloud provider object listings and consolidated metadata',
             'expected_stores':48,'verified_stores':len(rows),'errors':errors,'stores':rows,
             'primary_forcing_status':'Candidate only; no replacement of ongoing native NEX-GDDP-CMIP6 v2 retrieval'}
    (OUT/'pangeo_ensemble_source_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'verified_stores':len(rows),'errors':errors,'calendars':sorted({r['native_time_attributes'].get('calendar','standard') for r in rows}),
                      'units':{v:sorted({r['native_variable_attributes']['units'] for r in rows if r['variable']==v}) for v in VARIABLES},
                      'shapes':{r['model']:{'latitude':r['native_latitude_shape'],'longitude':r['native_longitude_shape'],'chunks':r['native_chunks']} for r in rows}}),flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
