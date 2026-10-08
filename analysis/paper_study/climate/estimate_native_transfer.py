"""Count exact public native-CMIP6 chunk bytes needed for required years."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date,datetime,timezone
from pathlib import Path
import json,requests
HERE=Path(__file__).resolve().parent
proof=json.loads((HERE/'pangeo_ensemble_source_verification.json').read_text())

def inspect(source):
    start=date.fromisoformat(source['actual_time_coverage']['first_date']);chunk=source['native_chunks'][0]
    years=range(1990,2015) if source['scenario']=='historical' else list(range(2015,2021))+list(range(2030,2061))+list(range(2070,2101))
    needed=set()
    for year in years:
        a=(date(year,1,1)-start).days;b=(date(year,12,31)-start).days
        needed.update(range(a//chunk,b//chunk+1))
    prefix=source['zstore'].removeprefix('gs://cmip6/').rstrip('/')+'/'+source['variable']+'/'
    response=requests.get('https://storage.googleapis.com/storage/v1/b/cmip6/o',params={'prefix':prefix,'maxResults':1000,'fields':'items(name,size),nextPageToken'},timeout=(20,90));response.raise_for_status();result=response.json();assert not result.get('nextPageToken')
    files={int(v['name'][len(prefix):].split('.')[0]):int(v['size']) for v in result.get('items',[]) if v['name'][len(prefix):].split('.')[0].isdigit()}
    missing=needed-set(files);assert not missing,(source['model'],source['scenario'],source['variable'],missing)
    return {'model':source['model'],'scenario':source['scenario'],'variable':source['variable'],'source_zstore':source['zstore'],'required_years':len(list(years)),'required_chunks':len(needed),'encoded_bytes':sum(files[c] for c in needed),'chunk_sizes_bytes':{str(c):files[c] for c in sorted(needed)}}
with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(inspect,proof['stores']))
result={'checked_utc':datetime.now(timezone.utc).isoformat(),'estimate_method':'Exact provider object sizes for the union of native time chunks touching every required calendar year; each chunk counted once, assuming shared resumable cache','stores':rows,'total_encoded_bytes':sum(r['encoded_bytes'] for r in rows),'total_chunks':sum(r['required_chunks'] for r in rows),'coordinate_and_metadata_bytes_excluded':'Small overhead, normally a few kilobytes per store'}
(HERE/'pangeo_required_chunk_transfer.json').write_text(json.dumps(result,indent=2)+'\n');print({k:result[k] for k in ['total_encoded_bytes','total_chunks']})
