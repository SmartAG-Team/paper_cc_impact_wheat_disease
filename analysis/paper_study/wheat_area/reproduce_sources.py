"""Verify archived provider downloads; optionally acquire missing advertised files."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import urllib.request
import zipfile

import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/wheat_area'


def run(download=False,extract=False):
    specs=json.loads((HERE/'source_download_spec.json').read_text())
    checks=[]
    for spec in specs:
        path=DATA/spec['file']
        if not path.exists():
            if not download:raise FileNotFoundError(path)
            # Public links are advertised by the provider. No guestbook or denied endpoint is bypassed.
            with urllib.request.urlopen(urllib.request.Request(spec['url'],headers={'User-Agent':'Mozilla/5.0'}),timeout=120) as response:
                content=response.read()
            if hashlib.sha256(content).hexdigest()!=spec['sha256']:
                raise ValueError(f"Source bytes changed for {path.name}; archived source version cannot be replaced")
            temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(content);temp.replace(path)
        content=path.read_bytes()
        assert len(content)==spec['bytes'],path.name
        assert hashlib.sha256(content).hexdigest()==spec['sha256'],path.name
        if 'md5' in spec:assert hashlib.md5(content).hexdigest()==spec['md5'],path.name
        checks.append(dict(file=path.name,bytes=len(content),sha256=spec['sha256'],status='verified_archived_bytes'))
    if extract:
        target=DATA/'spam2020_wheat';target.mkdir(exist_ok=True)
        for kind in ['harvested','physical']:
            with zipfile.ZipFile(DATA/f'spam2020V2r2_global_{kind}_area.geotiff.zip') as z:
                for name in z.namelist():
                    if '_WHEA_' in name and name.endswith('.tif'):
                        path=target/Path(name).name;content=z.read(name)
                        if path.exists():assert path.read_bytes()==content,path.name
                        else:path.write_bytes(content)
        with zipfile.ZipFile(DATA/'spam2020V2r2_global_harvested_area.csv.zip') as z:
            for name in z.namelist():
                if name.endswith('.csv'):
                    cols=['grid_code','whea']
                    if name.endswith('_TA.csv'):cols+=['x','y','FIPS0','FIPS1','FIPS2','ADM0_NAME','ADM1_NAME','ADM2_NAME']
                    frame=pd.read_csv(z.open(name),usecols=cols)
                    derived=target/f'{Path(name).stem}_wheat.parquet'
                    frame[['grid_code','whea']].to_parquet(derived,index=False)
                    if name.endswith('_TA.csv'):frame.to_parquet(target/'source_wheat_country_metadata.parquet',index=False)
        for system in ['ir','rf']:
            frame=pd.read_csv(DATA/f'MIRCA-OS_2020_{system}_v2.csv',encoding='latin1')
            frame.loc[frame.Crop.str.match(r'^Wheat[12]$')].to_csv(DATA/f'MIRCA-OS_2020_{system}_v2_wheat.csv',index=False)
    receipt=dict(status='verified_archived_source_bytes',completed_utc=datetime.now(timezone.utc).isoformat(),files=len(checks),checks=checks,
        external_download_requested=download,extract_requested=extract,
        metadata_limit='Volatile provider metadata may change while dataset bytes stay unchanged; changed snapshot bytes fail explicitly and are never substituted silently.')
    (HERE/'source_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],files=len(checks),extract=extract),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--extract',action='store_true');a=p.parse_args();run(a.download,a.extract)
