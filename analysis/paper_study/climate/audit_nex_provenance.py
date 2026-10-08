"""Independent audit of native NEX file identities, quarter metadata and terms."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,re,xml.etree.ElementTree as ET
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'analysis/paper_study/climate'
DATA=ROOT/'data/paper_study/climate'
MODELS=['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']
SCENARIOS=['ssp126','ssp245','ssp585']
VARIABLES={'tas':'K','tasmax':'K','pr':'kg m-2 s-1','hurs':'%'}

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hashes',action='store_true',help='Verify every component Parquet against its complete-byte digest')
    args=parser.parse_args()
    registry=DATA.parent/'wheat_area/europe_wheat_cells_025.parquet'
    cells=pd.read_parquet(registry);registry_hash=sha(registry)
    bounds={'north':float(cells.latitude.max()),'south':float(cells.latitude.min()),
            'west':float(cells.longitude.min()),'east':float(cells.longitude.max())}
    edge_bounds={'geospatial_lat_min':bounds['south']-.125,'geospatial_lat_max':bounds['north']+.125,
                 'geospatial_lon_min':bounds['west']-.125,'geospatial_lon_max':bounds['east']+.125}
    catalogs={};rows=[];errors=[];missing=[];transport=Counter();quarters=0;checks=0
    def check(condition,label,path):
        nonlocal checks
        checks+=1
        if not condition:errors.append({'path':str(path.relative_to(ROOT)),'check':label})
    def global_checks(attrs,model,scenario,filename,path):
        for key,value in {'version':'2.0','cmip6_source_id':model,'scenario':scenario,
                          'variant_label':'r1i1p1f1','activity':'NEX-GDDP-CMIP6','frequency':'day',
                          'source':'BCSD','resolution_id':'0.25 degree'}.items():
            check(attrs.get(key)==value,'native global '+key,path)
        check(filename in attrs.get('History',attrs.get('history','')),'native original dataset filename',path)
        for key,value in edge_bounds.items():check(abs(float(attrs.get(key,float('inf')))-value)<1e-7,'native '+key,path)
        check(attrs.get('cmip6_license') in ['CC-BY 4.0','CC-BY-SA 4.0'],'preserved native license',path)
    for model in MODELS:
        periods=[('historical',range(1990,2015))]+[(s,list(range(2015,2021))+list(range(2030,2061))+list(range(2070,2101))) for s in SCENARIOS]
        for scenario,years in periods:
            for variable,units in VARIABLES.items():
                catalog=DATA/'catalogs'/f'{model}_{scenario}_r1i1p1f1_{variable}.xml'
                parsed=ET.fromstring(catalog.read_bytes())
                paths={r.attrib['urlPath'] for r in parsed.iter() if 'urlPath' in r.attrib}
                catalog_hash=sha(catalog);catalogs[str(catalog.relative_to(ROOT))]=catalog_hash
                for year in years:
                    path=DATA/'nasa_v2/variables'/model/scenario/variable/f'{year}.json'
                    if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
                    info=json.loads(path.read_text());filename=f'{variable}_day_{model}_{scenario}_r1i1p1f1_gn_{year}_v2.0.nc'
                    source=f'AMES/NEX/GDDP-CMIP6/{model}/{scenario}/r1i1p1f1/{variable}/{filename}'
                    check(info.get('native_source_path')==source,'exact native source identity',path)
                    check(source in paths,'native file present in archived official catalog',path)
                    check(info.get('source_catalog_sha256')==catalog_hash,'official catalog checksum',path)
                    check(info.get('registry_sha256')==registry_hash,'unchanged source-cell registry',path)
                    check(info.get('version')=='2.0','declared version',path)
                    check(info.get('native_calendar')=='proleptic_gregorian','native calendar',path)
                    check(info.get('target_transform')==[.25,0,-180,0,-.25,90],'target transform',path)
                    actual_units=info.get('native_variable_attributes',{}).get('units')
                    check(actual_units==units or (variable=='pr' and actual_units in ['kg/m2/s','kg m**-2 s**-1']),'native variable units',path)
                    global_checks(info.get('native_global_attributes',{}),model,scenario,filename,path)
                    query=info.get('ncss_query',{})
                    for key,value in bounds.items():check(query.get(key)==value,'annual query '+key,path)
                    check(query.get('var')==variable,'annual query variable',path)
                    check(query.get('time_start')==f'{year}-01-01T12:00:00Z','annual query start',path)
                    check(query.get('time_end')==f'{year}-12-31T12:00:00Z','annual query end',path)
                    check(info.get('native_file_license')==info.get('native_global_attributes',{}).get('cmip6_license'),'native license preserved without blanket override',path)
                    parts=info.get('quarter_subsets')
                    if parts is not None:
                        transport['quarterly']+=1;check(len(parts)==4,'four quarter subsets',path)
                        for quarter,part in enumerate(parts,1):
                            quarters+=1;start=pd.Timestamp(year=year,month=3*quarter-2,day=1);end=start+pd.offsets.QuarterEnd()
                            q=part.get('query',{})
                            check(q==query|{'time_start':f'{start:%Y-%m-%d}T12:00:00Z','time_end':f'{end:%Y-%m-%d}T12:00:00Z'},'quarter exact bounds and dates',path)
                            check(part.get('native_source_path')==source,'quarter source identity',path)
                            check(part.get('native_calendar')=='proleptic_gregorian','quarter calendar',path)
                            check(bool(re.fullmatch('[0-9a-f]{64}',part.get('sha256',''))),'quarter byte digest retained',path)
                            check(part.get('bytes',0)>0,'quarter response bytes retained',path)
                            global_checks(part.get('native_global_attributes',{}),model,scenario,filename,path)
                            check(part.get('native_global_attributes',{}).get('cmip6_license')==info['native_file_license'],'same native terms across quarters',path)
                        check(info.get('download_bytes')==sum(p['bytes'] for p in parts),'quarter bytes reconcile',path)
                    else:transport['annual']+=1
                    check(bool(re.fullmatch('[0-9a-f]{64}',info.get('download_sha256',''))),'transport byte digest retained',path)
                    if args.hashes:
                        parquet=path.with_suffix('.parquet')
                        check(parquet.exists() and sha(parquet)==info.get('parquet_sha256'),'complete component Parquet byte checksum',path)
                    rows.append({'model':model,'scenario':scenario,'year':year,'variable':variable,'source_version':'2.0',
                        'variant_label':'r1i1p1f1','native_source_path':source,'native_license':info.get('native_file_license'),
                        'transport':'quarterly' if parts is not None else 'annual','native_calendar':info.get('native_calendar'),
                        'native_units':actual_units,'metadata_path':str(path.relative_to(ROOT)),'parquet_sha256':info.get('parquet_sha256')})
    licenses=Counter(r['native_license'] for r in rows)
    pd.DataFrame(rows).to_csv(OUT/'nex_native_file_provenance.csv',index=False)
    receipt={'checked_utc':datetime.now(timezone.utc).isoformat(),'audit_independence':'No retrieval functions imported; native identities, periods, grid edges, units, catalogs and quarter metadata reconstructed separately',
        'expected_components':2748,'available_components':len(rows),'missing_components':len(missing),
        'check_count':checks,'failed_checks':len(errors),'errors':errors,'missing':missing,'transport_components':dict(transport),
        'quarter_metadata_records_checked':quarters,'license_labels':dict(licenses),'complete_component_bytes_checked':args.hashes,
        'registry_sha256':registry_hash,'registry_cells':len(cells),'official_catalog_checksums':catalogs,
        'status':'Complete and passed' if len(rows)==2748 and not errors else 'Partial and passed' if not errors else 'Failed'}
    (OUT/'nex_native_provenance_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['missing','official_catalog_checksums','errors']}),flush=True)
    if errors:print(json.dumps(errors[:20]),flush=True);raise SystemExit(1)
    if len(rows)!=2748:raise SystemExit(2)

if __name__=='__main__':main()
