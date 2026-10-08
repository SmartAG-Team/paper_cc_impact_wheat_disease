"""Verify live Earth Engine NEX image versions and sample native-v2 equivalence."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import ee
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'analysis/paper_study/climate'
DATA = ROOT/'data/paper_study/climate'
MODELS = ['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']
SCENARIOS = ['historical','ssp245','ssp585']
VARIABLES = ['tas','tasmax','pr','hurs']
TRANSFORM = [0.25,0,-180,0,-0.25,90]

def scan(collection):
    periods = ee.Filter.Or(ee.Filter.date('1990-01-01','2021-01-01'),
                           ee.Filter.date('2030-01-01','2061-01-01'),
                           ee.Filter.date('2070-01-01','2101-01-01'))
    values = {}
    for model in MODELS:
        for scenario in SCENARIOS:
            subset = collection.filter(ee.Filter.eq('model',model)).filter(ee.Filter.eq('scenario',scenario)).filter(periods)
            values[f'{model}/{scenario}'] = ee.Dictionary({'images':subset.size(),
                'with_version':subset.aggregate_count('version'),
                'version_histogram':subset.aggregate_histogram('version'),
                'license_histogram':subset.aggregate_histogram('license')})
    return ee.Dictionary(values).getInfo()

def sample(collection):
    registry = pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet')
    targets = [(50.125,8.125),(37.125,-5.125),(65.125,25.125),(38.125,35.125),(55.125,50.125),(48.125,2.125)]
    indices = [int(((registry.latitude-y)**2+(registry.longitude-x)**2).idxmin()) for y,x in targets]
    cells = registry.loc[indices].drop_duplicates('cell_id')
    features = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.longitude),float(r.latitude)]),{'cell_id':r.cell_id}) for r in cells.itertuples()])
    images = []; selections = []; metadata = []
    for model in MODELS:
        for scenario in SCENARIOS:
            year = 1990 if scenario == 'historical' else 2030
            for suffix in ['01-01','02-15','08-01']:
                date = f'{year}-{suffix}'
                subset = collection.filter(ee.Filter.eq('model',model)).filter(ee.Filter.eq('scenario',scenario)).filterDate(date,ee.Date(date).advance(1,'day'))
                image = ee.Image(subset.first())
                index = len(images)
                images.append(image.select(VARIABLES).set('system:index',f'i{index:02d}'))
                metadata.append(image.toDictionary().set('projection',image.select('tas').projection().getInfo()))
                selections.append({'index':index,'model':model,'scenario':scenario,'date':date})
    metadata = ee.List(metadata).getInfo()
    stacked = ee.ImageCollection.fromImages(images).toBands()
    result = stacked.sampleRegions(collection=features,projection=ee.Projection('EPSG:4326',TRANSFORM),geometries=False).getInfo()
    native_cache = {}; rows = []
    for feature in result['features']:
        props = feature['properties'];cell_id=props['cell_id']
        for selection in selections:
            model=selection['model'];scenario=selection['scenario'];date=selection['date'];year=int(date[:4]);index=selection['index']
            for variable in VARIABLES:
                key=(model,scenario,variable,year)
                if key not in native_cache:
                    path=DATA/'nasa_v2/variables'/model/scenario/variable/f'{year}.parquet'
                    field={'tas':'tmean_c','tasmax':'tmax_c','pr':'precipitation_mm','hurs':'rh_mean_pct_raw'}[variable]
                    df=pd.read_parquet(path,columns=['date','cell_id',field],filters=[('cell_id','in',cells.cell_id.tolist()),('date','in',[s['date'] for s in selections if s['model']==model and s['scenario']==scenario])])
                    native_cache[key]=df.set_index(['date','cell_id'])[field]
                native=float(native_cache[key].loc[(date,cell_id)])
                ee_value=float(props[f'i{index:02d}_{variable}'])
                ee_converted=ee_value-273.15 if variable.startswith('tas') else ee_value*86400 if variable=='pr' else ee_value
                rows.append(selection|{'cell_id':cell_id,'variable':variable,'ee_native_value':ee_value,'native_v2_converted_value':native,'ee_converted_value':ee_converted,'difference':ee_converted-native,'ee_version':metadata[index].get('version')})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'ee_native_v2_value_comparison.csv',index=False)
    summary=frame.groupby('variable').difference.agg(['count','min','max',lambda x:float(np.abs(x).max())]).rename(columns={'<lambda_0>':'max_abs_difference'}).to_dict('index')
    return {'cells':cells[['cell_id','latitude','longitude']].to_dict('records'),'image_metadata':metadata,'selections':selections,'value_count':len(frame),'variable_comparison':summary,'numerical_units':{'tas':'degC','tasmax':'degC','pr':'mm/day','hurs':'percent'}}

def main():
    ee.Initialize(project='ee-gangzhaomodel');ee.data.setDeadline(420000)
    collection=ee.ImageCollection('NASA/GDDP-CMIP6')
    with ThreadPoolExecutor(max_workers=2) as pool:
        version_job=pool.submit(scan,collection);sample_job=pool.submit(sample,collection)
        versions=version_job.result();values=sample_job.result()
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'source_collection':'NASA/GDDP-CMIP6','source_catalog':'https://developers.google.com/earth-engine/datasets/catalog/NASA_GDDP-CMIP6','native_comparator_version':'2.0','version_scan':versions,'sample_comparison':values,'admissible_alternative_transport':all(set(x['version_histogram'])=={'2'} or set(x['version_histogram'])=={'2.0'} for x in versions.values()),'policy':'Versions 1.x must not be mixed with native version 2.0; identical isolated sample values alone do not establish full-file equivalence.'}
    (OUT/'ee_native_v2_compatibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'version_scan':versions,'comparison':values['variable_comparison'],'admissible_alternative_transport':result['admissible_alternative_transport']}),flush=True)

if __name__=='__main__':main()
