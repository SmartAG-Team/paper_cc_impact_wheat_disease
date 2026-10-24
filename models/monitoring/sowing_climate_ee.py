"""Complete prior-calendar-year temperature covariates for disease field sites."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import ee
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/disease_improvement/sowing_climate_20261022'


def main():
    fields_file=ROOT/'data/disease_improvement/station_preferred_ie_20261022/fields.csv'
    fields=pd.read_csv(fields_file)
    points=fields[['site_id','latitude','longitude','season_year']].drop_duplicates().copy()
    points['annual_temperature_year']=points.season_year-2
    points=points.drop(columns='season_year').drop_duplicates()
    if OUT.exists():
        if (OUT/'annual_temperature.csv').exists():
            raise FileExistsError('Completed extraction cannot be overwritten')
        original=json.loads((OUT/'registration.json').read_text())
        assert original['fields_sha256']==hashlib.sha256(fields_file.read_bytes()).hexdigest()
        (OUT/'registration.json').rename(OUT/'registration_attempt1.json')
        (OUT/'code_snapshot.py').rename(OUT/'code_snapshot_attempt1.py')
    else:
        OUT.mkdir(parents=True)
    registration=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        project='ee-gangzhaomodel',source='ECMWF/ERA5_LAND/MONTHLY_AGGR',band='temperature_2m',
        aggregation='12 monthly mean Kelvin values, weighted by Gregorian calendar days; subtract273.15',
        availability='Annual temperature year=harvestyear-2, completed before preceding autumn sowing',
        role='Explicit reanalysis fallback where complete-year DWD interpolation unavailable; all sites extracted for source comparison',
        fields_sha256=hashlib.sha256(fields_file.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),query_rows=len(points),
        disease_labels_used=False,authentication='Existing local EarthEngine credentials reused; no credentials exported')
    (OUT/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    (OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    ee.Initialize(project='ee-gangzhaomodel')
    records=[]
    for year,part in points.groupby('annual_temperature_year'):
        collection=ee.ImageCollection(registration['source']).filterDate(f'{year}-01-01',f'{year+1}-01-01').select('temperature_2m')
        assert collection.size().getInfo()==12
        image=ee.Image.cat([collection.filter(ee.Filter.calendarRange(month,month,'month')).first().rename(f'm{month:02d}') for month in range(1,13)])
        features=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.longitude,r.latitude]),dict(site_id=r.site_id,annual_temperature_year=int(year))) for r in part.itertuples(index=False)])
        cache=OUT/f'{year}_monthly_temperature.json'
        if cache.exists():
            response=json.loads(cache.read_text())
        else:
            response=image.reduceRegions(features,ee.Reducer.first(),crs=collection.first().projection(),scale=collection.first().projection().nominalScale(),tileScale=4).getInfo()
            cache.write_text(json.dumps(response,separators=(',',':'))+'\n')
        months=pd.date_range(f'{year}-01-01',periods=12,freq='MS')
        for feature in response['features']:
            row=feature['properties']
            values=np.array([row.get(f'm{m:02d}',np.nan) for m in range(1,13)],float)
            finite=np.isfinite(values)
            records.append(dict(site_id=row['site_id'],annual_temperature_year=int(year),
                era5_annual_mean_temperature_c=float(np.average(values-273.15,weights=months.days_in_month)) if finite.all() else None,
                valid_months=int(finite.sum()),expected_days=int(np.asarray(months.days_in_month).sum()),origin='ERA5_LAND_MONTHLY_AGGR'))
        print('extracted complete annual temperature',year,len(part),flush=True)
    result=pd.DataFrame(records)
    assert len(result)==len(points) and not result.duplicated(['site_id','annual_temperature_year']).any()
    result.to_csv(OUT/'annual_temperature.csv',index=False)
    print(result.valid_months.value_counts().to_dict(),flush=True)


if __name__=='__main__':
    main()
