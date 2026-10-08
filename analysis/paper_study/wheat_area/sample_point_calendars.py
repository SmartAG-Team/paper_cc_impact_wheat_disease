"""Sample provider calendar cells at trial coordinates without assessment values."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pyproj import Geod
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'data/paper_study/wheat_area'
GEOD = Geod(ellps='WGS84')


def run():
    assessments = ROOT/'data/paper_study/observations/assessments.csv'
    requests = ROOT/'data/paper_study/observations/corteva_weather_requests.csv'
    basf = pd.read_csv(assessments, usecols=['dataset_id','site_id','latitude','longitude','coordinate_basis'])
    basf = basf.loc[basf.dataset_id.eq('basf-wheat-diseases')].rename(columns={'site_id':'point_id','coordinate_basis':'coordinate_source'})
    basf = basf.drop_duplicates()
    corteva = pd.read_csv(requests, usecols=['dataset_id','location_id','latitude','longitude','coordinate_source']).rename(columns={'location_id':'point_id'}).drop_duplicates()
    points = pd.concat([basf,corteva],ignore_index=True)
    assert not points.duplicated(['dataset_id','point_id']).any()
    assert points.latitude.between(-90,90,inclusive='neither').all()
    assert points.longitude.between(-180,180,inclusive='left').all()
    points['native_calendar_row'] = np.floor((90-points.latitude)*2).astype(int)
    points['native_calendar_col'] = np.floor((points.longitude+180)*2).astype(int)
    points['native_calendar_latitude'] = 89.75-points.native_calendar_row*.5
    points['native_calendar_longitude'] = -179.75+points.native_calendar_col*.5
    _,_,distance = GEOD.inv(points.longitude.to_numpy(),points.latitude.to_numpy(),
        points.native_calendar_longitude.to_numpy(),points.native_calendar_latitude.to_numpy())
    points['point_to_native_cell_center_distance_km'] = distance/1000
    outputs=[]
    for crop,season in [('wwh','winter_wheat'),('swh','spring_wheat')]:
        for system,water in [('rf','rainfed'),('ir','irrigated')]:
            file=DATA/f'{crop}_{system}_ggcmi_crop_calendar_phase3_v1.01.nc4'
            with xr.open_dataset(file) as ds:
                frame=points.copy()
                row,col=frame.native_calendar_row.to_numpy(),frame.native_calendar_col.to_numpy()
                for old,new in [('planting_day','planting_doy'),('maturity_day','maturity_doy'),
                                ('growing_season_length','growing_season_length_days'),('data_source_used','provider_source_index')]:
                    frame[new]=ds[old].to_numpy()[row,col]
                frame['calendar_valid']=frame.planting_doy.between(1,366)&frame.maturity_doy.between(1,366)
                frame['native_missing_mask']=~frame.calendar_valid
                frame['crop_season'],frame['water_system']=season,water
                frame['calendar_version']='GGCMI Phase 3 v1.01'
                frame['calendar_doi']='10.5281/zenodo.5062513'
                frame['calendar_resolution_deg']=.5
                frame['calendar_assignment']='containing_native_cell_no_interpolation'
                frame['sowing_date_basis']='scenario_planting_calendar_not_observed_trial_sowing'
                frame['terminal_date_basis']='provider_maturity_not_observed_harvest'
                frame['harvest_doy']=np.nan
                frame['harvest_date_available']=False
                frame['source_file']=file.name
                frame['source_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
                outputs.append(frame)
    result=pd.concat(outputs,ignore_index=True)
    result.to_csv(DATA/'trial_point_calendar_scenarios.csv',index=False)
    result.to_parquet(DATA/'trial_point_calendar_scenarios.parquet',index=False)
    points.to_csv(DATA/'trial_calendar_points.csv',index=False)
    receipt=dict(status='verified_containing_calendar_cells',completed_utc=datetime.now(timezone.utc).isoformat(),
        points=len(points),datasets=points.groupby('dataset_id').size().to_dict(),calendar_scenario_rows=len(result),
        missing_calendar_rows=int(result.native_missing_mask.sum()),max_point_to_native_cell_center_distance_km=float(distance.max()/1000),
        observed_sowing_dates=False,observed_harvest_dates=False,assessment_outcomes_read=False,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [assessments,requests]})
    (HERE/'point_calendar_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':run()
