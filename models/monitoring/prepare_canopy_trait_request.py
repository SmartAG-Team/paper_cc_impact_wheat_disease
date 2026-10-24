"""Freeze strictly historical, geographically separated canopy/phenology requests."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .innovation_common import ROOT,DATA
from .inputs import sha

TRAIT=ROOT/'data/disease_improvement/phenology_transfer_20261022'


def prepare():
    out=ROOT/'data/disease_improvement/canopy_trait_request_20261022'
    if out.exists():raise FileExistsError('Fresh trait request destination required')
    source=TRAIT/'admitted_trait_cycles.json';geo=TRAIT/'trait_station_qualification.csv'
    records=json.loads(source.read_text());stations=pd.read_csv(geo)
    admitted=set(stations.loc[stations.minimum_disease_distance_km.gt(61),'station'].astype(int))
    records=[r for r in records if r['PEP_ID'] in admitted and 2003<=r['harvest_year']<=2012
        and r['SOWING_DATE']>='2002-07-04' and all(r['true_dates'][i] is not None for i in [1,2,3])]
    out.mkdir(parents=True)
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [source,geo,TRAIT/'registration.json',DATA/'fields.csv']},
        code_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        qualification='Allcyclesendbefore2013, harvest2003..2012, observedorderedBBCH31/51/87; all documented stationcoords>61kmfromeverydiseaselocation so10kmbuffer+pixelmarginremains>50km',
        dates='Fortnightly Feb1..Aug31, at least84daysaftersowing, no laterthanobservedBBCH87+14days orsourcecycleend',
        labels='Stagepassedlabelsderivefrom3observedeventdates; repeateddatecontextsarenotindependentnewphenologicalmeasurements; no diseaselabels',
        weights='Equalstationmasssplitamongcyclesandquerydates, then3stagecases; no class balancing',
        validation='Fixed20%stationholdoutseed20261022 beforeauxiliaryfit; calendar-eventCDFcomparison; no outerdiseaseoutcomes used',
        future_model='Onejointstage-orderedbinaryHGB400/depth3/leaves15/min100/L2=5/lr.05; stageindexmonotonedecreasing; 52canopyprefixfeatures+purepast14/42/84weather/calendar/latitude, no actualsowing input',
        auxiliary_screen='HeldstationmeanBrier mustbeatstation-weightedhistoricalcalendarCDFbeforefullrefit; no hyperparametertuning or diseasefitifauxiliaryscreenfails',
        timing='Allauxiliarylabels/weather/canopyqueriespre2013; currentdiseasequerycanopystrictpastwith7dayassumedlag',
        current_disease_information='Unchanged currentdateduntreatedlabels andsame-field-onlypriorhistory; no other-fieldcurrentseasondisease surveillance',
        prepared_only_no_fit=True)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    target=out/'code_snapshot'/Path(__file__).relative_to(ROOT);target.parent.mkdir(parents=True);shutil.copyfile(__file__,target)
    (out/'admitted_cycles.json').write_text(json.dumps(records,indent=2)+'\n')
    coordinates={}
    rows=[]
    for i,r in enumerate(records):
        station=int(r['PEP_ID']);site=f'dwd_trait_{station}'
        point=(float(r['lat']),float(r['lon']))
        assert station not in coordinates or coordinates[station]==point
        coordinates[station]=point
        start=max(pd.Timestamp(r['harvest_year'],2,1),pd.Timestamp(r['SOWING_DATE'])+pd.Timedelta(days=84))
        end=min(pd.Timestamp(r['harvest_year'],8,31),pd.Timestamp(r['true_dates'][3])+pd.Timedelta(days=14),pd.Timestamp(r['END_DATE']))
        for q in pd.date_range(start,end,freq='14D'):
            day=(q-pd.Timestamp(r['SOWING_DATE'])).days
            assert q<pd.Timestamp('2013-01-01') and 84<=day<r['length']
            rows.append(dict(cycle_index=i,station_id=station,site_id=site,date=str(q.date()),weather_row=int(r['row_start'])+day,
                stage31_passed=int(q>=pd.Timestamp(r['true_dates'][1])),stage51_passed=int(q>=pd.Timestamp(r['true_dates'][2])),
                stage87_passed=int(q>=pd.Timestamp(r['true_dates'][3]))))
    queries=pd.DataFrame(rows)
    cycle_count=queries.groupby('station_id').cycle_index.transform('nunique')
    date_count=queries.groupby('cycle_index').date.transform('size')
    queries['station_cycle_date_weight']=1/(cycle_count*date_count)
    np.testing.assert_allclose(queries.groupby('station_id').station_cycle_date_weight.sum(),1,atol=1e-15)
    ids=sorted(coordinates);rng=np.random.default_rng(20261022)
    held=set(rng.choice(ids,max(1,int(np.ceil(.2*len(ids)))),replace=False).tolist())
    queries['auxiliary_validation_station']=queries.station_id.isin(held)
    queries.to_csv(out/'queries.csv',index=False)
    pd.DataFrame([dict(site_id=f'dwd_trait_{station}',latitude=lat,longitude=lon,
        minimum_disease_distance_km=float(stations.loc[stations.station.eq(station),'minimum_disease_distance_km'].min()),
        auxiliary_validation_station=station in held) for station,(lat,lon) in sorted(coordinates.items())]).to_csv(out/'stations.csv',index=False)
    summary=dict(passed=True,cycles=len(records),stations=len(ids),query_dates=len(queries),
        auxiliary_train_stations=len(ids)-len(held),auxiliary_validation_stations=len(held),latest_query_date=queries.date.max(),
        source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},no_model_fit=True)
    (out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='artifacts_sha256'},indent=2),flush=True)


if __name__=='__main__':prepare()
