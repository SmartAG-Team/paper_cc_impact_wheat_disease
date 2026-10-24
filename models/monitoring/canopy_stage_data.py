"""Freeze historical crop-stage inputs and matching current-cohort inputs, without fitting."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .canopy_stage_context import stage_context
from .innovation_common import ROOT,DATA,PARENT
from .phenology_transfer_data import TRAIT
from .inputs import sha

REQUEST=ROOT/'data/disease_improvement/canopy_trait_request_20261022'
RAW=ROOT/'data/disease_improvement/canopy_trait_raw_20261022'
CURRENT_RAW=ROOT/'data/disease_improvement/canopy_regional_v2_20261022'
CURRENT_PREFIX=ROOT/'data/disease_improvement/canopy_prefix_20261022'


def _read_raw(directory):
    receipt=json.loads((directory/'verification.json').read_text())
    assert receipt['passed'] and receipt['source_unchanged']
    for name,digest in receipt['artifacts_sha256'].items():
        assert sha(directory/name)==digest,name
    frame=pd.read_csv(directory/'regional_canopy.csv',float_precision='round_trip')
    for name in ['period_start','period_end']:frame[name]=pd.to_datetime(frame[name],unit='ms')
    assert not frame.duplicated(['site_id','image_index']).any()
    assert (frame.period_end-frame.period_start).dt.days.between(1,4).all()
    for band in ['fpar','lai_uncertainty','fpar_uncertainty']:
        np.testing.assert_array_equal(frame.lai_count,frame[band+'_count'])
    return {site:g for site,g in frame.groupby('site_id')}


def build():
    out=ROOT/'data/disease_improvement/canopy_stage_context_20261022'
    if out.exists():raise FileExistsError('Fresh stage-context destination required')
    sources=[REQUEST/name for name in ['registration.json','verification.json','queries.csv','stations.csv','admitted_cycles.json']]
    sources += [directory/name for directory in [RAW,CURRENT_RAW] for name in ['registration.json','verification.json','regional_canopy.csv','raw/crop_2009.json']]
    sources += [TRAIT/'daily_inputs.npz',TRAIT/'manifest.json',DATA/'fields.csv',DATA/'daily_weather.parquet',
        PARENT/'training_observation_ledger.csv',CURRENT_PREFIX/'predictors.npz']
    codes=[Path(__file__),Path(__file__).with_name('canopy_stage_context.py'),Path(__file__).with_name('canopy_inputs.py')]
    out.mkdir(parents=True)
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        transformation='52unchangedqualifiedregionalcanopyprefixfeatures plus33past14/42/84dayweather/calendar/latitude;85total',
        weather='Everyweatherwindow endsdaybeforequery; exact84consecutivephysicaldays, no actualsowing/dateofevent/yearfeature',
        calendar='Common365daymonth/daycalendar;Feb29sharesFeb28; sine/cosineandlatitude/60',
        labels='ThreeobservedBBCH31/51/87passedlabels; sourceinternal85isBBCH87yellowripeness. Labelsnotreadbyfeaturefunction',
        sampling='OriginalfortnightcontextswithinFeb..Aug/observed87+14daywindow; auxiliaryretrospectivephenologypopulation,notoperationalhistoricalvalidation',
        auxiliary_map_timing='GlobCover2009released2010, afterearlyauxiliaryqueries; retrospectivepretrainingonly. Allmap/canopy/weather/phenologylabels precedecurrentdiseasequeries',
        source_transfer='HistoricalDWDweather andregionalcroplandcanopy→currentstation/reanalysisweather andregionalcroplandcanopy; Germanwheatstageobservationsnotcurrentfieldstages',
        missing='Allauxiliarycontextsretainedwithcanopyavailabilityflag. Currentunavailablecanopyqueriesmustretainfrozenweatherdevelopmentgates/baseline; no diseaseobservationremoved',
        current_information='Onlysame-fieldstrictlyearlierdiseasehistory remains permitted. No additionaldiseaselabels fromanyfieldareinputs',
        no_model_fit=True)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for code in codes:
        dest=out/'code_snapshot'/code.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code,dest)
    old=_read_raw(RAW);current=_read_raw(CURRENT_RAW)
    queries=pd.read_csv(REQUEST/'queries.csv');records=json.loads((REQUEST/'admitted_cycles.json').read_text())
    manifest=json.loads((TRAIT/'manifest.json').read_text())
    assert sha(TRAIT/'daily_inputs.npz')==manifest['artifacts']['daily_inputs.npz']
    with np.load(TRAIT/'daily_inputs.npz',allow_pickle=False) as source:weather=source['weather']
    auxiliary=[];aux_available=[];provenance=[]
    for i,row in enumerate(queries.itertuples()):
        record=records[row.cycle_index];query=pd.Timestamp(row.date)
        assert record['PEP_ID']==row.station_id and record['END_DATE']<'2013-01-01'
        assert query==pd.Timestamp(record['SOWING_DATE'])+pd.Timedelta(days=int(row.weather_row-record['row_start']))
        start=int(row.weather_row)-84;end=int(row.weather_row)
        assert record['row_start']<=start<end<=record['row_stop']
        dates=pd.date_range(query-pd.Timedelta(days=84),query-pd.Timedelta(days=1))
        value,names,available=stage_context(weather[start:end],dates,query,record['lat'],old[row.site_id])
        auxiliary.append(value);aux_available.append(available)
        provenance.append(dict(auxiliary_query_index=i,cycle_index=row.cycle_index,station_id=row.station_id,
            query_date=row.date,weather_row_start_inclusive=start,weather_row_stop_exclusive=end,
            weather_end=str(dates[-1].date()),qualified_canopy_available=available))
        if (i+1)%5000==0:print('auxiliary contexts',i+1,flush=True)
    aux=np.asarray(auxiliary,float);aux_available=np.asarray(aux_available,bool)
    np.savez_compressed(out/'auxiliary_predictors.npz',context=aux,canopy_available=aux_available,
        query_ids=np.arange(len(queries)))
    queries.to_csv(out/'auxiliary_queries.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'auxiliary_provenance.csv',index=False)
    del weather
    obs=pd.read_csv(PARENT/'training_observation_ledger.csv',usecols=['global_target_index','field_id','site_id','date'])
    fields=pd.read_csv(DATA/'fields.csv').set_index('field_id')
    daily=pd.read_parquet(DATA/'daily_weather.parquet')
    weather_by_site={site:(group[['tmean_c','tmax_c','tmin_c','precipitation_mm']].to_numpy(float),pd.DatetimeIndex(group.date))
        for site,group in daily.groupby('location_id')}
    values=[];available=[]
    for row in obs.itertuples():
        forcing,dates=weather_by_site[row.site_id]
        value,current_names,use=stage_context(forcing,dates,row.date,fields.loc[row.field_id,'latitude'],current[row.site_id])
        assert current_names==names
        values.append(value);available.append(use)
    values=np.asarray(values,float);available=np.asarray(available,bool)
    with np.load(CURRENT_PREFIX/'predictors.npz') as old_prefix:
        np.testing.assert_array_equal(obs.global_target_index,old_prefix['target_ids'])
        np.testing.assert_array_equal(values[:,:52],old_prefix['canopy'])
        np.testing.assert_array_equal(available,old_prefix['available'])
    assert values.shape==(2304,85) and aux.shape==(len(queries),85)
    assert np.isfinite(values).all() and np.isfinite(aux).all()
    np.savez_compressed(out/'predictors.npz',context=values,canopy_available=available,
        target_ids=obs.global_target_index.to_numpy(int))
    (out/'feature_schema.json').write_text(json.dumps(dict(feature_names=names,features=85,
        predictor_sowing_date=False,predictor_observed_stage=False,predictor_disease_label=False),indent=2)+'\n')
    verification=dict(passed=True,source_unchanged=all(sha(Path(p))==h for p,h in registration['source_sha256'].items()),
        auxiliary_rows=len(queries),auxiliary_stations=queries.station_id.nunique(),auxiliary_canopy_rows=int(aux_available.sum()),
        validation_rows=int(queries.auxiliary_validation_station.sum()),current_rows=len(obs),current_canopy_rows=int(available.sum()),
        current_canopy_features_bit_exact=True,no_model_fit=True,
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k!='artifacts_sha256'},indent=2),flush=True)


if __name__=='__main__':build()
