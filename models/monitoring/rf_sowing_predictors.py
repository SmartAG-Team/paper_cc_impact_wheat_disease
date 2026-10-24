"""Rebuild unchanged predictor schemas using independently estimated sowing dates."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features
from .infection_event_features import build_event_features,insert_events
from .inputs import WEATHER,sha
from .repairs import recorded_leaf_features
from .stage_features import build_stage_features
from .stage_experiment import augment

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/disease_improvement/rf_sowing_20261022'


def designs(raw,station,fields,queries,calendar,hp):
    if 'observed_percent' in queries:
        raise ValueError('Disease labels must be withheld from query features')
    config=dict(weather_cutoff_days=1,rain_variable='rain_mm',
        prior_dates=dict(zip(hp.global_target_index,hp.latest_same_date)))
    category,_=recorded_leaf_features(queries)
    process,first,ledger=None,None,{}
    for label,weather in [('process',raw),('station',station)]:
        x,names,p=build_features(weather,fields,queries,config)
        wc=[i for i,n in enumerate(names) if n.startswith('weather_')]
        pc=[i for i,n in enumerate(names) if n.startswith('process_')]
        base=np.column_stack([x[:,wc],category,x[:,pc]])
        assert base.shape==(len(queries),37)
        ledger[label]=p
        if label=='process':
            process=base
        else:
            stage,stage_names,sp=build_stage_features(weather,fields,queries,calendar)
            first=augment(base,stage);ledger['stage']=sp
    extra,event_names,ep=build_event_features(station,fields,queries,hp)
    ledger['event']=ep
    return dict(process=process,station=first,history_events=insert_events(first,extra)),ledger


def run():
    if OUT.exists():
        raise FileExistsError('A fresh predictor destination is required')
    parent=ROOT/'data/disease_improvement/station_preferred_ie_20261022'
    extension=ROOT/'data/disease_improvement/sowing_weather_extension_20261022'
    rf=ROOT/'outputs/disease_improvement/sowing_rf_20261022'
    fields=pd.read_csv(parent/'fields.csv')
    query=pd.read_csv(parent/'label_free_queries.csv')
    hp=pd.read_json(parent/'history_provenance.json')
    obs=pd.read_csv(parent/'training_observation_ledger.csv')
    original=dict(np.load(parent/'predictors.npz'))
    assert len(fields)==218 and fields.field_id.is_unique and len(query)==2304
    np.testing.assert_array_equal(query.global_target_index,original['target_ids'])
    np.testing.assert_array_equal(obs.global_target_index,original['target_ids'])
    calendar_files=[ROOT/'outputs/raw_metadata_recovery_20261022/stage_calendar.csv',
        ROOT/'data/disease_improvement/coordinate_augmentation_station_20261022/stage_calendar.csv']
    calendar=pd.concat([pd.read_csv(p) for p in calendar_files],ignore_index=True)
    assert not calendar.duplicated(['field_id','date']).any()
    raw=pd.read_parquet(ROOT/WEATHER)
    added_file=ROOT/'data/disease_improvement/coordinate_augmentation_20261022/daily_weather.parquet'
    added=pd.read_parquet(added_file)
    raw=pd.concat([raw,added.loc[~added.location_id.isin(raw.location_id)]],ignore_index=True)
    raw['date']=pd.to_datetime(raw.date)
    station=pd.read_parquet(parent/'daily_weather.parquet');station['date']=pd.to_datetime(station.date)
    predicted=pd.read_csv(rf/'field_estimated_sowing.csv').set_index('field_id')
    assert set(predicted.index)==set(fields.field_id)
    sources=[parent/n for n in ['fields.csv','label_free_queries.csv','history_provenance.json','training_observation_ledger.csv',
        'predictors.npz','daily_weather.parquet','manifest.json']]
    sources += [rf/n for n in ['field_estimated_sowing.csv','fitted_sowing_rf.joblib','registration.json','sowing_validation_metrics.csv']]
    sources += [extension/n for n in ['raw_era5_daily.parquet','daily_weather_extension.parquet','verification.json']]
    sources += [ROOT/WEATHER,added_file]+calendar_files
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['features','infection_event_features',
        'repairs','stage_features','stage_experiment','inputs']]
    codes += [ROOT/'model/seasonal_septoria'/n for n in ['core.py','wetness.py']]
    OUT.mkdir(parents=True)
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        change='Only estimated sowing dates and the17previously unsupported newlyrequiredweatherdays; all input schemas and diseasehistory preserved',
        auxiliary_model='Fixed pre2013RF, exactly annualmeanT+latitude, observedDWDsowinglabels1986–2012',
        sowing_is_estimated=True,disease_outcomes_used_to_estimate_sowing=False,
        station_preference='Existing qualified station forcing plus independently qualified newdays; explicit per-variable/day fallback',
        original_process_weather='OriginalERA5forcing; causal querycutoffday1',
        baseline_gate='All original2304process37/station53/event68 predictors reproduce exactly before sowing replacement',
        evaluation='User-authorized current retained dateduntreated cohort with original chronology and50km diseasefitexclusions; auxiliarysowinggeographictransferseparatelytested',
        eppo_alignment='Leafordinal=index+1; BASFflagverified, Cortevaflagrelativeconventionassumed. Nativecodes and numericseverityunchanged.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes})
    (OUT/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for p in codes:
        dest=OUT/'code_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    baseline,_=designs(raw,station,fields,query,calendar,hp)
    for key,matrix in baseline.items():
        np.testing.assert_array_equal(matrix,original[key])
    (OUT/'baseline_reproduction.json').write_text(json.dumps(dict(verified=True,rows=2304,
        process37_exact=True,station53_exact=True,event68_exact=True),indent=2)+'\n')
    newfields=fields.copy()
    newfields['previous_scenario_sowing_date']=newfields.sowing_date
    newfields['sowing_date']=newfields.field_id.map(predicted.estimated_sowing_date)
    newfields['sowing_basis']='rf_estimated_not_observed'
    newfields['first_forcing_date']=newfields.sowing_date
    newfields['forcing_days']=(pd.to_datetime(newfields.last_forcing_date)-pd.to_datetime(newfields.sowing_date)).dt.days+1
    for key in ['annual_temperature_source','annual_mean_temperature_c','annual_temperature_year',
                'outside_latitude_support','outside_temperature_support','outside_german_source_country']:
        newfields[key]=newfields.field_id.map(predicted[key])
    query['sowing_date']=query.field_id.map(predicted.estimated_sowing_date)
    raw_extra=pd.read_parquet(extension/'raw_era5_daily.parquet');raw_extra['date']=pd.to_datetime(raw_extra.date)
    station_extra=pd.read_parquet(extension/'daily_weather_extension.parquet');station_extra['date']=pd.to_datetime(station_extra.date)
    assert len(raw_extra)==len(station_extra)==17
    raw=pd.concat([raw,raw_extra],ignore_index=True).sort_values(['location_id','date']).reset_index(drop=True)
    station=pd.concat([station,station_extra],ignore_index=True).sort_values(['location_id','date']).reset_index(drop=True)
    assert not raw.duplicated(['location_id','date']).any() and not station.duplicated(['location_id','date']).any()
    rebuilt,provenance=designs(raw,station,newfields,query,calendar,hp)
    newfields.to_csv(OUT/'fields.csv',index=False);query.to_csv(OUT/'label_free_queries.csv',index=False)
    obs.to_csv(OUT/'training_observation_ledger.csv',index=False)
    calendar.to_csv(OUT/'stage_calendar.csv',index=False)
    (OUT/'history_provenance.json').write_bytes((parent/'history_provenance.json').read_bytes())
    raw.to_parquet(OUT/'raw_daily_weather.parquet',index=False)
    station.to_parquet(OUT/'station_daily_weather.parquet',index=False)
    for key,value in provenance.items():
        value.to_csv(OUT/(key+'_provenance.csv'),index=False)
    np.savez_compressed(OUT/'predictors.npz',target_ids=original['target_ids'],history=original['history'],**rebuilt)
    convention=obs[['global_target_index','source','leaf_index']].copy()
    convention['eppo_leaf_ordinal']=convention.leaf_index+1
    convention['flag_relative_offset_under_convention']=convention.leaf_index
    convention['flag_alignment_status']=np.where(convention.source.eq('BASF'),'source_flag_label_verified','user_authorized_EPPO_alignment_assumption')
    convention['cultivar_resistance']='unreported_not_imputed'
    convention.to_csv(OUT/'leaf_scoring_convention.csv',index=False)
    for path,digest in registration['source_sha256'].items():
        assert sha(ROOT/path)==digest
    result=dict(prepared=True,rows=2304,fields=218,added_weather_days=17,
        changed_rows={k:int(np.any(v!=original[k],axis=1).sum()) for k,v in rebuilt.items()},
        history35_bit_identical=True,original_target_ids_identical=True,
        original_observed_sowing_remains_missing=True,independent_external_validation=False)
    (OUT/'preparation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
