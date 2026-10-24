"""Create an immutable Irish precipitation-only predictor package; no model fitting."""
from pathlib import Path
import os,sys,json,hashlib,datetime,shutil,subprocess
os.environ['NUMBA_DISABLE_JIT']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
if (OUT/'code_snapshot').exists():sys.path.insert(0,str(OUT/'code_snapshot'))
import numpy as np
import pandas as pd
from analysis.disease_improvement.repairs import hybrid_weather,recorded_leaf_features
from analysis.disease_improvement.features import build_features
from analysis.disease_improvement.stage_features import build_stage_features
from analysis.disease_improvement.stage_experiment import augment
from analysis.disease_improvement.infection_event_features import EVENT_NAMES,build_event_features,insert_events
from analysis.disease_improvement.inputs import sha
BASE=ROOT/'data/disease_improvement/station_preferred_nl_20261022'
CAND=ROOT/'outputs/disease_improvement/knmi_station_candidate_20261022'
EVENT=ROOT/'data/disease_improvement/infection_events_20261022'
QA=ROOT/'outputs/weather_validation_20261022/irish_station_qualification'
META=ROOT/'outputs/disease_improvement/quality_weather_20261021_v2'
CAL=ROOT/'outputs/raw_metadata_recovery_20261022/stage_calendar.csv'
CODE=['analysis/disease_improvement/'+name+'.py' for name in ['__init__','repairs','features','stage_features','stage_experiment','history','infection_event_features','inputs','dwd_station','evaluation','models','robust_models']]+['model/seasonal_septoria/'+name+'.py' for name in ['__init__','core','wetness']]+['analysis/final_study/__init__.py','analysis/final_study/metrics.py']+[str((OUT/name).relative_to(ROOT)) for name in ['prep.py','verify.py']]

def writej(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def source_paths():
 return [BASE/n for n in ['daily_weather.parquet','manifest.json','feature_schema.json']]+[CAND/n for n in ['predictors.npz','training_observation_ledger.csv']]+[EVENT/n for n in ['manifest.json','READY.json','predictors.npz','label_free_queries.csv','fields.csv','history_provenance.json','feature_schema.json']]+[QA/n for n in ['manifest.json','READY.json','eligible_daily.parquet','seasonal_qualification.csv','qualification_policy.json','qualification_status.json']]+[META/'field_metadata_ledger.csv',META/'history_provenance.json',ROOT/'data/disease_improvement/coordinate_augmentation_station_20261022/fields.csv',CAL]
def verify_source_package(folder,key):
 manifest=json.loads((folder/'manifest.json').read_text())
 for name,checksum in manifest[key].items():assert sha(folder/name)==checksum,(folder,name)
 if (folder/'READY.json').exists():assert sha(folder/'manifest.json')==json.loads((folder/'READY.json').read_text())['manifest_sha256']

def load():
 weather=pd.read_parquet(BASE/'daily_weather.parquet');weather.date=pd.to_datetime(weather.date).dt.tz_localize(None)
 queries=pd.read_csv(EVENT/'label_free_queries.csv');assert 'observed_percent' not in queries
 fields=pd.read_csv(EVENT/'fields.csv');assert len(fields)==218 and not fields.field_id.duplicated().any()
 metadata=pd.concat([pd.read_csv(META/'field_metadata_ledger.csv'),pd.read_csv(ROOT/'data/disease_improvement/coordinate_augmentation_station_20261022/fields.csv')],ignore_index=True)
 pd.testing.assert_frame_equal(fields,metadata,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
 old=dict(np.load(CAND/'predictors.npz'));event=dict(np.load(EVENT/'predictors.npz'))
 assert len(queries)==2304 and set(queries.field_id)==set(fields.field_id)
 np.testing.assert_array_equal(old['target_ids'],queries.global_target_index)
 np.testing.assert_array_equal(old['target_ids'],event['target_ids'])
 assert old['history'].tobytes()==event['history'].tobytes()
 np.testing.assert_array_equal(event['history_events'],insert_events(old['station'],event['station_event_features']))
 selected=pd.read_parquet(QA/'eligible_daily.parquet');assert len(selected)==730 and selected.qualified_precipitation.all() and not selected.duplicated(['site_id','date']).any()
 selected.date=pd.to_datetime(selected.date).dt.tz_localize(None)
 qualification=pd.read_csv(QA/'seasonal_qualification.csv');qualified=qualification[qualification.qualified_precipitation]
 assert len(qualified)==5
 affected=queries[queries.field_id.isin(qualified.field_id)].copy();assert len(affected)==56
 positions=pd.Series(np.arange(len(queries)),index=queries.global_target_index).loc[affected.global_target_index].to_numpy()
 hp=pd.read_json(EVENT/'history_provenance.json');same=pd.read_json(META/'history_provenance.json')
 prior=dict(zip(same.global_target_index,same.latest_same_date))
 assert set(affected.global_target_index)<=set(prior)
 one=same.set_index('global_target_index').loc[affected.global_target_index,'latest_same_date']
 two=hp.set_index('global_target_index').loc[affected.global_target_index,'latest_same_date']
 np.testing.assert_array_equal(pd.to_datetime(one,format='mixed').to_numpy(),pd.to_datetime(two,format='mixed').to_numpy())
 config=dict(weather_cutoff_days=1,rain_variable='rain_mm',prior_dates=prior)
 return weather,queries,fields,old,event,selected,affected,positions,hp,pd.read_csv(CAL),config

def rebuild_station(weather,fields,queries,calendar,config):
 if 'observed_percent' in queries:raise ValueError('Query labels must be withheld from station features')
 features,names,provenance=build_features(weather,fields,queries,config)
 stage,stage_names,stage_provenance=build_stage_features(weather,fields,queries,calendar)
 wc=[i for i,n in enumerate(names) if n.startswith('weather_')];pc=[i for i,n in enumerate(names) if n.startswith('process_')]
 category,category_names=recorded_leaf_features(queries)
 design=augment(np.column_stack([features[:,wc],category,features[:,pc]]),stage)
 feature_names=[names[i] for i in wc]+category_names+stage_names+[names[i] for i in pc]
 assert design.shape==(len(queries),53) and len(wc)==12 and len(pc)==3
 return design,feature_names,provenance,stage_provenance

def main():
 if (OUT/'manifest.json').exists() or (OUT/'READY.json').exists():raise FileExistsError('Frozen package immutable; choose a new package')
 verify_source_package(QA,'files_sha256');verify_source_package(EVENT,'output_sha256')
 sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths()};codes={p:sha(ROOT/p) for p in CODE}
 writej('source_lock.json',dict(source_sha256=sources,code_sha256=codes,locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 weather,queries,fields,old,event,selected,affected,positions,hp,calendar,config=load()
 # Required pre-patch gate: all affected baseline designs must reproduce bit-for-bit.
 baseline,names,bp,bsp=rebuild_station(weather,fields,affected,calendar,config)
 baseline_events,event_names,bep=build_event_features(weather,fields,affected,hp)
 np.testing.assert_array_equal(baseline,old['station'][positions]);np.testing.assert_array_equal(baseline_events,event['station_event_features'][positions]);np.testing.assert_array_equal(insert_events(baseline,baseline_events),event['history_events'][positions])
 writej('baseline_reproduction.json',dict(result='PASS',affected_query_rows=len(affected),station53_bit_identical=True,event15_bit_identical=True,history_events68_bit_identical=True,gate_completed_before_weather_replacement=True))
 pairs=selected.rename(columns={'station_measured_total_precipitation_mm':'precipitation_mm_station','rain_hours_gt_0p1mm':'station_precip_hours_gt_0_1mm'}).copy()
 pairs['station_id_RR']=pairs.station_id.astype(str);pairs['distance_km_RR']=pairs.historical_distance_km
 forcing,replacements=hybrid_weather(weather,pairs,set(selected.site_id),source='MET_EIREANN',rainfall=True,meteorology=False)
 # Preserve existing station IDs on all fallback variables/dates.
 join=forcing[['location_id','date']].merge(pairs[['site_id','date','station_id']],left_on=['location_id','date'],right_on=['site_id','date'],how='left',validate='1:1')
 for variable,origin in [('precipitation_mm','MET_EIREANN'),('rain_mm','MET_EIREANN_phase_assisted'),('rain_hours_gt_0_1mm','MET_EIREANN')]:
  mask=forcing['origin_'+variable].eq(origin)
  ids=join.station_id.map(lambda x:str(int(x)) if pd.notna(x) else None)
  forcing.loc[mask,'station_id_'+variable]=ids.loc[mask].to_numpy()
  forcing['MET_EIREANN_station_id_'+variable]=pd.Series(np.where(mask,ids,None),index=forcing.index,dtype='object')
 assert len(forcing)==71965 and int(forcing.MET_EIREANN_total_precip_used.sum())==730
 forcing.to_parquet(OUT/'daily_weather.parquet',index=False);pairs.to_parquet(OUT/'qualified_paired_daily.parquet',index=False);replacements.to_csv(OUT/'replacement_counts.csv',index=False)
 # Include both affected sites' fallback dates so omissions remain explicit.
 cols=['location_id','date','tmin_c','precipitation_mm','rain_mm','rain_hours_gt_0_1mm','origin_precipitation_mm','origin_rain_mm','origin_rain_hours_gt_0_1mm','station_id_precipitation_mm','station_id_rain_mm','station_id_rain_hours_gt_0_1mm','MET_EIREANN_total_precip_used','MET_EIREANN_liquid_phase_assisted','MET_EIREANN_warm_liquid_assumption']
 day=forcing.loc[forcing.location_id.isin(selected.site_id),cols].copy()
 previous=weather[['location_id','date','precipitation_mm','rain_mm','rain_hours_gt_0_1mm']].rename(columns={v:'original_'+v for v in ['precipitation_mm','rain_mm','rain_hours_gt_0_1mm']})
 day=day.merge(previous,on=['location_id','date'],validate='1:1').merge(pairs.drop(columns=['season_year']),left_on=['location_id','date'],right_on=['site_id','date'],how='left',validate='1:1')
 day['rain_hours_fallback']=~day.origin_rain_hours_gt_0_1mm.eq('MET_EIREANN')
 day['liquid_rain_fallback']=~day.origin_rain_mm.eq('MET_EIREANN_phase_assisted')
 day['liquid_interpretation']='Measured station total; liquid rain estimated by existing Tmin>2C assumption or original liquid/total ratio; unsupported phase retains original rain'
 day['rain_hours_interpretation']='Count of complete station hourly precipitation>.1mm used only under existing warm-liquid or zero-total policy; otherwise original hours retained'
 day.to_csv(OUT/'daily_precipitation_provenance.csv',index=False)
 matrix,names,provenance,stage_provenance=rebuild_station(forcing,fields,affected,calendar,config)
 events,event_names,event_provenance=build_event_features(forcing,fields,affected,hp)
 station=old['station'].copy();station[positions]=matrix
 history_events=event['history_events'].copy();history_events[positions]=insert_events(matrix,events)
 other=np.ones(len(queries),bool);other[positions]=False
 assert station[other].tobytes()==old['station'][other].tobytes();assert history_events[other].tobytes()==event['history_events'][other].tobytes()
 # Only precipitation-dependent station means and state columns may change.
 fixed=[i for i,n in enumerate(names) if not ('weather_rain_' in n or n.startswith('process_'))]
 assert station[:,fixed].tobytes()==old['station'][:,fixed].tobytes()
 np.savez_compressed(OUT/'predictors.npz',target_ids=old['target_ids'],station=station,history_events=history_events,history=old['history'],process=old['process'])
 querycols=['global_target_index','field_id','site_id','date']
 for name,p in [('feature_provenance',provenance),('stage_provenance',stage_provenance),('event_provenance',event_provenance)]:
  p=p.merge(affected[querycols],on='global_target_index',validate='1:1')
  p['station_precipitation_interpretation']='Measured total; liquid rain is phase-assisted with original fallback'
  p['eligible_station_days_available']=p.apply(lambda r:int(selected.loc[selected.site_id.eq(r.site_id)&selected.date.between(r.weather_start,r.weather_end)].shape[0]),axis=1)
  p.to_csv(OUT/(name+'.csv'),index=False)
 affected[['global_target_index','field_id','site_id','date','season_year']].to_csv(OUT/'affected_queries.csv',index=False)
 for source,dest in [(CAND/'training_observation_ledger.csv','training_observation_ledger.csv'),(EVENT/'label_free_queries.csv','label_free_queries.csv'),(EVENT/'fields.csv','fields.csv'),(EVENT/'history_provenance.json','history_provenance.json')]:shutil.copyfile(source,OUT/dest)
 writej('feature_schema.json',dict(station_feature_names=names,station_columns=53,event_feature_names=event_names,history_event_feature_names=names[:-3]+event_names+names[-3:],history_events_columns=68,event_slice=[50,65],process_columns=37,history_columns=35,target_rows=2304,affected_rows=56,unchanged_rows=2248,weather_cutoff_days=1,history_anchors='Station uses latest earlier same-leaf date; event features use latest earlier same-leaf date, else latest earlier field date. Full original history provenance retained.',interpretation='Fixed infection proxies; no measured canopy leaf wetness or new stage data'))
 for name in CODE:
  dst=OUT/'code_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
 writej('registration.json',dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sources,code_sha256=codes,station_selection='Frozen source qualification independent of disease outcomes or model error',rain_policy='Existing repairs.hybrid_weather rainfall-only MET_EIREANN; total measured and liquid phase-assisted kept distinct',weather_cutoff_days=1,affected_query_rows=56,only_qualified_field_seasons=True,observed_labels_withheld_from_feature_queries=True,training_observation_ledger_copied_byte_identical=True,original_process37_and_history35_preserved=True,model_fitting_performed=False,publication_model_promoted=False))
 # Read-only verifier operates on serialized data and replays raw-source scalar checks.
 check=subprocess.run([sys.executable,str(OUT/'verify.py'),'--details'],cwd=ROOT,capture_output=True,text=True)
 if check.returncode:raise RuntimeError(check.stderr)
 verification=json.loads(check.stdout)
 pd.DataFrame(verification.pop('handchecks')).to_csv(OUT/'independent_handchecks.csv',index=False)
 pd.DataFrame(verification.pop('perturbations')).to_csv(OUT/'causality_perturbations.csv',index=False)
 writej('verification.json',verification);print(json.dumps(verification,indent=2))
 for p,digest in sources.items():assert sha(ROOT/p)==digest,p
 for p,digest in codes.items():assert sha(ROOT/p)==digest,p
 print(json.dumps(dict(status='PREPARED_VERIFIED_NOT_FROZEN',weather_rows=len(forcing),eligible_site_days=730,affected_query_rows=56,station_changed_rows=int(np.any(station!=old['station'],axis=1).sum()),history_events_changed_rows=int(np.any(history_events!=event['history_events'],axis=1).sum()),predictors_sha256=sha(OUT/'predictors.npz')),indent=2))
if __name__=='__main__':main()
