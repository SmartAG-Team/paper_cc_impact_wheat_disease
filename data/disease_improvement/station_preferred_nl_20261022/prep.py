"""Immutable KNMI precipitation-only weather and station-design extension."""
from pathlib import Path
import sys,json,datetime,shutil,subprocess,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from analysis.disease_improvement.inputs import sha
from analysis.disease_improvement.repairs import hybrid_weather,recorded_leaf_features
from analysis.disease_improvement.features import build_features
from analysis.disease_improvement.stage_features import build_stage_features
from analysis.disease_improvement.stage_experiment import augment
OUT=Path(__file__).resolve().parent
BASE=ROOT/'data/disease_improvement/station_preferred_augmented_20261022'
CAND=ROOT/'outputs/disease_improvement/station_candidate_20261022_v2'
META=ROOT/'outputs/disease_improvement/quality_weather_20261021_v2'
QA=ROOT/'outputs/weather_validation_20261022/knmi_season_qualification'
SITE='era5loc_c36c557e4afc'
def rebuild(weather,fields,queries,calendar,config):
 events,names,provenance=build_features(weather,fields,queries,config)
 stage,stage_names,stage_provenance=build_stage_features(weather,fields,queries,calendar)
 wc=[i for i,n in enumerate(names) if n.startswith('weather_')];pc=[i for i,n in enumerate(names) if n.startswith('process_')]
 assert len(wc)==12 and len(pc)==3
 category,category_names=recorded_leaf_features(queries)
 design=augment(np.column_stack([events[:,wc],category,events[:,pc]]),stage)
 feature_names=[names[i] for i in wc]+category_names+stage_names+[names[i] for i in pc]
 assert design.shape==(len(queries),53) and len(feature_names)==53
 return design,feature_names,provenance,stage_provenance

def main():
 if (OUT/'manifest.json').exists():raise FileExistsError('Completed dataset immutable; use new version')
 weather=pd.read_parquet(BASE/'daily_weather.parquet')
 obs=pd.read_csv(CAND/'training_observation_ledger.csv');old=np.load(CAND/'predictors.npz')
 np.testing.assert_array_equal(old['target_ids'],obs.global_target_index)
 fields=pd.read_csv(META/'field_metadata_ledger.csv')
 nlobs=obs.loc[obs.site_id.eq(SITE)&obs.field_id.isin(fields.field_id)].drop(columns='observed_percent').copy()
 assert len(nlobs)==19 and nlobs.source.eq('BASF').all()
 calendar=pd.read_csv(ROOT/'outputs/raw_metadata_recovery_20261022/stage_calendar.csv')
 prior=pd.read_json(META/'history_provenance.json')
 config=dict(weather_cutoff_days=1,rain_variable='rain_mm',prior_dates=dict(zip(prior.global_target_index,prior.latest_same_date)))
 selected=pd.read_parquet(QA/'qualified_selected_daily_precipitation.parquet')
 assert len(selected)==487 and selected.station_id.eq(280).all() and selected.qualified_precipitation.all()
 assert selected.complete_utc_precip_day.all()
 qa_manifest=json.loads((QA/'manifest.json').read_text())
 for file,checksum in qa_manifest['files_sha256'].items():assert sha(QA/file)==checksum
 pairs=selected[['site_id','date','station_id','station_measured_total_precip_mm','rainy_hour_count_gt0p1mm']].rename(columns={'station_measured_total_precip_mm':'precipitation_mm_station','rainy_hour_count_gt0p1mm':'station_precip_hours_gt_0_1mm'})
 assert not pairs.duplicated(['site_id','date']).any()
 pairs['station_id_RR']=pairs.station_id.astype(str);pairs['distance_km_RR']=selected.historical_distance_km.to_numpy()
 pairs.to_parquet(OUT/'qualified_paired_daily.parquet',index=False)
 forcing,ledger=hybrid_weather(weather,pairs,{SITE},source='KNMI',rainfall=True,meteorology=False)
 # Existing IDs remain unchanged; annotate only variables actually sourced from KNMI.
 for variable,origin in [('precipitation_mm','KNMI'),('rain_mm','KNMI_phase_assisted'),('rain_hours_gt_0_1mm','KNMI')]:
  mask=forcing['origin_'+variable].eq(origin);forcing.loc[mask,'station_id_'+variable]='280'
  forcing['KNMI_station_id_'+variable]=pd.Series(np.where(mask,'280',None),index=forcing.index,dtype='object')
 forcing.to_parquet(OUT/'daily_weather.parquet',index=False)
 ledger.to_csv(OUT/'replacement_counts.csv',index=False)
 # Per-day measurement/phase/fallback provenance, including unreplaced NL dates.
 day=forcing.loc[forcing.location_id.eq(SITE),['location_id','date','tmin_c','precipitation_mm','rain_mm','rain_hours_gt_0_1mm','origin_precipitation_mm','origin_rain_mm','origin_rain_hours_gt_0_1mm','station_id_precipitation_mm','station_id_rain_mm','station_id_rain_hours_gt_0_1mm','KNMI_total_precip_used','KNMI_liquid_phase_assisted','KNMI_warm_liquid_assumption']].copy()
 before=weather.loc[weather.location_id.eq(SITE),['date','precipitation_mm','rain_mm','rain_hours_gt_0_1mm']].rename(columns={n:'original_'+n for n in ['precipitation_mm','rain_mm','rain_hours_gt_0_1mm']})
 day=day.merge(before,on='date',validate='1:1').merge(pairs.drop(columns='site_id'),on='date',how='left',validate='1:1')
 day['liquid_interpretation']='Station total is measured; rain_mm uses existing warm-Tmin or original reanalysis phase-ratio estimate; unavailable phase retains original rain_mm'
 day.to_csv(OUT/'daily_precipitation_provenance.csv',index=False)
 oldnl,names,_,_=rebuild(weather,fields,nlobs,calendar,config)
 positions=pd.Series(np.arange(len(obs)),index=obs.global_target_index).loc[nlobs.global_target_index].to_numpy()
 np.testing.assert_array_equal(oldnl,old['station'][positions])
 newnl,names,provenance,stage_provenance=rebuild(forcing,fields,nlobs,calendar,config)
 station=old['station'].copy();station[positions]=newnl
 other=np.ones(len(obs),bool);other[positions]=False
 assert station[other].tobytes()==old['station'][other].tobytes()
 np.savez_compressed(OUT/'predictors.npz',target_ids=old['target_ids'],station=station)
 provenance=provenance.merge(nlobs[['global_target_index','field_id','site_id','date']],on='global_target_index',validate='1:1')
 provenance['station_id']='280';provenance['station_precipitation_interpretation']='Measured total; rain_mm phase-assisted estimate with fallback'
 provenance['station_precip_complete_days_available']=provenance.apply(lambda r:int(selected.loc[selected.field_id.eq(r.field_id)&selected.date.between(r.weather_start,r.weather_end)].shape[0]),axis=1)
 provenance.to_csv(OUT/'feature_provenance.csv',index=False)
 stage_provenance.to_csv(OUT/'stage_provenance.csv',index=False)
 prior.loc[prior.global_target_index.isin(nlobs.global_target_index)].to_json(OUT/'prior_history_provenance.json',orient='records',indent=2)
 (OUT/'feature_schema.json').write_text(json.dumps({'station_feature_names':names,'station_columns':53,'all_target_rows':2304,'patched_NL_rows':len(nlobs),'unchanged_other_rows':int(other.sum()),'design_rule':'12weather +22recorded-code +16stage/clock +3process-state','history_and_process_component':'Unchanged source arrays; only station design patched'},indent=2))
 shutil.copyfile(CAND/'training_observation_ledger.csv',OUT/'training_observation_ledger.csv')
 counts=forcing.groupby('location_id').agg(days=('date','size'),first_day=('date','min'),last_day=('date','max'))
 for name in ['precipitation_mm','rain_mm','rain_hours_gt_0_1mm','tmean_c','rh_mean_pct']:
  counts=counts.join(pd.crosstab(forcing.location_id,forcing['origin_'+name]).add_prefix(name+':'))
 counts['active_training_site']=counts.index.isin(obs.site_id);counts.reset_index().to_csv(OUT/'replacement_coverage.csv',index=False)
 sources=[BASE/'daily_weather.parquet',BASE/'manifest.json',CAND/'predictors.npz',CAND/'training_observation_ledger.csv',META/'field_metadata_ledger.csv',META/'history_provenance.json',META/'feature_schema.json',ROOT/'outputs/raw_metadata_recovery_20261022/stage_calendar.csv',QA/'manifest.json',QA/'qualification_status.json',QA/'qualified_selected_daily_precipitation.parquet',QA/'project_season_periods.csv',QA/'qualification_policy.json',QA/'verification_receipt.json']
 codes=[ROOT/'analysis/disease_improvement'/f'{n}.py' for n in ['repairs','features','stage_features','stage_experiment']]+[ROOT/'model/seasonal_septoria'/f'{n}.py' for n in ['core','wetness']]
 for file in codes:
  dest=OUT/'code_snapshot'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
 registration=dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},policy='QualifiedKNMI rain-only addition; no outcome-based selection; existing hybrid phase policy; no T/RH replacement; strictly before each assessment',model_fitting=False,external_disease_outcomes=False,blend_process_component_unchanged=True,history_arrays_unchanged=True,observations_unchanged=True,station_design_patched_only_NL=True,station_feature_source='OriginalNL observations with source-native recorded codes and existing earlier-only stage/history provenance')
 (OUT/'registration.json').write_text(json.dumps(registration,indent=2))
 # Separate independent verifier reads serialized output and original raw sources.
 subprocess.run([sys.executable,str(OUT/'verify.py')],cwd=ROOT,check=True)
 manifest=dict(weather_rows=len(forcing),weather_locations=int(forcing.location_id.nunique()),active_training_sites=int(obs.site_id.nunique()),active_station_precipitation_sites=int(forcing.loc[forcing.location_id.isin(obs.site_id)&forcing.origin_precipitation_mm.ne('reanalysis'),'location_id'].nunique()),precipitation_day_origins=forcing.origin_precipitation_mm.value_counts().to_dict(),qualified_KNMI_total_days=int(forcing.KNMI_total_precip_used.sum()),prediction_shape=list(station.shape),patched_target_ids=nlobs.global_target_index.astype(int).tolist(),other_target_rows_bit_identical=int(other.sum()),feature_schema_sha256=sha(OUT/'feature_schema.json'),source_sha256=registration['source_sha256'],code_sha256=registration['code_sha256'],liquid_precipitation='Explicitly phase-assisted; station total never relabeled as directly measured liquid rain',meteorology='Original temperature/humidity and all other nonprecipitation numerical variables unchanged',weather_cutoff_days=1,external_disease_outcomes_used=False,model_fitting_performed=False,frozen_inputs_modified=False,process_array_source_sha256=hashlib.sha256(old['process'].tobytes()).hexdigest(),history_array_source_sha256=hashlib.sha256(old['history'].tobytes()).hexdigest(),source_observation_ledger_copied_byte_identical=True)
 manifest['output_sha256']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in str(p)}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:manifest[k] for k in ['weather_rows','weather_locations','active_training_sites','active_station_precipitation_sites','qualified_KNMI_total_days','prediction_shape','patched_target_ids','other_target_rows_bit_identical']},indent=2))
if __name__=='__main__':main()
