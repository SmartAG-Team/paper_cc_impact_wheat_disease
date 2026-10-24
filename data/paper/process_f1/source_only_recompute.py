"""Immutable-source F1 reaggregation and unchanged-climate-equation replay."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import pandas as pd
import numpy as np

ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent/'models').is_dir())
OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from analysis.final_study.metrics import hierarchical_weights,observed_windows
from analysis.disease_improvement.climate_prefix import simulate_prefix
from analysis.methods.seasonal_model.disease.run import predict_record
from model.seasonal_septoria.regional import tpv_accumulation

SOURCES={
 'archived_assessment_predictions':'data/final_study/evidence/phenology_disease_coupling/results/assessment_predictions.csv',
 'qualified_F1_ids':'data/final_monitoring/qualified_flag_endpoint_ids.csv',
 'field_registry':'data/final_study/evidence/field_stage_extension/results/field_registry.csv',
 'weather':'data/final_study/inputs/paper_study/field_weather_completed/daily_weather.parquet',
 'calendar_membership':'data/final_study/evidence/seasonal_calibration_v1/field_season_membership.csv',
 'raw_BASF_snapshot':'data/final_study/evidence/seasonal_calibration_v1/basf_source_assessments_snapshot.csv',
 'original_evaluation_onsets':'data/final_study/evidence/nature_food_impact/derived/pooled_symptom_onset_membership.csv',
 'phenology':'data/final_climate/phenology_parameters.json',
 'disease':'data/final_climate/disease_parameters.json',
 'thresholds':'data/final_climate/stage_thresholds.json',
 'climate_adapter':'analysis/disease_improvement/climate_prefix.py',
 'field_predictor':'analysis/methods/seasonal_model/disease/run.py',
 'frozen_kernel':'analysis/final_climate/kernel.py',
 'disease_equations':'model/seasonal_septoria/overwinter.py',
 'host_equations':'model/seasonal_septoria/leaf_phenology.py',
 'phenology_equations':'model/seasonal_septoria/regional.py',
 'wetness_equations':'model/seasonal_septoria/wetness.py'}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(df,pred,truth='observed_percent',weight=None):
 w=hierarchical_weights(df) if weight is None else df[weight].to_numpy(float)
 o=df[truth].to_numpy(float);p=df[pred].to_numpy(float);err=p-o
 v=np.average((o-np.average(o,weights=w))**2,weights=w)
 return dict(records=len(df),fields=int(df.field_id.nunique()),sites=int(df.site_id.nunique()),
  coordinate_years=int(df.coordinate_year.nunique()),weight_sum=float(w.sum()),
  rmse=float(np.sqrt(np.average(err**2,weights=w))),mae=float(np.average(abs(err),weights=w)),
  bias=float(np.average(err,weights=w)),r2=float(1-np.average(err**2,weights=w)/v) if v>0 else None,
  observed_mean=float(np.average(o,weights=w)),prediction_mean=float(np.average(p,weights=w)))

raw=pd.read_csv(ROOT/SOURCES['archived_assessment_predictions'])
q=pd.read_csv(ROOT/SOURCES['qualified_F1_ids']).rename(columns={'global_target_index':'qualified_current_global_target_index'})
raw['archive_full_population_hierarchy_weight']=hierarchical_weights(raw)
raw['original_evaluation_hierarchy_weight']=np.nan
evaluation=~raw.partition.eq('calibration')
raw.loc[evaluation,'original_evaluation_hierarchy_weight']=hierarchical_weights(raw[evaluation])
raw['original_calibration_hierarchy_weight']=np.nan
raw.loc[~evaluation,'original_calibration_hierarchy_weight']=hierarchical_weights(raw[~evaluation])
flag=raw[raw.source.eq('BASF')&raw.leaf_index.eq(0)].copy()
flag=q.merge(flag,on=['field_id','date'],how='left',validate='one_to_one',indicator=True)
assert flag._merge.eq('both').all() and len(flag)==115
flag=flag.drop(columns='_merge')
flag['native_mapping']='BASF native code0; source organ LEAF, 1ST / FLAG LEAF; fixed anatomical F1 only'
assert flag.organ.str.contains('FLAG LEAF',regex=False).all()
assert flag.endpoint_series.str.contains('FLAG LEAF',regex=False).all()
flag['F1_only_hierarchy_weight']=hierarchical_weights(flag)
flag['F1_partition_hierarchy_weight']=np.nan
for partition,idx in flag.groupby('partition').groups.items():
 flag.loc[idx,'F1_partition_hierarchy_weight']=hierarchical_weights(flag.loc[idx])

registry=pd.read_csv(ROOT/SOURCES['field_registry'])
registry=registry[registry.field_id.isin(flag.field_id)].copy()
assert len(registry)==55 and not registry.field_id.duplicated().any()
weather=pd.read_parquet(ROOT/SOURCES['weather']);weather['date']=pd.to_datetime(weather.date)
phen=json.loads((ROOT/SOURCES['phenology']).read_text())
fit=json.loads((ROOT/SOURCES['disease']).read_text())['fitted']
thresholds={int(k):v for k,v in json.loads((ROOT/SOURCES['thresholds']).read_text())['all_stage_thresholds'].items()}
fields=('tmean_c','tmax_c','rh_mean_pct','precipitation_mm')
replay=[];checks=[];forcing_rows=[]
for i,meta in enumerate(registry.itertuples()):
 dates=pd.date_range(meta.sowing_date,meta.last_forcing_date)
 assert len(dates)==meta.forcing_days
 w=weather[weather.location_id.eq(meta.site_id)&weather.date.between(dates[0],dates[-1])].sort_values('date')
 assert pd.DatetimeIndex(w.date).equals(dates)
 assert np.isfinite(w[list(fields)]).all().all()
 forcing_rows.append(w[['location_id','date',*fields]].assign(field_id=meta.field_id,sowing_date=meta.sowing_date))
 forcing={k:w[k].to_numpy(float)[None,:] for k in fields}
 accumulation=tpv_accumulation(forcing['tmean_c'],forcing['tmax_c'],[meta.latitude],dates.dayofyear,np.ones((1,len(dates)),bool),phen)
 data=SimpleNamespace(temperature=forcing['tmean_c'],maximum_temperature=forcing['tmax_c'],
  humidity=forcing['rh_mean_pct'],rain=forcing['precipitation_mm'],
  metadata=pd.DataFrame({'field_index':[0],'forcing_days':[len(dates)]}),
  targets=pd.DataFrame({'value':[np.nan]}))
 original,host=predict_record(data,accumulation,thresholds,fit)
 adapter=simulate_prefix(dates,forcing,np.array([meta.sowing_date],dtype='datetime64[D]'),np.array([meta.latitude]),phen,fit,thresholds)
 part=flag[flag.field_id.eq(meta.field_id)]
 max_archived_error=0.;max_query_prefix_error=0.
 for r in part.itertuples():
  query=(pd.Timestamp(r.date)-dates[0]).days+1
  assert query==r.day_index
  # The state at index q is after processing the qth day (assessment date).
  # Index q-1 is the state after weather through the preceding date.
  qend=simulate_prefix(dates[:query],{k:v[:,:query] for k,v in forcing.items()},np.array([meta.sowing_date],dtype='datetime64[D]'),np.array([meta.latitude]),phen,fit,thresholds)
  qprev=simulate_prefix(dates[:query-1],{k:v[:,:query-1] for k,v in forcing.items()},np.array([meta.sowing_date],dtype='datetime64[D]'),np.array([meta.latitude]),phen,fit,thresholds)
  state_error=np.max(abs(qend['trajectory'].state-adapter['trajectory'].state[:,:query+1]))
  prev_error=np.max(abs(qprev['trajectory'].state-adapter['trajectory'].state[:,:query]))
  max_query_prefix_error=max(max_query_prefix_error,float(state_error),float(prev_error))
  archived_error=abs(100*original.damage[0,query,0]-r.raw_retained)
  max_archived_error=max(max_archived_error,float(archived_error))
  replay.append(dict(field_id=meta.field_id,date=r.date,archive_global_target_index=int(r.global_target_index),
   qualified_current_global_target_index=int(r.qualified_current_global_target_index),query_day_index=int(query),
   sowing_date=meta.sowing_date,latitude=float(meta.latitude),site_id=meta.site_id,
   original_field_day_end_percent=float(100*original.damage[0,query,0]),
   original_field_previous_day_percent=float(100*original.damage[0,query-1,0]),
   climate_prefix_day_end_percent=float(100*qend['trajectory'].damage[0,-1,0]),
   climate_prefix_previous_day_percent=float(100*qprev['trajectory'].damage[0,-1,0]),
   climate_prefix_day_end_status=str(qend['status'][0]),climate_prefix_previous_day_status=str(qprev['status'][0]),
   terminal_reached_by_query=bool(qend['complete'][0]),
   original_vs_archive_abs_error_pp=float(archived_error),
   query_prefix_vs_full_prefix_state_max_abs_error=float(max(state_error,prev_error)),
   stage85_day_index=int(adapter['host'].stage_day_index[85][0])))
 checks.append(dict(field_id=meta.field_id,forcing_days=int(len(dates)),flag_assessments=len(part),
  complete_by_last_field_assessment=bool(adapter['complete'][0]),
  maximum_original_field_vs_archive_abs_error_pp=max_archived_error,
  maximum_query_prefix_vs_full_prefix_state_error=max_query_prefix_error,
  original_vs_adapter_state_max_abs_error=float(np.max(abs(original.state-adapter['trajectory'].state)))))
 if (i+1)%10==0: print('replayed',i+1,'of',len(registry),'fields',flush=True)
replay=pd.DataFrame(replay);checks=pd.DataFrame(checks)
assert len(replay)==115 and replay.original_vs_archive_abs_error_pp.max()<1e-10
assert replay.query_prefix_vs_full_prefix_state_max_abs_error.max()==0
flag=flag.merge(replay.drop(columns=['qualified_current_global_target_index','site_id']),on=['field_id','date'],validate='one_to_one')
models=['raw_retained','leaf_mean','original_field_day_end_percent','original_field_previous_day_percent','climate_prefix_day_end_percent','climate_prefix_previous_day_percent']
parent_windows=observed_windows(raw,['raw_retained','leaf_mean'])
parent_windows['archive_full_window_hierarchy_weight']=hierarchical_weights(parent_windows)
parent_windows['original_evaluation_window_hierarchy_weight']=np.nan
parent_windows['original_calibration_window_hierarchy_weight']=np.nan
weval=~parent_windows.partition.eq('calibration')
parent_windows.loc[weval,'original_evaluation_window_hierarchy_weight']=hierarchical_weights(parent_windows[weval])
parent_windows.loc[~weval,'original_calibration_window_hierarchy_weight']=hierarchical_weights(parent_windows[~weval])
window_keys=['partition','field_id','coordinate_year','source','leaf_index','endpoint_series']
window_mass_cols=['archive_full_window_hierarchy_weight','original_evaluation_window_hierarchy_weight','original_calibration_window_hierarchy_weight']
allmetrics=[];allwindows=[]
for scope,df in [('all_including_calibration',flag),('calibration',flag[flag.partition.eq('calibration')]),('original_BASF2019_evaluation',flag[flag.partition.eq('reused_BASF2019')])]:
 parent_weight='original_evaluation_hierarchy_weight' if scope=='original_BASF2019_evaluation' else 'original_calibration_hierarchy_weight' if scope=='calibration' else 'archive_full_population_hierarchy_weight'
 for m in models:
  allmetrics.append(dict(population=scope,endpoint='assessment',weighting='F1-only conditional hierarchy',model=m,**score(df,m)))
  allmetrics.append(dict(population=scope,endpoint='assessment',weighting='full-parent relative hierarchy retained within F1',model=m,**score(df,m,weight=parent_weight)))
  severe=df[df.observed_percent.gt(75)]
  if len(severe):
   allmetrics.append(dict(population=scope,endpoint='severity_gt75_assessment',weighting='full-parent relative hierarchy retained within severe F1',model=m,**score(severe,m,weight=parent_weight)))
 wins=observed_windows(df,models).assign(population=scope)
 wins=wins.merge(parent_windows[window_keys+window_mass_cols],on=window_keys,validate='one_to_one')
 allwindows.append(wins)
 parent_window_weight='original_evaluation_window_hierarchy_weight' if scope=='original_BASF2019_evaluation' else 'original_calibration_window_hierarchy_weight' if scope=='calibration' else 'archive_full_window_hierarchy_weight'
 for m in models:
  allmetrics.append(dict(population=scope,endpoint='window_mean',weighting='F1-window conditional hierarchy',model=m,**score(wins,m)))
  allmetrics.append(dict(population=scope,endpoint='window_audpc',weighting='F1-window conditional hierarchy',model=m,**score(wins,m+'_audpc',truth='observed_audpc')))
  allmetrics.append(dict(population=scope,endpoint='window_mean',weighting='full-parent window relative hierarchy retained within F1',model=m,**score(wins,m,weight=parent_window_weight)))
  allmetrics.append(dict(population=scope,endpoint='window_audpc',weighting='full-parent window relative hierarchy retained within F1',model=m,**score(wins,m+'_audpc',truth='observed_audpc',weight=parent_window_weight)))
 onset=pd.read_csv(ROOT/SOURCES['original_evaluation_onsets']);onset=onset[onset.source.eq('BASF')&onset.leaf_index.eq(0)]
onset.to_csv(OUT/'original2019_F1_two_sided_onsets.csv',index=False)
flag.to_csv(OUT/'all115_F1_records_with_replay.csv',index=False)
flag[flag.partition.eq('reused_BASF2019')].to_csv(OUT/'evaluation70_F1_records_with_replay.csv',index=False)
flag[flag.partition.eq('calibration')].to_csv(OUT/'calibration45_F1_records_with_replay.csv',index=False)
windows=pd.concat(allwindows,ignore_index=True)
windows.to_csv(OUT/'F1_observed_windows.csv',index=False)
windows[windows.population.eq('original_BASF2019_evaluation')].to_csv(OUT/'evaluation23_F1_windows.csv',index=False)
pd.DataFrame(allmetrics).to_csv(OUT/'metrics.csv',index=False)
registry.to_csv(OUT/'field55_calendar_membership.csv',index=False)
pd.concat(forcing_rows,ignore_index=True).to_csv(OUT/'four_weather_forcing.csv',index=False)
checks.to_csv(OUT/'replay_field_checks.csv',index=False)
checks_summary=dict(qualified_current115_all_matched=True,qualified_mapping_BASF_native0_only=True,
 exact_field_date_identity=True,original_vs_archive_max_abs_error_pp=float(replay.original_vs_archive_abs_error_pp.max()),
 truncated_query_vs_full_prefix_max_state_error=float(replay.query_prefix_vs_full_prefix_state_max_abs_error.max()),
 fields=len(registry),records=len(flag),evaluation_records=70,calibration_records=45,evaluation_windows=23,
 severe_above75_evaluation_records=int((flag.partition.eq('reused_BASF2019')&flag.observed_percent.gt(75)).sum()),
 severe_above75_calibration_records=int((flag.partition.eq('calibration')&flag.observed_percent.gt(75)).sum()),
 original_evaluation_total_hierarchy_mass=float(raw.loc[evaluation,'original_evaluation_hierarchy_weight'].sum()),
 F1_evaluation_mass_within_original_parent=float(flag.loc[flag.partition.eq('reused_BASF2019'),'original_evaluation_hierarchy_weight'].sum()),
 severe_F1_evaluation_mass_within_original_parent=float(flag.loc[flag.partition.eq('reused_BASF2019')&flag.observed_percent.gt(75),'original_evaluation_hierarchy_weight'].sum()),
 records_terminal_reached_by_query=int(replay.terminal_reached_by_query.sum()),
 maximum_original_vs_adapter_state_error=float(checks.original_vs_adapter_state_max_abs_error.max()),
 disease_parameters_modified=False,phenology_parameters_modified=False,fit_performed=False,new_outcomes=False,
 source_only_reused_retrospective=True,
 timing='day_end uses through-query-date weather and state[q]; previous_day uses through-query-minus-1-date weather and state[q-1]; independent endpoints must not be pooled',
 scope='F1-only conditional-population metrics and full-parent-relative-mass sensitivity are explicitly separate; no physiological HAD/yield validation')
(OUT/'checks.json').write_text(json.dumps(checks_summary,indent=2)+'\n')
manifest=dict(root=str(ROOT),source_hashes={name:dict(path=p,sha256=digest(ROOT/p)) for name,p in SOURCES.items()},
 output_hashes={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'},checks=checks_summary)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(checks_summary,indent=2),flush=True)
