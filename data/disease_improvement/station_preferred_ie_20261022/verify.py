"""Read-only serialized weather/predictor verification, scalar references and leakage tests."""
from pathlib import Path
import os,sys,json,csv,datetime as dt,math,argparse
os.environ['NUMBA_DISABLE_JIT']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import prep
import numpy as np
import pandas as pd
from analysis.disease_improvement.history import build_history
from analysis.disease_improvement.infection_event_features import EVENT_NAMES,build_event_features,insert_events

def scalar_exposure(t,tx,rh,rain):
 # Independent fixed24-bin proxy reference retained in the frozen event package.
 hourly=np.array([t+max(tx-t,0)*np.cos(2*np.pi*(j+.5)/24) for j in range(24)])
 saturation=np.exp(17.625*hourly/(243.04+hourly));low,high=0.,float(max(saturation))
 for _ in range(30):
  vapor=(low+high)/2
  if np.minimum(vapor/saturation,1.).mean()<rh/100:low=vapor
  else:high=vapor
 humidity=100*np.minimum(((low+high)/2)/saturation,1.)
 if rh==0:humidity[:]=0
 if rh==100:humidity[:]=100
 wet=humidity>=90-1e-7;rain_fraction=-np.expm1(-rain/(24*.6))
 return float(np.mean(np.exp(-.5*((hourly-18)/8)**2)*(wet+(~wet)*rain_fraction)))

def scalar_events(weather,field,row,anchor):
 now=pd.Timestamp(row.date);start=pd.Timestamp(field.sowing_date);dates=pd.date_range(start,now-pd.Timedelta(days=1))
 d=weather.loc[weather.location_id.eq(field.site_id)].set_index('date').reindex(dates)
 rain=d.rain_mm.to_numpy(float);thermal=np.maximum(d.tmean_c.to_numpy(float),0)/18
 dose=np.array([scalar_exposure(x.tmean_c,x.tmax_c,x.rh_mean_pct,x.rain_mm) for x in d.itertuples()]);v={}
 for width in [7,21]:
  part=rain[-width:];wet=part>=.1;longest=run=0
  for state in wet:run=run+1 if state else 0;longest=max(longest,run)
  v.update({f'rain_max_{width}d_mm':float(max(part)),f'rain_days_{width}d':int(sum(wet)),f'rain_run_{width}d':longest,f'humid_hours_{width}d':float(d.rh_hours_ge_90pct.iloc[-width:].sum()),f'rain_hours_{width}d':float(d.rain_hours_gt_0_1mm.iloc[-width:].sum())})
 first=(pd.Timestamp(anchor)-start).days+1 if pd.notna(anchor) else max(0,len(dates)-21)
 ages=np.array([thermal[i+1:].sum() for i in range(len(thermal))]);rainy=np.flatnonzero(rain>=.1)
 v.update(gap_wet_dose=float(dose[first:].sum()),gap_thermal_units=float(thermal[first:].sum()),rain_age_days=len(dates)-1-int(rainy[-1]) if len(rainy) else len(dates),pending_infection_dose=float(dose[ages<20].sum()),matured_recent_dose=float(dose[(ages>=20)&(ages<40)].sum()))
 return np.array([v[n] for n in EVENT_NAMES])

def verify(details=False):
 prep.verify_source_package(prep.QA,'files_sha256')
 prep.verify_source_package(prep.EVENT,'output_sha256')
 # A missing product is a hard failure; verification never manufactures outputs.
 saved=dict(np.load(OUT/'predictors.npz'))
 weather,queries,fields,old,event,selected,affected,positions,hp,calendar,config=prep.load()
 new=pd.read_parquet(OUT/'daily_weather.parquet');new.date=pd.to_datetime(new.date).dt.tz_localize(None)
 assert set(saved)=={'target_ids','station','history_events','history','process'}
 expected_shapes={'target_ids':(2304,),'station':(2304,53),'history_events':(2304,68),'history':(2304,35),'process':(2304,37)}
 for key,shape in expected_shapes.items():assert saved[key].shape==shape and np.isfinite(saved[key]).all()
 for key in ['target_ids','history','process']:assert saved[key].tobytes()==old[key].tobytes(),key
 assert prep.sha(OUT/'training_observation_ledger.csv')==prep.sha(prep.CAND/'training_observation_ledger.csv')
 for name in ['label_free_queries.csv','fields.csv','history_provenance.json']:assert prep.sha(OUT/name)==prep.sha(prep.EVENT/name)
 mask=new.MET_EIREANN_total_precip_used.to_numpy(bool);assert len(new)==71965 and new.location_id.nunique()==133 and mask.sum()==730
 expected=weather[['location_id','date']].merge(selected[['site_id','date','station_id','station_measured_total_precipitation_mm','rain_hours_gt_0p1mm']],left_on=['location_id','date'],right_on=['site_id','date'],how='left',validate='1:1')
 np.testing.assert_array_equal(mask,expected.station_id.notna())
 precip=['precipitation_mm','rain_mm','rain_hours_gt_0_1mm']
 changed_columns=precip+['origin_'+n for n in precip]+['station_id_'+n for n in precip]
 fixed=[c for c in weather.columns if c not in changed_columns]
 pd.testing.assert_frame_equal(new[fixed],weather[fixed],check_exact=True)
 pd.testing.assert_frame_equal(new.loc[~mask,weather.columns],weather.loc[~mask],check_exact=True)
 checked=expected.loc[mask].copy();original=weather.loc[mask];actual=new.loc[mask]
 ratios=[];rain=[];hours=[];phase_valid=[];hours_used=[]
 for oldrow,pair in zip(original.itertuples(),checked.itertuples()):
  total=pair.station_measured_total_precipitation_mm
  ratio=min(1.,max(0.,oldrow.rain_mm/oldrow.precipitation_mm)) if oldrow.precipitation_mm>0 else float('nan')
  warm=oldrow.tmin_c>2
  if warm:ratio=1.
  if total==0:ratio=0.
  valid=math.isfinite(ratio);used=(warm and math.isfinite(pair.rain_hours_gt_0p1mm)) or total==0
  ratios.append(ratio);phase_valid.append(valid);hours_used.append(used)
  rain.append(total*ratio if valid else oldrow.rain_mm)
  hours.append(pair.rain_hours_gt_0p1mm if used else oldrow.rain_hours_gt_0_1mm)
 np.testing.assert_allclose(actual.precipitation_mm,checked.station_measured_total_precipitation_mm,rtol=0,atol=1e-12)
 np.testing.assert_allclose(actual.rain_mm,rain,rtol=0,atol=1e-12);np.testing.assert_array_equal(actual.rain_hours_gt_0_1mm,hours)
 np.testing.assert_array_equal(actual.MET_EIREANN_liquid_phase_assisted,phase_valid)
 np.testing.assert_array_equal(actual.origin_rain_hours_gt_0_1mm.eq('MET_EIREANN'),hours_used)
 for variable,origin in [('precipitation_mm','MET_EIREANN'),('rain_mm','MET_EIREANN_phase_assisted'),('rain_hours_gt_0_1mm','MET_EIREANN')]:
  use=new['origin_'+variable].eq(origin)
  assert new.loc[use,'station_id_'+variable].tolist()==expected.loc[use,'station_id'].map(lambda x:str(int(x))).tolist()
 # Independently parse provider CSV rows and verify every unique source day.
 raw={};daily={}
 for station in [375,1075]:
  path=prep.QA/'sourcecache'/'hly375.csv' if station==375 else next((prep.QA/'sourcecache').glob('hly1075_*.csv'))
  lines=path.read_text().splitlines();h=next(i for i,l in enumerate(lines) if l.startswith('date,'));values={}
  for r in csv.reader(lines[h+1:]):
   t=dt.datetime.strptime(r[0],'%d-%b-%Y %H:%M')
   if t.year<2016 or t.year>2019:continue
   values[t]=(int(r[1]),float(r[2]) if r[2].strip() else None)
  raw[station]=values
 for r in selected.itertuples():
  stamp=r.date.to_pydatetime();vals=[raw[r.station_id][stamp+dt.timedelta(hours=h)] for h in range(1,25)]
  assert all(code==0 and value is not None and value>=0 for code,value in vals)
  assert abs(math.fsum(x[1] for x in vals)-r.station_measured_total_precipitation_mm)<=1e-8
  assert sum(value>.1 for code,value in vals)==r.rain_hours_gt_0p1mm
  assert pd.Timestamp(r.latest_interval_end_utc).tz_localize(None)==r.date+pd.Timedelta(days=1)
 other=np.ones(2304,bool);other[positions]=False
 assert saved['station'][other].tobytes()==old['station'][other].tobytes()
 assert saved['history_events'][other].tobytes()==event['history_events'][other].tobytes()
 baseline,names,bp,bsp=prep.rebuild_station(weather,fields,affected,calendar,config)
 be,_,_=build_event_features(weather,fields,affected,hp)
 np.testing.assert_array_equal(baseline,old['station'][positions]);np.testing.assert_array_equal(be,event['station_event_features'][positions]);np.testing.assert_array_equal(insert_events(baseline,be),event['history_events'][positions])
 station,names,pro,sp=prep.rebuild_station(new,fields,affected,calendar,config);events,_,ep=build_event_features(new,fields,affected,hp)
 np.testing.assert_array_equal(station,saved['station'][positions]);np.testing.assert_array_equal(insert_events(station,events),saved['history_events'][positions])
 np.testing.assert_array_equal(np.column_stack([saved['history_events'][:,:50],saved['history_events'][:,-3:]]),saved['station'])
 fixed_cols=[i for i,n in enumerate(names) if not ('weather_rain_' in n or n.startswith('process_'))]
 assert saved['station'][:,fixed_cols].tobytes()==old['station'][:,fixed_cols].tobytes()
 expected_end=(pd.to_datetime(affected.date)-pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d').to_numpy()
 for p in [pro,sp,ep]:np.testing.assert_array_equal(p.weather_end.to_numpy(),expected_end)
 assert not sp.contemporaneous_or_future_stage_used.any()
 # First and last query of each changed field include distinct dates and sources.
 picks=sorted(set(affected.groupby('field_id').head(1).index)|set(affected.groupby('field_id').tail(1).index))
 id_to_pos=pd.Series(np.arange(len(affected)),index=affected.global_target_index)
 fields_idx=fields.set_index('field_id');hpidx=hp.set_index('global_target_index');hand=[];perturb=[]
 # Outcomes are read only for own/future-label invariance, never selection or fitting.
 context=pd.read_csv(prep.CAND/'training_observation_ledger.csv')
 for i in picks:
  row=queries.loc[i];local=int(id_to_pos.loc[row.global_target_index]);field=fields_idx.loc[row.field_id];h=hpidx.loc[row.global_target_index];anchor=h.latest_same_date if pd.notna(h.latest_same_date) else h.latest_any_date
  want=scalar_events(new,field,row,anchor);np.testing.assert_allclose(events[local],want,rtol=0,atol=1e-10)
  for name,actual_value,expected_value in zip(EVENT_NAMES,events[local],want):hand.append(dict(global_target_index=int(row.global_target_index),feature=name,actual=float(actual_value),independent_reference=float(expected_value),absolute_error=float(abs(actual_value-expected_value))))
  subset=new.loc[new.location_id.eq(row.site_id)].set_index('date');now=pd.Timestamp(row.date)
  for width in [7,21,42]:
   start=max(pd.Timestamp(field.sowing_date),now-pd.Timedelta(days=width));part=subset.loc[start:now-pd.Timedelta(days=1)]
   col=names.index(f'weather_rain_log1p_mean_{width}d');value=float(np.log1p(part.rain_mm.mean()));assert abs(station[local,col]-value)<=1e-12
   hand.append(dict(global_target_index=int(row.global_target_index),feature=names[col],actual=float(station[local,col]),independent_reference=value,absolute_error=float(abs(station[local,col]-value))))
  one=affected[affected.global_target_index.eq(row.global_target_index)]
  for period in ['assessment_date_only','assessment_and_future']:
   modified=new.copy();dates=modified.date.eq(now) if period=='assessment_date_only' else modified.date.ge(now);fmask=modified.location_id.eq(row.site_id)&dates
   modified.loc[fmask,['tmean_c','tmin_c','tmax_c','rh_mean_pct','rain_mm','precipitation_mm','rain_hours_gt_0_1mm','rh_hours_ge_90pct']]=999.
   test,_,_,_=prep.rebuild_station(modified,fields,one,calendar,config);ev,_,_=build_event_features(modified,fields,one,hp)
   np.testing.assert_array_equal(test[0],station[local]);np.testing.assert_array_equal(ev[0],events[local])
   perturb.append(dict(global_target_index=int(row.global_target_index),perturbation=period,weather_rows_changed=int(fmask.sum()),unchanged=True))
  changed=context.copy();lmask=changed.field_id.eq(row.field_id)&pd.to_datetime(changed.date).ge(now);changed.loc[lmask,'observed_percent']=100-changed.loc[lmask,'observed_percent']
  history,prior=build_history(changed,one,{})
  assert history.to_numpy()[0].tobytes()==old['history'][i].tobytes()
  ev,_,_=build_event_features(new,fields,one,prior);np.testing.assert_array_equal(ev[0],events[local])
  # The same-leaf history date in the station recipe is likewise invariant.
  changed_config=dict(config,prior_dates=dict(zip(prior.global_target_index,prior.latest_same_date)))
  test,_,_,_=prep.rebuild_station(new,fields,one,calendar,changed_config);np.testing.assert_array_equal(test[0],station[local])
  perturb.append(dict(global_target_index=int(row.global_target_index),perturbation='own_and_future_labels',weather_rows_changed=0,label_rows_changed=int(lmask.sum()),unchanged=True))
 first=affected.iloc[[0]]
 for fn in ['station','event']:
  try:
   if fn=='station':prep.rebuild_station(new,fields,first.assign(observed_percent=99.),calendar,config)
   else:build_event_features(new,fields,first.assign(observed_percent=99.),hp)
  except ValueError as e:assert 'labels' in str(e)
  else:raise AssertionError('Query labels accepted')
 bad=hp.copy();bad.loc[bad.global_target_index.eq(first.global_target_index.iloc[0]),['latest_same_date','latest_any_date']]=first.date.iloc[0]
 try:build_event_features(new,fields,first,bad)
 except ValueError as e:assert 'earlier' in str(e)
 else:raise AssertionError('Same-date history anchor accepted')
 registration=json.loads((OUT/'registration.json').read_text())
 for name,checksum in registration['source_sha256'].items():assert prep.sha(prep.ROOT/name)==checksum,name
 for name,checksum in registration['code_sha256'].items():
  assert prep.sha(prep.ROOT/name)==checksum,name
  assert prep.sha(OUT/'code_snapshot'/name)==checksum,name
 if (OUT/'manifest.json').exists():
  manifest=json.loads((OUT/'manifest.json').read_text());ready=json.loads((OUT/'READY.json').read_text());assert prep.sha(OUT/'manifest.json')==ready['manifest_sha256']
  for name,checksum in manifest['output_sha256'].items():assert prep.sha(OUT/name)==checksum,name
 result=dict(result='PASS',rows=2304,fields=218,weather_rows=71965,weather_sites=133,eligible_irish_station_days=730,affected_query_rows=56,unchanged_station_rows=2248,unchanged_history_events_rows=2248,baseline53_15_68_exact_before_patch=True,new53_68_exact_replay=True,process37_history35_targetids_byte_identical=True,nonprecipitation_weather_columns_exact=True,unreplaced_weather_rows_exact=int((~mask).sum()),scalar_raw_hour_days_verified=730,station_total_measured_days=730,liquid_phase_assisted_days=int(sum(phase_valid)),liquid_phase_fallback_days=int(len(phase_valid)-sum(phase_valid)),station_rain_hour_days=int(sum(hours_used)),rain_hour_fallback_days=int(len(hours_used)-sum(hours_used)),independent_scalar_targets=len(picks),independent_handchecked_values=len(hand),max_handcheck_absolute_error=max(x['absolute_error'] for x in hand),causal_perturbation_checks=len(perturb),query_label_rejection_checks=2,same_date_anchor_rejected=True,weather_cutoff_days=1,source_hashes_unchanged=True,model_fitted=False,post_freeze_manifest_verified=(OUT/'manifest.json').exists())
 if details:result.update(handchecks=hand,perturbations=perturb)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--details',action='store_true');a=p.parse_args();print(json.dumps(verify(a.details),indent=2,allow_nan=False))
