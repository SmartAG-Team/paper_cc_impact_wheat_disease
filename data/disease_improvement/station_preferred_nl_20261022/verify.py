"""Independent serialized-product checks against KNMI raw hours and frozen sources."""
from pathlib import Path
import sys,json,zipfile,csv,datetime,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from analysis.disease_improvement.features import build_features
from analysis.disease_improvement.repairs import recorded_leaf_features
from analysis.disease_improvement.stage_features import build_stage_features
from analysis.disease_improvement.stage_experiment import augment
from analysis.disease_improvement.inputs import sha
OUT=Path(__file__).resolve().parent
CAND=ROOT/'outputs/disease_improvement/station_candidate_20261022_v2'
QA=ROOT/'outputs/weather_validation_20261022/knmi_season_qualification'
META=ROOT/'outputs/disease_improvement/quality_weather_20261021_v2'
SITE='era5loc_c36c557e4afc'

def design(w,f,o,cal,prior):
 events,names,pro=build_features(w,f,o,dict(weather_cutoff_days=1,rain_variable='rain_mm',prior_dates=prior))
 wc=[names.index(n) for n in names if n.startswith('weather_')];pc=[names.index(n) for n in names if n.startswith('process_')]
 cat,_=recorded_leaf_features(o);stage,_,sp=build_stage_features(w,f,o,cal)
 return augment(np.column_stack((events[:,wc],cat,events[:,pc])),stage),pro,sp

def main():
 oldweather=pd.read_parquet(ROOT/'data/disease_improvement/station_preferred_augmented_20261022/daily_weather.parquet');newweather=pd.read_parquet(OUT/'daily_weather.parquet')
 original=np.load(CAND/'predictors.npz');new=np.load(OUT/'predictors.npz');obs=pd.read_csv(CAND/'training_observation_ledger.csv')
 assert new.files==['target_ids','station']
 assert newweather.shape[0]==oldweather.shape[0]==71965 and newweather.location_id.nunique()==133
 assert not newweather.duplicated(['location_id','date']).any()
 np.testing.assert_array_equal(new['target_ids'],original['target_ids']);assert new['station'].shape==(2304,53)
 assert (OUT/'training_observation_ledger.csv').read_bytes()==(CAND/'training_observation_ledger.csv').read_bytes()
 selected=pd.read_parquet(QA/'qualified_selected_daily_precipitation.parquet');assert len(selected)==487
 key=pd.MultiIndex.from_frame(selected[['site_id','date']]);wk=pd.MultiIndex.from_frame(oldweather[['location_id','date']]);replace=wk.isin(key)
 assert replace.sum()==487
 allow={'precipitation_mm','rain_mm','rain_hours_gt_0_1mm','origin_precipitation_mm','origin_rain_mm','origin_rain_hours_gt_0_1mm','station_id_precipitation_mm','station_id_rain_mm','station_id_rain_hours_gt_0_1mm'}
 for col in oldweather.columns:
  if col not in allow:pd.testing.assert_series_equal(newweather[col],oldweather[col],check_names=False)
  else:pd.testing.assert_series_equal(newweather.loc[~replace,col].reset_index(drop=True),oldweather.loc[~replace,col].reset_index(drop=True),check_names=False)
 # Parse raw archive independently with csv, without the qualification/preparation parser.
 z=zipfile.ZipFile(QA/'sourcecache/uurgeg_280_2011-2020.zip');lines=z.read(z.namelist()[0]).decode().splitlines()
 header=next(line for line in lines if line.startswith('# STN,')).lstrip('# ');names=[s.strip() for s in header.split(',')]
 raw={}
 for line in lines:
  parts=next(csv.reader([line]))
  if len(parts)!=len(names) or not parts[0].strip().isdigit():continue
  record=dict(zip(names,[p.strip() for p in parts]));date=pd.Timestamp(record['YYYYMMDD'])
  if (SITE,date) not in key:continue
  assert record['RH']!='';raw.setdefault(date,[]).append((int(record['HH']),int(record['RH'])))
 assert len(raw)==487
 current=newweather.loc[replace].set_index('date');before=oldweather.loc[replace].set_index('date')
 totals={};hours={}
 for date,values in raw.items():
  assert len(values)==24 and len({h for h,v in values})==24 and set(h for h,v in values)==set(range(1,25))
  assert all(v>=-1 for h,v in values)
  totals[date]=sum(max(0,v) for h,v in values)/10;hours[date]=sum(v>1 for h,v in values)
 for date,total in totals.items():assert abs(current.loc[date,'precipitation_mm']-total)<1e-12
 # Recompute the phase/fallback math directly from frozen original forcing.
 station_total=pd.Series(totals).reindex(current.index);station_hours=pd.Series(hours).reindex(current.index)
 warm=before.tmin_c.gt(2)
 ratio=pd.Series(np.nan,index=before.index)
 valid_original=before.precipitation_mm.gt(0)
 ratio.loc[valid_original]=(before.loc[valid_original,'rain_mm']/before.loc[valid_original,'precipitation_mm']).clip(0,1)
 ratio.loc[warm]=1.;ratio.loc[station_total.eq(0)]=0.
 liquid=ratio.notna();expected_rain=before.rain_mm.copy();expected_rain.loc[liquid]=station_total.loc[liquid]*ratio.loc[liquid]
 expected_hours=before.rain_hours_gt_0_1mm.copy();hours_used=warm|station_total.eq(0);expected_hours.loc[hours_used]=station_hours.loc[hours_used]
 np.testing.assert_allclose(current.rain_mm.to_numpy(),expected_rain.to_numpy(),atol=1e-12,rtol=0)
 np.testing.assert_array_equal(current.rain_hours_gt_0_1mm.to_numpy(),expected_hours.to_numpy())
 np.testing.assert_array_equal(current.KNMI_liquid_phase_assisted.to_numpy(),liquid.to_numpy())
 np.testing.assert_array_equal(current.KNMI_warm_liquid_assumption.to_numpy(),(liquid&warm&station_total.gt(0)).to_numpy())
 assert current.origin_precipitation_mm.eq('KNMI').all() and current.station_id_precipitation_mm.eq('280').all()
 assert current.loc[liquid,'origin_rain_mm'].eq('KNMI_phase_assisted').all()
 pd.testing.assert_series_equal(current.loc[~liquid,'rain_mm'],before.loc[~liquid,'rain_mm'],check_names=False)
 assert current.loc[hours_used,'origin_rain_hours_gt_0_1mm'].eq('KNMI').all()
 assert int(newweather.KNMI_total_precip_used.sum())==487
 assert int(newweather.loc[newweather.location_id.isin(obs.site_id)&newweather.origin_precipitation_mm.ne('reanalysis'),'location_id'].nunique())==57
 # Rebuild all changed station rows independently and preserve remaining matrix bytes.
 fields=pd.read_csv(META/'field_metadata_ledger.csv');cal=pd.read_csv(ROOT/'outputs/raw_metadata_recovery_20261022/stage_calendar.csv');h=pd.read_json(META/'history_provenance.json');prior=dict(zip(h.global_target_index,h.latest_same_date))
 queries=obs.loc[obs.site_id.eq(SITE)&obs.field_id.isin(fields.field_id)].drop(columns='observed_percent')
 position=pd.Series(np.arange(len(obs)),index=obs.global_target_index).loc[queries.global_target_index].to_numpy();mask=np.zeros(len(obs),bool);mask[position]=True
 assert len(position)==19 and new['station'][~mask].tobytes()==original['station'][~mask].tobytes()
 oldmatrix,_,_=design(oldweather,fields,queries,cal,prior);np.testing.assert_array_equal(oldmatrix,original['station'][position])
 matrix,pro,sp=design(newweather,fields,queries,cal,prior);np.testing.assert_array_equal(matrix,new['station'][position])
 expected_end=(pd.to_datetime(queries.date)-pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d').to_numpy()
 np.testing.assert_array_equal(pro.weather_end.to_numpy(),expected_end);np.testing.assert_array_equal(sp.weather_end.to_numpy(),expected_end)
 assert not sp.contemporaneous_or_future_stage_used.any()
 for r in sp.itertuples():assert pd.isna(r.latest_stage_date) or pd.Timestamp(r.latest_stage_date)<pd.Timestamp(queries.set_index('global_target_index').loc[r.global_target_index,'date'])
 saved=pd.read_csv(OUT/'feature_provenance.csv');np.testing.assert_array_equal(saved.global_target_index,queries.global_target_index)
 # Perturb the assessment date and every later day; earlier-only design must stay exact.
 future_checks=0
 for date,q in queries.groupby('date',sort=False):
  modified=newweather.copy();after=modified.location_id.eq(SITE)&pd.to_datetime(modified.date).ge(date)
  for col in ['tmean_c','tmin_c','tmax_c','rh_mean_pct','rain_mm','rain_hours_gt_0_1mm','rh_hours_ge_90pct']:modified.loc[after,col]=999.
  check,_,_=design(modified,fields,q,cal,prior)
  loc=pd.Series(np.arange(len(queries)),index=queries.global_target_index).loc[q.global_target_index].to_numpy()
  np.testing.assert_array_equal(check,matrix[loc]);future_checks+=1
 reg=json.loads((OUT/'registration.json').read_text())
 for path,v in reg['source_sha256'].items():assert sha(ROOT/path)==v
 for path,v in reg['code_sha256'].items():assert sha(ROOT/path)==v
 receipt=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),result='PASS',raw_hourly_days_verified=487,strict_gt0p1mm_counts_verified=True,phase_policy_independently_recomputed=True,rawsum_phase_abs_tolerance_mm=1e-12,measured_total_precip_days=487,liquid_phase_assisted_days=int(liquid.sum()),liquid_fallback_days=int((~liquid).sum()),rain_hour_station_days=int(hours_used.sum()),rain_hour_fallback_days=int((~hours_used).sum()),strict_warm_positive_liquid_assumption_days=int((liquid&warm&station_total.gt(0)).sum()),temperature_humidity_and_other_nonprecip_columns_identical=True,nonreplaced_weather_rows_identical=int((~replace).sum()),source_observation_ledger_byte_identical=True,baseline_NL_features_exact_reproduction=True,new_NL_features_exact_rebuild=True,NL_target_rows=19,unchanged_station_rows_byte_identical=2285,assessment_and_future_weather_perturbation_groups=future_checks,weather_cutoff_days=1,process_array_source_sha256=hashlib.sha256(original['process'].tobytes()).hexdigest(),history_array_source_sha256=hashlib.sha256(original['history'].tobytes()).hexdigest(),active_station_precipitation_sites=57,no_model_fitting=True)
 (OUT/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
