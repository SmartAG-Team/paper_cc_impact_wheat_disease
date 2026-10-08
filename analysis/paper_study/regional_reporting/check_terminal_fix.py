"""Independent retrospective replay of stratified archived baseline cells."""
from pathlib import Path
from datetime import datetime,timezone
import sys,os,json,hashlib,importlib.util
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/regional_reporting'
HERE.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT));os.environ['NUMBA_CACHE_DIR']=str(HERE/'numba_cache')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from model.seasonal_septoria.regional import simulate_grid_seasons,phenology_window
from model.seasonal_septoria.core import Parameters

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    b=ROOT/'analysis/paper_study/regional_v1/annual/era5/ERA5/baseline/original/winter_wheat_rainfed_sow+0_stage85'
    files=['model/seasonal_septoria/regional.py','analysis/paper_study/run_regional.py','model/seasonal_septoria/core.py','model/seasonal_septoria/host.py','process_model/parameters/calibration.json','analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json']
    snapshot={name:sha(ROOT/name) for name in files}
    (HERE/'terminal_fix_source_hashes.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    for name in files[:4]:
        p=HERE/'terminal_fix_source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    original_path=ROOT/'analysis/paper_study/model_review/source_snapshot_regional/model/seasonal_septoria/regional.py'
    spec=importlib.util.spec_from_file_location('model.seasonal_septoria.review_original_regional',original_path)
    original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
    cal=json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    parameters=Parameters(**json.loads((ROOT/'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json').read_text())['parameters'])
    results=[];samples=[];weather_sources={};archive_hashes={}
    def check(name,condition,details=None):
        if not bool(condition):raise AssertionError(name)
        results.append(dict(check=name,status='passed',details=details))
    for year in [1991,2020]:
        path=b/f'{year}.parquet';recpath=path.with_suffix('.json');archive_hashes[str(path.relative_to(ROOT))]=sha(path)
        source=pd.read_parquet(path);receipt=json.loads(recpath.read_text())
        check('archived_parquet_hash_'+str(year),sha(path)==receipt['parquet_sha256'])
        selected=[]
        for country in ['Spain','France','Germany','Poland','Russian Federation','Turkey','Finland','United Kingdom of Great Britain and Northern Ireland']:
            rows=source.loc[source.dominant_source_country.eq(country)&source.status.eq('complete')]
            if len(rows):selected.append(rows.sort_values('harvested_total_ha',ascending=False).iloc[0].cell_id)
        for status in ['complete','soft_dough_not_reached','missing_calendar']:
            rows=source.loc[source.status.eq(status)]
            if len(rows):
                selected.extend(rows.sort_values('latitude').iloc[[0,-1]].cell_id.tolist())
                selected.append(rows.sort_values('harvested_total_ha',ascending=False).iloc[0].cell_id)
        sample=source.set_index('cell_id').loc[list(dict.fromkeys(selected))].reset_index()
        samples.append(sample[['cell_id','harvest_year','dominant_source_country','latitude','longitude','harvested_total_ha','status','sowing_date']])
        eligible=sample.status.ne('missing_calendar');sample_valid=sample.loc[eligible].copy();ids=sample_valid.cell_id.tolist()
        pieces=[]
        for item in receipt['source_forcing']:
            p=ROOT/item['path'];actual=sha(p)
            check('source_forcing_hash_'+str(year)+'_'+p.stem,actual==item['sha256'])
            weather_sources[item['path']]=actual
            pieces.append(pq.read_table(p,columns=['date','cell_id','tmean_c','tmax_c','rh_mean_pct','precipitation_mm'],filters=[('cell_id','in',ids)]).to_pandas())
        weather=pd.concat(pieces,ignore_index=True);weather['date']=pd.to_datetime(weather.date)
        dates=pd.date_range(f'{year-1}-01-01',f'{year}-12-31')
        arrays={}
        for key in ['tmean_c','tmax_c','rh_mean_pct','precipitation_mm']:
            table=weather.pivot(index='cell_id',columns='date',values=key).reindex(index=ids,columns=dates)
            check('sample_forcing_complete_'+str(year)+'_'+key,np.isfinite(table.to_numpy()).all())
            arrays[key]=table.to_numpy(float)
        args=[dates,arrays['tmean_c'],arrays['tmax_c'],arrays['rh_mean_pct'],arrays['precipitation_mm'],sample_valid.latitude.to_numpy(),sample_valid.sowing_date.to_numpy(dtype='datetime64[D]'),parameters,cal]
        fixed=simulate_grid_seasons(*args);old=original.simulate_grid_seasons(*args)
        check('complete_flags_unchanged_'+str(year),np.array_equal(fixed['complete_season'],old['complete_season']))
        check('sample_archive_completion_'+str(year),np.array_equal(fixed['complete_season'],sample_valid.status.eq('complete').to_numpy()))
        for key in ['final_damage_percent','final_pycnidia_percent','endpoint_dates','primary_exposure_dates','first_visible_dates']:
            check('original_driver_exact_'+str(year)+'_'+key,np.array_equal(fixed[key],old[key],equal_nan=True))
        for leaf in range(7):
            archived=sample_valid[f'leaf{leaf+1}_final_damage_percent'].to_numpy()
            check('archived_damage_'+str(year)+'_leaf'+str(leaf+1),np.array_equal(fixed['final_damage_percent'][:,leaf],archived,equal_nan=True))
            for j,cutoff in enumerate([.1,1,5]):
                archived=sample_valid[f'leaf{leaf+1}_onset_{cutoff:g}pct_date'].to_numpy(dtype='datetime64[D]')
                check('archived_onset_'+str(year)+'_leaf'+str(leaf+1)+'_'+str(cutoff),np.array_equal(fixed['first_visible_dates'][:,leaf,j],archived,equal_nan=True))
        for key,column in [('endpoint_dates','endpoint_date'),('primary_exposure_dates','primary_exposure_date')]:
            check('archived_dates_'+str(year)+'_'+key,np.array_equal(fixed[key],sample_valid[column].to_numpy(dtype='datetime64[D]'),equal_nan=True))
        for stage in [0,10,31,51,85]:
            check('archived_stage_'+str(year)+'_'+str(stage),np.array_equal(fixed['stage_dates'][stage],sample_valid[f'bbch{stage}_date'].to_numpy(dtype='datetime64[D]'),equal_nan=True))
        check('missing_calendar_severity_missing_'+str(year),sample.loc[~eligible,'upper3_final_damage_percent'].isna().all())
    # Independent causality challenges cover hot days before/at/after endpoint,
    # pre-sowing weather and a crop that never reaches its terminal threshold.
    dates=pd.date_range('2020-01-01','2021-12-31');t=np.full((1,len(dates)),18.)
    tx=t+6;h=np.full_like(t,90.);rain=np.ones_like(t);sow=np.array(['2020-02-01'],dtype='datetime64[D]');lat=np.array([50.])
    base=simulate_grid_seasons(dates,t,tx,h,rain,lat,sow,parameters,cal,vernalization_required=False)
    end=int(np.flatnonzero(dates.to_numpy(dtype='datetime64[D]')==base['endpoint_dates'][0])[0])
    for mode,pos in [('pre_sowing',2),('post_terminal',end+1)]:
        hot=t.copy();hot_tx=tx.copy();hot[0,pos]=45.;hot_tx[0,pos]=50.
        out=simulate_grid_seasons(dates,hot,hot_tx,h,rain,lat,sow,parameters,cal,vernalization_required=False)
        check('hot_'+mode+'_accepted_and_unchanged',np.array_equal(base['final_damage_percent'],out['final_damage_percent']) and np.array_equal(base['endpoint_dates'],out['endpoint_dates']))
    for pos in [40,end]:
        hot=t.copy();hot_tx=tx.copy();hot[0,pos]=45.;hot_tx[0,pos]=50.
        try:simulate_grid_seasons(dates,hot,hot_tx,h,rain,lat,sow,parameters,cal,vernalization_required=False);rejected=False
        except ValueError as e:rejected='40' in str(e)
        check('hot_preterminal_or_terminal_rejected_'+str(pos),rejected)
    cold=np.full_like(t,-8.);cold_tx=cold+2;included=dates.to_numpy(dtype='datetime64[D]')[None,:]>=sow[:,None]
    _,complete,endpoint,unsupported=phenology_window(cold,cold_tx,lat,dates.dayofyear,included,cal,cal['thresholds'][-1]['Cumulative_t_pp_v_GDD'],False)
    check('incomplete_crop_retained',not complete[0] and not unsupported[0])
    cold[0,-1]=45.;cold_tx[0,-1]=50.
    _,complete,endpoint,unsupported=phenology_window(cold,cold_tx,lat,dates.dayofyear,included,cal,cal['thresholds'][-1]['Cumulative_t_pp_v_GDD'],False)
    check('hot_last_day_of_incomplete_window_unsupported',not complete[0] and unsupported[0])
    check('sources_unchanged_during_replay',all(sha(ROOT/name)==value for name,value in snapshot.items()))
    check('annual_archives_unchanged',all(sha(ROOT/name)==value for name,value in archive_hashes.items()))
    pd.concat(samples,ignore_index=True).to_csv(DATA/'baseline_terminal_fix_sample.csv',index=False)
    receipt=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks_passed=len(results),checks=results,
        source_hashes=snapshot,annual_archive_hashes=archive_hashes,forcing_hashes=weather_sources,
        sample_cell_years=sum(len(x) for x in samples),sampling='Largest wheat area by declared countries plus latitude extremes and largest area within completion/calendar strata; no disease values select samples',
        sample_scope='Stratified 1991 and 2020 cell replay; not full-domain numerical replay',retrospective_replay=True,parameter_refit=False,original_archives_preserved=True)
    (HERE/'terminal_fix_independent_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','checks_passed','sample_cell_years','sample_scope']},indent=2))

if __name__=='__main__':main()
