"""Independent regional TPV and endpoint checks; no observations or refits."""
from pathlib import Path
import sys,os,json,hashlib
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));os.environ['NUMBA_CACHE_DIR']=str(HERE/'numba_cache')
import numpy as np
import pandas as pd
from dataclasses import replace
from process_model.calibrate import transform
from model.seasonal_septoria.regional import tpv_accumulation,simulate_grid_seasons
from model.seasonal_septoria.core import Parameters,simulate_season
from model.seasonal_septoria.host import cohort_inputs

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    results=[];limitations=[]
    def check(name,condition,details=None):
        if not bool(condition):raise AssertionError(name)
        results.append(dict(check=name,status='passed',details=details))
    cal=json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    thresholds={x['BBCH']:x['Cumulative_t_pp_v_GDD'] for x in cal['thresholds']}
    rng=np.random.default_rng(20261005);dates=pd.date_range('2019-07-01','2021-12-31')
    n=12;doy=dates.dayofyear.to_numpy();lat=np.linspace(35,70,n)
    t=12+13*np.sin((doy[None,:]-110)*2*np.pi/365)+rng.normal(0,5,(n,len(dates)))
    t=np.clip(t,-12,39);tx=t+rng.uniform(0,18,t.shape)
    starts=pd.to_datetime(['2019-08-25','2019-10-13','2020-03-01','2020-10-01']*3)
    included=dates.to_numpy()[None,:]>=starts.to_numpy()[:,None]
    got=tpv_accumulation(t,tx,lat,doy,included,cal)
    differences=[]
    for i in range(n):
        raw=pd.DataFrame(dict(PEP_ID=i+1,DATE=dates[included[i]],LAT=lat[i],t_mean=t[i,included[i]],t_max=tx[i,included[i]],SOWING_DATE=starts[i],SOWING_KNOWN_AT=starts[i]))
        raw['GDD']=np.where(raw.t_mean>30,20-2*(raw.t_mean-30),np.clip(raw.t_mean,0,20))
        original=transform(raw,cal);difference=float(np.max(abs(got[i,included[i]]-original.Cumulative_t_pp_v_GDD)))
        differences.append(difference);check('original_TPV_equality_row_'+str(i),difference<1e-9,difference)
    check('pre_sowing_accumulation_zero',np.all(got[~included]==0))
    nover=tpv_accumulation(t,tx,lat,doy,included,cal,False)
    check('removing_vernalization_preserves_or_accelerates_accumulation',np.all(nover>=got-1e-9))
    h=rng.uniform(60,100,t.shape);rain=rng.exponential(2,t.shape);p=Parameters(alpha=.013,beta=.7,latent_days=20.)
    end_results={}
    for stage in [85,75]:
        out=simulate_grid_seasons(dates,t,tx,h,rain,lat,starts,p,cal,end_stage=stage)
        end_results[stage]=out
        value=thresholds[85] if stage==85 else (thresholds[51]+thresholds[85])/2
        reached=got>=value;complete=reached.any(axis=1);end=np.argmax(reached,axis=1)
        active,renewal=cohort_inputs(got,t,thresholds)
        valid=included & (np.arange(len(dates))[None,:]<=np.where(complete,end,len(dates)-1)[:,None])
        active &= valid[:,:,None];renewal[~active]=0
        direct=simulate_season(t,h,rain,active,renewal,p)
        expected_damage=direct.damage[np.arange(n),end+1,:7]*100
        expected_pycnidia=direct.pycnidia[np.arange(n),end+1,:7]*100
        expected_damage[~complete]=np.nan;expected_pycnidia[~complete]=np.nan
        check('direct_solver_damage_endpoint_'+str(stage),np.array_equal(out['final_damage_percent'],expected_damage,equal_nan=True))
        check('direct_solver_pycnidia_endpoint_'+str(stage),np.array_equal(out['final_pycnidia_percent'],expected_pycnidia,equal_nan=True))
        check('final_measurement_order_'+str(stage),np.all(out['final_pycnidia_percent'][complete]<=out['final_damage_percent'][complete]+1e-12))
        check('onset_within_sowing_endpoint_'+str(stage),all(np.isnat(x) or starts[i].to_datetime64().astype('datetime64[D]')<=x<=out['endpoint_dates'][i] for i in range(n) for x in out['first_visible_dates'][i].ravel() if complete[i]))
        check('no_pre_sowing_infection_'+str(stage),np.count_nonzero(direct.primary_flow[~included])==0)
        check('no_post_terminal_infection_'+str(stage),np.count_nonzero(direct.primary_flow[~valid])==0 and np.count_nonzero(direct.secondary_flow[~valid])==0)
        check('regional_mass_and_positivity_'+str(stage),out['mass_error']<1e-12 and out['minimum_state']>=-1e-13)
        changed_t=t.copy();changed_tx=tx.copy();changed_h=h.copy();changed_r=rain.copy()
        for i in range(n):
            if complete[i]:
                changed_t[i,end[i]+1:]=35.;changed_tx[i,end[i]+1:]=39.;changed_h[i,end[i]+1:]=0.;changed_r[i,end[i]+1:]=100.
        changed=simulate_grid_seasons(dates,changed_t,changed_tx,changed_h,changed_r,lat,starts,p,cal,end_stage=stage)
        check('post_endpoint_weather_cannot_change_final_severity_'+str(stage),np.array_equal(out['final_damage_percent'],changed['final_damage_percent'],equal_nan=True))
        check('post_endpoint_weather_cannot_change_first_visible_'+str(stage),np.array_equal(out['first_visible_dates'],changed['first_visible_dates'],equal_nan=True))
    check('GS75_sensitivity_not_later_than_soft_dough',np.all(end_results[75]['endpoint_dates']<=end_results[85]['endpoint_dates']))
    zero=simulate_grid_seasons(dates,t,tx,h,rain,lat,starts,replace(p,alpha=0,beta=0),cal)
    check('zero_external_pressure_gives_no_spontaneous_disease',np.nanmax(zero['final_damage_percent'])==0 and np.isnat(zero['first_visible_dates']).all())
    # An actual API limit: the crop-development pass uses all supplied post-sow
    # weather, and validates >40 C before its terminal event is located.
    hot=t.copy();hotx=tx.copy();out=end_results[85]
    eligible=np.flatnonzero(out['complete_season'])
    i=int(eligible[0]);end=int(np.flatnonzero(got[i]>=thresholds[85])[0])
    hot[i,end+1:]=41.;hotx[i,end+1:]=45.
    try:simulate_grid_seasons(dates,hot,hotx,h,rain,lat,starts,p,cal);rejected=False
    except ValueError as e:rejected='40' in str(e)
    limitations.append(dict(issue='A >40C mean after a completed crop endpoint still rejects the entire supplied season window',demonstrated=rejected,severity='Input-window robustness; does not change outputs of accepted records'))
    snapshots=json.loads((HERE/'regional_source_hashes.json').read_text())
    changes={k:dict(before=v,after=sha(ROOT/k)) for k,v in snapshots['source_hashes'].items() if sha(ROOT/k)!=v}
    check('regional_sources_unchanged_during_review',not changes)
    original=json.loads((HERE/'review_source_hashes.json').read_text())
    archive_changes={k:dict(before=v,after=sha(ROOT/k)) for k,v in original['prior_archive_hashes'].items() if sha(ROOT/k)!=v}
    check('all_original_frozen_archives_preserved',not archive_changes)
    receipt=dict(status='passed_with_reported_input_window_limit',completed_utc=datetime.now(timezone.utc).isoformat(),
        checks_passed=len(results),checks=results,source_changes=changes,prior_archive_changes=archive_changes,
        maximum_TPV_absolute_difference=float(max(differences)),limitations=limitations,
        external_outcomes_read=False,parameter_refitting_performed=False,
        interpretation='GS85 soft dough endpoint; GS75 interpolated sensitivity; removal of vernalization keeps original winter thresholds')
    (HERE/'regional_independent_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','checks_passed','maximum_TPV_absolute_difference','limitations','prior_archive_changes']},indent=2))

if __name__=='__main__':main()
