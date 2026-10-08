"""Paired current-kernel disease-weather/host-development model decomposition."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
from analysis.paper_study.nature_food_fix_20261007.climate import run as robustness
from model.seasonal_septoria.host import sowing_date_from_calendar
from model.seasonal_septoria.regional import phenology_window,tpv_accumulation
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.wetness import duration_exposure
from model.seasonal_septoria.overwinter import OverwinterParameters,simulate_overwinter
from model.seasonal_septoria.climate_alignment import apply_monthly_alignment

HERE=Path(__file__).resolve().parent
ROOT=robustness.ROOT
CLIMATE=HERE.parent
ORIGINAL=robustness.original
MODELS=ORIGINAL.MODELS
YEARS=list(range(1991,2021))
FIELDS=ORIGINAL.cache.FIELDS
METRICS=[f'F{leaf}_{event}_relative_anthesis_days' for leaf in (1,2,3) for event in ('infection','symptom')]+[
    'GS65_85_lost_had3','GS31_85_lost_had3','GS65_85_functional_lost_fraction','GS65_85_reference_had3','grain_fill_days']
COMPONENTS=['total_change','weather_alone','host_alone','interaction','weather_shapley','host_shapley']


def sha(path):return robustness.sha(path)


def write_json(path,value):return robustness.write_json(path,value)


def coefficients(model,draws):
    frame=pd.read_parquet(ORIGINAL.coefficient_path(model,'ssp585'))
    assert not frame.duplicated(['cell_id','month']).any()
    frame=frame.set_index(['cell_id','month']).reindex(pd.MultiIndex.from_product([draws.cell_id,range(1,13)],names=['cell_id','month']))
    return {k:frame[k].to_numpy().reshape(64,12) for k in ['temperature_offset','humidity_logit_offset','precipitation_ratio']}


def forcing(model,year,draws,coeff):
    prev,pw=ORIGINAL.load_cache_year(model,'ssp585',year-1,draws);cur,cw=ORIGINAL.load_cache_year(model,'ssp585',year,draws)
    dates=pd.DatetimeIndex(np.concatenate([prev,cur]))
    weather=apply_monthly_alignment({k:np.concatenate([pw[k],cw[k]],axis=1) for k in FIELDS},dates.month,coeff)
    sow=np.array([sowing_date_from_calendar(year,r.planting_doy,r.maturity_doy).to_datetime64().astype('datetime64[D]') for r in draws.itertuples()])
    return dict(dates=dates,weather=weather,sowing=sow)


def host_inputs(f,draws,phenology,fit,thresholds):
    dates=f['dates'];raw=f['weather'];sow=f['sowing'];days=len(dates)
    included=dates.to_numpy(dtype='datetime64[D]')[None,:]>=sow[:,None]
    finite=np.logical_and.reduce([np.isfinite(raw[k]) for k in FIELDS])
    finite&=(raw['tmax_c']>=raw['tmean_c']-.001)&(raw['rh_mean_pct']>=0)&(raw['rh_mean_pct']<=100)&(raw['precipitation_mm']>=0)
    gap=np.where((included&~finite).any(axis=1),(included&~finite).argmax(axis=1),days)
    prefix=included&(np.arange(days)[None,:]<gap[:,None])
    safe={k:np.where(finite,raw[k],0.) for k in FIELDS}
    mask,complete,endpoint,unsupported=phenology_window(safe['tmean_c'],safe['tmax_c'],draws.latitude.to_numpy(),dates.dayofyear,prefix,
        phenology,float(thresholds[85]),True)
    valid=complete&~unsupported;mask&=valid[:,None]
    t=np.where(mask,safe['tmean_c'],0.);tx=np.where(mask,safe['tmax_c'],0.)
    a=tpv_accumulation(t,tx,draws.latitude.to_numpy(),dates.dayofyear,mask,phenology,True)
    host=leaf_host(a,t,thresholds,rank_spacing_units=fit['rank_spacing_units'],forcing_mask=mask,juvenile_policy=fit['juvenile_policy'])
    return dict(host=host,mask=mask,valid=valid,accumulation=a)


def align_crop_day(weather_source,host_target,target_mask):
    """One-to-one real-day mapping by days since each crop-calendar sowing.

    Only target crop-window values are material. No leap-day interpolation,
    repeated weather, padding of active forcing or calendar-date assumptions.
    """
    target_dates=host_target['dates'].to_numpy(dtype='datetime64[D]')
    source_dates=weather_source['dates'].to_numpy(dtype='datetime64[D]')
    elapsed=(target_dates[None,:]-host_target['sowing'][:,None]).astype('timedelta64[D]').astype(int)
    source_index=(weather_source['sowing'][:,None]-source_dates[0]).astype('timedelta64[D]').astype(int)+elapsed
    in_range=(source_index>=0)&(source_index<len(source_dates))
    unavailable=target_mask&~in_range
    if unavailable.any():raise ValueError(f'Active crop days lack actual source forcing: {int(unavailable.sum())}.')
    field=np.arange(len(source_index))[:,None]
    clipped=np.clip(source_index,0,len(source_dates)-1)
    aligned={k:np.where(target_mask,np.asarray(weather_source['weather'][k])[field,clipped],0.) for k in FIELDS}
    assert all(np.isfinite(v).all() for v in aligned.values())
    assert all(np.all(np.diff(source_index[i,target_mask[i]])==1) for i in range(len(source_index)))
    return aligned,dict(active_target_days=int(target_mask.sum()),unavailable_active_weather_days=int(unavailable.sum()),
        alignment='elapsed calendar days from actual sowing; each active source day consumed once',
        source_date_origin=str(pd.Timestamp(source_dates[0]).date()),target_date_origin=str(pd.Timestamp(target_dates[0]).date()),
        source_forcing_days=len(source_dates),target_forcing_days=len(target_dates))


def hybrid(weather_source,host_target,host_state,draws,phenology,fit,thresholds):
    w,alignment=align_crop_day(weather_source,host_target,host_state['mask'])
    exposure=duration_exposure(w['tmean_c'],w['tmax_c'],w['rh_mean_pct'],w['precipitation_mm'],fit['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    host=host_state['host'];mask=host_state['mask']
    trajectory=simulate_overwinter(w['tmean_c'],exposure,w['precipitation_mm'],host.active,host.renewal,host.area,mask,
        OverwinterParameters(**fit['parameters']),initial_local_source=1.,imported_pressure=np.full_like(w['tmean_c'],fit['constant_imported_pressure']),detection_fraction=.001)
    rows,checks=robustness.simulation.simulate_draw_seasons(host_target['dates'],host_target['weather'],host_target['sowing'],draws.latitude.to_numpy(),
        phenology,fit,thresholds,trajectory_override=trajectory)
    assert np.array_equal(rows.valid_complete_season.to_numpy(),host_state['valid'])
    return rows[['status','valid_complete_season',*METRICS]],dict(**alignment,mass_error=float(checks['mass_error']),
        disease_weather_meanT_above40_active_days=int(((w['tmean_c']>40)&mask).sum()))


def inputs():
    draws=pd.read_csv(ORIGINAL.cache.DRAW_PATH)
    phenology=json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    fit=json.loads(ORIGINAL.FIT.read_text())['fitted']
    thresholds={int(k):v for k,v in json.loads(ORIGINAL.STAGES.read_text())['all_stage_thresholds'].items()}
    return draws,phenology,fit,thresholds


def pilot():
    draws,phenology,fit,thresholds=inputs();coeff=coefficients(MODELS[0],draws);records=[]
    for year in (1991,2020):
        r=forcing(MODELS[0],year,draws,coeff);f=forcing(MODELS[0],year+80,draws,coeff)
        rh=host_inputs(r,draws,phenology,fit,thresholds);fh=host_inputs(f,draws,phenology,fit,thresholds)
        for label,w,h,hs in [('Wfuture_Hreference',f,r,rh),('Wreference_Hfuture',r,f,fh)]:
            t=time.perf_counter();rows,details=hybrid(w,h,hs,draws,phenology,fit,thresholds)
            records.append(dict(reference_year=year,label=label,seconds=time.perf_counter()-t,complete=int(rows.valid_complete_season.sum()),**details));print(records[-1],flush=True)
    write_json(HERE/'pilot_cost.json',dict(pilot=records,planned_hybrid_batches=180,
        estimated_serial_hybrid_seconds=float(np.median([r['seconds'] for r in records])*180),
        outcomes_not_used_to_select_or_modify_decomposition=True))


def register():
    deps=[Path(__file__),HERE/'test_alignment.py',HERE/'pilot_cost.json',CLIMATE/'run.py',CLIMATE/'simulation.py',
        CLIMATE/'registration_before_robustness_results.json',CLIMATE/'independent_verification.json',CLIMATE/'draw_season_outputs.parquet',
        ORIGINAL.FIT,ORIGINAL.STAGES,ORIGINAL.cache.DRAW_PATH,ROOT/'process_model/parameters/calibration.json',ORIGINAL.HERE/'weather_cache_receipt.json',
        *[ROOT/f'model/seasonal_septoria/{name}.py' for name in ['overwinter','leaf_phenology','regional','host','wetness','climate_alignment']],
        *[ORIGINAL.coefficient_path(m,'ssp585') for m in MODELS]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in deps}
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),scenario='ssp585',reference_years=YEARS,future_year_offset=80,models=MODELS,
        spatial_draws=64,strata=16,draws_per_stratum=4,unique_cells=62,GCM_year_pairs=90,hybrid_trajectory_batches=180,
        corners=['Wreference_Hreference','Wfuture_Hreference','Wreference_Hfuture','Wfuture_Hfuture'],expected_corner_rows=23040,
        metrics=METRICS,source_parameters='Frozen selected current-kernel fit; local initial1; imported0.1; no refitting or validation selection.',
        host_factor='Complete host active/area/renewal arrays, calibrated stage crossings and GS85 crop mask, including host-temperature-driven juvenile tissue renewal.',
        disease_weather_factor='Actual aligned daily tmean/tmax/RH/rain enter exposure, source/residue thermal clocks and tissue progression; source parameters remain fixed.',
        calendar_alignment='Consecutive real weather days aligned one-to-one by elapsed days from each actual crop-calendar sowing date. Hybrid event offsets are expressed on the retained host trajectory.',
        leap_days='Actual daily chronology retained without interpolation, duplication or active-day padding.2020→2100 has a one-day month/day offset after reference Feb29. Values remain actual source days.',
        endpoint='Each chosen host trajectory retains its own GS85 endpoint and GS65–85/GS31–85 windows. Source weather may extend past its own-source GS85 because the target host window governs the hybrid.',
        high_temperature='Hybrid disease-weather meanT above40C is counted explicitly. The current tissue-progression kernel has a linear nonnegative-temperature clock without an upper40C guard. Host-development validity is determined from its own retained weather; no clipping or invented weather enters the hybrid. These component combinations have no physiological validation.',
        unavailable='Any active day without actual forcing aborts; host-invalid original seasons remain missing.',
        diagonal_corners='Reuse exactly the verified current-kernel baseline draw-season rows; no older model columns or reconstructions.',
        population='Same draw/GCM/relative year; common finite coverage across all four corners separately for each metric; same population for every decomposition term.',
        formulas={'weather_alone':'Y10−Y00','host_alone':'Y01−Y00','interaction':'Y11−Y10−Y01+Y00',
            'weather_shapley':'0.5[(Y10−Y00)+(Y11−Y01)]','host_shapley':'0.5[(Y01−Y00)+(Y11−Y10)]','total':'Y11−Y00'},
        estimator='Original area-time ratio estimator; stratified ratio-residual spatial MCSE; average GCM influence vectors before variance for shared draws.',
        interpretation='Order-invariant within-model factor decomposition. Hybrid combinations are mathematical component interventions, not physical causal effects, actual crop-production losses or independently validated predictions.',
        crop_production_scope='Climate/weather → modeled crop-infection timing → canopy HAD proxy. No actual-yield attribution or adaptation-effectiveness calculation.',
        randomness='None; original draws and weights retained.',source_sha256=hashes)
    write_json(HERE/'registration_before_decomposition_results.json',reg);print('REGISTERED90 paired GCM-years;180 hybrid batches',flush=True)


def run_model(model,reg):
    draws,phenology,fit,thresholds=inputs();coeff=coefficients(model,draws)
    baseline=pd.read_parquet(CLIMATE/'draw_season_outputs.parquet')
    baseline=baseline.loc[baseline.setting.eq('baseline')&baseline.model.eq(model)]
    reports=[]
    for i,year in enumerate(YEARS):
        path=HERE/'paired_outputs'/model/f'{year}.parquet';receipt=path.with_suffix('.json')
        if path.exists() or receipt.exists():
            assert path.exists() and receipt.exists();r=json.loads(receipt.read_text());assert sha(path)==r['output_sha256'];reports.append(r);continue
        started=time.perf_counter()
        ref=forcing(model,year,draws,coeff);future=forcing(model,year+80,draws,coeff)
        rh=host_inputs(ref,draws,phenology,fit,thresholds);fh=host_inputs(future,draws,phenology,fit,thresholds)
        corners={};checks={}
        for name,y in [('Wreference_Hreference',year),('Wfuture_Hfuture',year+80)]:
            corners[name]=baseline.loc[baseline.harvest_year.eq(y)].sort_values('spatial_draw_id')[['status','valid_complete_season',*METRICS]].reset_index(drop=True)
        for name,w,h,hs in [('Wfuture_Hreference',future,ref,rh),('Wreference_Hfuture',ref,future,fh)]:corners[name],checks[name]=hybrid(w,h,hs,draws,phenology,fit,thresholds)
        frames=[]
        for name,frame in corners.items():
            frame=frame.copy();frame['corner']=name;frame['model']=model;frame['reference_year']=year;frame['future_year']=year+80
            for k in ['spatial_draw_id','cell_id','stratum','area_mean_weight']:frame[k]=draws[k].to_numpy()
            frames.append(frame)
        output=pd.concat(frames,ignore_index=True);path.parent.mkdir(parents=True,exist_ok=True);output.to_parquet(path,index=False)
        r=dict(model=model,reference_year=year,rows=len(output),seconds=time.perf_counter()-started,checks=checks,
            output_sha256=sha(path),registration_sha256=sha(HERE/'registration_before_decomposition_results.json'))
        write_json(receipt,r);reports.append(r)
        if (i+1)%10==0:print(model,f'{i+1}/30 pairs',r['seconds'],flush=True)
    return reports


def aggregate(corners,draws):
    summaries=[];ensemble=[];draw_records=[];identities=[]
    for model in MODELS:
        arrays={}
        for name,g in corners.loc[corners.model.eq(model)].groupby('corner'):
            arrays[name]=g.sort_values(['spatial_draw_id','reference_year'])[METRICS].to_numpy(float).reshape(64,30,len(METRICS))
        y00=arrays['Wreference_Hreference'];y10=arrays['Wfuture_Hreference'];y01=arrays['Wreference_Hfuture'];y11=arrays['Wfuture_Hfuture']
        common=np.logical_and.reduce([np.isfinite(y) for y in [y00,y10,y01,y11]])
        values={'total_change':y11-y00,'weather_alone':y10-y00,'host_alone':y01-y00,'interaction':y11-y10-y01+y00,
            'weather_shapley':.5*((y10-y00)+(y11-y01)),'host_shapley':.5*((y01-y00)+(y11-y10))}
        error=np.where(common,values['weather_shapley']+values['host_shapley']-values['total_change'],0.)
        identities.append(float(np.max(np.abs(error))))
        for j,metric in enumerate(METRICS):
            z=common[:,:,j].mean(axis=1)
            for component in COMPONENTS:
                x=np.where(common[:,:,j],values[component][:,:,j],0.).mean(axis=1)
                s=ORIGINAL.ratio_summary(x,z,draws)
                summaries.append(dict(model=model,metric=metric,component=component,mean=s['mean'],spatial_mcse=s['spatial_mcse'],
                    common_four_corner_area_time_coverage=s['coverage'],common_draw_year_pairs=int(common[:,:,j].sum()),draws_complete_all30=int(common[:,:,j].all(axis=1).sum())))
                for k,draw in draws.iterrows():draw_records.append(dict(model=model,metric=metric,component=component,spatial_draw_id=int(draw.spatial_draw_id),
                    numerator=float(x[k]),denominator=float(z[k]),area_mean_weight=draw.area_mean_weight,stratum=draw.stratum))
    summary=pd.DataFrame(summaries);mom=pd.DataFrame(draw_records)
    indexed={(m,metric,c):g.sort_values('spatial_draw_id') for (m,metric,c),g in mom.groupby(['model','metric','component'])}
    for (metric,component),g in summary.groupby(['metric','component']):
        influences=[]
        for row in g.itertuples():
            p=indexed[(row.model,metric,component)];influences.append((p.numerator.to_numpy()-row.mean*p.denominator.to_numpy())/row.common_four_corner_area_time_coverage)
        ensemble.append(dict(metric=metric,component=component,gcm_mean=float(g['mean'].mean()),gcm_min=float(g['mean'].min()),gcm_max=float(g['mean'].max()),
            spatial_mcse_gcm_mean=robustness.shared_mcse(influences,draws),minimum_gcm_common_coverage=float(g.common_four_corner_area_time_coverage.min()),
            maximum_gcm_common_coverage=float(g.common_four_corner_area_time_coverage.max()),minimum_gcm_pairs=int(g.common_draw_year_pairs.min()),maximum_gcm_pairs=int(g.common_draw_year_pairs.max())))
    return summary,pd.DataFrame(ensemble),mom,max(identities)


def main():
    if sys.argv[1:]==['--pilot']:pilot();return
    if sys.argv[1:]==['--register']:register();return
    reg=json.loads((HERE/'registration_before_decomposition_results.json').read_text())
    for name,h in reg['source_sha256'].items():assert sha(ROOT/name)==h,name
    reports=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run_model,m,reg) for m in MODELS]):reports.extend(f.result())
    assert len(reports)==90
    corners=pd.concat([pd.read_parquet(HERE/'paired_outputs'/r['model']/f'{r["reference_year"]}.parquet') for r in reports],ignore_index=True)
    assert len(corners)==23040 and not corners.duplicated(['corner','model','reference_year','spatial_draw_id']).any()
    corners.to_parquet(HERE/'four_corner_draw_year_outputs.parquet',index=False)
    summary,ensemble,mom,error=aggregate(corners,pd.read_csv(ORIGINAL.cache.DRAW_PATH))
    for name,frame in [('decomposition_by_gcm',summary),('ensemble_decomposition',ensemble),('paired_draw_moments',mom)]:
        frame.to_parquet(HERE/f'{name}.parquet',index=False)
        if name!='paired_draw_moments':frame.to_csv(HERE/f'{name}.csv',index=False)
    for name,h in reg['source_sha256'].items():assert sha(ROOT/name)==h,name
    write_json(HERE/'completion_receipt.json',dict(status='complete',GCM_year_pairs=90,hybrid_trajectories=180,corner_draw_year_rows=23040,
        maximum_shapley_additivity_error=error,maximum_mass_error=max(c['mass_error'] for r in reports for c in r['checks'].values()),
        unavailable_active_weather_days=sum(c['unavailable_active_weather_days'] for r in reports for c in r['checks'].values()),
        source_sha256_verified=True,registration_sha256=sha(HERE/'registration_before_decomposition_results.json'),
        output_sha256={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.csv','.parquet')}))
    print('COMPLETE weather/host decomposition',flush=True)


if __name__=='__main__':main()
