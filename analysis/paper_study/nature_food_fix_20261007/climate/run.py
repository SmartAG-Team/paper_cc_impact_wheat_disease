"""Registered conditional climate robustness, isolated from manuscript/model sources."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import hashlib, importlib.util, json, platform, sys, time
import numpy as np
import pandas as pd
from analysis.paper_study.overwinter_leaf_model_20261006.climate import run as original
from model.seasonal_septoria.overwinter import _first_crossing
from model.seasonal_septoria.climate_alignment import apply_monthly_alignment
from model.seasonal_septoria.host import sowing_date_from_calendar

HERE=Path(__file__).resolve().parent
ROOT=original.ROOT
SOURCE=original.HERE.parent
MODELS=original.MODELS
SCENARIO='ssp585'
YEARS=list(range(1991,2021))+list(range(2071,2101))
PERIODS={'1991-2020':list(range(1991,2021)),'2071-2100':list(range(2071,2101))}
spec=importlib.util.spec_from_file_location('robustness_simulation',HERE/'simulation.py')
simulation=importlib.util.module_from_spec(spec);spec.loader.exec_module(simulation)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for block in iter(lambda:s.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def stage_profile_variants():
    """Minimum component-cost paths at early/late C37 within +1 day joint loss.

    Dynamic programming conditions on C37, while preserving strict ordering.
    Every alternative is checked by the sum of actual component losses.
    Disease and validation labels do not enter this construction.
    """
    frame=pd.read_csv(SOURCE/'phenology/ordered_threshold_joint_profiles.csv')
    stage_record=json.loads(original.STAGES.read_text())
    stages=[32,33,37,39]
    groups=[frame.loc[frame.event.eq(s)].sort_values('threshold') for s in stages]
    grid=groups[0].threshold.to_numpy(float)
    assert all(np.array_equal(g.threshold.to_numpy(float),grid) for g in groups)
    cost=np.array([g.loss_component_days.to_numpy(float) for g in groups])
    minimum=float(stage_record['minimum_loss_days'])
    admissible=groups[2].joint_profile_loss_days.to_numpy()<=minimum+1+1e-12
    points=np.flatnonzero(admissible)
    result={}
    for name,point in [('stage_profile_early37',int(points[0])),('stage_profile_late37',int(points[-1]))]:
        restricted=cost.copy();restricted[2,np.arange(len(grid))!=point]=np.inf
        dp=np.full_like(cost,np.inf);dp[0]=restricted[0]
        parent=np.full(cost.shape,-1,int)
        for k in range(1,4):
            for j in range(k,len(grid)):
                candidates=dp[k-1,:j]
                choice=int(np.argmin(candidates));parent[k,j]=choice
                dp[k,j]=restricted[k,j]+candidates[choice]
        last=int(np.argmin(dp[-1]));selected=[last]
        for k in range(3,0,-1):selected.append(int(parent[k,selected[-1]]))
        selected=selected[::-1]
        fitted={str(s):float(grid[j]) for s,j in zip(stages,selected)}
        loss=float(sum(cost[k,j] for k,j in enumerate(selected)))
        assert np.all(np.diff(list(fitted.values()))>0) and loss<=minimum+1+1e-12
        result[name]=dict(thresholds=fitted,joint_loss_days=loss,minimum_loss_days=minimum,
            tolerance_days=1.,selection='extreme admissible C37, then exact minimum ordered component-loss path',
            confidence_interval=False,validation_used=False,disease_labels_used=False)
    return result


def register():
    if (HERE/'registration_before_robustness_results.json').exists():
        raise FileExistsError('An immutable registration already exists; resume rather than replace it.')
    fit=json.loads(original.FIT.read_text())['fitted']
    full=json.loads((SOURCE/'disease/overwinter_source_model/spatial_0_training.json').read_text())
    selection=json.loads((SOURCE/'disease/overwinter_source_model/selection.json').read_text())
    score=selection['selected_score']
    tied=[s for s in selection['candidate_scores'] if abs(s['primary_distance_days']-score['primary_distance_days'])<1e-12 and s['candidate']!=score['candidate']]
    assert len(tied)==1
    alt=next(s for s in full['candidates'] if s['name']==tied[0]['candidate'])
    alt=dict(**alt,weather_preprocessing=fit['weather_preprocessing'])
    thresholds=json.loads(original.STAGES.read_text())['all_stage_thresholds']
    profiles=stage_profile_variants()
    settings={}
    def mechanism(name,record=None,source=1.,spacing=None,profile=None):
        value=json.loads(json.dumps(fit if record is None else record))
        if spacing is not None:value['rank_spacing_units']=spacing
        stage=dict(thresholds)
        if profile is not None:stage.update(profiles[profile]['thresholds'])
        settings[name]=dict(kind='mechanism',fitted=value,initial_local_source=source,
            thresholds=stage,detection_fraction=.001,reference_decline_end_factor=1.,functional_conversion=1.)
    mechanism('baseline');mechanism('primary_objective_tied_runnerup',record=alt)
    mechanism('local_source_off',source=0.)
    mechanism('rank_spacing80',spacing=80.);mechanism('rank_spacing160',spacing=160.)
    mechanism('stage_profile_early37',profile='stage_profile_early37')
    mechanism('stage_profile_late37',profile='stage_profile_late37')
    for name,cutoff in [('detection0p0001',.0001),('detection0p01',.01)]:
        settings[name]=dict(kind='daily_detection',detection_fraction=cutoff,reference_decline_end_factor=1.,functional_conversion=1.)
    for name,end,conversion in [('reference_decline_end0p5',.5,1.),('reference_decline_end0',0.,1.),
        ('functional_conversion0p5',1.,.5),('functional_conversion0p75',1.,.75),('decline_end0_conversion0p5',0.,.5)]:
        settings[name]=dict(kind='daily_had_transfer',detection_fraction=.001,reference_decline_end_factor=end,functional_conversion=conversion)
    dependencies=[HERE/'run.py',HERE/'simulation.py',HERE/'test_robustness.py',HERE/'pilot_cost.json',
        original.FIT,original.STAGES,SOURCE/'phenology/ordered_threshold_joint_profiles.csv',
        SOURCE/'disease/overwinter_source_model/full_calibration_training.json',SOURCE/'disease/overwinter_source_model/spatial_0_training.json',SOURCE/'disease/overwinter_source_model/selection.json',
        original.cache.DRAW_PATH,ROOT/'data/paper_study/regional_parameter_uncertainty/sampling_receipt.json',
        original.HERE/'run.py',original.HERE/'cache_weather.py',original.HERE/'weather_cache_receipt.json',
        original.HERE/'draw_season_outputs.parquet',ROOT/'process_model/parameters/calibration.json',
        *[ROOT/f'model/seasonal_septoria/{name}.py' for name in ['overwinter','leaf_phenology','regional','host','wetness','climate_alignment']],
        *[original.coefficient_path(model,SCENARIO) for model in MODELS]]
    sources={str(p.relative_to(ROOT)):sha(p) for p in dependencies}
    draws=pd.read_csv(original.cache.DRAW_PATH)
    record=dict(registered_utc=datetime.now(timezone.utc).isoformat(),models=MODELS,scenario=SCENARIO,
        periods=PERIODS,draws=64,unique_cells=int(draws.cell_id.nunique()),strata=16,draws_per_stratum=4,
        expected_batches=180,trajectory_settings=7,expected_mechanism_trajectory_batches=1260,
        reported_settings=len(settings),expected_draw_season_rows=180*64*len(settings),worker_count=3,
        settings=settings,stage_profile_construction=profiles,
        tied_runnerup=dict(selected_score=score,runnerup_score=tied[0],complete_selection_score_tie=False,
            note='Equal primary calibration objective; runner-up loses the prespecified balanced-error tie-break. No validation tuning.'),
        frozen_spatial_design='Original with-replacement 16 strata × 4 draws; duplicate cell IDs retained by unique draw IDs.',
        randomness='No new stochastic draws, no bootstrap, no RNG; all original draw weights and pairings retained.',
        climate_pairing='Same spatial_draw_id and GCM; 1991+i paired to 2071+i for i=0..29.',
        metric_populations=['within-setting finite historical/future pairs','intersection of those pairs across all settings for the same metric and GCM'],
        ratio_estimator='Weighted area-time mean on common paired coverage; mean over years for each draw, then weighted ratio.',
        mcse='Four independent draws per stratum, ratio residual linearization; average GCM influence vectors before variance (shared-draw covariance).',
        uncertainty_limits='MCSE is spatial Monte Carlo error, not parameter uncertainty; GCM range is not a confidence interval; settings are declared scenarios, not validated plausibility distributions.',
        reference_decline='Postprocessing reference weight: 1 before GS65, linearly declining in elapsed calendar days from GS65 to endpoint factor at GS85, multiplied by modeled unfolding area. Disease dynamics unchanged.',
        functional_conversion='Scalar c × symptom-gated modeled damage × reference area; c in {0.5,0.75,1}. Declared transfer sensitivity, not an observed damage-function calibration.',
        primary_outcomes='Modeled upper-leaf event timing relative to anthesis and normalized upper-three-leaf HAD proxy loss, GS65–85.',
        broader_window='GS31–85 also computed in every setting; avoids presenting a selected window alone.',
        yield_transfer='Diagnostic only: HAD proxy × fixed .0141/.018/.0207; conditional t/ha per unit nominal upper-three reference LAI. No European tonnage, absolute yield or policy benefit.',
        limitations=['Retrospective disease discrimination is poor on stricter validation.',
            'Disease magnitude, source-route attribution and reference senescence are unvalidated.',
            'Winter-wheat/rainfed calendar scenario applied to all-wheat area; no cultivar, sowing adaptation or management benefits estimated.',
            'Fixed positive sources tend to saturate event occurrence; conditional timing and burden are the modeled outputs.'],
        full_domain_census=False,source_sha256=sources,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    write_json(HERE/'registration_before_robustness_results.json',record)
    for p in dependencies:
        if p.suffix in ('.py','.json','.csv') and p.parent!=HERE:
            target=HERE/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    return record


def reference_decline(stage_indices,days,end_factor):
    anthesis=np.asarray(stage_indices[65],int)[:,None]
    end=np.asarray(stage_indices[85],int)[:,None]
    progress=np.clip((np.arange(days)[None,:]-anthesis)/np.maximum(end-anthesis,1),0.,1.)
    return 1.-(1.-end_factor)*progress


def shared_mcse(influences,draws):
    return float(np.sqrt(original.stratified_variance(np.mean(influences,axis=0),draws)))


def had_transfer(base,daily,decline_end,conversion):
    rows=base.copy()
    stage=daily['host'].stage_day_index
    days=daily['reference_area'].shape[1]
    reference=daily['reference_area']*reference_decline(stage,days,decline_end)[:,:,None]
    functional=daily['functional_loss']*conversion
    for i in range(len(rows)):
        end=int(stage[85][i])
        for window,code in [('GS31_85',31),('GS65_85',65)]:
            start=int(stage[code][i]);valid=bool(rows.valid_complete_season.iloc[i]) and start>=0
            ref=float(reference[i,start:end+1].sum()) if valid else np.nan
            lost=float((reference[i,start:end+1]*functional[i,start:end+1]).sum()) if valid else np.nan
            assert not valid or 0<=lost<=ref+1e-10
            rows.loc[i,f'{window}_reference_had3']=ref
            rows.loc[i,f'{window}_lost_had3']=lost
            rows.loc[i,f'{window}_functional_lost_fraction']=lost/ref if ref>0 else np.nan
            for label,slope in zip(['b0141','b0180','b0207'],original.SLOPES):
                rows.loc[i,f'conditional_yield_loss_{window}_{label}_t_ha_per_unit_lai']=lost*slope
    return rows


def run_gcm(model,registration):
    draws=pd.read_csv(original.cache.DRAW_PATH)
    phenology=json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    coeff_frame=pd.read_parquet(original.coefficient_path(model,SCENARIO))
    assert not coeff_frame.duplicated(['cell_id','month']).any()
    coeff=coeff_frame.set_index(['cell_id','month']).reindex(pd.MultiIndex.from_product([draws.cell_id,range(1,13)],names=['cell_id','month']))
    coeff={k:coeff[k].to_numpy().reshape(64,12) for k in ['temperature_offset','humidity_logit_offset','precipitation_ratio']}
    reports=[]
    for number,year in enumerate(YEARS):
        path=HERE/'annual_outputs'/model/f'{year}.parquet';receipt=path.with_suffix('.json')
        if path.exists() or receipt.exists():
            assert path.exists() and receipt.exists();r=json.loads(receipt.read_text());assert sha(path)==r['output_sha256'];assert r['registration_sha256']==sha(HERE/'registration_before_robustness_results.json');reports.append(r);continue
        started=time.perf_counter()
        prev,pw=original.load_cache_year(model,SCENARIO,year-1,draws);cur,cw=original.load_cache_year(model,SCENARIO,year,draws)
        dates=pd.DatetimeIndex(np.concatenate([prev,cur]));weather=apply_monthly_alignment({k:np.concatenate([pw[k],cw[k]],axis=1) for k in original.cache.FIELDS},dates.month,coeff)
        sow=np.array([sowing_date_from_calendar(year,r.planting_doy,r.maturity_doy).to_datetime64().astype('datetime64[D]') for r in draws.itertuples()])
        results=[];maximum_error=0.;baseline=None;baseline_daily=None
        for name,setting in registration['settings'].items():
            if setting['kind']=='mechanism':
                thresholds={int(k):v for k,v in setting['thresholds'].items()}
                rows,daily=simulation.simulate_draw_seasons(dates,weather,sow,draws.latitude.to_numpy(),phenology,setting['fitted'],thresholds,
                    return_daily=True,initial_local_source=setting['initial_local_source'],detection_fraction=setting['detection_fraction'])
                maximum_error=max(maximum_error,daily['mass_error'])
                if name=='baseline':baseline=rows;baseline_daily=daily
            elif setting['kind']=='daily_detection':
                assert baseline_daily is not None
                traj=baseline_daily['trajectory'];cutoff=setting['detection_fraction'];mask=baseline_daily['crop_mask']
                alt=replace(traj,infection_day=_first_crossing(traj.state[...,1:].sum(axis=-1),cutoff,mask),symptom_day=_first_crossing(traj.damage,cutoff,mask))
                baseline_setting=registration['settings']['baseline']
                rows,_=simulation.simulate_draw_seasons(dates,weather,sow,draws.latitude.to_numpy(),phenology,baseline_setting['fitted'],
                    {int(k):v for k,v in baseline_setting['thresholds'].items()},detection_fraction=cutoff,trajectory_override=alt)
            else:
                rows=had_transfer(baseline,baseline_daily,setting['reference_decline_end_factor'],setting['functional_conversion'])
            for k in ['spatial_draw_id','cell_id','stratum','draw_in_stratum','area_mean_weight','latitude','longitude']:rows[k]=draws[k].to_numpy()
            rows['setting']=name;rows['model']=model;rows['scenario']=SCENARIO;rows['harvest_year']=year;rows['period']='1991-2020' if year<=2020 else '2071-2100'
            results.append(rows)
        output=pd.concat(results,ignore_index=True)
        path.parent.mkdir(parents=True,exist_ok=True);output.to_parquet(path,index=False)
        daily_path=HERE/'daily_baseline'/model/f'{year}.npz';daily_path.parent.mkdir(parents=True,exist_ok=True)
        bd=baseline_daily
        np.savez_compressed(daily_path,dates=dates.to_numpy(dtype='datetime64[D]'),spatial_draw_id=draws.spatial_draw_id.to_numpy(),
            crop_mask=bd['crop_mask'],reference_area=bd['reference_area'].astype(np.float32),functional_loss=bd['functional_loss'].astype(np.float32),
            damage=bd['trajectory'].damage[:,1:,:3].astype(np.float32),affected=bd['trajectory'].state[:,1:,:3,1:].sum(axis=-1).astype(np.float32),
            BBCH31_index=bd['host'].stage_day_index[31],BBCH65_index=bd['host'].stage_day_index[65],BBCH85_index=bd['host'].stage_day_index[85])
        report=dict(model=model,year=year,settings=len(results),rows=len(output),maximum_mass_error=maximum_error,seconds=time.perf_counter()-started,
            output_sha256=sha(path),baseline_daily_sha256=sha(daily_path),registration_sha256=sha(HERE/'registration_before_robustness_results.json'))
        write_json(receipt,report);reports.append(report)
        if (number+1)%10==0:print(f'{model}: {number+1}/60 GCM-years; {len(output)} rows/year; {report["seconds"]:.2f} s/year',flush=True)
    return reports


def summarize(seasons,draws,registration):
    metrics=original.metric_columns(seasons)
    settings=list(registration['settings'])
    records=[];moments=[];periods=[];ensembles=[]
    for model in MODELS:
        ordered={}
        for setting in settings:
            a=seasons.loc[seasons.model.eq(model)&seasons.setting.eq(setting)&seasons.period.eq('1991-2020')].sort_values(['spatial_draw_id','harvest_year'])
            b=seasons.loc[seasons.model.eq(model)&seasons.setting.eq(setting)&seasons.period.eq('2071-2100')].sort_values(['spatial_draw_id','harvest_year'])
            assert len(a)==len(b)==1920
            assert np.array_equal(a.spatial_draw_id,b.spatial_draw_id)
            ordered[setting]={metric:(a[metric].to_numpy(float).reshape(64,30),b[metric].to_numpy(float).reshape(64,30)) for metric in metrics}
        for metric in metrics:
            common_all=np.logical_and.reduce([np.isfinite(ordered[s][metric][0])&np.isfinite(ordered[s][metric][1]) for s in settings])
            for setting in settings:
                a,b=ordered[setting][metric]
                masks={'within_setting_paired':np.isfinite(a)&np.isfinite(b),'all_settings_common_paired':common_all}
                for population,common in masks.items():
                    x=np.where(common,b-a,0.).mean(axis=1);z=common.mean(axis=1)
                    r=np.where(common,a,0.).mean(axis=1);f=np.where(common,b,0.).mean(axis=1)
                    d=original.ratio_summary(x,z,draws);rs=original.ratio_summary(r,z,draws);fs=original.ratio_summary(f,z,draws)
                    if np.isfinite(d['mean']):assert abs(fs['mean']-rs['mean']-d['mean'])<1e-10
                    records.append(dict(setting=setting,model=model,scenario=SCENARIO,metric=metric,population=population,
                        reference_period='1991-2020',future_period='2071-2100',reference_on_common=rs['mean'],future_on_common=fs['mean'],change=d['mean'],
                        spatial_mcse_change=d['spatial_mcse'],common_paired_area_time_coverage=d['coverage'],common_draw_year_pairs=int(common.sum()),
                        draws_complete_all30_pairs=int(common.all(axis=1).sum()),timing_conditioned_on_detection='relative_anthesis' in metric or 'day_after_sowing' in metric))
                    for i,row in draws.iterrows():moments.append(dict(setting=setting,model=model,metric=metric,population=population,
                        spatial_draw_id=int(row.spatial_draw_id),stratum=row.stratum,area_mean_weight=row.area_mean_weight,numerator=float(x[i]),
                        denominator=float(z[i]),reference_numerator=float(r[i]),future_numerator=float(f[i]),valid_year_pairs=int(common[i].sum())))
    summary=pd.DataFrame(records);mom=pd.DataFrame(moments)
    for (setting,metric,population),g in summary.groupby(['setting','metric','population'],sort=True):
        influences=[]
        for row in g.itertuples():
            part=mom.loc[mom.setting.eq(setting)&mom.model.eq(row.model)&mom.metric.eq(metric)&mom.population.eq(population)].sort_values('spatial_draw_id')
            influences.append((part.numerator.to_numpy()-row.change*part.denominator.to_numpy())/row.common_paired_area_time_coverage)
        ensembles.append(dict(setting=setting,scenario=SCENARIO,metric=metric,population=population,reference_period='1991-2020',future_period='2071-2100',
            gcm_mean_reference=float(g.reference_on_common.mean()),gcm_mean_future=float(g.future_on_common.mean()),gcm_mean_change=float(g.change.mean()),
            gcm_min_change=float(g.change.min()),gcm_max_change=float(g.change.max()),gcm_count=len(g),spatial_mcse_gcm_mean_change=shared_mcse(influences,draws),
            minimum_gcm_paired_coverage=float(g.common_paired_area_time_coverage.min()),maximum_gcm_paired_coverage=float(g.common_paired_area_time_coverage.max()),
            minimum_gcm_draw_year_pairs=int(g.common_draw_year_pairs.min()),maximum_gcm_draw_year_pairs=int(g.common_draw_year_pairs.max()),
            gcms_positive=int((g.change>0).sum()),gcms_negative=int((g.change<0).sum())))
    return summary,mom,pd.DataFrame(ensembles)


def finish(registration,reports):
    frames=[pd.read_parquet(HERE/'annual_outputs'/r['model']/f'{r["year"]}.parquet') for r in reports]
    seasons=pd.concat(frames,ignore_index=True).sort_values(['setting','model','harvest_year','spatial_draw_id']).reset_index(drop=True)
    assert len(seasons)==registration['expected_draw_season_rows'] and not seasons.duplicated(['setting','model','harvest_year','spatial_draw_id']).any()
    seasons.to_parquet(HERE/'draw_season_outputs.parquet',index=False)
    draws=pd.read_csv(original.cache.DRAW_PATH)
    summary,mom,ensemble=summarize(seasons,draws,registration)
    for name,frame in [('paired_changes_by_setting_gcm',summary),('paired_draw_moments',mom),('ensemble_paired_changes',ensemble)]:
        frame.to_parquet(HERE/f'{name}.parquet',index=False)
        if name!='paired_draw_moments':frame.to_csv(HERE/f'{name}.csv',index=False)
    coverage=seasons.groupby(['setting','model','period','status']).agg(rows=('spatial_draw_id','size'),area_time_coverage=('area_mean_weight','sum')).reset_index();coverage.area_time_coverage/=30
    coverage.to_csv(HERE/'coverage_by_setting_gcm_period_status.csv',index=False)
    focus=['F1_infection_relative_anthesis_days','F1_symptom_relative_anthesis_days','F2_infection_relative_anthesis_days','F2_symptom_relative_anthesis_days',
        'F3_infection_relative_anthesis_days','F3_symptom_relative_anthesis_days','GS65_85_lost_had3','GS31_85_lost_had3','GS65_85_functional_lost_fraction',
        'conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai','conditional_yield_loss_GS31_85_b0180_t_ha_per_unit_lai','any_top3_infection_before85','any_top3_symptom_before85']
    ensemble.loc[ensemble.metric.isin(focus)].to_csv(HERE/'headline_robustness.csv',index=False)
    old=pd.read_parquet(original.HERE/'draw_season_outputs.parquet')
    old=old.loc[old.scenario.eq(SCENARIO)&old.harvest_year.isin(YEARS)].sort_values(['model','harvest_year','spatial_draw_id']).reset_index(drop=True)
    base=seasons.loc[seasons.setting.eq('baseline')].sort_values(['model','harvest_year','spatial_draw_id']).reset_index(drop=True)
    metrics=original.metric_columns(old);errors={}
    for metric in metrics:
        a,b=old[metric].to_numpy(float),base[metric].to_numpy(float);assert np.array_equal(np.isfinite(a),np.isfinite(b)),metric
        errors[metric]=float(np.max(np.abs(a[np.isfinite(a)]-b[np.isfinite(b)]))) if np.isfinite(a).any() else 0.
    assert max(errors.values())<1e-10
    for name,h in registration['source_sha256'].items():assert sha(ROOT/name)==h,name
    write_json(HERE/'completion_receipt.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),settings=len(registration['settings']),
        GCM_year_batches=len(reports),trajectory_batches=7*len(reports),draw_season_rows=len(seasons),spatial_draws=64,full_domain_census=False,
        maximum_mass_error=max(r['maximum_mass_error'] for r in reports),baseline_replay_maximum_numeric_difference=max(errors.values()),
        baseline_replay_checked_metric_count=len(metrics),baseline_replay_same_missingness=True,registration_sha256=sha(HERE/'registration_before_robustness_results.json'),
        source_sha256_verified=True,total_worker_seconds=float(sum(r['seconds'] for r in reports)),
        output_sha256={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.csv','.parquet')}))
    print('COMPLETE',len(seasons),'rows;',len(registration['settings']),'settings; baseline replay error',max(errors.values()),flush=True)


def main():
    args=sys.argv[1:]
    if args==['--register']:
        r=register();print('REGISTERED',len(r['settings']),'settings; 1260 trajectory batches');return
    path=HERE/'registration_before_robustness_results.json'
    if not path.exists():raise RuntimeError('Run --register before any robustness outputs.')
    registration=json.loads(path.read_text())
    for name,h in registration['source_sha256'].items():assert sha(ROOT/name)==h,name
    reports=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_gcm,m,registration) for m in MODELS]
        for future in as_completed(futures):reports.extend(future.result())
    assert len(reports)==180
    finish(registration,reports)


if __name__=='__main__':main()
