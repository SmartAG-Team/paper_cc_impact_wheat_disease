"""Conditional mechanistic climate replay on 64 registered stratified draws."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

from model.seasonal_septoria.overwinter import OverwinterParameters,simulate_overwinter
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.regional import tpv_accumulation,phenology_window
from model.seasonal_septoria.host import sowing_date_from_calendar
from model.seasonal_septoria.wetness import duration_exposure
from model.seasonal_septoria.climate_alignment import apply_monthly_alignment
from . import cache_weather as cache

ROOT=cache.ROOT
HERE=Path(__file__).resolve().parent
MODELS,SCENARIOS,YEARS=cache.MODELS,cache.SCENARIOS,cache.YEARS
PERIODS={'1991-2020':list(range(1991,2021)),'2031-2060':list(range(2031,2061)),'2071-2100':list(range(2071,2101))}
FIT=HERE.parent/'disease/overwinter_source_model/frozen_selected_fit.json'
STAGES=HERE.parent/'phenology/calibrated_stage_thresholds.json'
SLOPES=[.0141,.018,.0207]


def sha(path):return cache.sha(path)


def frozen_json(path,value):return cache.frozen_json(path,value)


def simulate_draw_seasons(dates,weather,sowing,latitude,phenology,fit,thresholds,*,return_daily=False):
    """Respect actual dates and crop mask; no targets or observed stages enter."""
    dates=pd.DatetimeIndex(dates);raw={key:np.asarray(weather[key],float) for key in cache.FIELDS}
    sowing=np.asarray(sowing,dtype='datetime64[D]');fields,days=raw['tmean_c'].shape
    if not dates.equals(pd.date_range(dates[0],dates[-1])) or len(dates)!=days or sowing.shape!=(fields,):
        raise ValueError('Consecutive dated forcing and one calendar sowing date per draw are required.')
    included=dates.to_numpy(dtype='datetime64[D]')[None,:]>=sowing[:,None]
    finite=np.logical_and.reduce([np.isfinite(raw[key]) for key in cache.FIELDS])
    finite&=(raw['tmax_c']>=raw['tmean_c']-.001)&(raw['rh_mean_pct']>=0)&(raw['rh_mean_pct']<=100)&(raw['precipitation_mm']>=0)
    missing=included&~finite
    first_gap=np.where(missing.any(axis=1),missing.argmax(axis=1),days)
    prefix=included&(np.arange(days)[None,:]<first_gap[:,None])
    safe={key:np.where(finite,raw[key],0.) for key in cache.FIELDS}
    mask,complete,endpoint,unsupported=phenology_window(safe['tmean_c'],safe['tmax_c'],np.asarray(latitude,float),dates.dayofyear,
        prefix,phenology,float(thresholds[85]),True)
    valid=complete&~unsupported
    status=np.where(unsupported,'unsupported_mean_temperature_above40',np.where(complete,'complete',
        np.where(first_gap<days,'missing_weather_before_endpoint','soft_dough_not_reached')))
    mask&=valid[:,None]
    t=np.where(mask,safe['tmean_c'],0.);tx=np.where(mask,safe['tmax_c'],0.)
    a=tpv_accumulation(t,tx,np.asarray(latitude,float),dates.dayofyear,mask,phenology,True)
    host=leaf_host(a,t,thresholds,rank_spacing_units=fit['rank_spacing_units'],forcing_mask=mask,
        juvenile_policy=fit.get('juvenile_policy','handover_31_39'))
    exposure=duration_exposure(t,tx,np.where(mask,safe['rh_mean_pct'],0.),np.where(mask,safe['precipitation_mm'],0.),
        fit['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    trajectory=simulate_overwinter(t,exposure,np.where(mask,safe['precipitation_mm'],0.),host.active,host.renewal,host.area,mask,
        OverwinterParameters(**fit['parameters']),initial_local_source=1.,
        imported_pressure=np.full_like(t,fit['constant_imported_pressure']),detection_fraction=.001)
    mass_error=float(np.max(abs(trajectory.state.sum(axis=-1)-1.)))
    if mass_error>1e-10:raise AssertionError('Tissue mass accounting failed.')
    functional=trajectory.damage[:,1:,:3].copy()
    day=np.arange(1,days+1)[None,:,None]
    symptom=trajectory.symptom_day[:,:3]
    functional*=((symptom[:,None,:]>=0)&(day>=symptom[:,None,:]))
    functional*=mask[:,:,None]
    reference=host.area[:,:,:3]/3.
    rows=[]
    for field in range(fields):
        row=dict(status=str(status[field]),valid_complete_season=bool(valid[field]),calendar_sowing_date=pd.Timestamp(sowing[field]),
            BBCH31_date=pd.NaT,BBCH65_date=pd.NaT,BBCH85_date=pd.NaT,grain_fill_days=np.nan)
        stage_indices={stage:int(index[field]) for stage,index in host.stage_day_index.items()}
        for stage in [31,65,85]:
            index=stage_indices[stage];row[f'BBCH{stage}_date']=pd.NaT if index<0 else dates[index]
        anthesis=stage_indices[65];end=stage_indices[85];stem=stage_indices[31]
        gf=np.zeros(days,bool)
        if valid[field]:gf[anthesis:end+1]=True;row['grain_fill_days']=float(gf.sum())
        infected=[];expressed=[];events_in_grain=[]
        for leaf in range(3):
            key=f'F{leaf+1}';infection=int(trajectory.infection_day[field,leaf]);visible=int(trajectory.symptom_day[field,leaf])
            inf_date=pd.NaT if infection<0 or not valid[field] else dates[infection-1]
            sym_date=pd.NaT if visible<0 or not valid[field] else dates[visible-1]
            inf_overlap=(np.arange(1,days+1)>=infection)&(infection>=0)&gf
            sym_overlap=(np.arange(1,days+1)>=visible)&(visible>=0)&gf
            event_in_grain=bool(infection>=anthesis+1 and infection<=end+1 and infection>=0)
            row.update({f'{key}_infection_date':inf_date,f'{key}_symptom_date':sym_date,
                f'{key}_infection_day_after_sowing':np.nan if pd.isna(inf_date) else float((inf_date-pd.Timestamp(sowing[field])).days),
                f'{key}_symptom_day_after_sowing':np.nan if pd.isna(sym_date) else float((sym_date-pd.Timestamp(sowing[field])).days),
                f'{key}_infection_relative_anthesis_days':np.nan if pd.isna(inf_date) or anthesis<0 else float((inf_date-dates[anthesis]).days),
                f'{key}_symptom_relative_anthesis_days':np.nan if pd.isna(sym_date) or anthesis<0 else float((sym_date-dates[anthesis]).days),
                f'{key}_infection_before85':np.nan if not valid[field] else float(infection>=0 and infection<=end+1),
                f'{key}_symptom_before85':np.nan if not valid[field] else float(visible>=0 and visible<=end+1),
                f'{key}_infection_during_grain_fill':np.nan if not valid[field] else float(event_in_grain),
                f'{key}_infected_grain_fill_days':np.nan if not valid[field] else float(inf_overlap.sum()),
                f'{key}_symptomatic_grain_fill_days':np.nan if not valid[field] else float(sym_overlap.sum()),
                f'{key}_infected_grain_fill_fraction':np.nan if not valid[field] else float(inf_overlap.sum()/gf.sum()),
                f'{key}_symptomatic_grain_fill_fraction':np.nan if not valid[field] else float(sym_overlap.sum()/gf.sum())})
            infected.append(inf_overlap);expressed.append(sym_overlap);events_in_grain.append(event_in_grain)
        before=[row[f'F{leaf}_infection_before85'] for leaf in (1,2,3)]
        sym_before=[row[f'F{leaf}_symptom_before85'] for leaf in (1,2,3)]
        row.update(any_top3_infection_before85=np.nan if not valid[field] else float(any(before)),
            all_top3_infection_before85=np.nan if not valid[field] else float(all(before)),
            any_top3_symptom_before85=np.nan if not valid[field] else float(any(sym_before)),
            all_top3_symptom_before85=np.nan if not valid[field] else float(all(sym_before)),
            any_top3_infection_during_grain_fill=np.nan if not valid[field] else float(any(events_in_grain)),
            days_any_top3_infected_grain_fill=np.nan if not valid[field] else float(np.any(infected,axis=0).sum()),
            days_all_top3_infected_grain_fill=np.nan if not valid[field] else float(np.all(infected,axis=0).sum()),
            days_any_top3_symptomatic_grain_fill=np.nan if not valid[field] else float(np.any(expressed,axis=0).sum()),
            days_all_top3_symptomatic_grain_fill=np.nan if not valid[field] else float(np.all(expressed,axis=0).sum()))
        for window,start in [('GS31_85',stem),('GS65_85',anthesis)]:
            ref,loss=np.nan,np.nan
            if valid[field] and start>=0:
                ref=float(reference[field,start:end+1].sum());loss=float((reference[field,start:end+1]*functional[field,start:end+1]).sum())
                assert 0<=loss<=ref+1e-10
            row[f'{window}_reference_had3']=ref;row[f'{window}_lost_had3']=loss
            row[f'{window}_functional_lost_fraction']=np.nan if not np.isfinite(ref) or ref<=0 else loss/ref
            for label,slope in zip(['b0141','b0180','b0207'],SLOPES):row[f'conditional_yield_loss_{window}_{label}_t_ha_per_unit_lai']=loss*slope
        row['mean_temperature_grain_fill_c']=np.nan if not valid[field] else float(safe['tmean_c'][field,gf].mean())
        row['rainfall_grain_fill_mm']=np.nan if not valid[field] else float(safe['precipitation_mm'][field,gf].sum())
        row['exposure_grain_fill_effective_days']=np.nan if not valid[field] else float(exposure[field,gf].sum())
        rows.append(row)
    details=dict(mass_error=mass_error,complete_count=int(valid.sum()))
    if return_daily:details.update(trajectory=trajectory,host=host,accumulation=a,crop_mask=mask,functional_loss=functional,reference_area=reference)
    return pd.DataFrame(rows),details


def stratified_variance(influence,draws):
    values=np.asarray(influence,float);variance=0.
    for _,group in draws.groupby('stratum',sort=False):
        idx=group.index.to_numpy(int);n=len(idx);weight=float(group.area_mean_weight.sum())
        if n!=4:raise ValueError('Four independent registered draws per stratum are required.')
        variance+=weight**2*float(np.var(values[idx],ddof=1))/n
    return variance


def ratio_summary(numerator,denominator,draws):
    x,z=np.asarray(numerator,float),np.asarray(denominator,float);weight=draws.area_mean_weight.to_numpy(float)
    coverage=float(weight@z)
    if coverage<=0:return dict(mean=np.nan,spatial_mcse=np.nan,coverage=coverage)
    mean=float(weight@x)/coverage;influence=(x-mean*z)/coverage
    return dict(mean=mean,spatial_mcse=float(np.sqrt(stratified_variance(influence,draws))),coverage=coverage)


def load_cache_year(model,scenario,year,draws):
    path=cache.source_path(model,scenario,year);relative=path.relative_to(ROOT/'data/paper_study/climate/nasa_v2/daily')
    archive=HERE/'annual_weather_cache'/relative.with_suffix('.npz');receipt=json.loads(archive.with_suffix('.json').read_text())
    assert sha(archive)==receipt['cache_sha256']
    with np.load(archive) as loaded:
        dates=loaded['dates'].copy();ids=loaded['cell_ids'].tolist();mapping={cell:index for index,cell in enumerate(ids)}
        indices=draws.cell_id.map(mapping).to_numpy(int);weather={field:loaded[field][indices].copy() for field in cache.FIELDS}
    return dates,weather


def coefficient_path(model,scenario):return ROOT/'analysis/paper_study/regional_v1/bias_alignment'/model/scenario/'coefficients.parquet'


def run_gcm(model,configuration):
    draws=pd.read_csv(cache.DRAW_PATH);phenology=json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    fit=json.loads(FIT.read_text())['fitted'];thresholds={int(key):value for key,value in json.loads(STAGES.read_text())['all_stage_thresholds'].items()}
    reports=[]
    for scenario in SCENARIOS:
        frame=pd.read_parquet(coefficient_path(model,scenario))
        assert not frame.duplicated(['cell_id','month']).any()
        coeff=frame.set_index(['cell_id','month']).reindex(pd.MultiIndex.from_product([draws.cell_id,range(1,13)],names=['cell_id','month']))
        coeff={key:coeff[key].to_numpy().reshape(len(draws),12) for key in ['temperature_offset','humidity_logit_offset','precipitation_ratio']}
        for number,year in enumerate(YEARS):
            output=HERE/'annual_outputs'/model/scenario/f'{year}.parquet';receipt=output.with_suffix('.json')
            if output.exists() or receipt.exists():
                assert output.exists() and receipt.exists();report=json.loads(receipt.read_text())
                assert sha(output)==report['output_sha256'] and report['configuration_sha256']==sha(HERE/'configuration_before_climate_results.json')
                reports.append(report);continue
            previous,pw=load_cache_year(model,scenario,year-1,draws);current,cw=load_cache_year(model,scenario,year,draws)
            dates=pd.DatetimeIndex(np.concatenate([previous,current]));weather={key:np.concatenate([pw[key],cw[key]],axis=1) for key in cache.FIELDS}
            weather=apply_monthly_alignment(weather,dates.month,coeff)
            sow=np.array([sowing_date_from_calendar(year,row.planting_doy,row.maturity_doy).to_datetime64().astype('datetime64[D]') for row in draws.itertuples()])
            result,checks=simulate_draw_seasons(dates,weather,sow,draws.latitude.to_numpy(),phenology,fit,thresholds)
            for key in ['spatial_draw_id','cell_id','stratum','draw_in_stratum','area_mean_weight','stratum_ha','harvested_total_ha','latitude','longitude']:
                result[key]=draws[key].to_numpy()
            result['model']=model;result['scenario']=scenario;result['harvest_year']=year
            result['period']=next(period for period,years in PERIODS.items() if year in years)
            output.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(output,index=False)
            report=dict(model=model,scenario=scenario,harvest_year=year,spatial_draw_rows=len(result),
                status_counts=result.status.value_counts().to_dict(),maximum_mass_error=checks['mass_error'],
                valid_area_fraction=float(draws.area_mean_weight.to_numpy()@result.valid_complete_season.to_numpy()),
                configuration_sha256=sha(HERE/'configuration_before_climate_results.json'),output_sha256=sha(output),
                source_fit_sha256=configuration['source_sha256'][str(FIT.relative_to(ROOT))],full_domain_census=False)
            frozen_json(receipt,report);reports.append(report)
            if (number+1)%10==0:print(f'mechanistic replay {model}/{scenario}: {number+1}/90 seasons',flush=True)
    return reports


def metric_columns(frame):
    identity={'valid_complete_season','spatial_draw_id','draw_in_stratum','area_mean_weight','stratum_ha','harvested_total_ha','latitude','longitude','harvest_year'}
    return [key for key in frame.select_dtypes(include=['number']).columns if key not in identity]


def statistical_summaries(seasons,draws):
    metrics=metric_columns(seasons);annual=[];period_records=[];moments=[];paired_records=[];paired_moments=[]
    for (model,scenario,year),group in seasons.groupby(['model','scenario','harvest_year'],sort=True):
        group=group.sort_values('spatial_draw_id');base=dict(model=model,scenario=scenario,harvest_year=year,period=group.period.iloc[0],
            complete_season_draws=int(group.valid_complete_season.sum()),valid_area_fraction=float(draws.area_mean_weight@group.valid_complete_season.to_numpy()))
        for metric in metrics:
            y=group[metric].to_numpy(float);valid=np.isfinite(y);summary=ratio_summary(np.where(valid,y,0.),valid.astype(float),draws)
            base[metric]=summary['mean'];base[metric+'_area_coverage']=summary['coverage']
        annual.append(base)
    for (model,scenario,period),group in seasons.groupby(['model','scenario','period'],sort=True):
        group=group.sort_values(['spatial_draw_id','harvest_year']);assert len(group)==64*30
        for metric in metrics:
            y=group[metric].to_numpy(float).reshape(64,30);valid=np.isfinite(y)
            x=np.where(valid,y,0.).mean(axis=1);z=valid.mean(axis=1);summary=ratio_summary(x,z,draws)
            period_records.append(dict(model=model,scenario=scenario,period=period,metric=metric,mean=summary['mean'],spatial_mcse=summary['spatial_mcse'],
                metric_area_time_coverage=summary['coverage'],valid_draw_seasons=int(valid.sum()),draws_complete_all30=int(valid.all(axis=1).sum()),
                metric_conditioned_on_event_detection=('relative_anthesis' in metric or 'day_after_sowing' in metric)))
            for index,draw in draws.iterrows():moments.append(dict(model=model,scenario=scenario,period=period,metric=metric,spatial_draw_id=int(draw.spatial_draw_id),
                stratum=draw.stratum,area_mean_weight=draw.area_mean_weight,numerator=float(x[index]),denominator=float(z[index]),valid_years=int(valid[index].sum())))
    for model in MODELS:
        for scenario in SCENARIOS:
            reference=seasons.loc[seasons.model.eq(model)&seasons.scenario.eq(scenario)&seasons.period.eq('1991-2020')].sort_values(['spatial_draw_id','harvest_year'])
            for period in ['2031-2060','2071-2100']:
                future=seasons.loc[seasons.model.eq(model)&seasons.scenario.eq(scenario)&seasons.period.eq(period)].sort_values(['spatial_draw_id','harvest_year'])
                for metric in metrics:
                    a=reference[metric].to_numpy(float).reshape(64,30);b=future[metric].to_numpy(float).reshape(64,30);common=np.isfinite(a)&np.isfinite(b)
                    x=np.where(common,b-a,0.).mean(axis=1);z=common.mean(axis=1);summary=ratio_summary(x,z,draws)
                    rx=np.where(common,a,0.).mean(axis=1);fx=np.where(common,b,0.).mean(axis=1)
                    r=ratio_summary(rx,z,draws)['mean'];f=ratio_summary(fx,z,draws)['mean']
                    if np.isfinite(summary['mean']):assert abs((f-r)-summary['mean'])<1e-10
                    paired_records.append(dict(model=model,scenario=scenario,reference_period='1991-2020',future_period=period,metric=metric,
                        change=summary['mean'],spatial_mcse=summary['spatial_mcse'],reference_on_common=r,future_on_common=f,
                        common_paired_area_time_coverage=summary['coverage'],common_draw_year_pairs=int(common.sum()),draws_complete_all30_pairs=int(common.all(axis=1).sum())))
                    for index,draw in draws.iterrows():paired_moments.append(dict(model=model,scenario=scenario,future_period=period,metric=metric,
                        spatial_draw_id=int(draw.spatial_draw_id),stratum=draw.stratum,area_mean_weight=draw.area_mean_weight,
                        numerator=float(x[index]),denominator=float(z[index]),reference_numerator=float(rx[index]),future_numerator=float(fx[index]),valid_year_pairs=int(common[index].sum())))
    periods=pd.DataFrame(period_records);paired=pd.DataFrame(paired_records);pm=pd.DataFrame(moments);dm=pd.DataFrame(paired_moments)
    ensemble_period=[];ensemble_change=[]
    for (scenario,period,metric),group in periods.groupby(['scenario','period','metric']):
        influences=[]
        for row in group.itertuples():
            part=pm.loc[pm.model.eq(row.model)&pm.scenario.eq(scenario)&pm.period.eq(period)&pm.metric.eq(metric)].sort_values('spatial_draw_id')
            influences.append((part.numerator.to_numpy()-row.mean*part.denominator.to_numpy())/row.metric_area_time_coverage)
        ensemble_period.append(dict(scenario=scenario,period=period,metric=metric,gcm_mean=float(group['mean'].mean()),gcm_min=float(group['mean'].min()),
            gcm_max=float(group['mean'].max()),gcm_count=len(group),spatial_mcse_gcm_mean=float(np.sqrt(stratified_variance(np.mean(influences,axis=0),draws))),
            minimum_gcm_area_time_coverage=float(group.metric_area_time_coverage.min()),maximum_gcm_area_time_coverage=float(group.metric_area_time_coverage.max())))
    for (scenario,period,metric),group in paired.groupby(['scenario','future_period','metric']):
        influences=[]
        for row in group.itertuples():
            part=dm.loc[dm.model.eq(row.model)&dm.scenario.eq(scenario)&dm.future_period.eq(period)&dm.metric.eq(metric)].sort_values('spatial_draw_id')
            influences.append((part.numerator.to_numpy()-row.change*part.denominator.to_numpy())/row.common_paired_area_time_coverage)
        ensemble_change.append(dict(scenario=scenario,reference_period='1991-2020',future_period=period,metric=metric,gcm_mean_change=float(group.change.mean()),
            gcm_min_change=float(group.change.min()),gcm_max_change=float(group.change.max()),gcm_count=len(group),
            spatial_mcse_gcm_mean_change=float(np.sqrt(stratified_variance(np.mean(influences,axis=0),draws))),
            minimum_gcm_common_paired_coverage=float(group.common_paired_area_time_coverage.min()),maximum_gcm_common_paired_coverage=float(group.common_paired_area_time_coverage.max())))
    return dict(annual_area_weighted_means=pd.DataFrame(annual),period_means_by_gcm_ssp=periods,paired_period_changes_by_gcm_ssp=paired,
        draw_period_moments=pm,paired_draw_period_moments=dm,ensemble_period_means=pd.DataFrame(ensemble_period),ensemble_paired_period_changes=pd.DataFrame(ensemble_change))


def main():
    contract=HERE/'configuration_before_climate_results.json'
    if contract.exists():raise FileExistsError('Use a new immutable mechanistic climate archive.')
    cache_receipt=HERE/'weather_cache_receipt.json'
    if not cache_receipt.exists():raise FileNotFoundError('Complete verified raw annual weather cache first.')
    dependencies=[Path(__file__),HERE/'cache_weather.py',HERE/'test_climate.py',FIT,STAGES,HERE.parent/'disease/receipt.json',
        HERE.parent/'disease/independent_statistical_verification.json',cache.DRAW_PATH,
        ROOT/'data/paper_study/regional_parameter_uncertainty/sampling_receipt.json',cache_receipt,
        ROOT/'process_model/parameters/calibration.json',ROOT/'model/seasonal_septoria/overwinter.py',ROOT/'model/seasonal_septoria/leaf_phenology.py',
        ROOT/'model/seasonal_septoria/regional.py',ROOT/'model/seasonal_septoria/host.py',ROOT/'model/seasonal_septoria/wetness.py',
        ROOT/'model/seasonal_septoria/climate_alignment.py',*[coefficient_path(model,scenario) for model in MODELS for scenario in SCENARIOS]]
    hashes={str(path.relative_to(ROOT)):sha(path) for path in dependencies}
    draws=pd.read_csv(cache.DRAW_PATH);sampling=json.loads((ROOT/'data/paper_study/regional_parameter_uncertainty/sampling_receipt.json').read_text())
    configuration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),models=MODELS,scenarios=SCENARIOS,harvest_years=YEARS,periods=PERIODS,
        season_count=810,draw_count=64,unique_cells=62,expected_draw_season_rows=51840,strata=16,draws_per_stratum=4,replacement_sampling=True,
        eligible_calendar_cells=14932,area_registry_cells=14941,eligible_harvested_ha=sampling['eligible_harvested_ha'],
        source_weights='frozen spatial_draws.area_mean_weight; duplicate draw IDs retained',full_domain_census=False,
        model_source='frozen original28-field retrospective disease selection; no climate tuning',
        conditional_initial_local_source=1.,conditional_imported_pressure=json.loads(FIT.read_text())['fitted']['constant_imported_pressure'],
        juvenile_policy='handover_31_39',rainfall_intensity=json.loads(FIT.read_text())['fitted']['weather_preprocessing']['rain_rate_mm_hour'],
        calendar='unvalidated winter-wheat/rainfed scenario on all-wheat area; actual dates from registered calendars',
        development='frozen donor q1 T-P-V, vernalization required; stage-only ordered thresholds',
        origin='January1 preceding harvest year; actual sowing mask independent of array origin',
        endpoint='BBCH85; outside-crop weather cannot advance disease or residue clocks',
        invalid='missing weather before endpoint, unreached85, or meanT>40 before85; invalid outcomes remain missing',
        climate_alignment='existing historical-only monthly additive T/logit RH/multiplicative rain coefficients; mean alignment not full distributions',
        infection_definition='first modeled affected fraction >=0.001, conditional on inoculum; not validated observed infection date',
        symptom_definition='first modeled damage fraction >=0.001',functional_area='model damage as conditional functional loss, gated after modeled symptoms',
        reference_area='nominal maximum upper3LAI1; equal mature layer areas and modeled unfolding; no measured natural senescence',
        yield_slopes=SLOPES,yield_units='conditional t/ha per unit nominal upper3 reference LAI',coefficient_range_is_ci=False,
        reference_yield_invented=False,local_yield_calibration=False,
        periods_are_equal_year_area_time_ratio_estimators=True,
        paired_changes='same draw, same GCM/SSP, matched relative-year index; restrict to common valid metric coverage',
        spatial_mcse='four independent draws/stratum; ratio residual linearization; GCM-mean MCSE preserves shared draw covariance',
        climate_model_range_is_ci=False,legacy_calibration_bootstrap_reused=False,worker_count=3,source_sha256=hashes)
    frozen_json(contract,configuration)
    for path in dependencies:
        if path.suffix in ('.py','.json'):
            target=HERE/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    reports=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_gcm,model,configuration) for model in MODELS]
        for future in as_completed(futures):reports.extend(future.result())
    assert len(reports)==810
    frames=[pd.read_parquet(HERE/'annual_outputs'/record['model']/record['scenario']/f"{record['harvest_year']}.parquet") for record in reports]
    seasons=pd.concat(frames,ignore_index=True).sort_values(['model','scenario','harvest_year','spatial_draw_id']).reset_index(drop=True)
    assert len(seasons)==51840 and not seasons.duplicated(['model','scenario','harvest_year','spatial_draw_id']).any()
    seasons.to_parquet(HERE/'draw_season_outputs.parquet',index=False);seasons.to_csv(HERE/'draw_season_outputs.csv',index=False)
    summaries=statistical_summaries(seasons,draws)
    for name,frame in summaries.items():frame.to_parquet(HERE/f'{name}.parquet',index=False);frame.to_csv(HERE/f'{name}.csv',index=False)
    coverage=seasons.groupby(['model','scenario','period','status']).agg(draw_seasons=('spatial_draw_id','size'),
        sampled_area_time_weight=('area_mean_weight','sum')).reset_index();coverage.sampled_area_time_weight/=30
    coverage.to_csv(HERE/'coverage_by_period_and_status.csv',index=False)
    for name,value in hashes.items():assert sha(ROOT/name)==value,name
    frozen_json(HERE/'receipt.json',dict(status='complete',annual_seasons=810,draw_season_rows=51840,spatial_draws=64,unique_spatial_cells=62,
        eligible_harvested_ha=sampling['eligible_harvested_ha'],full_domain_census=False,conditional_model_scenario=True,
        disease_fit_validation_does_not_establish_route_identification=True,absolute_field_yield_predicted=False,legacy_bootstrap_reused=False,
        status_counts=seasons.status.value_counts().to_dict(),maximum_mass_error=max(record['maximum_mass_error'] for record in reports),
        configuration_sha256=sha(contract),source_sha256=hashes,
        output_sha256={str(path.relative_to(HERE)):sha(path) for path in HERE.rglob('*') if path.is_file()
            and 'source_snapshot' not in path.parts and '__pycache__' not in path.parts and 'annual_weather_cache' not in path.parts
            and path.name not in ('run.py','test_climate.py','cache_weather.py','receipt.json')}))


if __name__=='__main__':main()
