"""Independent paired ratio/MCSE arithmetic and actual climate prefix checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import run
from model.seasonal_septoria.overwinter import OverwinterParameters,simulate_overwinter
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.host import sowing_date_from_calendar
from model.seasonal_septoria.climate_alignment import apply_monthly_alignment
from model.seasonal_septoria.wetness import duration_exposure

HERE=Path(__file__).resolve().parent


def independent_summary(x,z,draws):
    area=draws.area_mean_weight.to_numpy(float)
    denominator=float(sum(area*z));mean=float(sum(area*x))/denominator
    residual=x-mean*z;variance=0.
    for stratum in draws.stratum.unique():
        idx=np.flatnonzero(draws.stratum.to_numpy()==stratum);n=len(idx);assert n==4
        mass=float(sum(area[idx]));center=float(sum(residual[idx]))/n
        sum_of_squares=float(sum((residual[idx]-center)**2))
        variance+=mass**2*sum_of_squares/(n*(n-1))
    return mean,float(np.sqrt(variance)/denominator),denominator,(x-mean*z)/denominator


def independent_influence_mcse(values,draws):
    variance=0.;area=draws.area_mean_weight.to_numpy(float)
    for stratum in draws.stratum.unique():
        idx=np.flatnonzero(draws.stratum.to_numpy()==stratum);n=len(idx);center=float(sum(values[idx]))/n
        variance+=float(sum(area[idx]))**2*float(sum((values[idx]-center)**2))/(n*(n-1))
    return float(np.sqrt(variance))


def main():
    receipt=json.loads((HERE/'receipt.json').read_text())
    for name,value in receipt['source_sha256'].items():assert run.sha(run.ROOT/name)==value,name
    for name,value in receipt['output_sha256'].items():assert run.sha(HERE/name)==value,name
    draws=pd.read_csv(run.cache.DRAW_PATH);seasons=pd.read_parquet(HERE/'draw_season_outputs.parquet')
    assert len(seasons)==51840 and seasons.groupby(['model','scenario','harvest_year']).size().eq(64).all()
    assert not seasons.duplicated(['model','scenario','harvest_year','spatial_draw_id']).any()
    assert draws.groupby('stratum').size().eq(4).all() and draws.cell_id.nunique()==62
    invalid=seasons.loc[~seasons.valid_complete_season]
    assert len(invalid)==27
    assert invalid[['GS65_85_lost_had3','any_top3_infection_before85','grain_fill_days']].isna().all().all()
    valid=seasons.loc[seasons.valid_complete_season]
    assert ((valid.BBCH85_date-valid.BBCH65_date).dt.days+1==valid.grain_fill_days).all()
    for leaf in (1,2,3):
        inf=pd.to_datetime(valid[f'F{leaf}_infection_date']);sym=pd.to_datetime(valid[f'F{leaf}_symptom_date'])
        start=valid.BBCH65_date;end=valid.BBCH85_date
        expected=np.where(inf.isna(),0.,np.maximum(0.,(end-pd.concat([inf,start],axis=1).max(axis=1)).dt.days+1))
        np.testing.assert_allclose(expected,valid[f'F{leaf}_infected_grain_fill_days'],rtol=0,atol=0)
        np.testing.assert_allclose(expected/valid.grain_fill_days,valid[f'F{leaf}_infected_grain_fill_fraction'],rtol=0,atol=1e-14)
        expected_sym=np.where(sym.isna(),0.,np.maximum(0.,(end-pd.concat([sym,start],axis=1).max(axis=1)).dt.days+1))
        np.testing.assert_allclose(expected_sym,valid[f'F{leaf}_symptomatic_grain_fill_days'],rtol=0,atol=0)
        observed=inf.notna();np.testing.assert_allclose(valid.loc[observed,f'F{leaf}_infection_day_after_sowing'],
            (inf[observed]-valid.loc[observed,'calendar_sowing_date']).dt.days,rtol=0,atol=0)
    for window in ('GS31_85','GS65_85'):
        assert (valid[f'{window}_lost_had3']<=valid[f'{window}_reference_had3']+1e-12).all()
        for label,slope in zip(['b0141','b0180','b0207'],run.SLOPES):
            np.testing.assert_allclose(valid[f'conditional_yield_loss_{window}_{label}_t_ha_per_unit_lai'],valid[f'{window}_lost_had3']*slope,rtol=0,atol=1e-14)
    metrics=run.metric_columns(seasons);metric_index={name:index for index,name in enumerate(metrics)}
    period_data={}
    for key,group in seasons.groupby(['model','scenario','period']):
        group=group.sort_values(['spatial_draw_id','harvest_year'])
        assert len(group)==64*30;period_data[key]=group[metrics].to_numpy(float).reshape(64,30,len(metrics))
    period=pd.read_csv(HERE/'period_means_by_gcm_ssp.csv');paired=pd.read_csv(HERE/'paired_period_changes_by_gcm_ssp.csv')
    period_influences={};pair_influences={};max_error=0.
    for row in period.itertuples():
        y=period_data[(row.model,row.scenario,row.period)][:,:,metric_index[row.metric]];finite=np.isfinite(y)
        x=np.where(finite,y,0.).mean(axis=1);z=finite.mean(axis=1);mean,error,coverage,influence=independent_summary(x,z,draws)
        discrepancy=max(abs(mean-row.mean),abs(error-row.spatial_mcse),abs(coverage-row.metric_area_time_coverage));max_error=max(max_error,discrepancy)
        assert discrepancy<1e-10
        assert int(finite.sum())==row.valid_draw_seasons and int(finite.all(axis=1).sum())==row.draws_complete_all30
        period_influences[(row.model,row.scenario,row.period,row.metric)]=influence
    for row in paired.itertuples():
        index=metric_index[row.metric];reference=period_data[(row.model,row.scenario,'1991-2020')][:,:,index]
        future=period_data[(row.model,row.scenario,row.future_period)][:,:,index];common=np.isfinite(reference)&np.isfinite(future)
        x=np.where(common,future-reference,0.).mean(axis=1);z=common.mean(axis=1);mean,error,coverage,influence=independent_summary(x,z,draws)
        r=independent_summary(np.where(common,reference,0.).mean(axis=1),z,draws)[0]
        f=independent_summary(np.where(common,future,0.).mean(axis=1),z,draws)[0]
        discrepancy=max(abs(mean-row.change),abs(error-row.spatial_mcse),abs(coverage-row.common_paired_area_time_coverage),
            abs(r-row.reference_on_common),abs(f-row.future_on_common));max_error=max(max_error,discrepancy)
        assert discrepancy<1e-10
        assert int(common.sum())==row.common_draw_year_pairs and int(common.all(axis=1).sum())==row.draws_complete_all30_pairs
        pair_influences[(row.model,row.scenario,row.future_period,row.metric)]=influence
    ep=pd.read_csv(HERE/'ensemble_period_means.csv');ed=pd.read_csv(HERE/'ensemble_paired_period_changes.csv')
    for row in ep.itertuples():
        g=period.loc[period.scenario.eq(row.scenario)&period.period.eq(row.period)&period.metric.eq(row.metric)]
        values=g['mean'].to_numpy();assert len(values)==3
        influence=np.mean([period_influences[(model,row.scenario,row.period,row.metric)] for model in run.MODELS],axis=0)
        np.testing.assert_allclose([row.gcm_mean,row.gcm_min,row.gcm_max,row.spatial_mcse_gcm_mean],
            [sum(values)/3,min(values),max(values),independent_influence_mcse(influence,draws)],rtol=0,atol=1e-10)
    for row in ed.itertuples():
        g=paired.loc[paired.scenario.eq(row.scenario)&paired.future_period.eq(row.future_period)&paired.metric.eq(row.metric)]
        values=g.change.to_numpy();assert len(values)==3
        influence=np.mean([pair_influences[(model,row.scenario,row.future_period,row.metric)] for model in run.MODELS],axis=0)
        np.testing.assert_allclose([row.gcm_mean_change,row.gcm_min_change,row.gcm_max_change,row.spatial_mcse_gcm_mean_change],
            [sum(values)/3,min(values),max(values),independent_influence_mcse(influence,draws)],rtol=0,atol=1e-10)
    annual=pd.read_csv(HERE/'annual_area_weighted_means.csv')
    groups={key:group.sort_values('spatial_draw_id') for key,group in seasons.groupby(['model','scenario','harvest_year'])}
    for row in annual.itertuples():
        group=groups[(row.model,row.scenario,row.harvest_year)]
        y=group[metrics].to_numpy(float);finite=np.isfinite(y);weight=draws.area_mean_weight.to_numpy(float)
        denominator=weight@finite;mean=(weight@np.where(finite,y,0.))/denominator
        np.testing.assert_allclose(mean,np.array([getattr(row,metric) for metric in metrics]),rtol=0,atol=1e-10)
    # A real corrected weather fixture reconciles prefix, endpoint and residue clocks.
    model,scenario,year=run.MODELS[0],run.SCENARIOS[0],1991
    previous,pw=run.load_cache_year(model,scenario,year-1,draws);current,cw=run.load_cache_year(model,scenario,year,draws)
    dates=pd.DatetimeIndex(np.concatenate([previous,current]));raw={key:np.concatenate([pw[key],cw[key]],axis=1)[:1] for key in run.cache.FIELDS}
    coefficient=pd.read_parquet(run.coefficient_path(model,scenario)).set_index(['cell_id','month']).loc[draws.cell_id.iloc[0]].sort_index()
    coeff={key:coefficient[key].to_numpy()[None,:] for key in ['temperature_offset','humidity_logit_offset','precipitation_ratio']}
    weather=apply_monthly_alignment(raw,dates.month,coeff)
    sow=np.array([sowing_date_from_calendar(year,draws.planting_doy.iloc[0],draws.maturity_doy.iloc[0]).to_datetime64().astype('datetime64[D]')])
    phenology=json.loads((run.ROOT/'process_model/parameters/calibration.json').read_text());fit=json.loads(run.FIT.read_text())['fitted']
    thresholds={int(key):value for key,value in json.loads(run.STAGES.read_text())['all_stage_thresholds'].items()}
    frame,details=run.simulate_draw_seasons(dates,weather,sow,draws.latitude.to_numpy()[:1],phenology,fit,thresholds,return_daily=True)
    assert frame.status.iloc[0]=='complete';mask=details['crop_mask'];t=np.where(mask,weather['tmean_c'],0.)
    before=dates.to_numpy(dtype='datetime64[D]')<sow[0];assert not details['trajectory'].state[0,1:][before,:,1:].any()
    np.testing.assert_array_equal(details['trajectory'].residue[0,1:][before],np.tile([.5,.5],(before.sum(),1)))
    end=int(np.flatnonzero(mask[0])[-1]);prefix=end-5;host=details['host']
    exposure=duration_exposure(t,np.where(mask,weather['tmax_c'],0.),np.where(mask,weather['rh_mean_pct'],0.),
        np.where(mask,weather['precipitation_mm'],0.),fit['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    short=simulate_overwinter(t[:,:prefix],exposure[:,:prefix],np.where(mask,weather['precipitation_mm'],0.)[:,:prefix],
        host.active[:,:prefix],host.renewal[:,:prefix],host.area[:,:prefix],mask[:,:prefix],OverwinterParameters(**fit['parameters']),
        imported_pressure=np.full_like(t[:,:prefix],fit['constant_imported_pressure']))
    np.testing.assert_array_equal(short.state,details['trajectory'].state[:,:prefix+1])
    np.testing.assert_array_equal(short.residue,details['trajectory'].residue[:,:prefix+1])
    expected_total=np.exp(-np.cumsum(np.maximum(t[0],0.)/18.)/fit['parameters']['residue_decay_reference_days'])
    np.testing.assert_allclose(details['trajectory'].residue[0,1:].sum(axis=1),expected_total,rtol=0,atol=1e-14)
    changed={key:array.copy() for key,array in weather.items()};changed['tmean_c'][:,end+1:]=41.;changed['tmax_c'][:,end+1:]=42.
    changed['rh_mean_pct'][:,end+1:]=100.;changed['precipitation_mm'][:,end+1:]=100.
    late,late_details=run.simulate_draw_seasons(dates,changed,sow,draws.latitude.to_numpy()[:1],phenology,fit,thresholds,return_daily=True)
    pd.testing.assert_frame_equal(frame,late);np.testing.assert_array_equal(details['trajectory'].state,late_details['trajectory'].state)
    np.testing.assert_array_equal(details['trajectory'].residue,late_details['trajectory'].residue)
    for name,value in receipt['source_sha256'].items():assert run.sha(run.ROOT/name)==value,name
    report=dict(status='passed',source_hashes_checked=len(receipt['source_sha256']),output_hashes_checked=len(receipt['output_sha256']),
        draw_season_rows=51840,invalid_cases_are_missing=len(invalid),leaf_overlap_rows_checked=len(valid)*3,
        independent_period_ratios_and_mcse=len(period),independent_paired_ratios_and_mcse=len(paired),
        independent_shared_draw_ensemble_rows=len(ep)+len(ed),annual_area_mean_rows_checked=len(annual),
        maximum_statistical_discrepancy=max_error,real_climate_prefix_and_endpoint_checks=True,
        real_climate_residue_conservation_check=True,duplicate_draw_ids_retained=True,
        no_absolute_yield_or_reference_yield_invented=True)
    target=HERE/'independent_climate_verification.json';text=json.dumps(report,indent=2)+'\n'
    if target.exists():assert target.read_text()==text
    else:target.write_text(text)
    print(text)


if __name__=='__main__':main()
