"""Numerical edge cases for the new sampled infection/yield climate analysis."""
import importlib
import numpy as np
import pandas as pd
import pytest


def module():
    try:
        return importlib.import_module('analysis.paper_study.manuscript_infection_yield_20261006.climate.run')
    except ModuleNotFoundError:
        pytest.fail('The isolated infection/yield climate engine is missing')


def fixture(humidity=100.):
    dates=pd.date_range('2000-01-01',periods=12)
    t=np.full((1,12),18.)
    weather=dict(tmean_c=t,tmax_c=t+2,rh_mean_pct=t*0+humidity,precipitation_mm=t*0)
    sow=np.array(['2000-01-06'],dtype='datetime64[D]')
    host=dict(flag_fraction=.3,leaf_interval=1.,expansion_units=1.)
    event=dict(event_model=dict(host_parameters=host,parameters=dict(dose_threshold=.5,dry_gap_days=3,
        delay_reference_days=1.),weather_preprocessing=dict(rain_rate_mm_hour=.46)),anthesis_threshold=20.,
        yield_slopes=[.0141,.018,.0207])
    progression=dict(components=[dict(weight=1.,fitted=dict(host_parameters=host,
        parameters=dict(initiation=.1,amplification=0.,latent_days=1.),weather_operator='daily_or'))])
    phenology=dict(photoperiod_onset_gdd=1e6,photoperiod_stop_gdd=2e6,
        vernalization_onset_tpp=1e6,vernalization_stop_tpp=2e6,
        thresholds=[dict(BBCH=k,Cumulative_t_pp_v_GDD=v) for k,v in [(10,1.),(31,5.),(51,8.),(85,40.)]])
    return dates,weather,sow,np.array([45.]),phenology,event,progression


def test_dates_use_global_weather_origin_and_actual_calendar_sowing():
    m=module();frame,details=m.simulate_draw_seasons(*fixture(),return_daily=True)
    row=frame.iloc[0]
    assert row.status=='complete'
    assert row.F1_infection_date==pd.Timestamp('2000-01-07')
    assert row.F1_symptom_date==pd.Timestamp('2000-01-08')
    assert row.F1_infection_day_after_sowing==1.
    assert row.F1_infection_relative_anthesis_days==0.
    assert row.grain_fill_days==2.
    assert row.F1_infected_grain_fill_fraction==1.
    assert row.F1_symptomatic_grain_fill_fraction==.5
    assert not details['functional_loss'][0,:7].any()


def test_dry_complete_season_is_zero_event_and_zero_conditional_loss():
    m=module();frame,_=m.simulate_draw_seasons(*fixture(humidity=0.))
    row=frame.iloc[0]
    assert row.status=='complete'
    assert row.any_top3_infection_before85==0.
    assert pd.isna(row.F1_infection_date)
    assert row.GS65_85_lost_had3==0.


def test_post_endpoint_weather_cannot_advance_events_or_functional_loss():
    m=module();args=fixture();first,_=m.simulate_draw_seasons(*args)
    args[1]['tmean_c'][:,8:]=41.;args[1]['tmax_c'][:,8:]=42.
    args[1]['precipitation_mm'][:,8:]=100.
    second,_=m.simulate_draw_seasons(*args)
    pd.testing.assert_frame_equal(first,second)


def test_missing_or_unsupported_pre_endpoint_is_invalid_not_zero_disease():
    m=module();args=fixture();args[1]['tmean_c'][:,5]=np.nan
    frame,_=m.simulate_draw_seasons(*args)
    assert frame.status.iloc[0]=='missing_weather_before_endpoint'
    assert pd.isna(frame.GS65_85_lost_had3.iloc[0])
    args=fixture();args[1]['tmean_c'][:,5]=41.;args[1]['tmax_c'][:,5]=42.
    frame,_=m.simulate_draw_seasons(*args)
    assert frame.status.iloc[0]=='unsupported_mean_temperature_above40'
    assert pd.isna(frame.any_top3_infection_before85.iloc[0])


def test_stratified_ratio_mcse_keeps_invalid_and_duplicate_draws():
    m=module();draws=pd.DataFrame(dict(spatial_draw_id=[0,1,2,3],stratum=['a']*4,
        cell_id=['duplicate','duplicate','b','c'],area_mean_weight=[.25]*4))
    full=m.ratio_summary(np.array([1.,2.,3.,4.]),np.ones(4),draws)
    assert full['mean']==2.5
    assert full['spatial_mcse']==pytest.approx(np.sqrt(5/12))
    incomplete=m.ratio_summary(np.array([1.,2.,3.,0.]),np.array([1.,1.,1.,0.]),draws)
    assert incomplete['mean']==2.
    assert incomplete['coverage']==.75
    assert incomplete['spatial_mcse']==pytest.approx(np.sqrt(8/27))
