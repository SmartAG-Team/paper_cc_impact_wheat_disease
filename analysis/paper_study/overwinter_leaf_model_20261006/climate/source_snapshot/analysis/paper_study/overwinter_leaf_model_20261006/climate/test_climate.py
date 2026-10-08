"""Calendar-origin, crop-horizon and sampling-error checks for the new replay."""
import importlib
import numpy as np
import pandas as pd
import pytest


def module():
    try:return importlib.import_module('analysis.paper_study.overwinter_leaf_model_20261006.climate.run')
    except ModuleNotFoundError:pytest.fail('The mechanistic climate replay is not implemented')


def fixture(humidity=100.):
    from model.seasonal_septoria.overwinter import OverwinterParameters
    dates=pd.date_range('2000-01-01',periods=20);t=np.full((1,20),18.)
    weather=dict(tmean_c=t,tmax_c=t+2,rh_mean_pct=t*0+humidity,precipitation_mm=t*0)
    sow=np.array(['2000-01-06'],dtype='datetime64[D]')
    parameters=OverwinterParameters(primary_scale=.1,secondary_scale=1.,latent_reference_days=1.)
    fit=dict(parameters=parameters.__dict__.copy(),rank_spacing_units=1.,constant_imported_pressure=.1,
        juvenile_policy='handover_31_39',weather_preprocessing=dict(rain_rate_mm_hour=.45))
    thresholds={10:1.,31:5.,32:6.,33:7.,37:8.,39:10.,51:20.,65:30.,85:100.}
    phenology=dict(photoperiod_onset_gdd=1e6,photoperiod_stop_gdd=2e6,vernalization_onset_tpp=1e6,
        vernalization_stop_tpp=2e6,thresholds=[dict(BBCH=k,Cumulative_t_pp_v_GDD=v) for k,v in thresholds.items()])
    return dates,weather,sow,np.array([45.]),phenology,fit,thresholds


def test_global_january_origin_and_calendar_sowing_are_distinct():
    m=module();frame,details=m.simulate_draw_seasons(*fixture(),return_daily=True);row=frame.iloc[0]
    assert row.status=='complete'
    assert row.calendar_sowing_date==pd.Timestamp('2000-01-06')
    assert row.BBCH65_date==pd.Timestamp('2000-01-07')
    assert row.BBCH85_date==pd.Timestamp('2000-01-11')
    assert row.grain_fill_days==5.
    assert row.F1_infection_date>=row.calendar_sowing_date
    assert row.F1_infection_day_after_sowing==(row.F1_infection_date-row.calendar_sowing_date).days
    assert not details['trajectory'].state[0,:7,:,1:].any()


def test_dry_valid_crop_is_zero_events_and_zero_conditional_loss():
    m=module();frame,_=m.simulate_draw_seasons(*fixture(humidity=0.));row=frame.iloc[0]
    assert row.status=='complete'
    assert row.any_top3_infection_before85==0.
    assert row.GS65_85_lost_had3==0.


def test_after_endpoint_forcing_does_not_age_residue_or_advance_disease():
    m=module();args=fixture();first,details=m.simulate_draw_seasons(*args,return_daily=True)
    args[1]['tmean_c'][:,12:]=41.;args[1]['tmax_c'][:,12:]=42.;args[1]['precipitation_mm'][:,12:]=100.
    second,changed=m.simulate_draw_seasons(*args,return_daily=True)
    pd.testing.assert_frame_equal(first,second)
    np.testing.assert_array_equal(details['trajectory'].state,changed['trajectory'].state)
    np.testing.assert_array_equal(details['trajectory'].residue,changed['trajectory'].residue)


def test_pre_endpoint_missing_and_unsupported_crops_are_not_zero_disease():
    m=module();args=fixture();args[1]['tmean_c'][:,5]=np.nan;frame,_=m.simulate_draw_seasons(*args)
    assert frame.status.iloc[0]=='missing_weather_before_endpoint'
    assert pd.isna(frame.GS65_85_lost_had3.iloc[0])
    args=fixture();args[1]['tmean_c'][:,5]=41.;args[1]['tmax_c'][:,5]=42.;frame,_=m.simulate_draw_seasons(*args)
    assert frame.status.iloc[0]=='unsupported_mean_temperature_above40'
    assert pd.isna(frame.any_top3_infection_before85.iloc[0])


def test_stratified_ratio_mcse_keeps_four_draws_even_with_a_duplicate_or_invalid():
    m=module();draws=pd.DataFrame(dict(spatial_draw_id=[0,1,2,3],stratum=['a']*4,
        cell_id=['duplicate','duplicate','b','c'],area_mean_weight=[.25]*4))
    full=m.ratio_summary(np.array([1.,2.,3.,4.]),np.ones(4),draws)
    assert full['mean']==2.5 and full['spatial_mcse']==pytest.approx(np.sqrt(5/12))
    incomplete=m.ratio_summary(np.array([1.,2.,3.,0.]),np.array([1.,1.,1.,0.]),draws)
    assert incomplete['mean']==2. and incomplete['coverage']==.75
    assert incomplete['spatial_mcse']==pytest.approx(np.sqrt(8/27))
