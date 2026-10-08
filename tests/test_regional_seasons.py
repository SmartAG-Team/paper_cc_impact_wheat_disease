import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.seasonal_septoria.core import Parameters, simulate_season
from model.seasonal_septoria.host import cohort_inputs
from model.seasonal_septoria.regional import tpv_accumulation, simulate_grid_seasons
from process_model.calibrate import transform

ROOT = Path(__file__).resolve().parents[1]
CAL = json.loads((ROOT/'process_model/parameters/calibration.json').read_text())


def test_vectorized_tpv_matches_copied_process_with_causal_gates():
    dates = pd.date_range('2019-08-01', '2020-12-31')
    doy = dates.dayofyear.to_numpy()
    t = np.vstack([8+16*np.sin(2*np.pi*(doy-110)/365), 10+12*np.sin(2*np.pi*(doy-110)/365)])
    tmax = t+9
    latitude = np.array([51., 60.])
    starts = pd.to_datetime(['2019-10-13','2020-03-20'])
    included = dates.to_numpy()[None,:] >= starts.to_numpy()[:,None]
    got = tpv_accumulation(t,tmax,latitude,doy,included,CAL)
    for i in range(2):
        part = pd.DataFrame({'PEP_ID':1,'DATE':dates[included[i]],'LAT':latitude[i],
            't_mean':t[i,included[i]],'t_max':tmax[i,included[i]],
            'SOWING_DATE':starts[i],'SOWING_KNOWN_AT':starts[i]})
        part['GDD'] = np.where(part.t_mean > 30,20-2*(part.t_mean-30),np.clip(part.t_mean,0,20))
        expected = transform(part,CAL).Cumulative_t_pp_v_GDD.to_numpy()
        np.testing.assert_allclose(got[i,included[i]],expected,rtol=0,atol=1e-10)
        assert np.all(got[i,~included[i]] == 0)


def test_regional_endpoint_equals_full_trajectory_at_soft_dough():
    dates = pd.date_range('2019-08-01','2020-12-31')
    t = np.full((1,len(dates)),18.)
    h,r = np.full_like(t,90.),np.ones_like(t)
    sow = np.array(['2019-10-01'],dtype='datetime64[D]')
    p = Parameters(alpha=.01,beta=1.,latent_days=20.)
    result = simulate_grid_seasons(dates,t,t+5,h,r,np.array([50.]),sow,p,CAL)
    included = dates.to_numpy()[None,:] >= sow[:,None]
    acc = tpv_accumulation(t,t+5,np.array([50.]),dates.dayofyear,included,CAL)
    end = np.flatnonzero(acc[0]>=CAL['thresholds'][-1]['Cumulative_t_pp_v_GDD'])[0]
    thresholds = {row['BBCH']:row['Cumulative_t_pp_v_GDD'] for row in CAL['thresholds']}
    active,renewal = cohort_inputs(acc,t,thresholds)
    active[:,end+1:] = False
    renewal[~active] = 0
    direct = simulate_season(t,h,r,active,renewal,p)
    np.testing.assert_allclose(result['final_damage_percent'][0],direct.damage[0,end+1,:7]*100)
    assert result['endpoint_dates'][0] == dates[end].to_datetime64().astype('datetime64[D]')
    assert result['complete_season'][0]
    assert result['mass_error'] < 1e-12


def test_zero_infection_and_incomplete_crop_have_distinct_missing_events():
    dates = pd.date_range('2020-03-01','2020-12-31')
    t = np.array([np.full(len(dates),18.),np.full(len(dates),-10.)])
    result = simulate_grid_seasons(dates,t,t+5,np.full_like(t,80.),np.zeros_like(t),
        np.array([50.,60.]),np.array(['2020-03-01','2020-03-01'],dtype='datetime64[D]'),
        Parameters(alpha=0.,beta=0.),CAL,vernalization_required=False)
    assert result['complete_season'].tolist() == [True,False]
    assert np.all(result['final_damage_percent'][0] == 0)
    assert np.isnan(result['final_damage_percent'][1]).all()
    assert np.isnat(result['first_visible_dates']).all()
    assert np.isnat(result['primary_exposure_dates']).all()


def test_tpv_rejects_unsupported_hot_mean_instead_of_negative_accumulation():
    with pytest.raises(ValueError,match='40'):
        tpv_accumulation(np.array([[41.]]),np.array([[45.]]),np.array([40.]),
            np.array([200]),np.ones((1,1),bool),CAL)


def test_hot_weather_after_completed_crop_does_not_reject_or_change_endpoint():
    dates = pd.date_range('2019-08-01','2020-12-31')
    t = np.full((1,len(dates)),18.)
    h,r = np.full_like(t,90.),np.ones_like(t)
    sow = np.array(['2019-10-01'],dtype='datetime64[D]')
    p = Parameters(alpha=.01,beta=1.)
    expected = simulate_grid_seasons(dates,t,t+5,h,r,np.array([50.]),sow,p,CAL)
    t[:,-10:] = 41.
    actual = simulate_grid_seasons(dates,t,t+5,h,r,np.array([50.]),sow,p,CAL)
    np.testing.assert_equal(actual['endpoint_dates'],expected['endpoint_dates'])
    np.testing.assert_allclose(actual['final_damage_percent'],expected['final_damage_percent'],rtol=0,atol=0)


def test_hot_weather_before_completed_crop_retains_explicit_unsupported_error():
    dates = pd.date_range('2020-03-01','2020-12-31')
    t = np.full((1,len(dates)),18.);t[:,4] = 41.
    with pytest.raises(ValueError,match='40'):
        simulate_grid_seasons(dates,t,t+5,np.full_like(t,80.),np.zeros_like(t),np.array([50.]),
            np.array(['2020-03-01'],dtype='datetime64[D]'),Parameters(),CAL,vernalization_required=False)
