import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.seasonal_septoria.core import Parameters,simulate_season
from model.seasonal_septoria.rolling_diagnostics import simulate_rolling_diagnostics
from model.seasonal_septoria.regional import simulate_grid_seasons
from model.seasonal_septoria.regional_compact import simulate_grid_seasons_compact


@pytest.mark.parametrize('step',[.25,.125])
def test_rolling_daily_events_equal_full_origin_flows_and_visible_states(step):
    rng=np.random.default_rng(32)
    t=rng.uniform(-4,28,(3,110));h=rng.uniform(30,99,t.shape);rain=rng.exponential(2,t.shape)
    active=rng.random((3,110,8))>.12;renewal=rng.uniform(0,.3,active.shape);renewal[~active]=0.
    endpoints=np.array([20,70,109]);cutoffs=[.1,1.,5.]
    for p in [Parameters(alpha=.03,beta=0.),Parameters(alpha=.03,beta=10.,latent_stages=6)]:
        full=simulate_season(t,h,rain,active,renewal,p,time_step=step)
        compact=simulate_rolling_diagnostics(t,h,rain,active,renewal,endpoints,[p],time_step=step)
        for field,end in enumerate(endpoints):
            np.testing.assert_allclose(compact['damage'][0,field],full.damage[field,end+1],rtol=0,atol=1e-14)
            np.testing.assert_allclose(compact['pycnidia'][0,field],full.pycnidia[field,end+1],rtol=0,atol=1e-14)
            for leaf in range(8):
                for j,cutoff in enumerate(cutoffs):
                    dates=np.flatnonzero(full.damage[field,1:end+2,leaf]*100>=cutoff)
                    assert compact['first_visible_days'][0,field,leaf,j]==(dates[0] if len(dates) else -1)
            dates=np.flatnonzero(np.cumsum(full.primary_flow[field,:end+1],axis=0).max(axis=1)>=.0001)
            assert compact['primary_exposure_days'][0,field]==(dates[0] if len(dates) else -1)


@pytest.mark.parametrize('spring',[False,True])
def test_complete_and_incomplete_regional_cases_preserve_all_scientific_outputs(spring):
    calibration=json.loads((Path(__file__).parents[1]/'process_model/parameters/calibration.json').read_text())
    dates=pd.date_range('2018-01-01','2019-12-31')
    t=np.vstack([12+9*np.sin(2*np.pi*(dates.dayofyear-100)/365),np.zeros(len(dates))])
    tx=t+5;h=np.full_like(t,87.);rain=np.full_like(t,1.5)
    args=(dates,t,tx,h,rain,[50.,63.],['2018-10-01','2018-10-01'],Parameters(alpha=.06,beta=1.),calibration)
    full=simulate_grid_seasons(*args,vernalization_required=not spring)
    compact=simulate_grid_seasons_compact(*args,vernalization_required=not spring)
    for key in ['final_damage_percent','final_pycnidia_percent']:
        np.testing.assert_allclose(compact[key],full[key],rtol=0,atol=1e-12,equal_nan=True)
    for key in ['first_visible_dates','primary_exposure_dates','endpoint_dates','complete_season']:
        np.testing.assert_array_equal(compact[key],full[key])
    for stage in full['stage_dates']:
        np.testing.assert_array_equal(compact['stage_dates'][stage],full['stage_dates'][stage])
    assert compact['mass_error']<1e-12
