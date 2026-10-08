import numpy as np
import pytest

from model.seasonal_septoria.core import Parameters,simulate_season
from model.seasonal_septoria.terminal_ensemble import simulate_terminal_ensemble


@pytest.mark.parametrize('time_step',[.25,.125])
def test_rolling_ensemble_matches_full_independent_trajectories(time_step):
    rng = np.random.default_rng(59)
    t = rng.uniform(-3,30,(3,85));h = rng.uniform(40,100,t.shape);r = rng.uniform(0,10,t.shape)
    active = rng.random((*t.shape,4))>.25
    renewal = rng.uniform(0,.08,active.shape)*active
    endpoints = np.array([23,84,49])
    parameters = [Parameters(alpha=.01,beta=0.),Parameters(alpha=.04,beta=10.),
                  Parameters(alpha=.002,beta=2.,latent_stages=6)]
    result = simulate_terminal_ensemble(t,h,r,active,renewal,endpoints,parameters,time_step=time_step)
    for i,parameter in enumerate(parameters):
        full = simulate_season(t,h,r,active,renewal,parameter,time_step=time_step)
        expected = full.damage[np.arange(3),endpoints+1]
        np.testing.assert_allclose(result['damage'][i],expected,rtol=0,atol=1e-14)
        expected_pyc = full.pycnidia[np.arange(3),endpoints+1]
        np.testing.assert_allclose(result['pycnidia'][i],expected_pyc,rtol=0,atol=1e-14)
    assert result['mass_error']<1e-12
    assert result['minimum_state']>=0


def test_future_weather_cannot_change_rolling_terminal_prediction():
    t = np.full((1,100),18.);h=np.full_like(t,90.);r=np.ones_like(t)
    active = np.ones((*t.shape,2),bool);renewal = np.zeros(active.shape)
    args = (active,renewal,np.array([30]),[Parameters()])
    expected = simulate_terminal_ensemble(t,h,r,*args)
    t[:,31:],h[:,31:],r[:,31:] = -30.,10.,100.
    actual = simulate_terminal_ensemble(t,h,r,*args)
    np.testing.assert_equal(actual['damage'],expected['damage'])


def test_endpoint_outside_forcing_is_rejected():
    t = np.ones((1,3));a=np.ones((1,3,1),bool)
    with pytest.raises(ValueError,match='endpoint'):
        simulate_terminal_ensemble(t,t*80,t,a,np.zeros(a.shape),np.array([3]),[Parameters()])
