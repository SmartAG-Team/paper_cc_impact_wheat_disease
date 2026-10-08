import numpy as np
import pytest

from model.seasonal_septoria.core import Parameters,simulate_season
from model.seasonal_septoria.counterfactual import aligned_crop_inputs,decompose_weather_host


def test_crop_alignment_preserves_full_simulation_and_excludes_post_endpoint_weather():
    weather = dict(tmean_c=np.full((2,20),16.),rh_mean_pct=np.full((2,20),87.),
        precipitation_mm=np.full((2,20),2.))
    active = np.zeros((2,20,3),bool);active[0,3:12] = True;active[1,5:15] = True
    renewal = np.zeros_like(active,dtype=float)
    base = aligned_crop_inputs(weather,active,renewal,[3,5],[11,14],[9,10])
    parameters = [Parameters(alpha=.1,beta=2.,latent_days=10)]
    result = decompose_weather_host(base,base,parameters)
    np.testing.assert_array_equal(result['weather_effect'],0.)
    np.testing.assert_array_equal(result['host_effect'],0.)
    full = simulate_season(weather['tmean_c'],weather['rh_mean_pct'],weather['precipitation_mm'],
        active,renewal,parameters[0])
    expected = np.array([full.damage[0,12,:3].mean(),full.damage[1,15,:3].mean()])*100
    np.testing.assert_allclose(result['y00'][0],expected,rtol=0,atol=1e-12)


def test_cross_host_duration_requires_real_weather_and_decomposition_is_symmetric():
    weather = dict(tmean_c=np.full((1,45),12.),rh_mean_pct=np.full((1,45),82.),precipitation_mm=np.full((1,45),1.))
    active = np.ones((1,45,3),bool);renewal = np.zeros_like(active,dtype=float)
    base = aligned_crop_inputs(weather,active,renewal,[0],[39],[40])
    warmer = {key:value.copy() for key,value in weather.items()};warmer['tmean_c'][:] = 20.
    future = aligned_crop_inputs(warmer,active,renewal,[0],[24],[40])
    p = [Parameters(alpha=.1,beta=1.,latent_days=15)]
    forward,backward = decompose_weather_host(base,future,p),decompose_weather_host(future,base,p)
    for key in ['weather_effect','host_effect','total_effect']:
        np.testing.assert_allclose(forward[key],-backward[key],rtol=0,atol=1e-12)
    with pytest.raises(ValueError,match='actual weather'):
        aligned_crop_inputs(weather,active,renewal,[10],[44],[40])
