"""Daily-input wetness proxies preserve meteorological constraints."""

import numpy as np
import pytest

from model.seasonal_septoria.wetness import humidity_cycle, duration_exposure


def test_inferred_humidity_cycle_preserves_the_supplied_daily_mean():
    cycle = humidity_cycle(np.array([18., 10., 25.]), np.array([24., 14., 30.]),
                           np.array([80., 95., 50.]))
    np.testing.assert_allclose(cycle.mean(axis=-1), [80., 95., 50.], atol=1e-5)
    assert cycle.min() >= 0 and cycle.max() <= 100


def test_temperature_amplitude_changes_humid_duration_at_same_daily_humidity():
    flat = duration_exposure([18.], [18.], [80.], [0.])
    varying = duration_exposure([18.], [24.], [80.], [0.])
    assert flat['humid_hours'][0] == 0.
    assert 0 < varying['humid_hours'][0] < 24.
    assert varying['exposure'][0] > flat['exposure'][0]


def test_uniform_humid_day_has_full_duration_without_inventing_rain():
    result = duration_exposure([18.], [18.], [95.], [0.])
    np.testing.assert_allclose(result['humid_hours'], [24.])
    np.testing.assert_allclose(result['rain_hours'], [0.])
    np.testing.assert_allclose(result['exposure'], [1.], atol=1e-6)


def test_proxy_rejects_impossible_daily_weather():
    with pytest.raises(ValueError):
        duration_exposure([18.], [16.], [80.], [1.])
    with pytest.raises(ValueError):
        duration_exposure([18.], [24.], [101.], [1.])
