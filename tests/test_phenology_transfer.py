import numpy as np
import pandas as pd


def test_development_clock_is_causal_and_winter_limits_thermal_progress():
    from models.monitoring.phenology_transfer import development_increments
    weather = np.tile([10., 15., 5., 0.], (90, 1))
    dates = pd.date_range('2014-11-01', periods=90)
    parameters = dict(photoperiod_onset_gdd=0., photoperiod_stop_gdd=10000.,
        vernalization_onset_tpp=0., vernalization_stop_tpp=10000.)
    first = development_increments(weather, dates, 51., parameters)
    altered = weather.copy(); altered[40:, :3] = [25., 30., 20.]
    second = development_increments(altered, dates, 51., parameters)
    np.testing.assert_array_equal(first[:40], second[:40])
    assert np.all(first[:, 1] < first[:, 0])
    assert np.all(first[:20, 2] < first[:20, 1])
    assert first[-1, 2] == first[-1, 1]


def test_development_probabilities_respect_stage_order_and_future_extension():
    from models.monitoring.phenology_transfer import summarize_development
    distributions = {model: {stage: dict(values=np.array([threshold, threshold+5.]),
        cumulative_weight=np.array([.5, 1.])) for stage, threshold in zip([10, 31, 51, 87], [20, 40, 60, 80])}
        for model in ['gdd', 'tp', 'tpv']}
    inc = np.ones((100, 3)); cumulative = np.cumsum(inc, axis=0)
    features, names, probabilities = summarize_development(cumulative, inc, distributions)
    assert features.shape == (100, 57) and len(names) == 57
    assert probabilities.shape == (100, 3, 4)
    assert np.all(np.diff(probabilities, axis=2) <= 0)
    assert np.all(np.diff(probabilities, axis=0) >= 0)
    assert probabilities[49, 0].tolist() == [1., 1., 0., 0.]
    extended, _, _ = summarize_development(np.r_[cumulative, cumulative[-1:]+30],
        np.r_[inc, [[30., 30., 30.]]], distributions)
    np.testing.assert_array_equal(features, extended[:100])
