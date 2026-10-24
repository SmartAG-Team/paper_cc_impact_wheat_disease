import numpy as np
import pandas as pd
import pytest


def _inputs():
    dates = pd.date_range('2018-03-01', periods=85)
    weather = pd.DataFrame(dict(location_id='s', date=dates, tmean_c=14., tmax_c=20.,
        rh_mean_pct=92., precipitation_mm=3.))
    fields = pd.DataFrame(dict(field_id=['f'], site_id=['s'], latitude=[50.], estimated_sowing_date=['2018-03-01']))
    queries = pd.DataFrame(dict(global_target_index=[1, 2], field_id='f', date=['2018-04-10', '2018-05-15']))
    distributions = {k: dict(values=np.array([v]), cumulative_weight=np.array([1.]))
        for k, v in zip([10, 31, 51, 87], [10, 80, 200, 3000])}
    prior = dict(distributions=dict(tpv=distributions), parameters=dict(photoperiod_onset_gdd=10.,
        photoperiod_stop_gdd=200., vernalization_onset_tpp=0., vernalization_stop_tpp=10.))
    return weather, fields, queries, prior


def test_scenario_changes_development_and_resistance_without_future_or_target_inputs():
    from models.monitoring.process_scenario import simulate_queries
    from models.seasonal_septoria.overwinter import OverwinterParameters
    weather, fields, queries, prior = _inputs()
    params = OverwinterParameters(primary_scale=.03, secondary_scale=1.)
    original = simulate_queries(weather, fields, queries, prior, params)
    changed = weather.copy()
    changed.loc[changed.date.ge(queries.date.iloc[0]), ['tmean_c', 'tmax_c', 'precipitation_mm']] = [18., 24., 0.]
    altered = simulate_queries(changed, fields, queries.assign(observed_percent=[100, 100], bbch=[99, 99]), prior, params)
    np.testing.assert_array_equal(original['damage'][0], altered['damage'][0])
    assert not np.array_equal(original['damage'][1], altered['damage'][1])
    resistant = simulate_queries(weather, fields, queries, prior, params, susceptibility_multiplier=.25)
    neutral = simulate_queries(weather, fields, queries, prior, params, susceptibility_multiplier=1.)
    np.testing.assert_array_equal(original['damage'], neutral['damage'])
    assert not np.array_equal(original['damage'], resistant['damage'])
    assert resistant['largest_mass_error'] < 1e-10
    assert np.isfinite(resistant['damage']).all()
    assert ((resistant['damage'] >= 0)&(resistant['damage'] <= 1)).all()
    warmer = weather.copy(); warmer['tmean_c'] += 2; warmer['tmax_c'] += 2
    warmed = simulate_queries(warmer, fields, queries, prior, params)
    assert not np.array_equal(original['area'], warmed['area'])


@pytest.mark.parametrize('invalid_date', ['2018-02-28', '2018-03-01'])
def test_mixed_query_batch_rejects_each_query_on_or_before_sowing(invalid_date):
    from models.monitoring.process_scenario import simulate_queries
    from models.seasonal_septoria.overwinter import OverwinterParameters
    weather, fields, queries, prior = _inputs()
    queries.loc[0, 'date'] = invalid_date
    with pytest.raises(ValueError, match='Every query must follow'):
        simulate_queries(weather, fields, queries, prior, OverwinterParameters())
