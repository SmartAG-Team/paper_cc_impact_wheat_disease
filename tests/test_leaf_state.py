import numpy as np
import pandas as pd
import pytest


def test_observation_update_conserves_mass_and_changes_visible_not_latent_truth():
    from models.monitoring.leaf_state import assimilate_visible
    state = np.array([[.3, .2, .1, .1, .1, .1, .1], [1., 0, 0, 0, 0, 0, 0]])
    before = state.copy()
    updated = assimilate_visible(state, np.array([.6, .25]))
    np.testing.assert_allclose(updated.sum(axis=1), 1)
    np.testing.assert_allclose(updated[:, 4:].sum(axis=1), [.6, .25])
    np.testing.assert_allclose(updated[0, :4], state[0, :4]*.4/.7)
    np.testing.assert_allclose(updated[1], [.75, 0, 0, 0, .25, 0, 0])
    np.testing.assert_array_equal(state, before)
    for invalid in [np.array([1.1, 0.]), np.array([np.nan, 0.])]:
        with pytest.raises(ValueError):
            assimilate_visible(state, invalid)


def test_daily_propagation_preserves_mass_and_has_expected_latent_transfer():
    from models.monitoring.leaf_state import advance_state
    from models.seasonal_septoria.core import Parameters
    state = np.array([[0., 1., 0., 0., 0., 0., 0.]])
    params = Parameters(alpha=0, beta=0)
    result = advance_state(state, 18., 90., 1., parameters=params, time_step=1.)
    q = 1-np.exp(-3/20)
    np.testing.assert_allclose(result, [[0, 1-q, q, 0, 0, 0, 0]])
    np.testing.assert_allclose(result.sum(axis=1), 1)
    # A full visible observation leaves no susceptible/latent tissue to reinfect.
    full = np.array([[0, 0, 0, 0, 1., 0, 0]])
    result = advance_state(full, 18., 90., 10.)
    np.testing.assert_allclose(result[:, 4:].sum(axis=1), 1)
    with pytest.raises(ValueError):
        advance_state(state, 18., 101., 0.)


def fixture():
    dates = pd.date_range('2020-10-01', '2020-10-15')
    weather = pd.DataFrame(dict(location_id='site', date=dates, tmean_c=18.,
        rh_mean_pct=90., rain_mm=1.))
    fields = pd.DataFrame(dict(field_id=['field'], site_id=['site'], source=['BASF'],
        sowing_date=['2020-10-01']))
    obs = pd.DataFrame(dict(global_target_index=[1, 2, 3], field_id='field', site_id='site',
        source='BASF', leaf_index=[0, 1, 0], endpoint_series=['flag', 'second', 'flag'],
        date=['2020-10-05', '2020-10-10', '2020-10-12'], observed_percent=[60., 20., 90.]))
    queries = obs.iloc[[1]].drop(columns='observed_percent').copy()
    return weather, fields, obs, queries


def test_state_inputs_use_only_strictly_earlier_observations_and_weather():
    from models.monitoring.leaf_state import build_state_features
    weather, fields, obs, queries = fixture()
    x, names, ledger = build_state_features(weather, fields, obs, queries)
    assert ledger.prior_target_ids.tolist() == [[1]]
    assert ledger.weather_end.tolist() == ['2020-10-09']
    assert x[0, names.index('state_any_assimilated')] == 1
    assert x[0, names.index('state_same_assimilated')] == 0
    changed = obs.copy()
    changed.loc[changed.date.ge('2020-10-10'), 'observed_percent'] = 0
    w2 = weather.copy()
    w2.loc[w2.date.ge('2020-10-10'), ['tmean_c', 'rh_mean_pct', 'rain_mm']] = [35, 40, 100]
    y, _, _ = build_state_features(w2, fields, changed, queries)
    np.testing.assert_array_equal(x, y)
    changed.loc[0, 'observed_percent'] = 0
    z, _, _ = build_state_features(weather, fields, changed, queries)
    assert not np.allclose(x, z)
    # Supplying the same query in a batch containing later queries cannot change its past.
    batch = obs.iloc[[1, 2]].drop(columns='observed_percent')
    b, _, _ = build_state_features(weather, fields, obs, batch)
    np.testing.assert_array_equal(x[0], b[0])
    with pytest.raises(ValueError, match='label'):
        build_state_features(weather, fields, obs, queries.assign(observed_percent=100.))
    with pytest.raises(ValueError, match='Duplicate'):
        build_state_features(weather, fields, pd.concat([obs, obs.iloc[[0]].assign(global_target_index=99)]), queries)
    with pytest.raises(ValueError, match='weather'):
        build_state_features(weather.iloc[1:], fields, obs, queries)
