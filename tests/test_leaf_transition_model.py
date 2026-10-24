import numpy as np
import pandas as pd
import pytest


def training_fixture():
    from models.monitoring.history import SAME_NAMES, OTHER_NAMES
    from models.monitoring.leaf_state import STATE_NAMES
    n = 40
    frame = pd.DataFrame(dict(global_target_index=np.arange(n),
        field_id=[f'field-{i}' for i in range(n)], site_id=[f'site-{i}' for i in range(n)],
        source='BASF', coordinate_year=[f'cluster-{i//2}' for i in range(n)],
        endpoint_series='flag', leaf_index=0, date='2019-06-01', season_year=2019,
        observed_percent=40.))
    h = pd.DataFrame(np.zeros((n, 35)), columns=SAME_NAMES+OTHER_NAMES)
    h['prior_assessment_available'] = 1
    h['latest_severity_fraction'] = .2
    z = np.zeros((n, len(STATE_NAMES)))
    z[:, 0] = .8; z[:, 4] = .2
    z[:, STATE_NAMES.index('state_same_assimilated')] = 1
    z[:, STATE_NAMES.index('state_any_assimilated')] = 1
    return np.zeros((n, 2)), h, z, frame


def test_transition_learns_change_relative_to_propagated_visible_state():
    from models.monitoring.leaf_transition_model import fit_transition, predict_transition
    x, h, z, frame = training_fixture()
    fitted = fit_transition(x, h, z, frame)
    q = frame.drop(columns='observed_percent')
    np.testing.assert_allclose(predict_transition(fitted, x, h, z, q), 40., atol=1e-7)
    shifted = z.copy(); shifted[:, 0] -= .1; shifted[:, 4] += .1
    np.testing.assert_allclose(predict_transition(fitted, x, h, shifted, q), 50., atol=1e-7)
    # Earlier other-leaf context is admissible; absent field history is not silently invented.
    bad = h.copy(); bad['prior_assessment_available'] = 0
    with pytest.raises(ValueError, match='history'):
        fit_transition(x, bad, z, frame)
    with pytest.raises(ValueError, match='label'):
        predict_transition(fitted, x, h, z, frame)
    with pytest.raises(ValueError, match='schema'):
        predict_transition(fitted, x, h, z[:, :-1], q)


def test_routing_changes_only_queries_with_strictly_earlier_field_information():
    from models.monitoring.leaf_transition_model import route_transition
    _, h, _, q = training_fixture()
    h = h.iloc[:3].copy()
    h.loc[:, 'prior_assessment_available'] = [0, 1, 0]
    h.loc[:, 'other_leaf_available'] = [0, 0, 1]
    parent = np.array([10., 20., 30.]); process = np.array([2., 4., 6.])
    prediction, mask = route_transition(parent, process, np.array([90., 80., 70.]), h)
    np.testing.assert_array_equal(prediction, [10., 42., 38.])
    np.testing.assert_array_equal(mask, [False, True, True])
    changed, _ = route_transition(parent, process, np.array([0., 80., 70.]), h)
    np.testing.assert_array_equal(prediction, changed)
    with pytest.raises(ValueError):
        route_transition(parent, process, np.array([np.nan, 80., 70.]), h)
