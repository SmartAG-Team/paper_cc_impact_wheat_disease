import numpy as np
import pandas as pd
import pytest


def context(anchor, available, age):
    from models.monitoring.history import SAME_NAMES, OTHER_NAMES
    h = pd.DataFrame(np.zeros((len(anchor), len(SAME_NAMES+OTHER_NAMES))), columns=SAME_NAMES+OTHER_NAMES)
    h['prior_assessment_available'] = available
    h['latest_severity_fraction'] = anchor
    h['days_since_latest_scaled'] = age
    return h


def test_recent_high_prior_uses_history_without_query_labels():
    from models.monitoring.prior_aware_blend import combine
    queries = pd.DataFrame(dict(leaf_index=[0]*5))
    h = context([1., 1., .5, 0., 1.], [1, 1, 1, 0, 1], [.2, 2., .2, 0., 1.])
    p, weights = combine([20.]*5, [90.]*5, h, queries)
    np.testing.assert_allclose(p, [90., 55., 55., 55., 90.])
    np.testing.assert_allclose(weights, [1., .5, .5, .5, 1.])
    with pytest.raises(ValueError, match='labels'):
        combine([20.]*5, [90.]*5, h, queries.assign(observed_percent=99.))


def test_other_leaf_high_prior_retains_equal_weights_and_bad_predictions_rejected():
    from models.monitoring.prior_aware_blend import combine
    q = pd.DataFrame(dict(leaf_index=[0]))
    h = context([0.], [0], [0.])
    h['other_leaf_available'] = 1
    h['other_code_1_available'] = 1
    h['other_code_1_severity'] = 1.
    h['other_code_1_age'] = .1
    prediction, weight = combine([20.], [90.], h, q)
    assert prediction[0] == 55. and weight[0] == .5
    with pytest.raises(ValueError, match='bounded'):
        combine([np.nan], [90.], h, q)
