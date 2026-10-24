import numpy as np
import pandas as pd
import pytest


def test_severe_priority_changes_only_historical_weights_above_the_fixed_threshold():
    from paper.scripts.endpoint_severe_priority import training_risk_weights

    training = pd.DataFrame(dict(observed_percent=[0., 75., 75.01, 100.]))
    scientific = np.array([1., 2., 3., 4.])
    actual = training_risk_weights(training, scientific)
    np.testing.assert_array_equal(actual, [1., 2., 6., 8.])
    np.testing.assert_array_equal(scientific, [1., 2., 3., 4.])


def test_severe_priority_changes_the_fitted_risk_optimum_without_query_labels():
    from paper.scripts.endpoint_severe_priority import fit_severe_priority
    from paper.scripts.endpoint_improvement import predict_endpoint

    training = pd.DataFrame(dict(field_id=['a', 'b', 'c', 'd'],
        date=['2014-06-01'] * 4, leaf_index=[0] * 4,
        observed_percent=[0., 0., 0., 100.]))
    original_weight = np.ones(4)
    fitted = fit_severe_priority(training, np.full(4, 20.), original_weight,
                                np.log([1., 1., 30., 30.]))
    # A constant estimate minimizes the fixed 2x severe risk near 40%,
    # whereas the unchanged ordinary MSE optimum is 25%.
    value = predict_endpoint(fitted['parameters'], training, np.full(4, 20.), growth=False)
    assert np.all((value > 39.) & (value < 41.))
    np.testing.assert_array_equal(original_weight, np.ones(4))


def test_severe_training_risk_rejects_invalid_weights():
    from paper.scripts.endpoint_severe_priority import training_risk_weights

    with pytest.raises(ValueError, match='positive'):
        training_risk_weights(pd.DataFrame(dict(observed_percent=[90.])), np.array([0.]))
