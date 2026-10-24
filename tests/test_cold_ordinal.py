import numpy as np
import pandas as pd


def test_soft_ordinal_loss_preserves_original_target_mass_and_mean():
    from models.monitoring.cold_ordinal import ordinal_cases
    x = np.arange(10.).reshape(5, 2)
    y = np.array([0., 1.3, 24.7, 87.5, 100.])
    weights = np.array([1., 3., .7, 2., .1])
    data = ordinal_cases(x, y, weights)
    np.testing.assert_allclose(np.bincount(data['target_index'], weights=data['weights']), weights,
        atol=1e-14, rtol=0)
    mass_positive = np.bincount(data['target_index'], weights=data['weights']*data['labels'])
    np.testing.assert_allclose(100*mass_positive/weights, y, atol=1e-13, rtol=0)
    assert np.all(np.bincount(data['target_index']) == 40)


def _frame(n):
    return pd.DataFrame(dict(global_target_index=np.arange(n), coordinate_year=np.arange(n).astype(str),
        field_id=np.arange(n).astype(str), source='test', leaf_index=1,
        observed_percent=np.repeat([0., 100.], n//2)))


def test_joint_cdf_learns_severity_and_is_monotone_in_threshold():
    from models.monitoring.cold_ordinal import fit_cold_ordinal, predict_cold_ordinal
    frame = _frame(80); x = np.repeat([0., 1.], 40)[:, None]
    fitted = fit_cold_ordinal(x, frame, np.ones(80), np.zeros(80, bool), members=2, iterations=200)
    result = predict_cold_ordinal(fitted, x[[0, -1]], frame.iloc[[0, -1]].drop(columns='observed_percent'),
        np.array([17., 17.]), np.array([False, False]))
    assert result['ordinal'][0] < 5 and result['ordinal'][1] > 95
    assert np.all(np.diff(result['threshold_probability'], axis=1) <= 1e-12)
    np.testing.assert_allclose(result['ordinal'], 5*result['threshold_probability'].sum(axis=1), atol=1e-12)


def test_warm_labels_are_excluded_and_warm_or_unseen_queries_keep_reference():
    from models.monitoring.cold_ordinal import fit_cold_ordinal, predict_cold_ordinal
    frame = _frame(40); available = np.tile([False, True], 20)
    frame.loc[~available, 'observed_percent'] = 100.
    fitted = fit_cold_ordinal(np.zeros((40, 1)), frame, np.ones(40), available, members=2, iterations=5)
    assert set(fitted['cold_training_ids']) == set(frame.loc[~available, 'global_target_index'])
    q = frame.iloc[:3].drop(columns='observed_percent').copy(); q.loc[q.index[-1], 'leaf_index'] = 6
    result = predict_cold_ordinal(fitted, np.zeros((3, 1)), q, np.array([17., 29., 43.]),
        np.array([False, True, False]))
    np.testing.assert_array_equal(result['ordinal'], [100., 29., 43.])
    np.testing.assert_array_equal(result['mean'], [100., 29., 43.])
