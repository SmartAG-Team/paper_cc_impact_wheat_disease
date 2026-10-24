import numpy as np
import pandas as pd


def test_ordered_category_fit_respects_scientific_weights():
    from models.monitoring.ordered_category_model import fit_ordered, predict_ordered
    frame = pd.DataFrame(dict(field_id=np.arange(20).astype(str), source='BASF', leaf_index=0,
        country='Germany', date='2014-05-01', global_target_index=np.arange(20),
        observed_percent=np.repeat([10., 90.], 10)))
    x = np.zeros((20, 2)); weights = np.repeat([9., 1.], 10)
    fitted = fit_ordered(x, frame, np.zeros(20), weights, 'direct', iterations=30)
    value = predict_ordered(fitted, x[:1], frame.iloc[:1], np.zeros(1))
    np.testing.assert_allclose(value, [18.], atol=.01)


def test_ordered_growth_restores_query_anchor_once_and_replays(tmp_path):
    import joblib
    from models.monitoring.ordered_category_model import fit_ordered, predict_ordered
    frame = pd.DataFrame(dict(field_id=np.arange(20).astype(str), source='BASF', leaf_index=0,
        country='Germany', date='2014-05-01', global_target_index=np.arange(20),
        observed_percent=np.repeat([50., 70.], 10)))
    x = np.zeros((20, 2)); baseline = np.repeat([20., 40.], 10)
    fitted = fit_ordered(x, frame, baseline, np.ones(20), 'growth', iterations=30)
    value = predict_ordered(fitted, x[:1], frame.iloc[:1], np.array([5.]))
    np.testing.assert_allclose(value, [35.], atol=.01)
    path = tmp_path/'ordered.joblib'; joblib.dump(fitted, path)
    np.testing.assert_array_equal(value, predict_ordered(joblib.load(path), x[:1], frame.iloc[:1], np.array([5.])))


def test_ordered_trees_learn_nominal_endpoint_effect_without_query_labels():
    from models.monitoring.ordered_category_model import fit_ordered, predict_ordered
    frame = pd.DataFrame(dict(field_id=np.arange(60).astype(str), source='BASF',
        leaf_index=np.tile([0, 1], 30), country='Germany', date='2014-05-01',
        observed_percent=np.tile([10., 90.], 30), global_target_index=np.arange(60)))
    x = np.zeros((60, 2))
    fitted = fit_ordered(x, frame, np.zeros(60), np.ones(60), iterations=150)
    query = frame.iloc[:2].copy(); query['observed_percent'] = -1000
    value = predict_ordered(fitted, x[:2], query, np.zeros(2))
    assert fitted['model'] is not None and value[0] < 25 and value[1] > 75
