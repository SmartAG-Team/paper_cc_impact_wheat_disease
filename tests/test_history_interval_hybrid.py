import numpy as np
import pandas as pd
import pytest


def test_event_head_is_used_only_when_strictly_earlier_field_context_is_available():
    from models.monitoring.history_interval_hybrid import history_prediction
    h = pd.DataFrame(dict(prior_assessment_available=[0., 1., 0.], other_leaf_available=[0., 0., 1.]))
    queries = pd.DataFrame(dict(leaf_index=[0, 1, 2]))
    pred, mask = history_prediction([10., 20., 30.], [80., 90., 100.], h, queries)
    np.testing.assert_array_equal(pred, [10., 90., 100.])
    np.testing.assert_array_equal(mask, [False, True, True])
    with pytest.raises(ValueError, match='labels'):
        history_prediction([10., 20., 30.], [80., 90., 100.], h, queries.assign(observed_percent=99.))
    h.loc[0, 'other_leaf_available'] = np.nan
    with pytest.raises(ValueError, match='binary'):
        history_prediction([10., 20., 30.], [80., 90., 100.], h, queries)


def test_future_inference_checks_domain_and_preserves_the_availability_switch(monkeypatch):
    from models.monitoring import history_interval_hybrid as module
    monkeypatch.setattr(module, 'predict_cluster_ensemble', lambda *args: np.array([20.]))
    monkeypatch.setattr(module, 'predict_ensemble', lambda fitted, *args: np.array([fitted['value']]))
    names = ['prior_assessment_available', 'other_leaf_available']
    h = pd.DataFrame([[0., 0.]], columns=names)
    queries = pd.DataFrame(dict(date=['2020-06-01'], season_year=[2020], source=['BASF'], leaf_index=[0]))
    fitted = dict(base=dict(process={}, history=dict(value=10.), history_names=names,
                            minimum_assessment_year=2020), events_history=dict(value=90.))
    inputs = [np.zeros((1, width)) for width in [37, 53, 68]]
    assert module.predict(fitted, *inputs, h, queries)[0] == 15.
    h.loc[0, 'other_leaf_available'] = 1.
    assert module.predict(fitted, *inputs, h, queries)[0] == 55.
    with pytest.raises(ValueError, match='domain'):
        module.predict(fitted, *inputs, h, queries.assign(source='unknown'))
    with pytest.raises(ValueError, match='domain'):
        module.predict(fitted, *inputs, h, queries.assign(date='2019-06-01'))
    with pytest.raises(ValueError, match='labels'):
        module.predict(fitted, *inputs, h, queries.assign(observed_percent=99.))
