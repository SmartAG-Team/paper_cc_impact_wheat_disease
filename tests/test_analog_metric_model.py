import numpy as np
import pandas as pd


def test_learned_analogs_identify_predictive_state_and_keep_unknown_endpoint_fallback():
    from models.monitoring.analog_metric_model import fit_analogs, predict_analogs
    rng = np.random.default_rng(19)
    signal = np.tile([-1., 1.], 40)
    x = np.column_stack([signal, rng.normal(size=(80, 12))])
    frame = pd.DataFrame(dict(field_id=np.repeat(np.arange(20).astype(str), 4),
        source='BASF', leaf_index=0, global_target_index=np.arange(80),
        observed_percent=50+40*signal))
    fit = fit_analogs(x, frame, np.full(80, 50.), np.ones(80), 'direct', epochs=250)
    query = pd.DataFrame(dict(field_id=['new']*3, source='BASF', leaf_index=[0, 0, 6]))
    q = np.column_stack([[-1., 1., 1.], rng.normal(size=(3, 12))])
    prediction = predict_analogs(fit, q, query, np.array([50., 50., 31.]))
    assert prediction[0] < 20 and prediction[1] > 80
    assert prediction[2] == 31
    assert fit['metric'][0] > np.mean(fit['metric'][1:])


def test_analog_training_cannot_use_any_observation_from_its_own_coordinate():
    from models.monitoring.analog_metric_model import training_neighbor_mask
    frame = pd.DataFrame(dict(field_id=['a', 'b', 'c', 'd', 'e'],
        coordinate_year=['x2014', 'x2015', 'y2014', 'z2014', 'q2014'],
        latitude=[51., 51., 52., 53., 54.], longitude=0.,
        source=['BASF', 'BASF', 'BASF', 'Corteva', 'BASF'], leaf_index=[0, 0, 0, 0, 1]))
    mask = training_neighbor_mask(frame)
    assert not mask[0, 0] and not mask[0, 1]
    assert mask[0, 2] and not mask[0, 3] and not mask[0, 4]
    np.testing.assert_array_equal(mask, mask.T)
