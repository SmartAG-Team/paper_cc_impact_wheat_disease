import numpy as np
import pandas as pd


def test_random_coefficient_filter_extrapolates_trend_and_withholds_current_day():
    from models.monitoring.random_coefficient_filter import predict_coefficients
    covariance = np.eye(14)*.5; covariance[7:, 7:] *= 4
    fitted = dict(link='linear', sources={'BASF': dict(covariance=covariance, noise=.0001)},
        supported={'BASF': [0]})
    frame = pd.DataFrame(dict(field_id=['a']*3+['b'], source='BASF', leaf_index=0,
        date=['2015-05-01', '2015-05-11', '2015-05-21', '2015-05-21'],
        observed_percent=[20., 40., 100., 100.]))
    clock = np.array([1.8, 2., 2.2, 2.2]); baseline = np.full(4, 40.)
    value = predict_coefficients(fitted, frame, baseline, clock)
    assert value[0] == value[3] == 40
    assert 55 < value[2] < 65
    changed = frame.copy(); changed.loc[2:, 'observed_percent'] = 0
    np.testing.assert_array_equal(value, predict_coefficients(fitted, changed, baseline, clock))


def test_training_learns_cross_category_error_covariance():
    from models.monitoring.random_coefficient_filter import fit_coefficients, predict_coefficients
    frame = pd.DataFrame(dict(field_id=np.repeat(np.arange(20).astype(str), 4), source='BASF',
        leaf_index=np.tile([0, 1, 0, 1], 20),
        date=np.tile(['2014-05-01', '2014-05-01', '2014-05-11', '2014-05-11'], 20),
        observed_percent=np.repeat(np.tile([20., 80.], 10), 4), global_target_index=np.arange(80)))
    clock = np.tile([1.8, 1.8, 2., 2.], 20)
    fitted = fit_coefficients(frame, np.full(80, 50.), clock, np.ones(80), 'linear')
    assert fitted['sources']['BASF']['covariance'][0, 1] > .01
    query = pd.DataFrame(dict(field_id='new', source='BASF', leaf_index=[0, 1],
        date=['2015-05-01', '2015-05-11'], observed_percent=[80., 0.]))
    value = predict_coefficients(fitted, query, np.full(2, 50.), np.array([1.8, 2.]))
    assert value[0] == 50 and value[1] > 65
