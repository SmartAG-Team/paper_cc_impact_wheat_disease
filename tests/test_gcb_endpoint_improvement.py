import numpy as np
import pandas as pd
import pytest


def _visits():
    return pd.DataFrame(dict(
        field_id=['a'] * 6 + ['b'],
        date=['2015-05-01', '2015-05-08', '2015-05-15',
              '2015-05-15', '2015-05-22', '2015-05-29', '2015-05-22'],
        leaf_index=[0, 0, 0, 1, 0, 1, 0],
        observed_percent=[10., 20., 80., 99., 90., 0., 90.],
    ))


def test_zero_extension_reproduces_the_retained_monitoring_filter():
    from models.monitoring.discrepancy_filter import predict_discrepancy
    from paper.scripts.endpoint_improvement import predict_endpoint

    frame = _visits()
    baseline = np.array([0., 20., 40., 60., 80., 100., 15.])
    kernel = np.log([2., 1., 30., 30.])
    expected = predict_discrepancy(kernel, frame, baseline, 'logit')
    actual = predict_endpoint(np.r_[kernel, 0., 0., 0.], frame, baseline, growth=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)


def test_native_growth_requires_two_strictly_earlier_same_code_dates():
    from paper.scripts.endpoint_improvement import native_growth

    value = native_growth(_visits())
    # logit(.20)-logit(.10)=log(2.25), with equal seven-day gaps.
    assert value[2] == pytest.approx(0.8109302162163288)
    np.testing.assert_array_equal(value[[0, 1, 3, 5, 6]], np.zeros(5))


def test_joint_model_withholds_the_whole_query_day_and_other_fields():
    from paper.scripts.endpoint_improvement import predict_endpoint

    frame = _visits()
    theta = np.r_[np.log([2., 1., 30., 30.]), -.1, .05, .2]
    baseline = np.full(len(frame), 40.)
    first = predict_endpoint(theta, frame, baseline, growth=True)
    changed = frame.copy()
    changed.loc[[2, 3, 4, 5, 6], 'observed_percent'] = [0., 0., 1., 100., 0.]
    second = predict_endpoint(theta, changed, baseline, growth=True)
    np.testing.assert_array_equal(first[:4], second[:4])
    assert first[4] != second[4]
    assert first[6] == second[6]


def test_training_calibration_learns_a_historical_mean_shift():
    from paper.scripts.endpoint_improvement import fit_endpoint, predict_endpoint

    frame = pd.DataFrame(dict(field_id=[f'f{i}' for i in range(12)],
        date=['2014-06-01'] * 12, leaf_index=[0] * 12,
        observed_percent=np.full(12, 40.)))
    baseline = np.full(12, 20.)
    fitted = fit_endpoint(frame, baseline, np.ones(12), np.log([1., 1., 30., 30.]), growth=False)
    forecast = predict_endpoint(fitted['parameters'], frame, baseline, growth=False)
    assert np.mean((forecast - 40.) ** 2) < 1.
    assert fitted['objective'] < fitted['initial_objective']


def test_endpoint_fitting_rejects_nonpositive_scientific_weights():
    from paper.scripts.endpoint_improvement import fit_endpoint

    frame = _visits()
    with pytest.raises(ValueError, match='positive'):
        fit_endpoint(frame, np.full(len(frame), 40.), np.zeros(len(frame)),
                     np.log([1., 1., 30., 30.]), growth=False)


def test_visit_context_labels_whole_days_and_excludes_same_day_native_records():
    from paper.scripts.endpoint_improvement import visit_context

    frame = _visits()
    frame.loc[6, 'date'] = '2015-05-15'
    context = visit_context(frame)
    np.testing.assert_array_equal(context.first_visit, [True, False, False, False, False, False, True])
    np.testing.assert_array_equal(context.last_visit, [False, False, False, False, False, True, True])
    np.testing.assert_array_equal(context.only_visit, [False, False, False, False, False, False, True])
    assert context.loc[2, 'prior_same_nativecode_count'] == 2
    assert context.loc[3, 'prior_same_nativecode_count'] == 0
    assert context.loc[2, 'prior_same_nativecode_lastdate'] == '2015-05-08'
