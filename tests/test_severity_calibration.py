import joblib
import numpy as np
import pytest
from scipy.special import expit, logit


def test_bounded_calibration_learns_known_severity_map():
    from models.monitoring.severity_calibration import fit_calibration, predict_calibration
    p = np.linspace(1., 99., 120)
    y = 100*expit(-1.+.6*logit(p/100))
    fit = fit_calibration(p, y, np.ones(120), np.ones(120, int), 'logit')
    actual = predict_calibration(fit, np.array([10., 50., 90.]))
    expected = 100*expit(-1.+.6*logit(np.array([.1, .5, .9])))
    np.testing.assert_allclose(actual, expected, atol=.5)
    assert fit['converged']
    assert fit['objective'] < fit['initial_objective']*.05


def test_weighted_isotonic_pooling_and_bounded_extrapolation():
    from models.monitoring.severity_calibration import fit_calibration, predict_calibration
    p = np.array([10., 20., 20., 40.]); y = np.array([5., 40., 20., 90.])
    fit = fit_calibration(p, y, np.array([1., 3., 1., 1.]), np.ones(4, int), 'isotonic')
    values = predict_calibration(fit, np.array([0., 10., 20., 40., 100.]))
    np.testing.assert_allclose(values, [5., 5., 35., 90., 90.])
    assert (np.diff(values) >= 0).all()


def test_missing_oob_rows_cannot_enter_calibration_or_receive_fitted_fallback():
    from models.monitoring.severity_calibration import fit_calibration, predict_calibration
    p = np.array([10., 30., 70., np.nan]); y = np.array([5., 20., 50., 100.])
    counts = np.array([2, 1, 4, 0]); weights = np.array([1., 2., 3., 1000.])
    for kind in ['logit', 'isotonic']:
        a = fit_calibration(p, y, weights, counts, kind)
        changed = y.copy(); changed[-1] = 0.
        b = fit_calibration(p, changed, weights*17., counts, kind)
        np.testing.assert_allclose(predict_calibration(a, np.array([0., 50., 100.])),
            predict_calibration(b, np.array([0., 50., 100.])), atol=1e-8)
        np.testing.assert_array_equal(a['calibration_rows'], [0, 1, 2])
    with pytest.raises(ValueError, match='zero-count'):
        fit_calibration(np.nan_to_num(p), y, weights, counts, 'logit')
    with pytest.raises(ValueError, match='positive-count'):
        fit_calibration(p, y, weights, np.ones(4, int), 'logit')


def test_saved_calibration_is_monotone_and_requires_no_query_outcomes(tmp_path):
    from models.monitoring.severity_calibration import fit_calibration, predict_calibration
    p = np.linspace(1., 99., 35); y = np.minimum(100., .75*p+3.)
    for kind in ['logit', 'isotonic']:
        fit = fit_calibration(p, y, np.ones(35), np.ones(35, int), kind)
        query = np.linspace(0., 100., 501)
        values = predict_calibration(fit, query)
        assert ((values >= 0) & (values <= 100)).all() and (np.diff(values) >= 0).all()
        path = tmp_path/f'{kind}.joblib'; joblib.dump(fit, path)
        np.testing.assert_array_equal(values, predict_calibration(joblib.load(path), query))


def test_no_oob_support_preserves_original_forecast_without_fitted_calibration():
    from models.monitoring.severity_calibration import fit_calibration, predict_calibration
    for kind in ['logit', 'isotonic']:
        fit = fit_calibration(np.full(3, np.nan), np.array([2., 30., 90.]), np.ones(3), np.zeros(3, int), kind)
        assert fit['fit_status'] == 'no_honest_calibration_support'
        assert fit['covered_count'] == 0 and len(fit['calibration_rows']) == 0
        p = np.array([0., 30., 100.])
        np.testing.assert_array_equal(predict_calibration(fit, p), p)
