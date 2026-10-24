import joblib
import numpy as np
import pandas as pd
import pytest


def _prior():
    from models.monitoring.phenology_transfer import MODELS, STAGES
    distributions = {m: {s: dict(values=np.array([value]), cumulative_weight=np.array([1.]))
        for s, value in zip(STAGES, [20., 80., 160., 300.])} for m in MODELS}
    return dict(parameters=dict(photoperiod_onset_gdd=20., photoperiod_stop_gdd=160.,
        vernalization_onset_tpp=20., vernalization_stop_tpp=80.), distributions=distributions,
        trait_coordinates=pd.DataFrame(dict(latitude=[47., 54.], longitude=[9., 12.])))


def _inputs():
    dates = pd.date_range('2017-09-01', periods=55)
    weather = pd.DataFrame(dict(location_id='site', date=dates, tmean_c=12., tmax_c=18.,
        rh_mean_pct=85., precipitation_mm=np.tile([0., 2., 0., 1., 5.], 11)))
    fields = pd.DataFrame(dict(field_id=['field'], site_id=['site'], latitude=[50.], longitude=[10.],
        sowing_date=['2017-09-04'], estimated_sowing_date=['2017-09-02']))
    query = pd.DataFrame(dict(global_target_index=[1, 2], field_id='field', source='BASF', leaf_index=[0, 1],
        date=['2017-09-20', '2017-10-05']))
    return weather, fields, query


def test_features_are_four_weather_only_and_strictly_causal():
    from models.monitoring.autonomous_weather import build_autonomous_features
    weather, fields, query = _inputs()
    a = build_autonomous_features(weather, fields, query, _prior())
    changed = weather.copy()
    future = changed.date.ge(pd.Timestamp(query.date.iloc[0]))
    changed.loc[future, ['tmean_c', 'tmax_c', 'rh_mean_pct', 'precipitation_mm']] = [22., 30., 95., 20.]
    b = build_autonomous_features(changed, fields, query, _prior())
    for key in ['baseline', 'augmented', 'phase']:
        np.testing.assert_array_equal(a[key][0], b[key][0])
    assert not np.array_equal(a['augmented'][1], b['augmented'][1])
    changed = weather.copy()
    for column in ['tmin_c', 'rain_mm', 'rain_hours_gt_0_1mm', 'rh_hours_ge_90pct', 'satellite']:
        changed[column] = np.nan
    other_query = query.assign(observed_percent=[100., 0.], bbch=[99., 10.], prior_severity=[1., 99.])
    c = build_autonomous_features(changed, fields.assign(observed_sowing_date='2017-08-01'), other_query, _prior())
    for key in ['baseline', 'augmented', 'phase']:
        np.testing.assert_array_equal(a[key], c[key])
    assert a['baseline'].shape == (2, 34)
    assert a['augmented'].shape == (2, 129)
    assert a['provenance'].weather_end.tolist() == ['2017-09-19', '2017-10-04']


def test_sowing_fallback_uses_only_query_weather_prefix():
    from models.monitoring.autonomous_weather import build_autonomous_features
    weather, fields, query = _inputs()
    weather.loc[weather.date.eq('2017-09-03'), 'rh_mean_pct'] = np.nan
    a = build_autonomous_features(weather, fields, query, _prior())
    assert a['provenance'].sowing_fallback.all()
    assert a['provenance'].weather_start.eq('2017-09-04').all()
    later = weather.copy()
    later.loc[later.date.eq('2017-10-04'), 'tmax_c'] = np.nan
    b = build_autonomous_features(later, fields, query.iloc[:1], _prior())
    np.testing.assert_array_equal(a['augmented'][:1], b['augmented'])
    with pytest.raises(ValueError, match='Complete four-variable'):
        build_autonomous_features(later, fields, query, _prior())


def test_delay_bank_matches_gamma_cdf_and_has_no_future_exposure():
    from models.monitoring.autonomous_weather import delayed_exposure
    dose = np.array([1., 0., 0.]); thermal = np.ones(3)
    stage = np.ones((3, 4))
    x = delayed_exposure(dose, thermal, stage, 2)
    z = 3.*2./10.
    cdf = 1-np.exp(-z)*(1+z+z*z/2)
    np.testing.assert_allclose(x[:4], [1-cdf, cdf, cdf*np.exp(-2/21), -np.expm1(-.01*cdf)])
    np.testing.assert_allclose(x, delayed_exposure(np.r_[dose, 1e6], np.r_[thermal, 100.],
        np.r_[stage, [[0., 0., 0., 0.]]], 2))
    assert (x >= 0).all() and np.isfinite(x).all()
    y = delayed_exposure(2*dose, thermal, stage, 2)
    assert np.all(y >= x)


def test_weather_only_models_learn_without_observation_anchors(tmp_path):
    from models.monitoring.autonomous_weather import fit_autonomous, predict_autonomous
    count = 120
    x = np.tile(np.array([[0.], [1.]]), (count//2, 1))
    phase = np.tile([0., 0., 1., 0.], (count, 1))
    frame = pd.DataFrame(dict(global_target_index=np.arange(count), coordinate_year=np.arange(count).astype(str),
        observed_percent=np.where(x[:, 0] > .5, 90., 10.)))
    model = fit_autonomous(x, x, frame, np.ones(count), phase, members=2, iterations=80)
    values = predict_autonomous(model, x[:2], x[:2], phase[:2])
    assert model['history_conditioned'] is False
    for value in values.values():
        assert value[0] < 25 and value[1] > 75
    path = tmp_path/'fit.joblib'; joblib.dump(model, path)
    replay = predict_autonomous(joblib.load(path), x[:2], x[:2], phase[:2])
    for name in values:
        np.testing.assert_array_equal(values[name], replay[name])
    assert np.all(model['growth']['training_anchor'] == 0)
