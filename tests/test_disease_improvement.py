"""Scientific boundary tests for the disease improvement experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def observations():
    return pd.DataFrame(dict(
        global_target_index=[1, 2, 3, 4, 5], field_id=['a']*5,
        site_id=['s']*5, source=['BASF']*5, coordinate_year=['s|2018']*5,
        endpoint_series=['leaf0', 'leaf1', 'leaf0', 'leaf1', 'leaf0'],
        leaf_index=[0, 1, 0, 1, 0], date=['2018-05-01', '2018-05-02',
        '2018-05-08', '2018-05-08', '2018-05-20'],
        observed_percent=[10., 30., 50., 80., 100.], season_year=[2018]*5))


def weather():
    return pd.DataFrame(dict(location_id=['s']*40,
        date=pd.date_range('2018-04-20', periods=40), tmean_c=[15.]*40,
        tmax_c=[20.]*40, rh_mean_pct=[85.]*40, precipitation_mm=[0.]*40, rain_mm=[0.]*40,
        rain_hours_gt_0_1mm=[0.]*40, rh_hours_ge_90pct=[8.]*40))


def test_history_ignores_same_day_and_future_labels_and_preserves_source():
    from models.monitoring.history import build_history
    obs = observations()
    query = obs.iloc[[2]].drop(columns='observed_percent')
    h, provenance = build_history(obs, query, {})
    altered = obs.copy()
    altered.loc[altered.date.ge('2018-05-08'), 'observed_percent'] = 0
    h2, _ = build_history(altered, query, {})
    np.testing.assert_array_equal(h, h2)
    assert h.loc[0, 'latest_severity_fraction'] == .1
    assert h.loc[0, 'other_leaf_latest_mean'] == .3
    assert set(provenance.iloc[0].prior_target_ids) == {1, 2}
    bad = obs.copy(); bad.loc[0, 'source'] = 'Corteva'
    with pytest.raises(ValueError, match='metadata'):
        build_history(bad, query, {})


def test_weather_features_use_actual_gap_and_preserve_event_order():
    from models.monitoring.features import build_features
    obs = observations(); query = obs.iloc[[2]]
    fields = pd.DataFrame(dict(field_id=['a'], site_id=['s'], sowing_date=['2018-04-20']))
    w = weather(); w.loc[w.date.eq('2018-05-04'), 'precipitation_mm'] = 12.
    w.loc[w.date.eq('2018-05-04'), 'rain_mm'] = 12.
    config = {'window_days': [7, 21], 'include_process': True,
              'prior_dates': {3: '2018-05-01'}}
    x, names, provenance = build_features(w, fields, query, config)
    assert np.isfinite(x).all()
    assert x[0, names.index('gap_rain_total_mm')] == 12
    assert x[0, names.index('gap_days')] == 7
    future = w.copy(); future.loc[future.date.gt('2018-05-08'), ['precipitation_mm', 'rain_mm']] = 99
    x2, _, _ = build_features(future, fields, query, config)
    np.testing.assert_allclose(x, x2)
    assert provenance.iloc[0].weather_end == '2018-05-08'
    moved = w.copy(); moved[['precipitation_mm', 'rain_mm']] = 0
    moved.loc[moved.date.eq('2018-05-08'), ['precipitation_mm', 'rain_mm']] = 12
    x3, _, _ = build_features(moved, fields, query, config)
    assert x3[0, names.index('gap_rain_total_mm')] == 12
    assert x3[0, names.index('rain_age_days')] != x[0, names.index('rain_age_days')]
    with pytest.raises(ValueError, match='weather'):
        build_features(w.loc[~w.date.eq('2018-05-04')], fields, query, config)


def test_loading_keeps_benchmark_and_flags_unresolved_measurements():
    from models.monitoring.inputs import load_observations
    obs, fields, provenance = load_observations(Path(__file__).resolve().parents[1])
    assert len(obs) == 2287 and obs.global_target_index.is_unique
    assert len(fields) == 216
    assert obs.measurement_qualified.eq(False).all()
    assert obs.loc[obs.global_target_index.eq(1036), 'after_grain_record'].item()
    assert fields.cultivar.isna().all()
    assert provenance['independent_validation_available'] is False


def test_score_requires_complete_predictions_and_preserves_hierarchy():
    from models.monitoring.evaluation import score_predictions
    d = observations(); d['prediction'] = d.observed_percent
    result = score_predictions(d, 'prediction')
    assert result['rmse'] == result['integral_rmse'] == 0
    d.loc[0, 'prediction'] = np.nan
    with pytest.raises(ValueError, match='finite|Finite|complete'):
        score_predictions(d, 'prediction')


def test_group_temporal_split_has_no_year_or_spatial_overlap():
    from models.monitoring.evaluation import temporal_spatial_splits
    d = observations()
    copies = []
    for year in [2014, 2015, 2016]:
        for site, lat in [('s0', 40.), ('s1', 45.), ('s2', 50.)]:
            a = d.copy(); a['season_year'] = year; a['site_id'] = site
            a['latitude'] = lat; a['longitude'] = 0.
            a['field_id'] = f'{site}|{year}'
            a['global_target_index'] += len(copies)*10
            copies.append(a)
    frame = pd.concat(copies, ignore_index=True)
    splits = list(temporal_spatial_splits(frame, [2016], folds=3, buffer_km=50))
    assert sum(len(test) for _, _, test in splits) == 15
    for _, train, test in splits:
        assert frame.iloc[train].season_year.max() < frame.iloc[test].season_year.min()
        assert not set(frame.iloc[train].site_id) & set(frame.iloc[test].site_id)
    changed = frame.copy(); changed['observed_percent'] = 0
    other = list(temporal_spatial_splits(changed, [2016], folds=3, buffer_km=50))
    for first, second in zip(splits, other):
        np.testing.assert_array_equal(first[1], second[1])


def test_split_model_bounds_sparse_leaf_fallback_and_prediction_no_truth():
    from models.monitoring.history import build_history
    from models.monitoring.models import fit_candidate, predict_candidate
    obs = observations(); h, _ = build_history(obs, obs, {})
    x = np.arange(len(obs)*2).reshape(-1, 2).astype(float)
    fitted = fit_candidate(x, h, obs, {'structure': 'split', 'estimator': 'hgb',
        'min_samples_leaf': 2, 'max_iter': 10})
    query = obs.drop(columns='observed_percent').copy()
    query.loc[0, 'leaf_index'] = 6; query.loc[0, 'source'] = 'Corteva'
    p = predict_candidate(fitted, x, h, query)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 100)).all()
    changed = query.assign(observed_percent=10000)
    np.testing.assert_array_equal(p, predict_candidate(fitted, x, h, changed))


def test_promotion_rejects_adaptive_evidence_and_severe_deterioration():
    from models.monitoring.evaluation import promotion_decision
    current = {'rmse': 20., 'integral_rmse': 500.}
    candidate = {'rmse': 17., 'integral_rmse': 450.}
    interval = [-4., -1.]
    assert not promotion_decision(current, candidate, interval,
        independent=False, subgroup_deltas=dict(cold=0, warm=0, BASF=0, Corteva=0),
        severe_bias_delta=-1, new_site_ratio=.85)['promote']
    assert not promotion_decision(current, candidate, interval,
        independent=True, subgroup_deltas=dict(cold=0, warm=0, BASF=0, Corteva=0),
        severe_bias_delta=1, new_site_ratio=.85)['promote']
    assert promotion_decision(current, candidate, interval,
        independent=True, subgroup_deltas=dict(cold=.5, warm=.5, BASF=.5, Corteva=.5),
        severe_bias_delta=-1, new_site_ratio=.85)['promote']


def test_promotion_requires_new_site_and_each_named_subgroup():
    from models.monitoring.evaluation import promotion_decision
    ref, candidate = {'rmse': 20., 'integral_rmse': 500.}, {'rmse': 17., 'integral_rmse': 450.}
    groups = dict(cold=0., warm=0., BASF=0., Corteva=0.)
    assert not promotion_decision(ref, candidate, [-4., -1.], independent=True,
        subgroup_deltas=groups, severe_bias_delta=-1.)['promote']
    assert not promotion_decision(ref, candidate, [-4., -1.], independent=True,
        subgroup_deltas={'cold': 0}, severe_bias_delta=-1., new_site_ratio=.85)['promote']


def test_invalid_history_dates_fail_instead_of_silently_routing_cold():
    from models.monitoring.history import build_history
    obs = observations(); obs.loc[0, 'date'] = 'NaT'
    with pytest.raises(ValueError, match='date'):
        build_history(obs, observations().iloc[[2]], {})
    query = observations().iloc[[2]].copy(); query['date'] = 'NaT'
    with pytest.raises(ValueError, match='date'):
        build_history(observations(), query, {})
    duplicate = observations().iloc[[0, 0]].copy().reset_index(drop=True)
    duplicate['global_target_index'] = [100, 101]
    duplicate['date'] = ['2018-05-01 01:00', '2018-05-01 23:00']
    with pytest.raises(ValueError, match='Duplicate'):
        build_history(duplicate, observations().iloc[[2]], {})


def test_split_heads_preserve_original_training_hierarchy_weights():
    from models.monitoring.models import fit_candidate, predict_candidate
    from models.monitoring.history import SAME_NAMES
    frame = observations()
    frame['field_id'] = ['a', 'a', 'a', 'b', 'b']
    frame['leaf_index'] = 0
    frame['observed_percent'] = [0., 100., 50., 100., 50.]
    history = pd.DataFrame(np.zeros((5, 8)), columns=SAME_NAMES)
    history['prior_assessment_available'] = [0, 0, 1, 0, 1]
    x = np.zeros((5, 1))
    fitted = fit_candidate(x, history, frame, dict(structure='split', max_iter=1, min_samples_leaf=20))
    p = predict_candidate(fitted, x, history, frame)
    assert p[0] == pytest.approx(100*5/7)


def test_primary_selection_excludes_later_trained_crop_features():
    from models.monitoring.run import CANDIDATES, PRIMARY_CANDIDATES
    assert PRIMARY_CANDIDATES
    assert all(CANDIDATES[name]['features'] in ['weather', 'weather_events', 'weather_process']
               for name in PRIMARY_CANDIDATES)
    assert any(CANDIDATES[name]['structure'] == 'process' for name in PRIMARY_CANDIDATES)


def test_primary_designs_are_invariant_to_all_crop_columns():
    from models.monitoring.run import assemble_designs, CANDIDATES, PRIMARY_CANDIDATES
    legacy = np.ones((4, 38)); categories = np.eye(4)
    events = np.ones((4, 4)); names = ['rain_total_7d_mm', 'process_damage',
        'process_infectious', 'process_latent']
    first = assemble_designs(legacy, categories, events, names)
    legacy[:, 12:] = 10000
    second = assemble_designs(legacy, categories, events, names)
    for name in PRIMARY_CANDIDATES:
        key = CANDIDATES[name]['features']
        np.testing.assert_array_equal(first[key], second[key])


def test_process_correction_uses_process_state_as_baseline():
    from models.monitoring.models import fit_candidate, predict_candidate
    from models.monitoring.history import SAME_NAMES
    frame = observations(); frame['leaf_index'] = 0
    history = pd.DataFrame(np.zeros((5, 8)), columns=SAME_NAMES)
    x = np.array([0., .1, .2, .3, .4])[:, None]
    fitted = fit_candidate(x, history, frame, dict(structure='process',
        process_baseline_column=0, max_iter=1, min_samples_leaf=20))
    result = predict_candidate(fitted, x, history, frame)
    offset = (frame.observed_percent.to_numpy()/100-x[:, 0]).mean()
    np.testing.assert_allclose(result, 100*np.clip(x[:, 0]+offset, 0, 1))


def test_direct_head_does_not_force_an_uncalibrated_process_baseline():
    from models.monitoring.models import fit_candidate, predict_candidate
    from models.monitoring.history import SAME_NAMES
    frame=observations();frame['leaf_index']=0
    history=pd.DataFrame(np.zeros((5,8)),columns=SAME_NAMES)
    x=np.linspace(.2,.9,5)[:,None]
    fitted=fit_candidate(x,history,frame,dict(structure='direct',max_iter=1,min_samples_leaf=20))
    changed_history=history.copy()
    changed_history['prior_assessment_available']=1
    changed_history['latest_severity_fraction']=.9
    prediction=predict_candidate(fitted,np.ones((5,1)),changed_history,frame)
    np.testing.assert_allclose(prediction,np.repeat(frame.observed_percent.mean(),5))


def test_diagnostic_subsets_retain_full_cohort_weights_and_report_rapid_rises():
    from models.monitoring.evaluation import score_subset
    from models.monitoring.run import _scores
    from paper.analyses.source_support.metrics import hierarchical_weights
    d = observations(); d['field_id'] = ['a', 'a', 'a', 'b', 'b']
    d['leaf_index'] = 0; d['prediction'] = 20.
    mask = np.array([True, False, True, False, True])
    weights = hierarchical_weights(d)
    value = score_subset(d, mask, 'prediction')
    assert value['rmse'] == pytest.approx(np.sqrt(np.average(
        (d.loc[mask].prediction-d.loc[mask].observed_percent)**2, weights=weights[mask])))
    d['prior_assessment_available'] = [False, True, True, False, True]
    d['observed_change_pp'] = [np.nan, 20, 45, np.nan, 50]
    d['observed_percent'] = [0, 1, 50, 80, 100]
    scores = _scores(d, ['prediction'])
    assert {'observed_above_75', 'observed_at_most_5', 'observed_increase_above_40'} <= set(scores.population)
    assert scores.loc[scores.population.eq('observed_increase_above_40'), 'n'].item() == 2


def test_selector_uses_inner_predictions_and_integral_guard():
    from models.monitoring.run import select_candidate
    d = observations(); d['reference'] = d.observed_percent+10
    d['better'] = d.observed_percent+5
    d['worse'] = d.observed_percent+20
    selected, rows = select_candidate(d, ['reference', 'better', 'worse'])
    assert selected == 'better'
    assert set(r['candidate'] for r in rows) == {'reference', 'better', 'worse'}
    d.loc[0, 'better'] = np.nan
    with pytest.raises(ValueError, match='complete|Complete'):
        select_candidate(d, ['reference', 'better'])


def test_selector_keeps_reference_when_cluster_uncertainty_spans_zero():
    from models.monitoring.run import select_candidate
    parts = []
    for site in range(4):
        part = observations().iloc[:3].copy()
        part['field_id'] = f'f{site}'; part['site_id'] = f's{site}'
        part['coordinate_year'] = f's{site}|2018'
        part['leaf_index'] = 0; part['endpoint_series'] = 'same'
        part['observed_percent'] = 50.; part['reference'] = 60.
        part['candidate'] = 69. if site == 3 else 50.
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    selected, rows = select_candidate(frame, ['reference', 'candidate'])
    assert selected == 'reference'
    row = next(r for r in rows if r['candidate'] == 'candidate')
    assert row['paired_upper'] > 0


def test_prepare_registers_before_fitting_and_refuses_overwrite(tmp_path):
    from models.monitoring.run import run_experiment
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'run'
    result = run_experiment(root, output, years=[2015], prepare_only=True, quality_policy='retain')
    import json
    protocol = json.loads((output/'registration.json').read_text())
    assert protocol['independent_validation'] is False
    assert protocol['status'] == 'registered_before_fitting'
    assert result['trained'] is False
    assert (output/'observation_ledger.csv').is_file()
    assert (output/'acquisition_metadata_template.csv').is_file()
    assert (output/'feature_provenance.csv').is_file()
    with pytest.raises(FileExistsError):
        run_experiment(root, output, years=[2015], prepare_only=True, quality_policy='retain')


def test_full_year_run_is_independently_verifiable_and_detects_metric_tampering(tmp_path):
    from models.monitoring.run import run_experiment
    from models.monitoring.verify import verify_run
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'fitted'
    result = run_experiment(root, output, years=[2015], spatial=False, quality_policy='retain')
    assert result['trained'] and not result['promote']
    checked = verify_run(root, output)
    assert checked['verified'] and checked['assessments'] == 426
    scores = pd.read_csv(output/'metrics.csv')
    scores.loc[0, 'rmse'] += 1
    scores.to_csv(output/'metrics.csv', index=False)
    with pytest.raises(ValueError, match='metric'):
        verify_run(root, output)


def test_quality_screen_quarantines_chronology_without_selecting_residuals():
    from models.monitoring.inputs import load_observations, quality_screen
    root = Path(__file__).resolve().parents[1]
    obs, fields, _ = load_observations(root)
    keep, excluded = quality_screen(obs, fields)
    assert excluded.global_target_index.tolist() == [1036]
    assert len(keep) == 2286
    assert excluded.reason.str.contains('grain').all()
    changed = obs.assign(selected_pipeline=obs.observed_percent+1000)
    again, excluded_again = quality_screen(changed, fields)
    assert set(again.global_target_index) == set(keep.global_target_index)
    assert excluded_again.global_target_index.tolist() == [1036]
    bad = obs.copy(); bad.loc[0, 'observed_percent'] = 101
    _, exclusions = quality_screen(bad, fields)
    assert 0 in set(exclusions.global_target_index)


def test_liquid_rain_and_previous_day_weather_cutoff():
    from models.monitoring.features import build_features
    fields = pd.DataFrame(dict(field_id=['a'], site_id=['s'], sowing_date=['2018-04-20']))
    queries = observations().iloc[[2]]
    w = weather()
    w.loc[w.date.eq('2018-05-04'), 'precipitation_mm'] = 30.
    w.loc[w.date.eq('2018-05-07'), ['precipitation_mm', 'rain_mm']] = 4.
    config = dict(weather_cutoff_days=1, prior_dates={3: '2018-05-01'})
    x, names, provenance = build_features(w, fields, queries, config)
    assert x[0, names.index('gap_rain_total_mm')] == 4
    assert provenance.iloc[0].weather_end == '2018-05-07'
    assert provenance.iloc[0].rain_variable == 'rain_mm'
    changed = w.copy()
    changed.loc[changed.date.ge('2018-05-08'), ['precipitation_mm', 'rain_mm']] = 100
    y, _, _ = build_features(changed, fields, queries, config)
    np.testing.assert_allclose(x, y)
    assert x[0, names.index('weather_rain_log1p_mean_7d')] == pytest.approx(np.log1p(4/7))


def test_separate_source_models_do_not_mix_measurement_targets():
    from models.monitoring.models import fit_candidate, predict_candidate
    from models.monitoring.history import SAME_NAMES
    first = observations(); first['observed_percent'] = 10.; first['source'] = 'BASF'
    other = first.copy(); other['global_target_index'] += 100; other['source'] = 'Corteva'
    other['field_id'] = 'b'; other['coordinate_year'] = 't|2018'; other['observed_percent'] = 90.
    frame = pd.concat([first, other], ignore_index=True)
    h = pd.DataFrame(np.zeros((10, 8)), columns=SAME_NAMES); x = np.zeros((10, 1))
    spec = dict(structure='residual', source_specific=True, max_iter=1, min_samples_leaf=20)
    fitted = fit_candidate(x, h, frame, spec)
    prediction = predict_candidate(fitted, x, h, frame)
    np.testing.assert_allclose(prediction[:5], 10.)
    np.testing.assert_allclose(prediction[5:], 90.)
