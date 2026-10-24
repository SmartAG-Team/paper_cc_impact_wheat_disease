"""Replay fixed first-visit screens without changing completed experiment files."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model.seasonal_septoria.wetness import duration_exposure
from .evaluation import paired_interval, score_predictions, score_subset
from .field_history_models import predict_field_history
from .history_blind_model import predict_blind
from .infection_event_experiment import populations
from .inputs import sha
from .season_exposure_features import build_season_exposure
from .verify_french_weather import write_verification_receipt

ROOT = Path(__file__).resolve().parents[2]


def manual_weights(frame):
    """Independent explicit hierarchy, avoiding the scoring helper."""
    weights = np.empty(len(frame))
    for _, site in frame.reset_index(drop=True).groupby('coordinate_year'):
        nf = site.field_id.nunique()
        for _, field in site.groupby('field_id'):
            nl = field.leaf_index.nunique()
            for _, leaf in field.groupby('leaf_index'):
                weights[leaf.index] = 1 / (nf * nl * len(leaf))
    return weights


def manual_integral(frame, name):
    frame = frame.copy()
    if 'partition' not in frame:
        frame['partition'] = 'development'
    keys = ['partition', 'field_id', 'coordinate_year', 'source', 'leaf_index', 'endpoint_series']
    rows = []
    for key, part in frame.groupby(keys):
        part = part.sort_values('date')
        days = (pd.to_datetime(part.date) - pd.to_datetime(part.date.iloc[0])).dt.days.to_numpy()
        if len(part) < 2 or days[-1] <= 0:
            continue
        error = (part[name] - part.observed_percent).to_numpy()
        integral = sum((days[i+1]-days[i])*(error[i]+error[i+1])/2 for i in range(len(days)-1))
        rows.append(dict(zip(keys, key), error=integral))
    windows = pd.DataFrame(rows)
    return np.sqrt(np.average(windows.error**2, weights=manual_weights(windows)))


def verify(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    parent = root/'data/final_monitoring/cohort'
    reg = json.loads((output/'registration.json').read_text())
    for path, digest in reg['source_sha256'].items():
        assert sha(root/path) == digest, path
    for path, digest in reg['code_sha256'].items():
        assert sha(root/path) == digest == sha(output/'code_snapshot'/path), path
    obs = pd.read_csv(parent/'training_observation_ledger.csv')
    original = pd.read_csv(parent/'assessment_predictions.csv')
    frame = pd.read_csv(output/'assessment_predictions.csv')
    assert len(obs) == 2304 and obs.global_target_index.is_unique
    assert len(frame) == 2017 and frame.global_target_index.is_unique
    pd.testing.assert_frame_equal(frame[original.columns], original)
    arrays = np.load(parent/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    bundle = joblib.load(parent/'fitted_candidate.joblib')
    history = pd.DataFrame(arrays['history'], columns=bundle['base']['history_names'])
    assert arrays['station'].shape == (2304, 53) and history.shape == (2304, 35)
    positions = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    splits = [s for s in json.loads((parent/'split_membership.json').read_text())
              if s['fold'].startswith('temporal')]
    assert len(splits) == 5
    assert {s['fold'] for s in splits} == {'temporal'+str(y) for y in range(2015, 2020)}
    tests = [i for s in splits for i in s['test']]
    assert tests == frame.global_target_index.tolist() and len(set(tests)) == 2017
    blind = 'history_blind_component' in frame
    name = 'history_blind_first' if blind else 'season_exposure_first'
    component = 'history_blind_component' if blind else 'new_first_component'
    features = arrays['station']
    scalar_checks = perturbations = 0
    if not blind:
        new = np.load(output/'predictors.npz')
        np.testing.assert_array_equal(new['target_ids'], arrays['target_ids'])
        np.testing.assert_array_equal(new['history'], arrays['history'])
        np.testing.assert_array_equal(np.delete(new['first'], np.s_[50:56], axis=1), arrays['station'])
        fields = pd.read_csv(root/'data/disease_improvement/infection_events_20261022/fields.csv')
        weather = pd.read_parquet(root/'data/disease_improvement/station_preferred_ie_20261022/daily_weather.parquet')
        query = obs.drop(columns='observed_percent')
        rebuilt, names, provenance = build_season_exposure(weather, fields, query)
        np.testing.assert_array_equal(rebuilt, new['season_exposure'])
        pd.testing.assert_frame_equal(provenance, pd.read_csv(output/'feature_provenance.csv'), check_dtype=False)
        assert names == json.loads((output/'feature_schema.json').read_text())['new_feature_names']
        fields = fields.set_index('field_id')
        dates = pd.to_datetime(weather.date)
        selected = np.linspace(0, len(obs)-1, 30).astype(int)
        for pos in selected:
            row = obs.iloc[pos]
            field = fields.loc[row.field_id]
            start, end = pd.Timestamp(field.sowing_date), pd.Timestamp(row.date)
            daily = weather.loc[weather.location_id.eq(field.site_id) & dates.ge(start) & dates.lt(end)].copy()
            assert len(daily) == (end-start).days
            dose = duration_exposure(daily.tmean_c, daily.tmax_c, daily.rh_mean_pct, daily.rain_mm)['exposure']
            direct = [daily.rain_mm.sum(), daily.rain_mm.ge(.1).sum(), daily.rain_hours_gt_0_1mm.sum(),
                      daily.rh_hours_ge_90pct.sum(), np.sum(dose),
                      np.sum(np.asarray(dose)[pd.to_datetime(daily.date).lt(end-pd.Timedelta(days=42))])]
            np.testing.assert_allclose(direct, rebuilt[pos], atol=1e-10, rtol=1e-12)
            scalar_checks += 6
        for pos in selected[::6]:
            row = query.iloc[[pos]]
            cutoff = pd.Timestamp(row.date.iloc[0])
            changed = weather.copy()
            changed.loc[dates.ge(cutoff), 'rain_mm'] += 999
            probe = build_season_exposure(changed, fields.reset_index(), row)[0]
            np.testing.assert_array_equal(probe, rebuilt[[pos]])
            perturbations += 1
        try:
            build_season_exposure(weather, fields.reset_index(), obs.iloc[[0]])
        except ValueError:
            pass
        else:
            raise AssertionError('Query labels accepted')
        features = new['first']
        assert features.shape == (2304, 59)
    has_history = history.prior_assessment_available.eq(1) | history.other_leaf_available.eq(1)
    checked_members = checked_rows = 0
    support = []
    for split in splits:
        train, test = [positions.loc[split[k]].to_numpy() for k in ['train', 'test']]
        year = int(obs.iloc[test].season_year.iloc[0])
        assert obs.iloc[test].season_year.eq(year).all()
        assert obs.iloc[train].season_year.lt(year).all()
        assert set(split['train']) == set(obs.loc[obs.season_year.lt(year), 'global_target_index'])
        fitted = joblib.load(output/(split['fold']+'_first.joblib'))
        assert fitted['training_ids'] == split['train']
        assert fitted['seed'] == 20261022 and fitted['fraction'] == .8
        clusters = np.asarray(sorted(obs.iloc[train].coordinate_year.unique()))
        count = max(1, int(np.ceil(.8*len(clusters))))
        assert fitted['clusters_per_member'] == count and len(fitted['members']) == 20
        rng = np.random.default_rng(20261022)
        query = obs.iloc[test].drop(columns='observed_percent')
        h = history.iloc[test] if not blind else pd.DataFrame(np.zeros((len(test), 35)), columns=history.columns)
        predictions = []
        for member in fitted['members']:
            chosen = rng.choice(clusters, count, replace=False)
            sampled = obs.iloc[train].loc[obs.iloc[train].coordinate_year.isin(chosen)]
            assert member['training_ids'] == sampled.global_target_index.tolist()
            assert member['spec'] == {'structure': 'field_split'}
            assert member['n_features'] == features.shape[1]
            if blind:
                assert member['models']['history'] is None
                estimator = member['models']['none']
                baseline = np.average(sampled.observed_percent, weights=manual_weights(sampled))/100
                np.testing.assert_allclose(estimator._baseline_prediction.ravel(), [baseline], atol=1e-14)
            for estimator in member['models'].values():
                if estimator is None:
                    continue
                params = estimator.get_params()
                expected = dict(max_iter=200, max_leaf_nodes=15, max_depth=3, min_samples_leaf=10,
                    learning_rate=.05, l2_regularization=1., early_stopping=False, random_state=20261021)
                assert all(params[k] == v for k, v in expected.items())
            predictions.append(predict_field_history(member, features[test], h, query))
            checked_members += 1
        part = frame.set_index('global_target_index').loc[split['test']]
        replay = np.mean(predictions, axis=0)
        np.testing.assert_allclose(replay, part[component], atol=1e-10, rtol=0)
        first = ~has_history.iloc[test].to_numpy()
        np.testing.assert_array_equal(first, ~part.ie_uses_event_head.to_numpy())
        np.testing.assert_allclose(part.loc[first, name], .5*(part.loc[first, 'candidate_process']+replay[first]), atol=1e-10)
        np.testing.assert_array_equal(part.loc[~first, name], part.loc[~first, 'irish_station_candidate'])
        if blind:
            np.testing.assert_allclose(predict_blind(fitted, features[test], query), replay, atol=1e-10)
            try:
                predict_blind(fitted, features[test], obs.iloc[test])
            except ValueError:
                pass
            else:
                raise AssertionError('Query labels accepted')
        earlier_first = obs.iloc[train].loc[~has_history.iloc[train].to_numpy()]
        for row in part.loc[first].itertuples():
            category = earlier_first.loc[earlier_first.source.eq(row.source) & earlier_first.leaf_index.eq(row.leaf_index)]
            all_category = obs.iloc[train].loc[obs.iloc[train].source.eq(row.source) & obs.iloc[train].leaf_index.eq(row.leaf_index)]
            support.append(dict(global_target_index=row.Index, fold=split['fold'], source=row.source,
                leaf_index=row.leaf_index, observed_percent=row.observed_percent,
                prior_first_field_rows=len(earlier_first),
                prior_first_field_severe_rows=int(earlier_first.observed_percent.gt(75).sum()),
                prior_same_source_code_rows=len(category),
                prior_same_source_code_max_percent=float(category.observed_percent.max()) if len(category) else np.nan,
                prior_same_source_code_severe_rows=int(category.observed_percent.gt(75).sum()),
                all_assessment_same_source_code_rows=len(all_category),
                all_assessment_same_source_code_severe_rows=int(all_category.observed_percent.gt(75).sum())))
        checked_rows += len(test)
    support = pd.DataFrame(support)
    if not blind:
        saved = pd.read_csv(output/'training_support_audit.csv')
        pd.testing.assert_frame_equal(support[saved.columns], saved, check_dtype=False)
    assert len(support) == 494
    severe = support.loc[support.observed_percent.gt(75)]
    assert len(severe) == 13
    assert severe.prior_same_source_code_severe_rows.eq(0).sum() == 7
    assert severe.all_assessment_same_source_code_rows.eq(0).sum() == 2
    hh = history.iloc[positions.loc[frame.global_target_index]].reset_index(drop=True)
    masks = populations(frame, hh)
    metrics = pd.read_csv(output/'metrics.csv')
    models = [name, 'irish_station_candidate', 'reference_replay', 'current']
    assert len(metrics) == 4 and set(metrics.model) == set(models)
    for row in metrics.itertuples(index=False):
        score = score_predictions(frame, row.model)
        for key, value in score.items():
            if value is not None:
                np.testing.assert_allclose(getattr(row, key), value, atol=1e-9)
        direct = np.sqrt(np.average((frame[row.model]-frame.observed_percent)**2, weights=manual_weights(frame)))
        np.testing.assert_allclose(score['rmse'], direct, atol=1e-10)
        np.testing.assert_allclose(score['integral_rmse'], manual_integral(frame, row.model), atol=1e-9)
    groups = pd.read_csv(output/'subgroup_metrics.csv')
    assert len(groups) == 18 and set(zip(groups.population, groups.model)) == {(p, m) for p in masks for m in models[:2]}
    for row in groups.itertuples(index=False):
        for key, value in score_subset(frame, masks[row.population], row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row, key), value, atol=1e-9)
    interval = json.loads((output/'paired_interval.json').read_text())
    np.testing.assert_allclose(paired_interval(frame, name, 'irish_station_candidate'), [interval['lower'], interval['upper']], atol=1e-10)
    old, newscore = [score_predictions(frame, m) for m in models[1::-1]]
    gates = dict(overall_rmse_improves=newscore['rmse'] < old['rmse'],
        first_field_rmse_improves=score_subset(frame, masks['first_field'], name)['rmse'] < score_subset(frame, masks['first_field'], models[1])['rmse'],
        integral_noninferiority=newscore['integral_rmse'] <= old['integral_rmse'],
        severe_bias_noninferiority=abs(score_subset(frame, masks['severe_above75'], name)['bias']) <= abs(score_subset(frame, masks['severe_above75'], models[1])['bias']))
    for key in ['cold', 'warm', 'BASF', 'Corteva']:
        gates[key+'_rmse_noninferiority'] = score_subset(frame, masks[key], name)['rmse']-score_subset(frame, masks[key], models[1])['rmse'] <= 1
    decision = json.loads((output/'screen_decision.json').read_text())
    assert gates == decision['gates'] and decision['proceed_to_spatial'] == all(gates.values())
    assert not decision['proceed_to_spatial'] and not decision['publication_model_promoted']
    result = dict(verified=True, experiment=name, all_registered_hashes_match=True,
        predictor_and_unique_identity_alignment=True, evaluation_rows=checked_rows,
        first_field_rows=494, later_field_rows_unchanged=1523, earlier_year_training_only=True,
        replayed_members=checked_members, cluster_draws_and_estimator_settings_match=True,
        blind_all_assessment_baselines_verified=blind, cumulative_scalar_checks=scalar_checks,
        same_and_future_weather_perturbation_checks=perturbations, query_label_rejection=True,
        severe_first_rows=13, missing_prior_same_source_code_severe_first_examples=7,
        absent_prior_source_code_categories=2, severe_training_support=severe.where(pd.notna(severe), None).to_dict('records'),
        metric_rows_recomputed=4, independent_rmse_and_integral_arithmetic=True,
        subgroup_rows_recomputed=18, paired_interval_recomputed=True, screen_guards_recomputed=8,
        proceed_to_spatial=False, independent_external_validation=False, publication_model_promoted=False,
        verifier_sha256=sha(__file__))
    # JSON nulls preserve the distinction between unavailable support and zero severity.
    result['severe_training_support'] = json.loads(severe.to_json(orient='records'))
    write_verification_receipt(output, result, Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = verify(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'severe_training_support'}, indent=2))
