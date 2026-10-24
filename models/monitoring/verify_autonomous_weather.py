"""Replay the autonomous experiment and independently check saved fit contracts.

No model fitting, source mutation, imagery requests or output replacement.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .autonomous_weather import build_autonomous_features, predict_autonomous, OUTPUTS, WEATHER
from .autonomous_weather_data import AUTONOMOUS, PHENO, SOWING, EXTENSION
from .sowing_distribution_data import _calendar_date
from .innovation_common import ROOT, DATA, load_cohort, goal_decision
from .evaluation import score_predictions, score_subset, paired_interval
from .inputs import sha


def verify(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/autonomous_weather_20261022').resolve()
    result_path = out/'independent_verification.json'
    if result_path.exists():
        raise FileExistsError('Preserve existing verification receipt')
    registration = json.loads((out/'registration.json').read_text())
    receipt = json.loads((out/'verification.json').read_text())
    assert receipt['passed'] and receipt['fitted_splits'] == 20
    for path, digest in registration['source_sha256'].items():
        assert sha(ROOT/path) == digest, path
    for path, digest in registration['code_sha256'].items():
        assert sha(ROOT/path) == sha(out/'code_snapshot'/path) == digest, path
    for path, digest in receipt['artifacts_sha256'].items():
        assert sha(out/path) == digest, path
    data_registration = json.loads((AUTONOMOUS/'registration.json').read_text())
    for path, digest in data_registration['code_sha256'].items():
        assert sha(AUTONOMOUS/'code_snapshot'/path) == digest
    amendment = json.loads((AUTONOMOUS/'data_dependency_amendment.json').read_text())
    dependency = amendment['added_dependency']
    assert sha(ROOT/dependency['path']) == sha(ROOT/dependency['snapshot']) == dependency['sha256']
    builder = ROOT/'analysis/disease_improvement/autonomous_weather_data.py'
    original = AUTONOMOUS/'code_snapshot'/builder.relative_to(ROOT)
    assert builder.read_text().replace("        Path(__file__).with_name('phenology_phase_growth.py'),\n", '') == original.read_text()
    obs, _, _, parents, splits, position = load_cohort()
    assert json.loads((out/'split_membership.json').read_text()) == splits
    arrays = dict(np.load(AUTONOMOUS/'predictors.npz'))
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    fields = pd.read_csv(DATA/'fields.csv')
    sowing = pd.read_csv(SOWING/'field_sowing_prior.csv')
    fields = fields.merge(pd.DataFrame(dict(field_id=sowing.field_id, estimated_sowing_date=[
        _calendar_date(day, year) for day, year in zip(sowing.conditional_empirical_mean_day, sowing.season_year)])),
        on='field_id', validate='one_to_one')
    weather = pd.read_parquet(DATA/'daily_weather.parquet', columns=['location_id', 'date', *WEATHER])
    extension = pd.read_parquet(EXTENSION, columns=['location_id', 'date', *WEATHER])
    weather.date, extension.date = pd.to_datetime(weather.date), pd.to_datetime(extension.date)
    known = pd.MultiIndex.from_frame(weather[['location_id', 'date']])
    extension = extension.loc[~pd.MultiIndex.from_frame(extension[['location_id', 'date']]).isin(known)]
    rebuilt = build_autonomous_features(pd.concat([weather, extension], ignore_index=True), fields,
        obs.drop(columns='observed_percent'), joblib.load(PHENO/'development_prior.joblib'))
    for key in ['baseline', 'augmented', 'phase']:
        np.testing.assert_array_equal(rebuilt[key], arrays[key])
    pd.testing.assert_frame_equal(rebuilt['provenance'], pd.read_csv(AUTONOMOUS/'query_provenance.csv'), check_dtype=False)
    saved = {label: pd.read_csv(out/name, float_precision='round_trip').set_index('global_target_index')
        for label, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    max_error = 0.
    heads, models = 0, 0
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        ti, vi = [position.loc[split[key]].to_numpy() for key in ['train', 'test']]
        train = obs.iloc[ti]
        fit = joblib.load(out/f'{split["fold"]}_model.joblib')
        assert fit['train_ids'] == fit['training_ids'] == split['train']
        assert fit['test_ids'] == split['test'] and fit['history_conditioned'] is False
        assert fit['satellite_used'] is False and fit['observed_bbch_used'] is False
        assert fit['runtime_weather'] == WEATHER and fit['baseline_features'] == 34 and fit['augmented_features'] == 129
        weights = hierarchical_weights(train)
        np.testing.assert_array_equal(fit['growth']['training_weight'], weights)
        np.testing.assert_array_equal(fit['growth']['training_anchor'], np.zeros(len(train)))
        np.testing.assert_array_equal(fit['growth']['training_phase'], arrays['phase'][ti])
        rng = np.random.default_rng(20261022)
        clusters = np.unique(train.coordinate_year.astype(str))
        assert len(fit['growth']['members']) == len(fit['baseline']) == 20
        for member, reference in zip(fit['growth']['members'], fit['baseline']):
            keep = rng.choice(clusters, max(1, int(np.ceil(.8*len(clusters)))), replace=False)
            assert member['clusters'] == keep.tolist()
            local = train.coordinate_year.astype(str).isin(keep).to_numpy()
            assert member['training_ids'] == train.loc[local].global_target_index.astype(int).tolist()
            w = weights[local]/weights[local].mean()
            y = train.observed_percent.to_numpy(float)[local]
            gate = np.maximum(arrays['phase'][ti[local]], .01); gate /= gate.sum(axis=1, keepdims=True)
            for head, mass, size in [(reference, w, 34), (member['shared'], w, 129),
                    *[(h, w*gate[:, k], 129) for k, h in enumerate(member['stage'])]]:
                assert head.n_features_in_ == size and head.max_iter == 200 and head.max_depth == 3
                assert head.min_samples_leaf == 10 and head.l2_regularization == 3.
                assert head.learning_rate == .05 and head.max_leaf_nodes == 15 and not head.early_stopping
                expected = float(np.sum(mass*y)/np.sum(mass))
                error = abs(float(head._baseline_prediction.ravel()[0])-expected)
                max_error = max(max_error, error); assert error < 1e-10
                heads += 1
        predictions = predict_autonomous(fit, arrays['baseline'][vi], arrays['augmented'][vi], arrays['phase'][vi])
        for name in OUTPUTS:
            np.testing.assert_array_equal(predictions[name], saved[label].loc[split['test'], name].to_numpy())
        models += 1
        print('verified', split['fold'], flush=True)
    metrics = pd.read_csv(out/'metrics.csv'); groups = pd.read_csv(out/'subgroup_metrics.csv')
    for label, frame in saved.items():
        frame = frame.reset_index()
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        parent = parents[label].loc[frame.global_target_index].reset_index()
        pd.testing.assert_frame_equal(frame[parent.columns], parent, check_dtype=False, rtol=1e-12, atol=1e-12)
        for name in OUTPUTS:
            scored = score_predictions(frame, name)
            row = metrics.loc[metrics.evaluation.eq(label)&metrics.model.eq(name)].iloc[0]
            severe = score_subset(frame, frame.observed_percent.gt(75), name)
            severe_row = groups.loc[groups.evaluation.eq(label)&groups.model.eq(name)&groups.population.eq('severe_above75')].iloc[0]
            for key in ['rmse', 'mae', 'bias', 'r2', 'integral_rmse']:
                np.testing.assert_allclose(scored[key], row[key], rtol=1e-12, atol=1e-10)
            for key in ['rmse', 'mae', 'bias']:
                np.testing.assert_allclose(severe[key], severe_row[key], rtol=1e-12, atol=1e-10)
    monitoring = json.loads((out/'all_goal_decisions.json').read_text())
    comparisons = json.loads((out/'autonomous_comparisons.json').read_text())
    severe = groups.loc[groups.population.eq('severe_above75')]
    for name in OUTPUTS:
        assert monitoring[name] == goal_decision(metrics, severe, name)
    for name in OUTPUTS[1:]:
        decision = goal_decision(metrics, severe, name, reference=OUTPUTS[0])
        assert comparisons[name]['achieved'] == decision['achieved']
        assert comparisons[name]['gates'] == decision['gates']
        for label, frame in saved.items():
            np.testing.assert_array_equal(paired_interval(frame.reset_index(), name, OUTPUTS[0]),
                comparisons[name]['paired_rmse_difference_intervals'][label])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True, models=models, heads=heads,
        all2304features_rebuilt_exact=True, original20memberships_and_scientific_weights=True,
        saved_predictions_replayed_exact=True, fitted_initial_mean_max_error=max_error,
        saved_metrics_and_both_separate_goal_sets_recomputed=True, runtime_satellite_or_observations=False,
        verifier_sha256=sha(Path(__file__)), not_independent_external_validation=True)
    result_path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    with threadpool_limits(limits=1):
        verify(parser.parse_args().output)
