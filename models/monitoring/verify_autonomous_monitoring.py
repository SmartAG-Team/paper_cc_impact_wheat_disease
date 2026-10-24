"""Read-only replay of honest OOB calibration and same-field posterior forecasts."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .autonomous_monitoring import coordinate_oob, monitoring_forecast
from .autonomous_monitoring_experiment import PRIOR, PREFIX, NAMES, PRIMARY
from .autonomous_weather import predict_autonomous, OUTPUTS
from .autonomous_weather_data import AUTONOMOUS
from .innovation_common import ROOT, load_cohort, goal_decision
from .evaluation import score_predictions, score_subset
from .inputs import sha


def verify():
    out = ROOT/'outputs/disease_improvement/autonomous_monitoring_20261022'
    destination = out/'independent_verification.json'
    if destination.exists():
        raise FileExistsError('Preserve existing verification receipt')
    registration = json.loads((out/'registration.json').read_text())
    receipt = json.loads((out/'verification.json').read_text())
    assert receipt['passed']
    for path, digest in registration['source_sha256'].items():
        assert sha(ROOT/path) == digest, path
    for path, digest in registration['code_sha256'].items():
        assert sha(ROOT/path) == sha(out/'code_snapshot'/path) == digest, path
    for name, digest in receipt['artifacts_sha256'].items():
        assert sha(out/name) == digest, name
    obs, _, _, parents, splits, position = load_cohort()
    assert json.loads((out/'split_membership.json').read_text()) == splits
    arrays = dict(np.load(AUTONOMOUS/'predictors.npz'))
    metrics = pd.read_csv(out/'metrics.csv'); groups = pd.read_csv(out/'subgroup_metrics.csv')
    all_frames, checked, objective_error, first_error = {}, 0, 0., 0.
    missing = 0
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        if not (out/filename).exists():
            continue
        frame = pd.read_csv(out/filename, float_precision='round_trip').set_index('global_target_index')
        expected = parents[label].loc[frame.index]
        pd.testing.assert_frame_equal(frame[expected.columns], expected, check_dtype=False, rtol=1e-12, atol=1e-12)
        assert len(frame) == 2017 and frame.index.is_unique
        all_frames[label] = frame
    for split in splits:
        path = out/f'{split["fold"]}_model.joblib'
        if not path.exists():
            continue
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        fit = joblib.load(path); prior = joblib.load(fit['prior_path'])
        assert fit['train_ids'] == prior['train_ids'] == split['train']
        assert fit['test_ids'] == prior['test_ids'] == split['test']
        assert sha(Path(fit['prior_path'])) == fit['prior_sha256']
        ti, vi = [position.loc[split[key]].to_numpy() for key in ['train', 'test']]
        train = obs.iloc[ti]
        oob = coordinate_oob(prior, arrays['baseline'][ti], arrays['augmented'][ti], arrays['phase'][ti], train)
        covered = oob['count'] > 0
        np.testing.assert_array_equal(oob['count'], fit['oob']['count'])
        missing += int((~covered).sum())
        for name in OUTPUTS:
            np.testing.assert_array_equal(oob['predictions'][name], fit['oob']['predictions'][name])
            assert np.isnan(oob['predictions'][name][~covered]).all()
        weights = hierarchical_weights(train)
        np.testing.assert_array_equal(weights, fit['training_weights'])
        assert fit['calibration_ids'] == train.loc[covered].global_target_index.astype(int).tolist()
        for source in OUTPUTS:
            baseline = oob['predictions'][source][covered]
            for link in ['linear', 'logit']:
                name = f'weather_prior_{PREFIX[source]}_{link}'
                params = fit['filters'][name]['parameters']
                predictions = monitoring_forecast(params, train.loc[covered], baseline, link)
                error = predictions/100.-train.loc[covered].observed_percent.to_numpy()/100.
                mass = weights[covered]/weights[covered].sum()
                objective = np.sum(mass*error**2)+1e-5*np.sum((params-np.log([1., 1., 30., 30.]))**2)
                objective_error = max(objective_error, abs(objective-fit['filters'][name]['objective']))
        context = np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        qpos = pd.Series(np.arange(len(context)), index=context).loc[vi].to_numpy()
        context_frame = obs.iloc[context]
        base = predict_autonomous(prior, arrays['baseline'][context], arrays['augmented'][context], arrays['phase'][context])
        dates = pd.to_datetime(context_frame.date)
        first = dates.eq(dates.groupby(context_frame.field_id).transform('min')).to_numpy()
        for source in OUTPUTS:
            np.testing.assert_array_equal(base[source][qpos], all_frames[label].loc[split['test'], source])
            for link in ['linear', 'logit']:
                name = f'weather_prior_{PREFIX[source]}_{link}'
                predicted = monitoring_forecast(fit['filters'][name]['parameters'], context_frame, base[source], link)
                first_error = max(first_error, np.max(np.abs(predicted[first]-base[source][first])))
                np.testing.assert_array_equal(predicted[qpos], all_frames[label].loc[split['test'], name])
        checked += 1
        print('verified', split['fold'], flush=True)
    assert checked == receipt['fitted_splits'] and objective_error < 1e-12 and first_error < 1e-10
    for label, frame in all_frames.items():
        frame = frame.reset_index()
        for name in [*NAMES, *OUTPUTS]:
            score = score_predictions(frame, name)
            severe = score_subset(frame, frame.observed_percent.gt(75), name)
            saved = metrics.loc[metrics.evaluation.eq(label)&metrics.model.eq(name)].iloc[0]
            saved_severe = groups.loc[groups.evaluation.eq(label)&groups.model.eq(name)&groups.population.eq('severe_above75')].iloc[0]
            for key in ['rmse', 'mae', 'bias', 'r2', 'integral_rmse']:
                np.testing.assert_allclose(score[key], saved[key], rtol=1e-12, atol=1e-10)
            for key in ['rmse', 'mae', 'bias']:
                np.testing.assert_allclose(severe[key], saved_severe[key], rtol=1e-12, atol=1e-10)
    decisions = json.loads((out/'all_goal_decisions.json').read_text())
    severe = groups.loc[groups.population.eq('severe_above75')]
    for name in NAMES:
        assert decisions[name] == goal_decision(metrics, severe, name)
    if checked == 5:
        assert set(all_frames) == {'temporal'}
        assert not json.loads((out/'screen_decision.json').read_text())['continue_spatial']
        keys = ['temporal_ten_percent_rmse', 'temporal_severe_rmse_improves', 'temporal_severe_absolute_bias_improves']
        assert not any(all(d['gates'][k] for k in keys) for d in decisions.values())
    else:
        assert checked == 20 and set(all_frames) == {'temporal', 'spatial_50km'}
    result = dict(passed=True, fitted_splits=checked, oob_and_query_predictions_replayed_exact=True,
        countzero_calibration_records=missing, alloriginalscientificweights_and_memberships=True,
        covariance_training_objective_max_error=objective_error, first_visit_prior_max_error=first_error,
        sources_codes_snapshots_and_artifact_hashes_verified=True, all_metrics_and_goal_gates_recomputed=True,
        model_refitting=False, not_independent_external_validation=True, verifier_sha256=sha(Path(__file__)))
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify()
