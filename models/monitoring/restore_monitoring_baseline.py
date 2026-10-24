"""Restore the user-selected monitoring result without refitting or deletion."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .discrepancy_experiment import forecast_baseline
from .discrepancy_filter import predict_discrepancy
from .evaluation import score_predictions, score_subset, paired_interval
from .innovation_common import ROOT, load_cohort
from .inputs import sha

SOURCE = ROOT/'data/final_monitoring/folds'
OUT = ROOT/'data/final_monitoring/evaluation'
MODEL = 'discrepancy_logit'
ACTIVE = ROOT/'data/final_monitoring/selection.json'


def restore():
    if OUT.exists():
        raise FileExistsError('Preserve an existing restoration receipt')
    receipt = json.loads((SOURCE/'verification.json').read_text()); assert receipt['passed']
    registration = json.loads((SOURCE/'registration.json').read_text())
    for name, digest in receipt['artifacts_sha256'].items():
        assert sha(SOURCE/name) == digest, name
    for key in ['source_sha256', 'code_sha256']:
        for name, digest in registration[key].items(): assert sha(ROOT/name) == digest, name
    obs, arrays, history, original, splits, position = load_cohort()
    frames = {label: pd.read_csv(SOURCE/name, float_precision='round_trip')
        for label, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    maximum_error = 0.
    for split in splits:
        fitted = joblib.load(SOURCE/f'{split["fold"]}_model.joblib')
        assert fitted['train_ids'] == split['train'] and fitted['test_ids'] == split['test']
        qi = position.loc[split['test']].to_numpy()
        ci = np.flatnonzero(obs.field_id.isin(obs.iloc[qi].field_id.unique()))
        within = pd.Series(np.arange(len(ci)), index=ci).loc[qi].to_numpy()
        baseline = forecast_baseline(fitted['base'], arrays, history, obs, ci)
        predicted = predict_discrepancy(fitted['corrections']['logit']['parameters'], obs.iloc[ci], baseline, 'logit')[within]
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        recorded = frames[label].set_index('global_target_index').loc[split['test'], MODEL].to_numpy()
        error = float(np.max(np.abs(predicted-recorded))); maximum_error = max(maximum_error, error)
        np.testing.assert_allclose(predicted, recorded, rtol=1e-14, atol=1e-12)
        print('replayed', split['fold'], flush=True)
    OUT.mkdir(parents=True)
    metrics, groups, intervals = [], [], []
    diagnostic_names = [MODEL, 'irish_station_candidate', 'reference_replay']
    recorded_metrics = pd.read_csv(SOURCE/'metrics.csv', float_precision='round_trip')
    for label, frame in frames.items():
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        assert set(frame.global_target_index) == set(original[label].index)
        frame['prediction_percent'] = frame[MODEL]
        frame['selected_model'] = MODEL
        frame.to_csv(OUT/('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
        for name in diagnostic_names:
            values = score_predictions(frame, name)
            if name == MODEL:
                expected = recorded_metrics.loc[recorded_metrics.evaluation.eq(label)&recorded_metrics.model.eq(name)].iloc[0]
                for key, value in values.items(): np.testing.assert_allclose(expected[key], value, atol=1e-9, rtol=1e-12)
            metrics.append(dict(evaluation=label, model=name, **values))
        masks = dict(severe_above75=frame.observed_percent.gt(75), first_field=~frame.ie_uses_event_head.astype(bool),
            earlier_field=frame.ie_uses_event_head.astype(bool), BASF=frame.source.eq('BASF'), Corteva=frame.source.eq('Corteva'))
        masks.update({f'year_{year}': frame.season_year.eq(year) for year in sorted(frame.season_year.unique())})
        for population, mask in masks.items():
            groups.append(dict(evaluation=label, model=MODEL, population=population, **score_subset(frame, mask, MODEL)))
        for reference in diagnostic_names[1:]:
            low, high = paired_interval(frame, MODEL, reference)
            intervals.append(dict(evaluation=label, model=MODEL, reference=reference, lower95=low, upper95=high,
                scope='Descriptive site-resampling of fixed predictions after adaptive development; not external confirmation'))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(OUT/'subgroup_metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv', index=False)
    decision = json.loads((SOURCE/'all_goal_decisions.json').read_text())[MODEL]
    source_paths = [SOURCE/name for name in receipt['artifacts_sha256']] + [SOURCE/'verification.json', Path(__file__)]
    source_paths += [ROOT/'publication/gcb_20261023/Manuscript.docx', ROOT/'data/final_climate/scientific_manifest.json', ROOT/'data/final_models/manifest.json']
    selected = dict(schema_version=1, selected_utc=datetime.now(timezone.utc).isoformat(),
        selected_model=MODEL, information_setting='rolling_same_field_monitoring',
        source_experiment=str(SOURCE.relative_to(ROOT)), evaluation_bundle=str(OUT.relative_to(ROOT)),
        fold_model_pattern=str((SOURCE/'{fold}_model.joblib').relative_to(ROOT)), correction_key='logit',
        prediction_column='prediction_percent', source_prediction_column=MODEL,
        selection_basis='User requested rollback to the previously discussed 17.15/17.31 monitoring results',
        previous_autonomous_experiments='Preserved as separate research diagnostics; not the active monitoring choice',
        additional_model_search_requested=False, satellite_inputs_used=False,
        same_field_earlier_observations_used=True, current_season_other_fields_used=False,
        autonomous_climate_validation=False, independent_external_validation=False,
        original_research_goal=decision, publication_model_promoted=False,
        climate_runtime_replaced=False, full_data_final_fit_available=False,
        scope='Verified outer-fold monitoring model/results restored as working baseline; no all-data deployment model is asserted',
        metrics=[r for r in metrics if r['model'] == MODEL],
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_paths})
    if ACTIVE.exists(): shutil.copyfile(ACTIVE, OUT/'previous_active_monitoring_model.json')
    (OUT/'selection.json').write_text(json.dumps(selected, indent=2)+'\n')
    ACTIVE.write_text(json.dumps(selected, indent=2)+'\n')
    verification = dict(passed=True, saved_fold_replays=20, maximum_prediction_difference=maximum_error,
        original_memberships_support_preserved=True, no_new_model_fitting=True,
        previous_experiments_preserved=True, selected_by_user=True,
        source_unchanged=all(sha(ROOT/p) == digest for p, digest in selected['source_sha256'].items()),
        active_selection_sha256=sha(ACTIVE), artifacts_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()})
    assert verification['source_unchanged']
    (OUT/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(pd.DataFrame(selected['metrics']).to_string(index=False), flush=True)
    return OUT


if __name__ == '__main__':
    with threadpool_limits(limits=1): restore()
