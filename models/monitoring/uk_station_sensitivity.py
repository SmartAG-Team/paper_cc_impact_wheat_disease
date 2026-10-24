"""Frozen-model response to provisional Wellesbourne rainfall; no refitting."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import score_predictions, score_subset
from .features import build_features
from .field_history_models import predict_ensemble
from .infection_event_features import build_event_features, insert_events
from .inputs import sha
from .repairs import hybrid_weather, recorded_leaf_features
from .robust_models import predict_cluster_ensemble
from .stage_experiment import augment
from .stage_features import build_stage_features

ROOT = Path(__file__).resolve().parents[2]
SITE = 'era5loc_10922b9e995a'
SCENARIOS = ['provisional_history_only', 'provisional_process_only', 'provisional_both']


def rebuild(weather, fields, queries, calendar, provenance):
    if 'observed_percent' in queries:
        raise ValueError('Feature queries must withhold labels')
    same = provenance.set_index('global_target_index').loc[queries.global_target_index, 'latest_same_date']
    prior = dict(zip(queries.global_target_index, same))
    events, names, ledger = build_features(weather, fields, queries,
        dict(weather_cutoff_days=1, rain_variable='rain_mm', prior_dates=prior))
    weather_columns = [i for i, n in enumerate(names) if n.startswith('weather_')]
    process_columns = [i for i, n in enumerate(names) if n.startswith('process_')]
    assert len(weather_columns) == 12 and len(process_columns) == 3
    category, category_names = recorded_leaf_features(queries)
    process = np.column_stack([events[:, weather_columns], category, events[:, process_columns]])
    stages, stage_names, stage_ledger = build_stage_features(weather, fields, queries, calendar)
    station = augment(process, stages)
    extra, extra_names, event_ledger = build_event_features(weather, fields, queries, provenance)
    event_station = insert_events(station, extra)
    return dict(process=process, station=station, history_events=event_station), (ledger, stage_ledger, event_ledger)


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    parent = root/'data/final_monitoring/cohort'
    process_parent = root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    data = root/'data/disease_improvement/station_preferred_ie_20261022'
    qa = root/'outputs/weather_validation_20261022/uk_hydrology_rainfall_qualification'
    output = Path(output or root/'outputs/disease_improvement/uk_station_sensitivity_20261022').resolve()
    if output.exists():
        raise FileExistsError('A fresh diagnostic destination is required')
    obs = pd.read_csv(parent/'training_observation_ledger.csv')
    arrays = np.load(parent/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    assert len(obs) == 2304 and obs.global_target_index.is_unique
    weather = pd.read_parquet(data/'daily_weather.parquet')
    fields = pd.read_csv(data/'fields.csv')
    provenance = pd.read_json(data/'history_provenance.json')
    calendar_file = root/'outputs/raw_metadata_recovery_20261022/stage_calendar.csv'
    calendar = pd.read_csv(calendar_file)
    daily = pd.read_csv(qa/'project_daily_weather_pairs.csv')
    daily = daily.loc[daily.accepted_day].copy()
    assert len(daily) == 956 and daily.site_id.eq(SITE).all()
    assert not daily.duplicated(['site_id', 'date']).any()
    assert daily.good_complete_intervals.eq(96).all()
    assert np.isfinite(daily.station_rainfall_mm).all() and daily.station_rainfall_mm.ge(0).all()
    affected = obs.loc[obs.site_id.eq(SITE)].drop(columns='observed_percent').copy()
    positions = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    loc = positions.loc[affected.global_target_index].to_numpy()
    splits = json.loads((parent/'split_membership.json').read_text())
    relevant = [s for s in splits if set(s['test']) & set(affected.global_target_index)]
    source_files = [data/n for n in ['manifest.json', 'daily_weather.parquet', 'fields.csv', 'history_provenance.json']]
    source_files += [parent/n for n in ['predictors.npz', 'training_observation_ledger.csv', 'split_membership.json',
        'assessment_predictions.csv', 'spatial_predictions.csv', 'fitted_candidate_lock.json', 'fitted_candidate.joblib']]
    source_files += [qa/n for n in ['manifest.json', 'project_daily_weather_pairs.csv', 'project_diagnostic_summary.json']]
    source_files += [calendar_file]
    for split in relevant:
        source_files += [parent/(split['fold']+'_'+head+'.joblib') for head in ['first', 'event']]
        source_files += [process_parent/(split['fold']+'_process.joblib')]
    code = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in ['evaluation', 'features',
        'field_history_models', 'models', 'infection_event_features', 'repairs', 'robust_models',
        'stage_features', 'stage_experiment', 'inputs']]
    code += [root/'model/seasonal_septoria'/n for n in ['core.py', 'wetness.py']]
    code += [root/'analysis/final_study/metrics.py']
    output.mkdir(parents=True)
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        question='Can the independently documented local gauge discrepancy materially change existing large disease errors under frozen fitted models?',
        status='PROVISIONAL_FORCING_SENSITIVITY_ONLY', station_replacement_qualified=False,
        station_selection='Previously identified closest public gauge at this location; fixed956GoodCompleteUTCdays; all retained regardless of disease response',
        scenarios=SCENARIOS, branch_decomposition='Apply the same fixed provisional forcing to learned component, process component, and both; no selection or adoption among scenarios',
        forcing='Existing hybrid_weather rainfall-only phase-assisted rule; missing and unknown phase fallback; zero measured total sets rain duration0, wet-day duration retains its existing value',
        phase_warning='Liquid phase and rain duration are not measured by the supplied daily total',
        geography_warning='Historical source corroborates station1165 location/elevation; continuous2014–2017 position validity remains unproven',
        timing_warning='End-labelled15minute convention empirically reconciles provider daily totals; explicit endpoint documentation unresolved',
        fitting_performed=False, hyperparameter_search=False, weather_cutoff_days=1,
        frozen_training_weather_preserved=True, inference_weather_changed=True,
        interpretation='Input sensitivity under fixed parameters, not a causal error decomposition or consistently retrained replacement candidate',
        independent_external_validation=False, publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)): sha(p) for p in source_files},
        code_sha256={str(p.relative_to(root)): sha(p) for p in code})
    (output/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for path in code:
        dest = output/'code_snapshot'/path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    old, _ = rebuild(weather, fields, affected, calendar, provenance)
    for key in old:
        np.testing.assert_array_equal(old[key], arrays[key][loc])
    (output/'baseline_reproduction.json').write_text(json.dumps(dict(verified=True,
        query_rows=len(affected), process37_exact=True, station53_exact=True, history_events68_exact=True,
        completed_before_replacement=True), indent=2)+'\n')
    pairs = daily[['site_id', 'date', 'station_rainfall_mm']].rename(columns={'station_rainfall_mm': 'precipitation_mm_station'})
    forcing, counts = hybrid_weather(weather, pairs, {SITE}, rainfall=True, meteorology=False, source='EA_PROVISIONAL_DIAGNOSTIC')
    assert int(forcing.EA_PROVISIONAL_DIAGNOSTIC_total_precip_used.sum()) == 956
    for column in ['tmean_c', 'tmin_c', 'tmax_c', 'rh_mean_pct', 'rh_hours_ge_90pct']:
        np.testing.assert_array_equal(forcing[column], weather[column])
    new, ledgers = rebuild(forcing, fields, affected, calendar, provenance)
    np.savez_compressed(output/'predictors.npz', target_ids=affected.global_target_index.to_numpy(),
        history=arrays['history'][loc], **new)
    counts.to_csv(output/'provisional_replacement_counts.csv', index=False)
    columns = ['location_id', 'date', 'precipitation_mm', 'rain_mm', 'rain_hours_gt_0_1mm',
        'origin_precipitation_mm', 'origin_rain_mm', 'origin_rain_hours_gt_0_1mm',
        'EA_PROVISIONAL_DIAGNOSTIC_total_precip_used', 'EA_PROVISIONAL_DIAGNOSTIC_liquid_phase_assisted']
    forcing.loc[forcing.location_id.eq(SITE), columns].to_csv(output/'provisional_daily_precipitation.csv', index=False)
    for label, ledger in zip(['feature', 'stage', 'event'], ledgers):
        ledger.to_csv(output/(label+'_provenance.csv'), index=False)
    bundle = joblib.load(parent/'fitted_candidate.joblib')
    history = pd.DataFrame(arrays['history'], columns=bundle['base']['history_names'])
    selected_position = pd.Series(np.arange(len(affected)), index=affected.global_target_index)
    frames = {label: pd.read_csv(parent/filename) for label, filename in
        [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    for frame in frames.values():
        for name in SCENARIOS:
            frame[name] = frame.irish_station_candidate.copy()
    impacts = []
    for split in relevant:
        ids = [i for i in split['test'] if i in selected_position.index]
        idx, subset = positions.loc[ids].to_numpy(), selected_position.loc[ids].to_numpy()
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        frame = frames[label]
        rowpos = pd.Series(np.arange(len(frame)), index=frame.global_target_index).loc[ids].to_numpy()
        query = obs.iloc[idx].drop(columns='observed_percent')
        h = history.iloc[idx]
        assert obs.set_index('global_target_index').loc[split['train'], 'season_year'].max() < query.season_year.min()
        heads = {head: joblib.load(parent/(split['fold']+'_'+head+'.joblib')) for head in ['first', 'event']}
        process = joblib.load(process_parent/(split['fold']+'_process.joblib'))
        for fitted in [*heads.values(), process]:
            assert fitted['training_ids'] == split['train']
        baseline_process = predict_cluster_ensemble(process, arrays['process'][idx], h, query)
        before = [predict_ensemble(heads[head], arrays[key][idx], h, query)
                  for head, key in [('first', 'station'), ('event', 'history_events')]]
        after = [predict_ensemble(heads[head], new[key][subset], h, query)
                 for head, key in [('first', 'station'), ('event', 'history_events')]]
        mask = (h.prior_assessment_available.eq(1) | h.other_leaf_available.eq(1)).to_numpy()
        old_history, new_history = np.where(mask, before[1], before[0]), np.where(mask, after[1], after[0])
        new_process = predict_cluster_ensemble(process, new['process'][subset], h, query)
        np.testing.assert_allclose(baseline_process, frame.candidate_process.iloc[rowpos], atol=1e-10)
        np.testing.assert_allclose(old_history, frame.ie_selected_history.iloc[rowpos], atol=1e-10)
        values = [.5*(baseline_process+new_history), .5*(new_process+old_history), .5*(new_process+new_history)]
        for name, value in zip(SCENARIOS, values):
            frame.loc[rowpos, name] = value
        part = query[['global_target_index', 'field_id', 'date', 'season_year', 'source', 'leaf_index']].copy()
        part['evaluation'], part['fold'] = label, split['fold']
        part['has_earlier_field_history'] = mask
        part['old_process'], part['new_process'] = baseline_process, new_process
        part['old_history'], part['new_history'] = old_history, new_history
        impacts.append(part)
        print('replayed', split['fold'], len(ids), flush=True)
    impact = pd.concat(impacts, ignore_index=True)
    impact.to_csv(output/'component_input_sensitivity.csv', index=False)
    metrics, groups = [], []
    for label, frame in frames.items():
        for name in ['irish_station_candidate']+SCENARIOS:
            metrics.append(dict(evaluation=label, model=name, **score_predictions(frame, name)))
        frame.to_csv(output/('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
        first = ~frame.ie_uses_event_head
        masks = dict(affected_site=frame.site_id.eq(SITE), affected_first=frame.site_id.eq(SITE)&first,
            affected_severe_first=frame.site_id.eq(SITE)&first&frame.observed_percent.gt(75),
            affected_other_first=frame.site_id.eq(SITE)&first&frame.observed_percent.le(75))
        for pop, mask in masks.items():
            for name in ['irish_station_candidate']+SCENARIOS:
                groups.append(dict(evaluation=label, population=pop, model=name, **score_subset(frame, mask, name)))
    pd.DataFrame(metrics).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(output/'subgroup_metrics.csv', index=False)
    temporal = frames['temporal']
    severe = temporal.loc[temporal.site_id.eq(SITE)&~temporal.ie_uses_event_head&temporal.observed_percent.gt(75)]
    assert len(severe) == 4
    severe.to_csv(output/'four_severe_first_visit_cases.csv', index=False)
    for path, digest in registration['source_sha256'].items():
        assert sha(root/path) == digest
    return pd.DataFrame(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output).to_string(index=False))
