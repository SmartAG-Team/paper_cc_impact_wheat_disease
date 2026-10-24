"""Verify provisional forcing sensitivity without modifying source packages."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import distances_km, score_predictions, score_subset
from .field_history_models import predict_ensemble
from .inputs import sha
from .repairs import hybrid_weather
from .robust_models import predict_cluster_ensemble
from .uk_station_sensitivity import ROOT, SITE, SCENARIOS, rebuild
from .verify_first_visit_screens import manual_integral, manual_weights
from .verify_french_weather import write_verification_receipt


def verify(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    parent = root/'data/final_monitoring/cohort'
    process_parent = root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    data = root/'data/disease_improvement/station_preferred_ie_20261022'
    qa = root/'outputs/weather_validation_20261022/uk_hydrology_rainfall_qualification'
    registration = json.loads((output/'registration.json').read_text())
    for path, digest in registration['source_sha256'].items():
        assert sha(root/path) == digest, path
    for path, digest in registration['code_sha256'].items():
        assert sha(root/path) == digest == sha(output/'code_snapshot'/path), path
    assert registration['status'] == 'PROVISIONAL_FORCING_SENSITIVITY_ONLY'
    for key in ['station_replacement_qualified', 'fitting_performed', 'hyperparameter_search',
                'independent_external_validation', 'publication_model_promoted']:
        assert registration[key] is False
    obs = pd.read_csv(parent/'training_observation_ledger.csv')
    arrays, changed = np.load(parent/'predictors.npz'), np.load(output/'predictors.npz')
    assert obs.global_target_index.is_unique and len(obs) == 2304
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    query = obs.loc[obs.site_id.eq(SITE)].drop(columns='observed_percent')
    assert len(query) == 344 and query.field_id.nunique() == 21
    np.testing.assert_array_equal(query.global_target_index, changed['target_ids'])
    loc = position.loc[query.global_target_index].to_numpy()
    np.testing.assert_array_equal(changed['history'], arrays['history'][loc])
    weather = pd.read_parquet(data/'daily_weather.parquet')
    fields = pd.read_csv(data/'fields.csv')
    history_provenance = pd.read_json(data/'history_provenance.json')
    calendar = pd.read_csv(root/'outputs/raw_metadata_recovery_20261022/stage_calendar.csv')
    original, _ = rebuild(weather, fields, query, calendar, history_provenance)
    for key, values in original.items():
        np.testing.assert_array_equal(values, arrays[key][loc])
    daily = pd.read_csv(qa/'project_daily_weather_pairs.csv')
    daily = daily.loc[daily.accepted_day]
    assert len(daily) == 956 and daily.site_id.eq(SITE).all()
    pairs = daily[['site_id', 'date', 'station_rainfall_mm']].rename(columns={'station_rainfall_mm': 'precipitation_mm_station'})
    forcing, counts = hybrid_weather(weather, pairs, {SITE}, rainfall=True, meteorology=False, source='EA_PROVISIONAL_DIAGNOSTIC')
    pd.testing.assert_frame_equal(counts, pd.read_csv(output/'provisional_replacement_counts.csv'), check_dtype=False)
    assert counts.replaced_days.tolist() == [956, 919, 424]
    # Independently reconstruct phase partition and dry-day duration behavior.
    source = weather.copy()
    source['date'] = pd.to_datetime(source.date)
    pair_index = pairs.copy()
    pair_index['date'] = pd.to_datetime(pair_index.date)
    aligned = source[['location_id', 'date']].merge(pair_index, left_on=['location_id', 'date'], right_on=['site_id', 'date'], how='left', validate='1:1')
    liquid, durations = weather.rain_mm.to_numpy().copy(), weather.rain_hours_gt_0_1mm.to_numpy().copy()
    total = weather.precipitation_mm.to_numpy().copy()
    phase_assisted = zero_duration = 0
    for i in np.flatnonzero(aligned.precipitation_mm_station.notna()):
        measured = aligned.precipitation_mm_station.iloc[i]
        ratio = np.clip(liquid[i]/total[i], 0, 1) if total[i] > 0 else np.nan
        if weather.tmin_c.iloc[i] > 2:
            ratio = 1.
        if measured == 0:
            ratio = 0.
            durations[i] = 0.
            zero_duration += 1
        if np.isfinite(ratio):
            liquid[i] = measured*ratio
            phase_assisted += 1
        total[i] = measured
    assert phase_assisted == 919 and zero_duration == 424
    for key, values in [('rain_mm', liquid), ('precipitation_mm', total), ('rain_hours_gt_0_1mm', durations)]:
        np.testing.assert_array_equal(forcing[key], values)
    for key in ['tmean_c', 'tmin_c', 'tmax_c', 'rh_mean_pct', 'rh_hours_ge_90pct']:
        np.testing.assert_array_equal(weather[key], forcing[key])
    reconstructed, provenance = rebuild(forcing, fields, query, calendar, history_provenance)
    for key, values in reconstructed.items():
        np.testing.assert_array_equal(changed[key], values)
    for label, ledger in zip(['feature', 'stage', 'event'], provenance):
        pd.testing.assert_frame_equal(ledger, pd.read_csv(output/(label+'_provenance.csv')), check_dtype=False)
    np.testing.assert_array_equal(changed['station'][:, 12:50], arrays['station'][loc, 12:50])
    by_location = forcing.loc[forcing.location_id.eq(SITE)].copy()
    by_location['date'] = pd.to_datetime(by_location.date)
    weather_scalars = 0
    field_index = fields.set_index('field_id')
    for pos, row in enumerate(query.itertuples(index=False)):
        end = pd.Timestamp(row.date)-pd.Timedelta(days=1)
        start = pd.Timestamp(field_index.loc[row.field_id, 'sowing_date'])
        values = []
        for width in [7, 21, 42]:
            days = by_location.loc[by_location.date.between(max(start, end-pd.Timedelta(days=width-1)), end)]
            assert len(days) == min(width, (end-start).days+1)
            values.extend([days.tmean_c.mean(), days.rh_mean_pct.mean(), np.log1p(days.rain_mm.mean())])
        # Builder orders by width and temperature/humidity/rain, as verified independently here.
        np.testing.assert_allclose(changed['station'][pos, :9], values, atol=1e-12, rtol=0)
        weather_scalars += 9
    # A same-date or later weather perturbation must not change an individual query.
    perturbations = 0
    for target_id in [1141, 1633, 1905, 1906]:
        q = query.loc[query.global_target_index.eq(target_id)]
        altered = forcing.copy()
        future = pd.to_datetime(altered.date).ge(pd.Timestamp(q.date.iloc[0]))
        altered.loc[future, 'rain_mm'] += 999
        probe, _ = rebuild(altered, fields, q, calendar, history_provenance)
        at = np.flatnonzero(changed['target_ids'] == target_id)
        for key in probe:
            np.testing.assert_array_equal(probe[key], changed[key][at])
        perturbations += 1
    bundle = joblib.load(parent/'fitted_candidate.joblib')
    history = pd.DataFrame(arrays['history'], columns=bundle['base']['history_names'])
    subposition = pd.Series(np.arange(len(query)), index=query.global_target_index)
    splits = json.loads((parent/'split_membership.json').read_text())
    frames = {}
    member_calls = folds = 0
    minimum_distance = np.inf
    minimum_full_fold_distance = np.inf
    all_impacts = []
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame, old = pd.read_csv(output/filename), pd.read_csv(parent/filename)
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        pd.testing.assert_frame_equal(frame[old.columns], old)
        other = frame.site_id.ne(SITE)
        for name in SCENARIOS:
            np.testing.assert_array_equal(frame.loc[other, name], old.loc[other, 'irish_station_candidate'])
        assert other.sum() == 1758
        seen = []
        for split in splits:
            if (split['fold'].startswith('temporal')) != (label == 'temporal'):
                continue
            ids = [i for i in split['test'] if i in subposition.index]
            if not ids:
                continue
            seen += ids
            train, idx = position.loc[split['train']].to_numpy(), position.loc[ids].to_numpy()
            assert obs.iloc[train].season_year.max() < obs.iloc[idx].season_year.min()
            assert set(split['train']).isdisjoint(split['test'])
            if label != 'temporal':
                distance = distances_km(obs.iloc[train][['latitude', 'longitude']], obs.iloc[idx][['latitude', 'longitude']]).min()
                minimum_distance = min(distance, minimum_distance)
                assert distance >= 50
                held = position.loc[split['test']].to_numpy()
                full_distance = distances_km(obs.iloc[train][['latitude', 'longitude']], obs.iloc[held][['latitude', 'longitude']]).min()
                minimum_full_fold_distance = min(full_distance, minimum_full_fold_distance)
                assert full_distance >= 50
            h, q = history.iloc[idx], obs.iloc[idx].drop(columns='observed_percent')
            at = subposition.loc[ids].to_numpy()
            part = frame.set_index('global_target_index').loc[ids]
            results = {}
            for head, key, folder in [('first', 'station', parent), ('event', 'history_events', parent), ('process', 'process', process_parent)]:
                fitted = joblib.load(folder/(split['fold']+'_'+head+'.joblib'))
                assert fitted['training_ids'] == split['train'] and len(fitted['members']) == 20
                predict = predict_cluster_ensemble if head == 'process' else predict_ensemble
                before = predict(fitted, arrays[key][idx], h, q)
                after = predict(fitted, changed[key][at], h, q)
                results[head] = (before, after)
                member_calls += 40
            mask = (h.prior_assessment_available.eq(1) | h.other_leaf_available.eq(1)).to_numpy()
            p, pp = results['process']
            sh = np.where(mask, results['event'][0], results['first'][0])
            nh = np.where(mask, results['event'][1], results['first'][1])
            np.testing.assert_allclose(part.irish_station_candidate, .5*(p+sh), atol=1e-10)
            for name, expected in zip(SCENARIOS, [.5*(p+nh), .5*(pp+sh), .5*(pp+nh)]):
                np.testing.assert_allclose(part[name], expected, atol=1e-10)
            saved = pd.read_csv(output/'component_input_sensitivity.csv')
            saved = saved.loc[saved.fold.eq(split['fold'])].set_index('global_target_index').loc[ids]
            for column, expected in [('old_process', p), ('new_process', pp), ('old_history', sh), ('new_history', nh)]:
                np.testing.assert_allclose(saved[column], expected, atol=1e-10)
            all_impacts.extend((label, i) for i in ids)
            folds += 1
        assert len(seen) == len(set(seen)) == 259
        frames[label] = frame
    impacts = pd.read_csv(output/'component_input_sensitivity.csv')
    assert len(impacts) == 518 and set(zip(impacts.evaluation, impacts.global_target_index)) == set(all_impacts)
    metrics = pd.read_csv(output/'metrics.csv')
    names = ['irish_station_candidate']+SCENARIOS
    assert len(metrics) == 8 and set(zip(metrics.evaluation, metrics.model)) == {(e, n) for e in frames for n in names}
    for row in metrics.itertuples(index=False):
        frame = frames[row.evaluation]
        for key, value in score_predictions(frame, row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row, key), value, atol=1e-9)
        manual_rmse = np.sqrt(np.average((frame[row.model]-frame.observed_percent)**2, weights=manual_weights(frame)))
        np.testing.assert_allclose(manual_rmse, row.rmse, atol=1e-10)
        np.testing.assert_allclose(manual_integral(frame, row.model), row.integral_rmse, atol=1e-9)
    groups = pd.read_csv(output/'subgroup_metrics.csv')
    populations = ['affected_site', 'affected_first', 'affected_severe_first', 'affected_other_first']
    assert len(groups) == 32 and set(zip(groups.evaluation, groups.population, groups.model)) == {(e, p, n) for e in frames for p in populations for n in names}
    for row in groups.itertuples(index=False):
        frame = frames[row.evaluation]
        mask = frame.site_id.eq(SITE)
        if row.population != 'affected_site':
            mask &= ~frame.ie_uses_event_head
        if row.population == 'affected_severe_first':
            mask &= frame.observed_percent.gt(75)
        if row.population == 'affected_other_first':
            mask &= frame.observed_percent.le(75)
        for key, value in score_subset(frame, mask, row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row, key), value, atol=1e-9)
    assert set(pd.read_csv(output/'four_severe_first_visit_cases.csv').global_target_index) == {1141, 1633, 1905, 1906}
    result = dict(verified=True, status='PROVISIONAL_FORCING_SENSITIVITY_ONLY',
        station_replacement_qualified=False, fitting_performed=False,
        complete_site_days=956, phase_assisted_liquid_days=919, dry_day_duration_updates=424,
        source_and_code_hashes_match=True, baseline_exact=True, unchanged_history_and_stage_columns=True,
        weather_means_independent_scalar_checks=weather_scalars, cutoff_perturbations=perturbations,
        model_folds_replayed=folds, member_prediction_calls=member_calls, strict_earlier_year_training=True,
        minimum_relevant_spatial_distance_km=float(minimum_distance),
        minimum_full_fold_spatial_distance_km=float(minimum_full_fold_distance),
        evaluation_rows_per_design=2017, relevant_rows_per_design=259, unchanged_other_rows_per_design=1758,
        metric_rows_recomputed=8, subgroup_rows_recomputed=32, independent_rmse_and_integrals=True,
        independent_external_validation=False, publication_model_promoted=False,
        verifier_dependencies_sha256={str((Path(__file__).parent/n).relative_to(root)): sha(Path(__file__).parent/n)
            for n in ['verify_first_visit_screens.py', 'verify_french_weather.py']},
        verifier_sha256=sha(__file__))
    write_verification_receipt(output, result, Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.output), indent=2))
