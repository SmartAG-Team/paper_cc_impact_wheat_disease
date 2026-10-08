"""Complete, version-labelled validation evidence; no model fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from analysis.phenology.verify_archived_validation import verify as verify_german
from analysis.paper_study.infection_priority_20261006.anthesis_clock import run as clock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT/'analysis/paper_study/overwinter_leaf_model_20261006'
GERMAN = ROOT/'analysis/phenology/heldout_validation'
EVENTS = [10, 31, 32, 33, 37, 39, 51, 65, 85]
PARTITIONS = ['calibration', 'reused_development_2019', 'reused_external_strict', 'reused_external_full']
LABELS = dict(zip(PARTITIONS, ['BASF 2017–18 calibration', 'BASF 2019', 'Corteva location-disjoint', 'Corteva full transfer']))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field_evidence():
    stages = pd.read_csv(STUDY/'phenology/normalized_stage_dates.csv', parse_dates=['date'])
    replay = pd.read_csv(STUDY/'manuscript/replay/full_season_top3_timing_and_conditional_yield.csv')
    assert len(replay) == 218 and replay.field_id.is_unique
    raw_path = ROOT/'data/paper_study/observations/corteva_external_assessments.csv'
    raw = pd.read_csv(raw_path, usecols=lambda c: c in clock.intervals.STAGE_COLUMNS,
        dtype={'physical_unit': str})
    full_fields = set(replay.loc[replay.source.eq('Corteva'), 'field_id'])
    full_units = {field.split('|')[1] for field in full_fields}
    normalized, rejected, _ = clock.build_stage_constraints(raw, 'Corteva', full_units)
    normalized = normalized.loc[normalized.field_id.isin(full_fields)].copy()
    normalized['partition'] = 'reused_external_full'
    stages = pd.concat([stages, normalized], ignore_index=True)
    bounds = []
    for partition, group in stages.groupby('partition'):
        constraints = clock.intervals.make_constraints(group, events=EVENTS)
        invalid = constraints.field_id.isin(set(group.loc[~group.stage_order_valid, 'field_id']))
        constraints.loc[invalid, ['eligible_constraint', 'genuinely_bracketed']] = False
        constraints.loc[invalid, 'reason'] = 'temporally_inconsistent_stage_order'
        constraints['partition'] = partition
        constraints['bracket_width_days'] = (constraints.upper_inclusive-constraints.lower_exclusive).dt.days.where(
            constraints.genuinely_bracketed)
        bounds.append(constraints)
    bounds = pd.concat(bounds, ignore_index=True)
    observed = pd.read_csv(STUDY/'phenology/predicted_stage_dates.csv', parse_dates=['predicted_date', 'forcing_end'])
    full = []
    for event in EVENTS:
        rows = replay[['field_id', f'BBCH{event}_date', 'forcing_end_date', 'reused_validation_status']].copy()
        rows = rows.rename(columns={f'BBCH{event}_date': 'predicted_date', 'forcing_end_date': 'forcing_end'})
        rows['event'] = event
        rows['partition'] = rows.reused_validation_status.map({
            'original_calibration': 'calibration', 'reused_BASF2019': 'reused_development_2019',
            'reused_strict_Corteva': 'reused_external_strict'})
        rows['q'] = 1.
        rows['prediction_status'] = 'reached'
        full.append(rows)
    full = pd.concat(full, ignore_index=True)
    for column in ['predicted_date', 'forcing_end']:
        full[column] = pd.to_datetime(full[column])
    assert full.predicted_date.notna().all()
    # Existing daily threshold crossings cannot move when later weather is added.
    reconcile = observed.merge(full[['field_id', 'event', 'predicted_date']],
        on=['field_id', 'event'], suffixes=('_old', '_complete'), validate='1:1')
    matched = reconcile.predicted_date_old.notna()
    assert reconcile.loc[matched, 'predicted_date_old'].eq(reconcile.loc[matched, 'predicted_date_complete']).all()
    source = replay.set_index('field_id').source
    strict_fields = set(observed.loc[observed.partition.eq('reused_external_strict'), 'field_id'])
    full['partition'] = full.field_id.map(lambda field: 'calibration' if field in set(
        observed.loc[observed.partition.eq('calibration'), 'field_id']) else 'reused_development_2019'
        if source[field] == 'BASF' else 'reused_external_strict' if field in strict_fields else 'outside_strict_subset')
    expanded = full.loc[full.field_id.isin(full_fields)].copy()
    expanded['partition'] = 'reused_external_full'
    full = pd.concat([full.loc[full.partition.isin(PARTITIONS)], expanded], ignore_index=True)
    scores = []
    for scope, predictions in [('observed_horizon', observed), ('full_season', full)]:
        for partition, pred in predictions.groupby('partition'):
            selected = bounds.loc[bounds.partition.eq(partition)]
            scored = clock.score_clock(selected, pred.drop(columns=['partition', 'reused_validation_status'], errors='ignore'))
            scored['partition'] = partition
            scored['forcing_scope'] = scope
            scores.append(scored)
    scores = pd.concat(scores, ignore_index=True)
    previous = pd.read_csv(STUDY/'phenology/stage_censoring_scores.csv')
    old = scores.loc[scores.forcing_scope.eq('observed_horizon') & scores.event.isin([31,32,33,37,39])]
    check = previous.merge(old, on=['partition', 'field_id', 'event'], suffixes=('_old', '_new'), validate='1:1')
    assert len(check) == len(previous)
    assert check.compatible_old.eq(check.compatible_new).all()
    np.testing.assert_array_equal(check.distance_days_old, check.distance_days_new)
    coverage = []
    metrics = []
    totals = dict(calibration=28, reused_development_2019=45, reused_external_strict=143, reused_external_full=145)
    for partition in PARTITIONS:
        for event in EVENTS:
            selected = bounds.loc[bounds.partition.eq(partition) & bounds.event.eq(event)]
            eligible = selected.loc[selected.eligible_constraint]
            coverage.append(dict(partition=partition, event=event, registered_fields=totals[partition],
                constrained_fields=len(eligible), genuine_two_sided=int(eligible.genuinely_bracketed.sum()),
                left_censored=int(eligible.censoring.eq('left').sum()),
                right_censored=int(eligible.censoring.eq('right').sum()),
                unavailable_or_rejected_fields=totals[partition]-len(eligible)))
    for (scope, partition, event), group in scores.groupby(['forcing_scope','partition','event']):
        for kind, selected in [('all_constraints',group), ('genuine_two_sided',group.loc[group.genuinely_bracketed]),
                ('left_censored',group.loc[group.censoring.eq('left')]), ('right_censored',group.loc[group.censoring.eq('right')])]:
            metrics.append(dict(forcing_scope=scope, partition=partition, event=event, scope=kind,
                n_field_events=len(selected), mean_distance_days=selected.distance_days.mean(),
                mean_signed_distance_days=selected.signed_distance_days.mean(),
                compatible_fraction=selected.compatible.mean(),
                missing_predictions=int(selected.predicted_date.isna().sum()),
                lower_bound_distances=int(selected.distance_is_lower_bound.sum()),
                median_bracket_width_days=selected.bracket_width_days.median() if len(selected) else np.nan,
                evaluation_role='calibration' if partition=='calibration' else 'reused_validation',
                overlaps_strict_subset=partition=='reused_external_full'))
    bounds.to_csv(HERE/'field_stage_constraints_all.csv',index=False)
    scores.to_csv(HERE/'field_stage_scores_all.csv',index=False)
    pd.DataFrame(metrics).to_csv(HERE/'field_stage_metrics_all.csv',index=False)
    pd.DataFrame(coverage).to_csv(HERE/'field_stage_coverage_all.csv',index=False)
    return dict(predictions_reconciled=int(matched.sum()), prior_stage_score_rows_reconciled=len(previous),
        events=EVENTS, fields_by_partition=totals, scores=len(scores), coverage_rows=len(coverage))


def main():
    verification = verify_german(GERMAN)
    (HERE/'german_independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    german_metrics = pd.read_csv(GERMAN/'metrics_long.csv')
    german_metrics.to_csv(HERE/'german_stage_metrics_all.csv',index=False)
    for cohort in ['calibration','validation','testing']:
        events = pd.read_csv(GERMAN/f'{cohort}_tpv_event_dates.csv.gz')
        events['cohort'] = cohort
        events.to_parquet(HERE/f'german_{cohort}_tpv_events.parquet',index=False)
    result = field_evidence()
    for name in ['assessment_detection_metrics','onset_bracket_metrics','observed_window_occurrence_metrics']:
        rows = pd.read_csv(STUDY/'disease'/f'{name}.csv')
        rows.loc[rows.scenario.eq('baseline')].to_csv(HERE/f'disease_{name}_baseline.csv',index=False)
    history = []
    legacy_paths = ['analysis/paper_study/seasonal_calibration_v1/severity_metrics.csv',
        'analysis/paper_study/corteva_external_seasonal_v1/severity_metrics.csv',
        'analysis/paper_study/accuracy_review_20261006_verified/verified_onset_metrics.csv',
        'analysis/primary_secondary/external_french/frozen_transfer_metrics.csv',
        'analysis/primary_secondary/external_french/transfer_v2/frozen_transfer_metrics.csv']
    for relative in legacy_paths:
        path = ROOT/relative
        rows = pd.read_csv(path)
        rows['source_path'] = relative
        rows['evidence_role'] = 'archived model-version results; not current seasonal-model validation'
        if 'external_french' in relative:
            rows['evidence_version'] = 'BASF-conditioned hidden-state v2' if '/transfer_v2/' in relative else 'BASF-conditioned hidden-state v1'
            rows.to_csv(HERE/('french_transfer_v2.csv' if '/transfer_v2/' in relative else 'french_transfer_v1.csv'),index=False)
        else:
            history.append(rows)
    pd.concat(history,ignore_index=True).to_csv(HERE/'archived_seasonal_model_results.csv',index=False)
    inventory = []
    sources = ['analysis/phenology/heldout_validation/metrics_long.csv',
        'analysis/phenology/heldout_validation/validation_summary.json',
        'analysis/phenology/heldout_validation/PROVENANCE.json',
        *[f'analysis/phenology/heldout_validation/{cohort}_all_model_event_dates.csv.gz' for cohort in ['calibration','validation','testing']],
        *[f'analysis/paper_study/overwinter_leaf_model_20261006/phenology/{name}' for name in ['predicted_stage_dates.csv','normalized_stage_dates.csv','stage_censoring_scores.csv']],
        'analysis/paper_study/overwinter_leaf_model_20261006/manuscript/replay/full_season_top3_timing_and_conditional_yield.csv',
        'data/paper_study/observations/corteva_external_assessments.csv', *legacy_paths]
    for relative in sources:
        inventory.append(dict(source_path=relative,sha256=sha(ROOT/relative)))
    pd.DataFrame(inventory).to_csv(HERE/'validation_source_inventory.csv',index=False)
    receipt = dict(status='verified',german=verification,field=result,
        prior_datasets=['German station network','BASF 2017–2018','BASF 2019',
            'Corteva location-disjoint subset','Corteva full source transfer','French plot-leaf transfer 2018–2019'],
        current_runtime_or_parameters_changed=False,calibration_performed=False,
        legacy_model_results_pooled_with_current=False, source_sha256={r['source_path']:r['sha256'] for r in inventory},
        output_sha256={p.name:sha(p) for p in sorted(HERE.glob('*')) if p.suffix in ['.csv','.parquet']})
    (HERE/'evidence_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='verified',german_metrics_recomputed=verification['recomputed_numeric_metrics'],field=result)))


if __name__=='__main__':main()
