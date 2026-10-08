"""Stage-only BBCH 65 clock with frozen development and sowing assumptions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from analysis.paper_study.phenology_assumptions_20261006.partial_endpoint_inclusive import run as intervals

ARCHIVE = ROOT / 'analysis/paper_study/phenology_assumptions_20261006/partial_endpoint_inclusive'
DEFAULT_DEST = Path(__file__).resolve().parent
FRACTIONS = np.linspace(.02, .60, 201)
LOSS_TOLERANCES_DAYS = (0., 1., 2.)


def predict_clock(accumulation, metadata, thresholds):
    """First inclusive daily crossings at frozen q=1; unreached events stay missing."""
    values = np.asarray(accumulation, float)
    if (values.ndim != 2 or not np.isfinite(values).all()
            or metadata.field_id.duplicated().any()
            or not all(np.isfinite(v) and v > 0 for v in thresholds.values())):
        raise ValueError('Finite development inputs and positive thresholds required')
    for row in metadata.itertuples(index=False):
        if not 0 <= row.field_index < len(values) or not 0 < row.forcing_days <= values.shape[1]:
            raise ValueError('Clock membership lies outside its archived forcing array')
        valid = values[int(row.field_index), :int(row.forcing_days)]
        if np.any(np.diff(valid) < -1e-10):
            raise ValueError('Development accumulation must be nondecreasing')
    return intervals.first_event_dates(values, metadata, thresholds, q=1.)


def build_stage_constraints(rows, source, strict_units=None):
    """Project stage-only columns and reject impossible chronological sequences."""
    projected = intervals.project_stage_rows(rows, source, strict_units)
    normalized, rejected = intervals.normalize_stage_dates(projected)
    contradictions = set()
    for field, group in normalized.groupby('field_id', sort=True):
        previous_lower = -np.inf
        for row in group.sort_values('date').itertuples(index=False):
            if np.isfinite(row.stage_to) and row.stage_to < previous_lower:
                contradictions.add(field)
                break
            if np.isfinite(row.stage_from):
                previous_lower = max(previous_lower, row.stage_from)
    normalized['stage_order_valid'] = ~normalized.field_id.isin(contradictions)
    constraints = intervals.make_constraints(normalized, events=[65])
    if contradictions:
        conflict = normalized.loc[normalized.field_id.isin(contradictions)].copy()
        conflict['reason'] = 'temporally_inconsistent_stage_order'
        rejected = pd.concat([rejected, conflict], ignore_index=True)
        invalid = constraints.field_id.isin(contradictions)
        constraints.loc[invalid, ['eligible_constraint', 'genuinely_bracketed']] = False
        constraints.loc[invalid, 'reason'] = 'temporally_inconsistent_stage_order'
    if not constraints.empty:
        constraints['bracket_width_days'] = (
            constraints.upper_inclusive - constraints.lower_exclusive).dt.days.where(
                constraints.genuinely_bracketed)
    return normalized, rejected, constraints


def score_clock(constraints, predictions):
    return intervals.score_constraints(constraints, predictions)


def fit_threshold_grid(constraints, accumulation, metadata, c51, c85, fractions=None):
    """Select C65 on original calibration stages using equal-field censored loss."""
    if ('value' in constraints or not constraints.source.eq('BASF').all()
            or not constraints.season_year.isin([2017, 2018]).all()):
        raise ValueError('Only original BASF 2017-2018 stage bounds may fit C65')
    fractions = FRACTIONS if fractions is None else np.asarray(fractions, float)
    if (not np.isfinite([c51, c85]).all() or not 0 < c51 < c85
            or fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
            or np.any((fractions < .02-1e-12) | (fractions > .60+1e-12))
            or np.any(np.diff(fractions) <= 0)):
        raise ValueError('Ordered finite thresholds and unique fractions in [.02,.60] required')
    rows, losses = [], []
    for fraction in fractions:
        threshold = float(c51+fraction*(c85-c51))
        scored = score_clock(constraints,
            predict_clock(accumulation, metadata, {65: threshold}))
        rows.append(dict(fraction=float(fraction), threshold_65=threshold,
            loss_days=intervals.equal_field_event_loss(scored), n_fields=scored.field_id.nunique(),
            n_bracketed_fields=int(scored.genuinely_bracketed.sum()),
            compatibility_fraction=float(scored.compatible.mean()),
            missing_predicted_events=int(scored.prediction_status.ne('reached').sum()),
            lower_bound_distances=int(scored.distance_is_lower_bound.sum())))
        losses.append(scored.set_index('field_id').distance_days.rename(float(fraction)))
    curve = pd.DataFrame(rows)
    minimum = float(curve.loss_days.min())
    minima = curve.loc[np.isclose(curve.loss_days, minimum, atol=1e-12, rtol=0)].copy()
    center = float(minima.fraction.median())
    minima['distance_to_plateau_center'] = (minima.fraction-center).abs().round(12)
    chosen = minima.sort_values(['distance_to_plateau_center', 'fraction']).iloc[0]
    bands = []
    for tolerance in LOSS_TOLERANCES_DAYS:
        part = curve.loc[curve.loss_days.le(minimum+tolerance+1e-12)]
        bands.append(dict(loss_tolerance_days=tolerance, n_grid_values=len(part),
            fraction_low=float(part.fraction.min()), fraction_high=float(part.fraction.max()),
            threshold_low=float(part.threshold_65.min()), threshold_high=float(part.threshold_65.max()),
            interpretation='grid loss tolerance set; not a confidence interval'))
    broad = (bands[0]['fraction_high']-bands[0]['fraction_low'] >= .10-1e-12
        or bands[1]['fraction_high']-bands[1]['fraction_low'] >= .20-1e-12
        or np.isclose(chosen.fraction, fractions[[0, -1]]).any()
        or int(chosen.n_bracketed_fields) < 10)
    selected = dict(fraction=float(chosen.fraction), threshold_65=float(chosen.threshold_65),
        minimum_loss_days=minimum, exact_minimum_count=len(minima), loss_tolerance_sets=bands,
        n_constrained_fields=int(chosen.n_fields), n_bracketed_fields=int(chosen.n_bracketed_fields),
        calibration_compatibility=float(chosen.compatibility_fraction),
        poorly_identified=bool(broad), primary_interpretation='stage-calibrated flowering proxy',
        q=1., c51=float(c51), c85=float(c85),
        selected_from='original BASF 2017-2018 stage bounds only',
        tie_break='grid minimizer closest to median minimizing fraction, then smaller fraction',
        numeric_bbch_interpolation=False)
    return selected, curve, pd.DataFrame(losses)


def _metric_rows(scores, constraints, partition, total_fields):
    records = []
    for scope, part in [('all_constraints', scores),
                        ('genuinely_bracketed', scores.loc[scores.genuinely_bracketed])]:
        widths = part.bracket_width_days.dropna()
        records.append(dict(partition=partition, evidence_role=(
            'calibration selection evidence' if partition == 'calibration'
            else 'reused development evidence'), scope=scope, n_original_fields=total_fields,
            n_constrained_fields=part.field_id.nunique(),
            n_locations=part.site_id.nunique(), n_coordinate_years=part.coordinate_year.nunique(),
            mean_distance_days=None if part.empty else float(part.distance_days.mean()),
            mean_signed_distance_days=None if part.empty else float(part.signed_distance_days.mean()),
            compatibility_fraction=None if part.empty else float(part.compatible.mean()),
            missing_predicted_events=int(part.prediction_status.ne('reached').sum()),
            lower_bound_distances=int(part.distance_is_lower_bound.sum()),
            median_bracket_width_days=None if widths.empty else float(widths.median()),
            minimum_bracket_width_days=None if widths.empty else float(widths.min()),
            maximum_bracket_width_days=None if widths.empty else float(widths.max()),
            stage_order_conflict_fields=int(constraints.reason.eq(
                'temporally_inconsistent_stage_order').sum()),
            unconstrained_fields=total_fields-int(constraints.eligible_constraint.sum())))
    return records


def _clock_inputs(partition, original, expected):
    meta = original.loc[original.partition.eq(partition)].sort_values('field_index').copy()
    if set(meta.field_id) != set(expected.field_id):
        raise ValueError(f'Original field registry changed: {partition}')
    paired = meta.merge(expected[['field_id', 'sowing_date', 'first_forcing_date',
        'last_forcing_date', 'forcing_days']], on='field_id', validate='one_to_one',
        suffixes=('_clock', '_original'))
    for column in ['sowing_date', 'first_forcing_date', 'last_forcing_date', 'forcing_days']:
        if not paired[column+'_clock'].eq(paired[column+'_original']).all():
            raise ValueError(f'Frozen forcing identity changed: {partition}, {column}')
    if not np.array_equal(meta.field_index.to_numpy(), np.arange(len(meta))):
        raise ValueError('Archived development field indices are not contiguous')
    with np.load(ARCHIVE / f'{partition}_development_inputs.npz') as inputs:
        accumulation = inputs['accumulation'].copy()
    if len(accumulation) != len(meta):
        raise ValueError('Archived development accumulation membership changed')
    return meta.reset_index(drop=True), accumulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    dest = args.output.resolve()
    if (dest/'configuration_before_fitting.json').exists():
        raise FileExistsError('Use a new flowering-clock archive')
    dest.mkdir(parents=True, exist_ok=True)
    paths = dict(
        basf=ROOT/'analysis/paper_study/seasonal_calibration_v1/basf_source_assessments_snapshot.csv',
        basf_registry=ROOT/'analysis/paper_study/seasonal_calibration_v1/field_season_membership.csv',
        corteva=ROOT/'data/paper_study/observations/corteva_external_assessments.csv',
        corteva_registry=ROOT/'analysis/paper_study/corteva_external_seasonal_v1/external_input_membership.csv',
        strict_registry=ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv',
        phenology=ROOT/'process_model/parameters/calibration.json',
        clock_membership=ARCHIVE/'original_field_input_membership.csv',
        accumulation_calibration=ARCHIVE/'calibration_development_inputs.npz',
        accumulation_2019=ARCHIVE/'reused_development_2019_development_inputs.npz',
        accumulation_external=ARCHIVE/'reused_external_strict_development_inputs.npz',
        accumulation_configuration=ARCHIVE/'configuration_before_calibration.json',
        accumulation_receipt=ARCHIVE/'receipt.json')
    dependencies = [Path(__file__), ROOT/'tests/test_anthesis_clock.py',
        Path(intervals.__file__), Path(intervals.base.__file__)]
    hashes = {str(p.relative_to(ROOT)): intervals.sha(p) for p in [*paths.values(), *dependencies]}
    parameters = json.loads(paths['phenology'].read_text())
    thresholds = {int(row['BBCH']): float(row['Cumulative_t_pp_v_GDD'])
                  for row in parameters['thresholds'] if int(row['BBCH']) in [10, 31, 51, 85]}
    if thresholds != {10: 142.01, 31: 542.3, 51: 986.65, 85: 1760.39}:
        raise ValueError('The original T-P-V thresholds changed')
    config = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        parameter='C65 only', fraction_grid=FRACTIONS.tolist(),
        parameterization='C65 = C51 + fraction*(C85-C51); thermal-index parameterization',
        numeric_bbch_interpolation=False, fixed_q=1., fixed_thresholds=thresholds,
        sowing_and_weather_horizons='unchanged original field registries and archived development arrays',
        objective='mean field distance to the valid BBCH65 censoring interval in days',
        partial_endpoint_rule='stage_to<65 gives exclusive lower date; stage_from>=65 gives inclusive upper date',
        rejection_rule='missing-both and invalid/reversed rows; incompatible same-date intersections; fields with impossible nondecreasing stage order',
        missing_prediction_rule='event later than forcing end; upper-bound distance is a minimum bound; right-only missing events remain compatible',
        stage_column_whitelist=intervals.STAGE_COLUMNS,
        disease_value_or_score_eligibility_used=False, observed_stages_assimilated_into_disease=False,
        calibration_fields=28, calibration_years=[2017, 2018],
        reused_evaluation_fields=dict(reused_development_2019=45, reused_external_strict=143),
        evaluation_stage_values_used_for_selection=False,
        tie_break='grid minimizer closest to median minimizing fraction, then smaller fraction',
        loss_tolerances_days=list(LOSS_TOLERANCES_DAYS),
        loss_tolerance_ranges_are_confidence_intervals=False,
        poor_identification_rule='exact-minimum fraction span >=0.10, one-day tolerance span >=0.20, selected grid boundary, or fewer than 10 genuine calibration brackets',
        diagnostic_grid_sensitivity='all registered C65 fractions; never selected on reused evaluations',
        grain_fill_clock='BBCH65-to-BBCH85 development-window proxy; no yield conversion or maturity claim',
        input_and_source_sha256=hashes)
    intervals.freeze_json(dest/'configuration_before_fitting.json', config)
    original = pd.read_csv(paths['clock_membership'], dtype={'source_unit': str})
    basf_registry = pd.read_csv(paths['basf_registry'], dtype={'source_unit': str})
    cal_registry = basf_registry.loc[basf_registry.season_year.isin([2017, 2018])]
    meta, accumulation = _clock_inputs('calibration', original, cal_registry)
    if len(meta) != 28:
        raise ValueError('Calibration no longer contains exactly 28 original field-seasons')
    raw = pd.read_csv(paths['basf'], usecols=lambda c: c in intervals.STAGE_COLUMNS,
                      dtype={'physical_unit': str})
    projected = intervals.project_stage_rows(raw, 'BASF')
    calibration = intervals.calibration_stage_rows(projected)
    calibration, excluded_cal = intervals.restrict_membership(calibration, meta)
    stages, rejected, constraints = build_stage_constraints(calibration, 'BASF')
    constraints.to_csv(dest/'calibration_constraints_before_fitting.csv', index=False)
    meta.to_csv(dest/'calibration_input_membership.csv', index=False)
    selected, curve, field_losses = fit_threshold_grid(constraints, accumulation, meta,
        thresholds[51], thresholds[85])
    curve.to_csv(dest/'calibration_threshold_grid.csv', index=False)
    field_losses.rename_axis('fraction').to_csv(dest/'calibration_field_loss_by_fraction.csv')
    selected['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    selected['input_and_source_sha256'] = hashes
    intervals.freeze_json(dest/'threshold_selection.json', selected)
    frozen_hash = intervals.sha(dest/'threshold_selection.json')
    print(f"FROZEN C65={selected['threshold_65']:.6f}, fraction={selected['fraction']:.4f}; "
          f"calibration censoring distance={selected['minimum_loss_days']:.4f} days; "
          f"poorly identified={selected['poorly_identified']}", flush=True)
    strict = set(pd.read_csv(paths['strict_registry'], usecols=['source_unit'],
                            dtype={'source_unit': str}).source_unit)
    external_registry = pd.read_csv(paths['corteva_registry'], dtype={'source_unit': str})
    external_registry = external_registry.loc[external_registry.source_unit.isin(strict)]
    external_raw = pd.read_csv(paths['corteva'], usecols=lambda c: c in intervals.STAGE_COLUMNS,
                              dtype={'physical_unit': str})
    external = intervals.project_stage_rows(external_raw, 'Corteva', strict)
    reused_2019 = projected.loc[projected.partition.eq('reused_development_2019')].copy()
    selected_thresholds = dict(thresholds)
    selected_thresholds[65] = selected['threshold_65']
    groups = [('calibration', calibration, meta, accumulation, stages, rejected, constraints, excluded_cal)]
    for partition, rows, registry, count in [
        ('reused_development_2019', reused_2019, basf_registry.loc[basf_registry.season_year.eq(2019)], 45),
        ('reused_external_strict', external, external_registry, 143)]:
        group_meta, group_accumulation = _clock_inputs(partition, original, registry)
        if len(group_meta) != count:
            raise ValueError(f'Original evaluation membership changed: {partition}')
        rows, excluded = intervals.restrict_membership(rows, group_meta)
        normalized, invalid, bounds = build_stage_constraints(rows,
            'Corteva' if partition == 'reused_external_strict' else 'BASF', strict)
        groups.append((partition, rows, group_meta, group_accumulation, normalized, invalid, bounds, excluded))
    all_events, all_bounds, all_scores, all_stages, all_rejected, all_excluded = [], [], [], [], [], []
    wide_events, sensitivity, metric_rows = [], [], []
    for partition, _, group_meta, group_accumulation, normalized, invalid, bounds, excluded in groups:
        pred = predict_clock(group_accumulation, group_meta, selected_thresholds)
        scored = score_clock(bounds, pred)
        metric_rows.extend(_metric_rows(scored, bounds, partition, len(group_meta)))
        pred = pred.merge(group_meta, on='field_id', validate='many_to_one')
        pred['partition'] = partition
        pred['clock_role'] = 'flowering/development proxy; independent of disease values'
        scored['partition'] = partition
        all_events.append(pred); all_scores.append(scored); all_bounds.append(bounds)
        all_stages.append(normalized); all_rejected.append(invalid); all_excluded.append(excluded)
        wide = pred.pivot(index='field_id', columns='event', values='predicted_date')
        wide.columns = [f'bbch{event}_date' for event in wide.columns]
        wide = group_meta.merge(wide.reset_index(), on='field_id', validate='one_to_one')
        status = pred.loc[pred.event.eq(65), ['field_id', 'prediction_status']].rename(
            columns={'prediction_status': 'anthesis_prediction_status'})
        wide = wide.merge(status, on='field_id', validate='one_to_one')
        wide = wide.merge(bounds[['field_id', 'lower_exclusive', 'upper_inclusive',
            'censoring', 'eligible_constraint', 'genuinely_bracketed', 'reason', 'bracket_width_days']],
            on='field_id', how='left', validate='one_to_one')
        wide['eligible_constraint'] = wide.eligible_constraint.eq(True)
        wide['genuinely_bracketed'] = wide.genuinely_bracketed.eq(True)
        wide['reason'] = wide.reason.fillna('no_valid_stage_dates')
        wide['censoring'] = wide.censoring.fillna('none')
        wide['partition'], wide['threshold_65'], wide['fraction'] = partition, selected['threshold_65'], selected['fraction']
        wide['anthesis_proxy_date'] = wide.bbch65_date
        wide['grain_fill_window_proxy_days'] = (wide.bbch85_date-wide.bbch65_date).dt.days
        part_sensitivity = []
        for fraction in FRACTIONS:
            threshold = float(thresholds[51]+fraction*(thresholds[85]-thresholds[51]))
            events = predict_clock(group_accumulation, group_meta, {65: threshold})
            events['fraction'], events['threshold_65'], events['partition'] = fraction, threshold, partition
            part_sensitivity.append(events)
        part_sensitivity = pd.concat(part_sensitivity, ignore_index=True)
        date_range = part_sensitivity.groupby('field_id').agg(
            anthesis_grid_earliest_date=('predicted_date', 'min'),
            anthesis_grid_latest_reached_date=('predicted_date', 'max'),
            anthesis_grid_unreached_values=('predicted_date', lambda x: int(x.isna().sum()))).reset_index()
        wide = wide.merge(date_range, on='field_id', validate='one_to_one')
        wide['anthesis_grid_reached_date_span_days'] = (
            wide.anthesis_grid_latest_reached_date-wide.anthesis_grid_earliest_date).dt.days
        wide_events.append(wide); sensitivity.append(part_sensitivity)
    event_table = pd.concat(all_events, ignore_index=True)
    score_table = pd.concat(all_scores, ignore_index=True)
    metric_table = pd.DataFrame(metric_rows)
    event_table.to_csv(dest/'field_event_predictions_long.csv', index=False)
    pd.concat(wide_events, ignore_index=True).to_csv(dest/'field_events.csv', index=False)
    pd.concat(sensitivity, ignore_index=True).to_csv(dest/'grid_sensitivity_field_events.csv', index=False)
    pd.concat(all_bounds, ignore_index=True).to_csv(dest/'bbch65_stage_constraints.csv', index=False)
    pd.concat(all_stages, ignore_index=True).to_csv(dest/'normalized_stage_dates.csv', index=False)
    pd.concat(all_rejected, ignore_index=True).to_csv(dest/'rejected_stage_rows.csv', index=False)
    pd.concat(all_excluded, ignore_index=True).to_csv(dest/'original_membership_excluded_stage_rows.csv', index=False)
    score_table.to_csv(dest/'censored_timing_scores.csv', index=False)
    metric_table.to_csv(dest/'bbch65_censoring_metrics.csv', index=False)
    pd.DataFrame(selected['loss_tolerance_sets']).to_csv(dest/'grid_loss_tolerance_sets.csv', index=False)
    independent = []
    for row in score_table.itertuples(index=False):
        if pd.isna(row.predicted_date):
            distance = (max(0, (pd.Timestamp(row.forcing_end)+pd.Timedelta(days=1)-row.upper_inclusive).days)
                        if pd.notna(row.upper_inclusive) else 0)
        elif pd.notna(row.lower_exclusive) and row.predicted_date <= row.lower_exclusive:
            distance = (row.lower_exclusive+pd.Timedelta(days=1)-row.predicted_date).days
        elif pd.notna(row.upper_inclusive) and row.predicted_date > row.upper_inclusive:
            distance = (row.predicted_date-row.upper_inclusive).days
        else:
            distance = 0
        independent.append(float(distance))
    delta = float(np.max(np.abs(np.asarray(independent)-score_table.distance_days)))
    if delta != 0:
        raise ValueError('Independent censoring arithmetic differs')
    intervals.verify_dependencies(hashes)
    if intervals.sha(dest/'threshold_selection.json') != frozen_hash:
        raise ValueError('The frozen C65 changed during reused evaluation')
    snapshot = dest/'source_snapshot'
    for source in dependencies:
        target = snapshot/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    intervals.freeze_json(dest/'verification.json', dict(
        independent_censoring_rows=len(score_table), maximum_distance_delta_days=delta,
        original_membership_counts={p: len(m) for p, _, m, *_ in groups},
        original_sowing_and_forcing_identity_preserved=True,
        selected_threshold_sha256=frozen_hash, input_and_source_hashes_preserved=True,
        disease_values_used=False, observed_stage_assimilation_into_disease=False))
    all_metrics = metric_table.loc[metric_table.scope.eq('all_constraints')].set_index('partition')
    cal = all_metrics.loc['calibration']; dev = all_metrics.loc['reused_development_2019']; ext = all_metrics.loc['reused_external_strict']
    exact, one_day, two_day = selected['loss_tolerance_sets']
    summary = (
        f"The flowering clock retains q=1, the original sowing scenarios, the original daily forcing horizons, and fixed T-P-V thresholds C10=142.01, C31=542.30, C51=986.65 and C85=1760.39. A single BBCH65 development threshold is constrained between C51 and C85 using 201 declared thermal-index fractions from 0.02 to 0.60. Numeric BBCH codes are not interpolated.\n\n"
        f"The original 28 BASF 2017–2018 field-seasons supply {int(cal.n_constrained_fields)} usable flowering constraints, including {selected['n_bracketed_fields']} genuine date brackets. The selected fraction is {selected['fraction']:.4f}, corresponding to C65={selected['threshold_65']:.2f}. Mean equal-field censoring distance is {selected['minimum_loss_days']:.2f} days, and calibration compatibility is {100*selected['calibration_compatibility']:.1f}%. Exact grid minima occupy {exact['n_grid_values']} thresholds spanning {exact['threshold_low']:.2f}–{exact['threshold_high']:.2f}. One-day and two-day loss-tolerance sets span {one_day['threshold_low']:.2f}–{one_day['threshold_high']:.2f} and {two_day['threshold_low']:.2f}–{two_day['threshold_high']:.2f}, respectively. These sets describe the calibration loss surface and are not confidence intervals.\n\n"
        f"The reused BASF 2019 registry retains 45 fields, with {int(dev.n_constrained_fields)} valid flowering constraints, mean censored distance {dev.mean_distance_days:.2f} days and compatibility {100*dev.compatibility_fraction:.1f}%. The strict reused Corteva registry retains 143 fields, with {int(ext.n_constrained_fields)} valid constraints, mean distance {ext.mean_distance_days:.2f} days and compatibility {100*ext.compatibility_fraction:.1f}%. Genuine bracket widths and their associated timing scores remain separate from one-sided constraints. Reused evaluation stages do not determine the selected threshold.\n\n"
        f"{'The flowering threshold is weakly identified under the declared grid diagnostics. ' if selected['poorly_identified'] else ''}C65 is a stage-calibrated flowering proxy rather than an observed field-specific anthesis date. The full registered grid supplies a field-level sensitivity range; thresholds not reached by the original forcing horizon remain explicitly right-censored. Missing predicted events retain minimum one-sided distances to observed upper bounds and remain compatible with right-only observations.\n\n"
        "The BBCH65-to-BBCH85 interval is a development-window proxy for grain-fill exposure. It does not establish physiological maturity, measured yield loss, or a disease-to-yield conversion. No disease score, score-eligibility flag, or held-out stage record enters the flowering fit or any disease forecast. The original phenology runtime, disease runtime, manuscript and regional projections remain unchanged.\n")
    (dest/'scientific_summary.txt').write_text(summary)
    intervals.freeze_json(dest/'receipt.json', dict(status='complete', threshold_65=selected['threshold_65'],
        fraction=selected['fraction'], poorly_identified=selected['poorly_identified'],
        original_field_counts={p: len(m) for p, _, m, *_ in groups},
        fixed_q=1., fixed_thresholds=thresholds, threshold_selection_sha256=frozen_hash,
        reused_evaluation_stages_used_for_selection=False, untouched_test_evaluated=False,
        disease_values_used=False, yield_loss_estimated=False, shared_runtime_changed=False))
    print(metric_table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
