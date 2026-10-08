"""Bounded development-index calibration using archived crop-stage bounds only."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
DEST = Path(__file__).resolve().parent
EVENTS = (31, 51, 85)
Q_GRID = np.linspace(.6, 1.6, 401)
STAGE_COLUMNS = ["dataset_id", "physical_unit", "season_year", "site_id", "date",
                 "stage_from", "stage_to", "country", "latitude", "longitude", "sowing_date"]
IDENTITY_COLUMNS = ["source", "partition", "field_id", "coordinate_year", "dataset_id",
                    "physical_unit", "season_year", "site_id", "country", "latitude", "longitude"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Frozen study artifact already exists: {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def project_stage_rows(rows, source, strict_units=None):
    """Select an explicit whitelist; disease scores and score eligibility are unused."""
    projected = rows[[c for c in STAGE_COLUMNS if c in rows]].copy()
    required = set(STAGE_COLUMNS) - {"sowing_date"}
    if not required.issubset(projected):
        raise ValueError(f"Missing stage/identity columns: {required - set(projected)}")
    projected["physical_unit"] = projected.physical_unit.astype(str)
    projected["season_year"] = pd.to_numeric(projected.season_year).astype(int)
    projected["date"] = pd.to_datetime(projected.date).dt.normalize()
    projected["source"] = source
    if source == "BASF":
        if not projected.season_year.isin([2017, 2018, 2019]).all():
            raise ValueError("Unexpected BASF year outside original development membership")
        projected["partition"] = np.where(projected.season_year.isin([2017, 2018]),
            "calibration", "reused_development_2019")
    elif source == "Corteva":
        if strict_units is None:
            raise ValueError("Explicit strict external registry is required")
        projected = projected.loc[projected.physical_unit.isin(set(strict_units))].copy()
        projected["partition"] = "reused_external_strict"
    else:
        raise ValueError("Unknown source")
    projected["field_id"] = (projected.dataset_id.astype(str) + "|" +
        projected.physical_unit + "|" + projected.season_year.astype(str))
    projected["coordinate_year"] = (projected.site_id.astype(str) + "|" +
                                    projected.season_year.astype(str))
    return projected.reset_index(drop=True)


def calibration_stage_rows(rows):
    if "value" in rows:
        raise ValueError("Disease values must be removed before the calibration boundary")
    return rows.loc[rows.source.eq("BASF") & rows.season_year.isin([2017, 2018])].copy()


def restrict_membership(rows, membership):
    """Preserve original model fields and their archived daily forcing horizons."""
    if membership.field_id.duplicated().any():
        raise ValueError("Original field membership has duplicate identifiers")
    frame = rows.merge(membership[["field_id", "first_forcing_date", "last_forcing_date"]],
                       on="field_id", how="left", validate="many_to_one")
    frame["first_forcing_date"] = pd.to_datetime(frame.first_forcing_date)
    frame["last_forcing_date"] = pd.to_datetime(frame.last_forcing_date)
    frame["reason"] = ""
    frame.loc[frame.first_forcing_date.isna(), "reason"] = "outside_original_model_field_membership"
    outside = frame.first_forcing_date.notna() & ~frame.date.between(
        frame.first_forcing_date, frame.last_forcing_date)
    frame.loc[outside, "reason"] = "outside_original_forcing_horizon"
    excluded = frame.loc[frame.reason.ne("")].copy()
    selected = frame.loc[frame.reason.eq("")].drop(
        columns=["first_forcing_date", "last_forcing_date", "reason"]).copy()
    return selected, excluded


def input_placeholders(rows):
    """Zero is a preparation placeholder, never an observed disease value."""
    projected = rows[[c for c in STAGE_COLUMNS if c in rows]].copy()
    identities = [c for c in STAGE_COLUMNS if c not in ["stage_from", "stage_to"] and c in projected]
    frame = projected[identities].drop_duplicates().copy()
    frame["organ"], frame["metric"], frame["unit"] = "F1", "infection_percent_unspecified_basis", "percent"
    frame["value"] = 0.
    frame["endpoint_series"] = frame.physical_unit.astype(str) + "|stage_input_placeholder"
    return frame


def normalize_stage_dates(rows):
    if "value" in rows:
        raise ValueError("Numeric disease values are forbidden at the stage normalization boundary")
    frame = rows.copy()
    frame["stage_from"] = pd.to_numeric(frame.stage_from, errors="coerce")
    frame["stage_to"] = pd.to_numeric(frame.stage_to, errors="coerce")
    frame["reason"] = ""
    missing = frame.stage_from.isna() | frame.stage_to.isna()
    frame.loc[missing, "reason"] = "missing_stage_endpoint"
    outside = (~missing) & (~frame.stage_from.between(0, 99) | ~frame.stage_to.between(0, 99))
    frame.loc[outside, "reason"] = "stage_outside_supported_range"
    reversed_range = (~missing) & frame.stage_from.gt(frame.stage_to)
    frame.loc[reversed_range, "reason"] = "reversed_stage_interval"
    rejected = [frame.loc[frame.reason.ne("")].copy()]
    accepted = []
    for (_, _), group in frame.loc[frame.reason.eq("")].groupby(["field_id", "date"], sort=True):
        if len(group[IDENTITY_COLUMNS].drop_duplicates()) != 1:
            raise ValueError("Conflicting field/date identity metadata")
        lower, upper = float(group.stage_from.max()), float(group.stage_to.min())
        if lower > upper:
            conflict = group.copy()
            conflict["reason"] = "conflicting_date_stage_ranges"
            rejected.append(conflict)
            continue
        row = group.iloc[0][IDENTITY_COLUMNS + ["date"]].to_dict()
        row.update(stage_from=lower, stage_to=upper, source_row_count=len(group),
                   unique_source_stage_intervals=len(group[["stage_from", "stage_to"]].drop_duplicates()))
        accepted.append(row)
    normalized = pd.DataFrame(accepted, columns=IDENTITY_COLUMNS + ["date", "stage_from", "stage_to",
        "source_row_count", "unique_source_stage_intervals"])
    normalized["date"] = pd.to_datetime(normalized.date)
    return normalized, pd.concat(rejected, ignore_index=True)


def make_constraints(stages, events=EVENTS):
    rows = []
    for field, group in stages.groupby("field_id", sort=True):
        identity = group.iloc[0][IDENTITY_COLUMNS].to_dict()
        for event in events:
            negatives = group.loc[group.stage_to.lt(event), "date"]
            positives = group.loc[group.stage_from.ge(event), "date"]
            lower = pd.NaT if negatives.empty else negatives.max()
            upper = pd.NaT if positives.empty else positives.min()
            available = not (pd.isna(lower) and pd.isna(upper))
            inconsistent = available and pd.notna(lower) and pd.notna(upper) and lower >= upper
            kind = ("none" if not available else "left" if pd.isna(lower)
                    else "right" if pd.isna(upper) else "interval")
            rows.append(dict(**identity, event=int(event), lower_exclusive=lower, upper_inclusive=upper,
                censoring=kind, eligible_constraint=bool(available and not inconsistent),
                genuinely_bracketed=bool(kind == "interval" and not inconsistent),
                reason=("no_event_bound" if not available else
                        "temporally_inconsistent_event_bounds" if inconsistent else ""),
                n_distinct_stage_dates=len(group), n_lower_bound_dates=len(negatives),
                n_upper_bound_dates=len(positives)))
    return pd.DataFrame(rows)


def first_event_dates(accumulation, metadata, thresholds, q):
    if not np.isfinite(q) or q <= 0:
        raise ValueError("Development-index multiplier must be positive and finite")
    predictions = []
    for meta in metadata.itertuples(index=False):
        days, index = int(meta.forcing_days), int(meta.field_index)
        scaled = np.asarray(accumulation[index, :days]) * q
        for event, threshold in thresholds.items():
            found = np.flatnonzero(scaled >= threshold)
            predicted = (pd.NaT if not len(found) else pd.Timestamp(meta.sowing_date) +
                         pd.Timedelta(days=int(found[0])))
            predictions.append(dict(field_id=meta.field_id, event=int(event), q=float(q),
                predicted_date=predicted, forcing_end=pd.Timestamp(meta.last_forcing_date),
                prediction_status="not_reached_by_forcing_end" if pd.isna(predicted) else "reached"))
    return pd.DataFrame(predictions)


def censoring_distance(predicted, lower, upper, forcing_end):
    lower, upper, predicted = [pd.NaT if value is None else pd.Timestamp(value)
                               for value in [lower, upper, predicted]]
    missing = pd.isna(predicted)
    if pd.isna(lower) and pd.isna(upper):
        raise ValueError("Unconstrained events cannot enter the loss")
    if pd.notna(lower) and pd.notna(upper) and lower >= upper:
        raise ValueError("Temporal event bounds are inconsistent")
    if missing:
        # The unobserved prediction date is strictly later than the forcing
        # horizon. This is a bound, not an imputed event date.
        prediction_bound = pd.Timestamp(forcing_end) + pd.Timedelta(days=1)
        signed = float(max(0, (prediction_bound - upper).days)) if pd.notna(upper) else 0.
    else:
        minimum = lower + pd.Timedelta(days=1) if pd.notna(lower) else pd.NaT
        signed = (float((predicted - minimum).days) if pd.notna(minimum) and predicted < minimum
                  else float((predicted - upper).days) if pd.notna(upper) and predicted > upper else 0.)
    return dict(signed_distance_days=signed, distance_days=abs(signed), compatible=bool(signed == 0),
                distance_is_lower_bound=bool(missing),
                prediction_status="not_reached_by_forcing_end" if missing else "reached")


def score_constraints(constraints, predictions):
    eligible = constraints.loc[constraints.eligible_constraint].copy()
    joined = eligible.merge(predictions.drop(columns="prediction_status"), on=["field_id", "event"],
        how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("A constrained event lacks prediction/horizon membership")
    joined = joined.drop(columns="_merge")
    distances = pd.DataFrame([censoring_distance(row.predicted_date, row.lower_exclusive,
        row.upper_inclusive, row.forcing_end) for row in joined.itertuples(index=False)], index=joined.index)
    return pd.concat([joined, distances], axis=1)


def equal_field_event_loss(scores):
    if scores.empty or not np.isfinite(scores.distance_days).all():
        raise ValueError("Calibration must retain finite explicitly censored event distances")
    return float(scores.groupby("field_id").distance_days.mean().mean())


def choose_q(curve):
    minimum = float(curve.loss_days.min())
    tied = curve.loc[np.isclose(curve.loss_days, minimum, atol=1e-12, rtol=0)].copy()
    tied["distance_to_one"] = (tied.q - 1.).abs().round(12)
    return float(tied.sort_values(["distance_to_one", "q"]).iloc[0].q)


def fit_grid(constraints, accumulation, metadata, thresholds):
    if not constraints.source.eq("BASF").all() or not constraints.season_year.isin([2017, 2018]).all():
        raise ValueError("Only original BASF 2017-2018 stages may enter q fitting")
    if "value" in constraints:
        raise ValueError("Disease values crossed the phenology fitting boundary")
    rows, field_losses = [], []
    for q in Q_GRID:
        scored = score_constraints(constraints, first_event_dates(accumulation, metadata, thresholds, q))
        rows.append(dict(q=float(q), loss_days=equal_field_event_loss(scored),
            n_constrained_field_events=len(scored), n_fields=scored.field_id.nunique(),
            missing_predicted_events=int(scored.prediction_status.ne("reached").sum()),
            compatibility_fraction=float(scored.compatible.mean())))
        field_losses.append(scored.groupby("field_id").distance_days.mean().rename(float(q)))
    curve = pd.DataFrame(rows)
    return choose_q(curve), curve, pd.DataFrame(field_losses)


def stability_checks(field_losses, constraints):
    identities = constraints[["field_id", "site_id", "season_year"]].drop_duplicates().set_index("field_id")
    rows = []
    for kind, labels in [("year_only", sorted(identities.season_year.unique())),
                         ("leave_location_out", sorted(identities.site_id.unique()))]:
        for label in labels:
            selected = identities.index[identities.season_year.eq(label) if kind == "year_only"
                else identities.site_id.ne(label)]
            selected = [field for field in selected if field in field_losses]
            if not selected:
                continue
            curve = pd.DataFrame(dict(q=field_losses.index.astype(float),
                                      loss_days=field_losses[selected].mean(axis=1).to_numpy()))
            chosen = choose_q(curve)
            rows.append(dict(check=kind, label=str(label), selected_q=chosen,
                selected_loss_days=float(curve.loc[curve.q.eq(chosen), "loss_days"].iloc[0]),
                n_fields=len(selected)))
    return pd.DataFrame(rows)


def metric_summary(scores):
    rows = []
    for (model, partition), group in scores.groupby(["model", "partition"]):
        for scope, selected in [("all_constraints", group),
                                ("genuinely_bracketed", group.loc[group.genuinely_bracketed])]:
            for event in ["all", *EVENTS]:
                part = selected if event == "all" else selected.loc[selected.event.eq(event)]
                if part.empty:
                    continue
                by_field = part.groupby("field_id")
                rows.append(dict(model=model, partition=partition, scope=scope, event=str(event),
                    n_field_events=len(part), n_fields=part.field_id.nunique(),
                    n_locations=part.site_id.nunique(),
                    equal_field_event_distance_days=equal_field_event_loss(part),
                    mean_signed_distance_days=float(by_field.signed_distance_days.mean().mean()),
                    compatibility_fraction=float(part.compatible.mean()),
                    equal_field_compatibility_fraction=float(by_field.compatible.mean().mean()),
                    missing_predicted_events=int(part.prediction_status.ne("reached").sum()),
                    n_lower_bound_distances=int(part.distance_is_lower_bound.sum())))
    return pd.DataFrame(rows)


def verify_dependencies(hashes):
    for relative, digest in hashes.items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Frozen dependency changed: {relative}")


def prepare_group(rows, original_membership, calendars, weather, phenology):
    from model.seasonal_septoria.field_data import prepare_fields
    from analysis.paper_study.run_structural_canopy import development_inputs

    data = prepare_fields(input_placeholders(rows), calendars, weather)
    expected = original_membership.loc[original_membership.field_id.isin(rows.field_id.unique())].copy()
    if set(data.metadata.field_id) != set(expected.field_id):
        raise ValueError("Original model field eligibility was not preserved")
    paired = data.metadata.merge(expected[["field_id", "sowing_date", "forcing_days", "last_forcing_date"]],
        on="field_id", validate="one_to_one", suffixes=("_new", "_original"))
    for column in ["sowing_date", "forcing_days", "last_forcing_date"]:
        if not paired[f"{column}_new"].eq(paired[f"{column}_original"]).all():
            raise ValueError(f"Original phenology forcing identity changed: {column}")
    accumulation, thresholds = development_inputs(data, weather, phenology)
    stages, rejected = normalize_stage_dates(rows)
    return data.metadata, accumulation, {event: thresholds[event] for event in EVENTS}, stages, rejected


def main():
    paths = dict(basf=ROOT / "analysis/paper_study/seasonal_calibration_v1/basf_source_assessments_snapshot.csv",
        basf_membership=ROOT / "analysis/paper_study/seasonal_calibration_v1/field_season_membership.csv",
        corteva=ROOT / "data/paper_study/observations/corteva_external_assessments.csv",
        corteva_membership=ROOT / "analysis/paper_study/corteva_external_seasonal_v1/external_input_membership.csv",
        strict_registry=ROOT / "data/paper_study/observations/corteva_location_disjoint_external_units.csv",
        weather=ROOT / "data/paper_study/field_weather_completed/daily_weather.parquet",
        calendars=ROOT / "data/paper_study/wheat_area/trial_point_calendar_scenarios.csv",
        phenology=ROOT / "process_model/parameters/calibration.json")
    dependencies = [Path(__file__), ROOT / "tests/test_phenology_assumptions.py", *paths.values(),
        ROOT / "analysis/paper_study/run_structural_canopy.py",
        ROOT / "analysis/paper_study/run_empirical_benchmarks.py",
        ROOT / "analysis/paper_study/calibrate_seasonal.py",
        *sorted((ROOT / "model/seasonal_septoria").glob("*.py")),
        *sorted((ROOT / "process_model").glob("*.py")),
        *sorted((ROOT / "process_model/_support").glob("*.py"))]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in dependencies}
    contract = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        q_grid=dict(lower=.6, upper=1.6, points=401, spacing=.0025),
        selection="minimum mean of within-field mean event censoring distance in days; nearest-to-one then smaller-q tie break",
        calibration_source="Original BASF 2017-2018 stage rows within original 28-field model membership and forcing horizons",
        events=list(EVENTS), numeric_bbch_interpolation=False, disease_values_read=False,
        stage_column_whitelist=STAGE_COLUMNS,
        placeholder_semantics="Synthetic zero is solely required by input preparation; it is not an observed disease value",
        date_level_stage_rule="intersection of valid ranges; reversed rows and conflicting date-level ranges rejected",
        temporal_bounds="stage_to<event supplies exclusive lower bound; stage_from>=event supplies inclusive upper bound; contradictory time bounds rejected",
        missing_predictions="right-censored after forcing end; minimum distance to an observed upper bound is scored in days; right-only compatible missing events remain explicit",
        stability_checks=["same one-parameter fit by calibration year", "leave-one-calibration-location-out"],
        prior_development_exposure=True, untouched_test=False,
        original_membership_frozen=True, observed_stage_assimilated_into_disease=False,
        input_and_source_sha256=hashes)
    freeze_json(DEST / "configuration_before_calibration.json", contract)
    weather, calendars = pd.read_parquet(paths["weather"]), pd.read_csv(paths["calendars"])
    phenology = json.loads(paths["phenology"].read_text())
    membership = pd.read_csv(paths["basf_membership"])
    raw_basf = pd.read_csv(paths["basf"], usecols=lambda c: c in STAGE_COLUMNS,
                           dtype={"physical_unit": "string"})
    basf = project_stage_rows(raw_basf, "BASF")
    basf, basf_excluded = restrict_membership(basf, membership)
    calibration = calibration_stage_rows(basf)
    meta, accumulation, thresholds, stages, rejected = prepare_group(
        calibration, membership, calendars, weather, phenology)
    constraints = make_constraints(stages)
    constraints.to_csv(DEST / "calibration_constraints_before_fitting.csv", index=False)
    meta.to_csv(DEST / "calibration_input_membership.csv", index=False)
    np.savez_compressed(DEST / "calibration_development_inputs.npz", accumulation=accumulation)
    q, curve, field_losses = fit_grid(constraints, accumulation, meta, thresholds)
    curve.to_csv(DEST / "calibration_q_grid.csv", index=False)
    field_losses.rename_axis("q").to_csv(DEST / "calibration_field_loss_by_q.csv")
    stability = stability_checks(field_losses, constraints)
    stability.to_csv(DEST / "calibration_q_stability.csv", index=False)
    best = float(curve.loss_days.min())
    plateau = curve.loc[np.isclose(curve.loss_days, best, atol=1e-12, rtol=0), "q"]
    near = curve.loc[curve.loss_days.le(best + 1.), "q"]
    selected = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), selected_q=q,
        calibration_loss_days=best, exact_minimum_q_range=[float(plateau.min()), float(plateau.max())],
        within_one_day_loss_q_range=[float(near.min()), float(near.max())],
        thresholds=thresholds, fitted_parameter_count=1, evaluation_stages_used_to_select_q=False,
        disease_values_used_to_select_q=False, parameter_applied_to_disease_model=False,
        q_calibration_grid_sha256=sha(DEST / "calibration_q_grid.csv"))
    freeze_json(DEST / "frozen_q_before_evaluation.json", selected)
    frozen_hash = sha(DEST / "frozen_q_before_evaluation.json")
    print(f"Frozen stage-only q={q:.4f}; calibration distance={best:.4f} days", flush=True)

    strict = set(pd.read_csv(paths["strict_registry"], usecols=["source_unit"],
                            dtype={"source_unit": "string"}).source_unit.astype(str))
    external_membership = pd.read_csv(paths["corteva_membership"])
    strict_membership = external_membership.loc[external_membership.source_unit.astype(str).isin(strict)].copy()
    external = project_stage_rows(pd.read_csv(paths["corteva"], usecols=lambda c: c in STAGE_COLUMNS,
        dtype={"physical_unit": "string"}), "Corteva", strict_units=strict)
    external, external_excluded = restrict_membership(external, strict_membership)
    groups = [("calibration", calibration, membership, (meta, accumulation, thresholds, stages, rejected)),
        ("reused_development_2019", basf.loc[basf.season_year.eq(2019)], membership, None),
        ("reused_external_strict", external, strict_membership, None)]
    scores, all_stages, all_constraints, flags, exclusions, inputs, predictions = [], [], [], [], [], [], []
    for partition, rows, members, prepared in groups:
        meta, accumulation, thresholds, stages, rejected = prepared or prepare_group(
            rows, members, calendars, weather, phenology)
        meta = meta.copy()
        meta["partition"] = partition
        inputs.append(meta)
        if prepared is None:
            np.savez_compressed(DEST / f"{partition}_development_inputs.npz", accumulation=accumulation)
        cons = make_constraints(stages)
        all_constraints.append(cons)
        all_stages.append(stages)
        flags.append(make_constraints(stages, events=[37, 39]))
        exclusions.append(rejected)
        for model, multiplier in [("copied_tpv_q1", 1.), ("stage_calibrated_q", q)]:
            pred = first_event_dates(accumulation, meta, thresholds, multiplier)
            pred["model"], pred["partition"] = model, partition
            predictions.append(pred)
            scored = score_constraints(cons, pred.drop(columns="partition"))
            scores.append(scored)
    scores = pd.concat(scores, ignore_index=True)
    metrics = metric_summary(scores)
    normalized = pd.concat(all_stages, ignore_index=True)
    constraints = pd.concat(all_constraints, ignore_index=True)
    flag_constraints = pd.concat(flags, ignore_index=True)
    pd.concat(inputs).to_csv(DEST / "original_field_input_membership.csv", index=False)
    pd.concat(predictions).to_csv(DEST / "frozen_event_predictions.csv", index=False)
    normalized.to_csv(DEST / "normalized_field_date_stages.csv", index=False)
    constraints.to_csv(DEST / "event_constraints.csv", index=False)
    flag_constraints.to_csv(DEST / "flag_stage_constraints.csv", index=False)
    pd.concat(exclusions).to_csv(DEST / "rejected_stage_rows.csv", index=False)
    pd.concat([basf_excluded, external_excluded]).to_csv(DEST / "original_membership_excluded_stage_rows.csv", index=False)
    scores.to_csv(DEST / "censoring_compatibility_and_distances.csv", index=False)
    metrics.to_csv(DEST / "event_censoring_metrics.csv", index=False)
    counts = []
    for partition, group in constraints.groupby("partition"):
        for event in EVENTS:
            part = group.loc[group.event.eq(event)]
            counts.append(dict(partition=partition, event=event, n_fields_with_valid_stages=part.field_id.nunique(),
                n_eligible_constraints=int(part.eligible_constraint.sum()),
                n_genuinely_bracketed=int(part.genuinely_bracketed.sum()),
                n_left_censored=int((part.eligible_constraint & part.censoring.eq("left")).sum()),
                n_right_censored=int((part.eligible_constraint & part.censoring.eq("right")).sum()),
                n_temporal_conflicts=int(part.reason.eq("temporally_inconsistent_event_bounds").sum()),
                n_unconstrained=int(part.reason.eq("no_event_bound").sum())))
    pd.DataFrame(counts).to_csv(DEST / "event_constraint_counts.csv", index=False)
    flag_counts = flag_constraints.groupby(["partition", "event"]).agg(
        n_fields_with_valid_stages=("field_id", "nunique"), n_eligible_constraints=("eligible_constraint", "sum"),
        n_genuinely_bracketed=("genuinely_bracketed", "sum"),
        n_left_censored=("censoring", lambda c: int(c.eq("left").sum())),
        n_right_censored=("censoring", lambda c: int(c.eq("right").sum()))).reset_index()
    flag_counts.to_csv(DEST / "flag_stage_constraint_counts.csv", index=False)
    verify_dependencies(hashes)
    if sha(DEST / "frozen_q_before_evaluation.json") != frozen_hash:
        raise ValueError("Frozen q changed during reused stage evaluation")
    freeze_json(DEST / "receipt.json", dict(status="complete", selected_q=q,
        input_and_source_sha256=hashes, frozen_q_sha256=frozen_hash,
        original_fields=dict(calibration=28, reused_development_2019=45, reused_external_strict=143),
        thresholds_unchanged=True, model_sources_unchanged=True, disease_values_read=False,
        numeric_bbch_interpolation=False, observed_stages_assimilated_into_disease=False,
        prior_development_exposure=True, untouched_test_evaluated=False,
        model_q_changed=False, n_scored_field_events=len(scores)))
    print(metrics.loc[metrics.event.eq("all")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
