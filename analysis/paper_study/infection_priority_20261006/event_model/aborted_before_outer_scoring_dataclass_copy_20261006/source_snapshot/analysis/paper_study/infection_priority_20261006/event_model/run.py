"""Finite retrospective nested evaluation of effective STB infection events.

Run from the workspace with .venv/bin/python -m
analysis.paper_study.infection_priority_20261006.event_model.run. All new
artifacts remain in this directory; prior archives and deployment are retained.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paper_study.calibrate_seasonal import sha
from analysis.paper_study.structural_evaluation.membership import (
    spatial_folds, chronological_folds, inner_folds, guard_nested_membership,
    guard_outer_membership,
)
from analysis.paper_study.structural_evaluation.run import (
    DEFAULT_PATHS, SEED, VALIDATION_LABEL, ROOT, load_inputs, subset_fields,
    estimate_training_rain_rate, freeze_membership_plan, write_frozen_json,
    save_frozen_csv, _save_frame, verify_dependencies,
)
from model.seasonal_septoria.infection_events import (
    EventParameters, simulate_events, symptom_brackets, onset_distance,
    equal_hierarchy_weights,
)
from model.seasonal_septoria.structural import canopy_host
from model.seasonal_septoria.wetness import duration_exposure
from model.seasonal_septoria.publication_model import predict_publication_model

DESTINATION = Path(__file__).resolve().parent
ORIGINAL = ROOT / "analysis/paper_study/nested_canopy_daily_20261006"
EVENT_MODEL = "duration_dose_event"
HOST_MODEL = "matched_phenology_only"
PHENOLOGY_MODEL = "independently_selected_phenology_only"
V6_MODEL = "frozen_v6_full_data_descriptive"
SIGN_COLUMNS = ["global_target_index", "field_id", "field_index", "site_id", "source",
    "dataset_id", "season_year", "coordinate_year", "endpoint_series", "leaf_index",
    "date", "day_index", "value", "metric"]


def candidate_grid():
    return [dict(name=f"flag{flag:g}_dose{dose:g}_gap{gap}_delay{delay}",
        host_parameters=dict(flag_fraction=flag, leaf_interval=120., expansion_units=100.),
        parameters=asdict(EventParameters(dose, gap, float(delay))))
        for flag in (.3, .6, .9) for dose in (.1, .5, 1.) for gap in (1, 3)
        for delay in (15, 20, 25, 30)]


def phenology_grid():
    return [dict(name=f"phenology_flag{flag:g}_delay{delay}",
        host_parameters=dict(flag_fraction=flag, leaf_interval=120., expansion_units=100.),
        parameters=asdict(EventParameters(.1, 1, float(delay))))
        for flag in (.3, .6, .9) for delay in (15, 20, 25, 30)]


def assert_same_target_membership(fresh, stored):
    keys = ["global_target_index", "field_id", "endpoint_series", "date"]
    pd.testing.assert_frame_equal(fresh[keys], stored[keys], check_dtype=False)


def perturb_positive_magnitudes(targets):
    changed = targets.copy()
    changed["value"] = changed.value.astype(float)
    positive = changed.value.gt(0)
    changed.loc[positive, "value"] = np.linspace(.00001, 100., int(positive.sum()))
    return changed


def _finite(value):
    return None if not np.isfinite(value) else float(value)


def rank_candidates(brackets, candidate_onsets, forcing_end_days):
    """Select on pooled two-sided Top3 brackets with fixed-order tie breaking."""
    primary = brackets.censoring.eq("two_sided").to_numpy()
    if not primary.any():
        raise ValueError("Informative two-sided symptom brackets are required for selection.")
    weights = equal_hierarchy_weights(brackets.loc[primary])
    records = []
    for order, (name, days) in enumerate(candidate_onsets.items()):
        distance = onset_distance(brackets, days, forcing_end_days)
        record = dict(candidate=name, candidate_order=order,
            primary_distance_days=float(weights @ distance[primary]),
            primary_compatible_fraction=float(weights @ (distance[primary] == 0)),
            informative_two_sided_count=int(primary.sum()),
            missed_positive_count=int(((days < 0) & brackets.observed_positive.to_numpy()).sum()))
        for censoring in ("left", "right"):
            selected = brackets.censoring.eq(censoring).to_numpy()
            record[f"{censoring}_count"] = int(selected.sum())
            record[f"{censoring}_distance_days"] = (float(
                equal_hierarchy_weights(brackets.loc[selected]) @ distance[selected]) if selected.any() else None)
        records.append(record)
    return sorted(records, key=lambda x: (x["primary_distance_days"], x["candidate_order"]))


def predict_event_record(data, accumulation, thresholds, fitted, *, phenology_only=False):
    if data.targets.value.notna().any():
        raise ValueError("Prediction disease values must be redacted.")
    active = canopy_host(accumulation, data.temperature, thresholds,
        **fitted["host_parameters"])[0]
    # Field-specific forcing ends are respected even though the arrays are padded.
    lengths = data.metadata.set_index("field_index").forcing_days.to_dict()
    for field, length in lengths.items():
        active[int(field), int(length):] = False
    exposure = duration_exposure(data.temperature, data.maximum_temperature,
        data.humidity, data.rain, fitted["weather_preprocessing"]["rain_rate_mm_hour"])["exposure"]
    return simulate_events(data.temperature, exposure, active,
        EventParameters(**fitted["parameters"]), phenology_only=phenology_only)


def detection_metrics(frame):
    observed = frame.observed_positive.to_numpy(bool)
    predicted = frame.predicted_positive.to_numpy(bool)
    weights = equal_hierarchy_weights(frame)
    tp, fp = observed & predicted, ~observed & predicted
    tn, fn = ~observed & ~predicted, observed & ~predicted
    positive_mass, negative_mass = float(weights @ observed), float(weights @ ~observed)
    return dict(n=len(frame), positive_count=int(observed.sum()), negative_count=int((~observed).sum()),
        true_positive=int(tp.sum()), false_positive=int(fp.sum()), true_negative=int(tn.sum()),
        false_negative=int(fn.sum()), weighted_accuracy=float(weights @ (observed == predicted)),
        sensitivity=float(weights @ tp)/positive_mass if positive_mass else None,
        specificity=float(weights @ tn)/negative_mass if negative_mass else None,
        false_positive_rate=float(weights @ fp)/negative_mass if negative_mass else None,
        specificity_identifiable=bool(negative_mass > 0))


def _training_record(destination, training, weather, stage, identity, grid):
    preprocessing = dict(weather_operator="duration_proxy", **estimate_training_rain_rate(training.metadata, weather))
    record = dict(status="fit_complete", stage=stage, identity=identity,
        weather_preprocessing=preprocessing, finite_candidates=grid,
        training_fields=training.metadata.field_id.astype(str).tolist(),
        training_sites=sorted(training.targets.site_id.unique().tolist()),
        training_target_count=len(training.targets), sign_only_targets=True,
        training_positive_count=int(training.targets.value.gt(0).sum()),
        calibration_labels_sha256=hashlib.sha256(training.targets[SIGN_COLUMNS].to_csv(index=False).encode()).hexdigest(),
        disease_percentage_fit=False, evaluation_observations_used=False,
        prior_outcome_exposure=True, validation_label=VALIDATION_LABEL)
    write_frozen_json(destination, record)
    return preprocessing


def _fitted(candidate, preprocessing):
    return dict(**candidate, weather_preprocessing=preprocessing,
        validation_label=VALIDATION_LABEL, conditional_on_inoculum_presence=True,
        calibrated_occurrence_probability=False, disease_percentage_fit=False)


def _bracket_end(brackets, data):
    lengths = data.metadata.set_index("field_index").forcing_days
    return brackets.field_index.map(lengths).to_numpy(int)


def select_grid(destination, data, accumulation, thresholds, weather, partitions, grid, identity, *, stage,
                phenology_only=False):
    """Every validation simulation receives physically isolated, redacted data."""
    pooled_brackets, pooled_days = [], {item["name"]: [] for item in grid}
    candidate_predictions = []
    for partition in partitions:
        training = subset_fields(data, partition.train)
        validation = subset_fields(data, partition.test, redact_values=True)
        truth = subset_fields(data, partition.test)
        brackets = symptom_brackets(truth.targets)
        end = _bracket_end(brackets, truth)
        brackets["forcing_end_day"] = end
        brackets["inner_fold"] = partition.name
        # A second copy changes only positive magnitudes. Selection labels must
        # remain identical; values below zero and missing values remain invalid.
        changed = perturb_positive_magnitudes(truth.targets)
        pd.testing.assert_frame_equal(symptom_brackets(truth.targets), symptom_brackets(changed))
        record_path = destination / f"{partition.name}_training_record.json"
        preprocessing = _training_record(record_path, training, weather,
            dict(stage=stage, fold=partition.name), identity, grid)
        fields = np.sort(data.targets.iloc[partition.test].field_index.unique())
        # Weather inference is fixed by the train subset for all 72 candidates.
        exposure = duration_exposure(validation.temperature, validation.maximum_temperature,
            validation.humidity, validation.rain, preprocessing["rain_rate_mm_hour"])["exposure"]
        hosts = {flag: canopy_host(accumulation[fields], validation.temperature, thresholds,
            flag_fraction=flag, leaf_interval=120., expansion_units=100.)[0] for flag in (.3, .6, .9)}
        for active in hosts.values():
            for meta in validation.metadata.itertuples():
                active[meta.field_index, meta.forcing_days:] = False
        field, leaf = brackets.field_index.to_numpy(int), brackets.leaf_index.to_numpy(int)
        for candidate in grid:
            trajectory = simulate_events(validation.temperature, exposure,
                hosts[candidate["host_parameters"]["flag_fraction"]], EventParameters(**candidate["parameters"]),
                phenology_only=phenology_only)
            days = trajectory.symptom_day[field, leaf]
            pooled_days[candidate["name"]].append(days)
            part = brackets.copy()
            part["candidate"] = candidate["name"]
            part["predicted_symptom_day"] = days
            part["predicted_effective_infection_day"] = trajectory.infection_day[field, leaf]
            part["onset_distance_days"] = onset_distance(brackets, days, end)
            part["training_record_sha256"] = sha(record_path)
            candidate_predictions.append(part)
        pooled_brackets.append(brackets)
    pooled = pd.concat(pooled_brackets, ignore_index=True)
    onsets = {name: np.concatenate(items) for name, items in pooled_days.items()}
    ranking = rank_candidates(pooled, onsets, pooled.forcing_end_day.to_numpy(int))
    chosen = next(item for item in grid if item["name"] == ranking[0]["candidate"])
    record = dict(selected=chosen, ranking=ranking,
        selection_criterion="pooled inner two-sided Top3 first-symptom bracket distance",
        weighting="equal coordinate-year; equal fields within coordinate-year; equal leaves within field",
        tie_breaking="fixed candidate enumeration order", missed_positive_minimum_penalty_days=30.,
        positive_magnitude_invariance=True, outer_test_outcomes_used=False,
        inner_fold_count=len(partitions), phenology_only=phenology_only, validation_label=VALIDATION_LABEL)
    write_frozen_json(destination / "selection.json", record)
    _save_frame(destination / "all_inner_candidate_bracket_predictions.parquet", pd.concat(candidate_predictions, ignore_index=True))
    return record


def assessment_predictions(truth, trajectory, model, fold, kind):
    part = truth.targets[SIGN_COLUMNS].copy()
    field, day, leaf = [part[column].to_numpy(int) for column in ("field_index", "day_index", "leaf_index")]
    part["observed_positive"] = part.value.gt(0)
    part = part.drop(columns="value")
    part["predicted_positive"] = trajectory.symptomatic[field, day, leaf]
    part["predicted_symptom_day"] = trajectory.symptom_day[field, leaf]
    part["predicted_effective_infection_day"] = trajectory.infection_day[field, leaf]
    part["model"], part["outer_fold"], part["evaluation_kind"] = model, fold, kind
    part["validation_label"] = VALIDATION_LABEL
    return part


def bracket_predictions(truth, trajectory, model, fold, kind):
    part = symptom_brackets(truth.targets, upper_three=False)
    field, leaf = part.field_index.to_numpy(int), part.leaf_index.to_numpy(int)
    part["forcing_end_day"] = _bracket_end(part, truth)
    part["predicted_symptom_day"] = trajectory.symptom_day[field, leaf]
    part["predicted_effective_infection_day"] = trajectory.infection_day[field, leaf]
    part["onset_distance_days"] = onset_distance(part, part.predicted_symptom_day.to_numpy(), part.forcing_end_day.to_numpy())
    part["onset_compatible"] = part.onset_distance_days.eq(0)
    part["predicted_positive"] = part.predicted_symptom_day.ge(0) & part.predicted_symptom_day.le(part.last_assessment_day)
    part["model"], part["outer_fold"], part["evaluation_kind"] = model, fold, kind
    part["validation_label"] = VALIDATION_LABEL
    return part


def daily_and_events(truth, trajectory, model, fold, kind):
    daily, events = [], []
    for meta in truth.metadata.itertuples():
        field, length = int(meta.field_index), int(meta.forcing_days)
        dates = pd.date_range(meta.sowing_date, periods=length)
        for leaf in range(7):
            infection, symptom = int(trajectory.infection_day[field, leaf]), int(trajectory.symptom_day[field, leaf])
            events.append(dict(field_id=meta.field_id, site_id=meta.site_id, season_year=meta.season_year,
                leaf_index=leaf, predicted_effective_infection_day=infection, predicted_symptom_day=symptom,
                predicted_effective_infection_date=None if infection < 0 else str(dates[infection-1].date()),
                predicted_symptom_date=None if symptom < 0 else str(dates[symptom-1].date()),
                model=model, outer_fold=fold, evaluation_kind=kind, forcing_days=length))
            if leaf < 3:
                daily.append(pd.DataFrame(dict(field_id=meta.field_id, site_id=meta.site_id,
                    season_year=meta.season_year, leaf_index=leaf, date=dates,
                    day_index=np.arange(1, length+1),
                    effective_infected=trajectory.infected[field, 1:length+1, leaf],
                    symptomatic=trajectory.symptomatic[field, 1:length+1, leaf],
                    model=model, outer_fold=fold, evaluation_kind=kind)))
    return pd.concat(daily, ignore_index=True), pd.DataFrame(events)


def summarize(assessments, brackets):
    detection, onset, occurrence = [], [], []
    for kind, kind_group in assessments.groupby("evaluation_kind"):
        for model, model_group in kind_group.groupby("model"):
            sources = ["pooled", *sorted(model_group.source.unique())]
            for source in sources:
                selected = model_group if source == "pooled" else model_group.loc[model_group.source.eq(source)]
                selected_brackets = brackets.loc[brackets.evaluation_kind.eq(kind) & brackets.model.eq(model)]
                if source != "pooled":
                    selected_brackets = selected_brackets.loc[selected_brackets.source.eq(source)]
                for scope in ("top3", "all_ordinal_leaves"):
                    part = selected.loc[selected.leaf_index.lt(3)] if scope == "top3" else selected
                    bpart = selected_brackets.loc[selected_brackets.leaf_index.lt(3)] if scope == "top3" else selected_brackets
                    base = dict(evaluation_kind=kind, model=model, source=source, leaf_scope=scope)
                    detection.append(dict(**base, **detection_metrics(part)))
                    for censoring, censored in bpart.groupby("censoring"):
                        weight = equal_hierarchy_weights(censored)
                        onset.append(dict(**base, censoring=censoring, n=len(censored),
                            distance_days=float(weight @ censored.onset_distance_days.to_numpy()),
                            compatible_fraction=float(weight @ censored.onset_compatible.to_numpy()),
                            missed_positive_count=int((censored.observed_positive & censored.predicted_symptom_day.lt(0)).sum()),
                            persistence_violation_count=int(censored.persistence_violations.sum())))
                    occurrence.append(dict(**base, occurrence_level="leaf_observed_window", **detection_metrics(bpart)))
                    field_part = bpart.groupby(["coordinate_year", "field_id", "outer_fold"], as_index=False).agg(
                        observed_positive=("observed_positive", "any"), predicted_positive=("predicted_positive", "any"))
                    field_part["leaf_index"] = 0
                    occurrence.append(dict(**base, occurrence_level="field_observed_window", **detection_metrics(field_part)))
    return pd.DataFrame(detection), pd.DataFrame(onset), pd.DataFrame(occurrence)


def independent_checks(assessments, brackets, detection):
    """Recompute bracket errors and raw confusion counts without model helpers."""
    errors = []
    for row in brackets.itertuples():
        day = int(row.predicted_symptom_day)
        if day == -1:
            expected = max(30., row.forcing_end_day+1-row.upper_day) if pd.notna(row.upper_day) else 0.
        elif row.censoring == "left":
            expected = max(0., day-row.upper_day)
        elif row.censoring == "right":
            expected = max(0., row.lower_day+1-day)
        else:
            expected = max(0., row.lower_day+1-day, day-row.upper_day)
        errors.append(abs(expected-row.onset_distance_days))
    for metric in detection.itertuples():
        part = assessments.loc[assessments.evaluation_kind.eq(metric.evaluation_kind) & assessments.model.eq(metric.model)]
        if metric.source != "pooled":
            part = part.loc[part.source.eq(metric.source)]
        if metric.leaf_scope == "top3":
            part = part.loc[part.leaf_index.lt(3)]
        counts = [sum(bool(o) and bool(p) for o, p in zip(part.observed_positive, part.predicted_positive)),
            sum(not bool(o) and bool(p) for o, p in zip(part.observed_positive, part.predicted_positive)),
            sum(not bool(o) and not bool(p) for o, p in zip(part.observed_positive, part.predicted_positive)),
            sum(bool(o) and not bool(p) for o, p in zip(part.observed_positive, part.predicted_positive))]
        if counts != [metric.true_positive, metric.false_positive, metric.true_negative, metric.false_negative]:
            raise AssertionError("Independent confusion counts disagree.")
    if max(errors, default=0.) != 0.:
        raise AssertionError("Independent onset arithmetic disagrees.")
    return dict(status="passed", bracket_rows_checked=len(brackets), maximum_distance_error_days=max(errors, default=0.),
        confusion_metric_rows_checked=len(detection), no_midpoint_imputation=True)


def _v6_descriptive(data, accumulation, thresholds):
    redacted = subset_fields(data, np.arange(len(data.targets)), redact_values=True)
    result = predict_publication_model(redacted, accumulation, thresholds, return_daily=True)
    daily = result["daily_percent"]
    # Exact sign cutoff matches the signs supplied by source observations. V6
    # is already fit to these outcomes and has no independent event validation.
    onset = np.full((len(data.metadata), 8), -1, int)
    for meta in data.metadata.itertuples():
        for leaf in range(8):
            found = np.flatnonzero(daily[meta.field_index, 1:meta.forcing_days+1, leaf] > 0.)
            if len(found):
                onset[meta.field_index, leaf] = int(found[0]+1)
    from model.seasonal_septoria.infection_events import EventTrajectory
    boundary = np.arange(data.temperature.shape[1]+1)[None, :, None]
    trajectory = EventTrajectory(np.full_like(onset, -1), onset,
        np.zeros_like(daily, bool), (onset[:, None, :] >= 0) & (boundary >= onset[:, None, :]))
    a = assessment_predictions(data, trajectory, V6_MODEL, "full_data", "full_data_descriptive_reuse")
    b = bracket_predictions(data, trajectory, V6_MODEL, "full_data", "full_data_descriptive_reuse")
    a["predicted_effective_infection_day"] = np.nan
    b["predicted_effective_infection_day"] = np.nan
    return a, b


def main():
    destination = DESTINATION
    if (destination / "configuration_before_fitting.json").exists():
        raise FileExistsError("The event experiment is immutable; use a new dated directory for another run.")
    grid = candidate_grid()
    host_grid = phenology_grid()
    dependencies = [Path(__file__), ROOT / "tests/test_infection_events.py", *DEFAULT_PATHS.values(),
        ORIGINAL / "nested_membership_before_fitting.csv", ORIGINAL / "all_source_target_membership.parquet",
        ROOT / "model/seasonal_septoria/publication_model.json",
        ROOT / "analysis/paper_study/run_structural_canopy.py",
        *sorted((ROOT / "analysis/paper_study/structural_evaluation").glob("*.py")),
        *sorted((ROOT / "model/seasonal_septoria").glob("*.py")),
        *sorted((ROOT / "process_model").glob("*.py")),
        *sorted((ROOT / "process_model/_support").glob("*.py"))]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in dependencies}
    configuration = dict(validation_label=VALIDATION_LABEL, registered_utc=datetime.now(timezone.utc).isoformat(),
        prior_historical_outcome_exposure=True, untouched_test=False, directly_observed_infection_dates=False,
        candidates=grid, candidate_count=len(grid), independently_selected_phenology_candidates=host_grid,
        independently_selected_phenology_candidate_count=len(host_grid),
        random_seed=SEED, original_outer_membership_archive=str(ORIGINAL.relative_to(ROOT)),
        weather_operator="duration_proxy", rainfall_intensity="train-only median warm positive-rain precipitation / rain_hours",
        infection_event="first active-leaf daily cumulative effective exposure reaching positive dose threshold",
        dry_gap="reset unconsumed dose after declared consecutive zero-exposure days",
        symptom_delay="after infection-day boundary, accumulate max(Tmean,0)/18 until delay_reference_days",
        host_leaf_interval=120., host_expansion_units=100., first_symptom_persistence_assumption=True,
        positive_label="source disease percentage > 0; magnitudes discarded",
        selection="pooled inner Top3 two-sided onset bracket calendar-day distance",
        missed_positive_minimum_penalty_days=30., deterministic_conditional_on_inoculum=True,
        calibrated_occurrence_probability=False, actual_infection_date_validation=False,
        phenology_comparator="independently selected host and delay on the same inner objective; immediate host-availability event",
        additional_matched_phenology_comparator="event model selected host and delay; immediate host-availability event",
        v6_comparator="frozen full-data fit, descriptive reuse only; exact >0 score cutoff; no independent test",
        full_data_refit="saved after completed outer evaluation; not an evaluation prediction or deployment replacement",
        input_and_source_sha256=hashes)
    write_frozen_json(destination / "configuration_before_fitting.json", configuration)
    for path in dependencies:
        if path.suffix in (".py", ".json"):
            snapshot = destination / "source_snapshot" / path.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
    data, weather, accumulation, thresholds = load_inputs(DEFAULT_PATHS)
    # All subsequent model fitting/scoring sees signs only. Source provenance
    # and numeric severity remain intact in the original immutable archives.
    original_targets = pd.read_parquet(ORIGINAL / "all_source_target_membership.parquet")
    assert_same_target_membership(data.targets, original_targets)
    data.targets = data.targets[SIGN_COLUMNS].copy()
    data.targets["value"] = data.targets.value.gt(0).astype(int)
    outers = spatial_folds(data.targets, seed=SEED) + chronological_folds(data.targets)
    plan = freeze_membership_plan(destination, data.targets, outers)
    pd.testing.assert_frame_equal(pd.read_csv(destination / "nested_membership_before_fitting.csv"),
        pd.read_csv(ORIGINAL / "nested_membership_before_fitting.csv"))
    _save_frame(destination / "sign_only_source_targets.parquet", data.targets)
    _save_frame(destination / "all_source_field_seasons.parquet", data.metadata)
    source_counts = []
    for scope in ("top3", "all_ordinal_leaves"):
        selected = data.targets.loc[data.targets.leaf_index.lt(3)] if scope == "top3" else data.targets
        for source, group in selected.groupby("source"):
            fields = group.groupby("field_id").value.max()
            brackets = symptom_brackets(group, upper_three=scope == "top3")
            source_counts.append(dict(source=source, leaf_scope=scope, field_windows=len(fields),
                positive_field_windows=int(fields.sum()), zero_only_field_windows=int(fields.eq(0).sum()),
                leaf_windows=len(brackets), positive_leaf_windows=int(brackets.observed_positive.sum()),
                zero_only_leaf_windows=int((~brackets.observed_positive).sum()),
                assessment_count=len(group), positive_assessment_count=int(group.value.sum()),
                zero_assessment_count=int(group.value.eq(0).sum()),
                two_sided_brackets=int(brackets.censoring.eq("two_sided").sum()),
                left_censored_brackets=int(brackets.censoring.eq("left").sum()),
                right_censored_brackets=int(brackets.censoring.eq("right").sum()),
                later_positive_to_zero_returns=int(brackets.persistence_violations.sum())))
    save_frozen_csv(destination / "source_positivity_and_censoring_counts.csv", pd.DataFrame(source_counts))
    identity = dict(configuration_sha256=sha(destination / "configuration_before_fitting.json"),
        membership_sha256=sha(destination / "nested_membership_before_fitting.csv"),
        sign_targets_sha256=sha(destination / "sign_only_source_targets.parquet"))
    verify_dependencies(hashes)
    frozen = {}
    for outer in outers:
        guard_outer_membership(data.targets, outer)
        for inner in plan[outer.name]:
            guard_nested_membership(data.targets, outer, inner)
        folder = destination / "outer_folds" / outer.name
        folder.mkdir(parents=True, exist_ok=True)
        frozen[outer.name] = {}
        for model, settings, host_only in ((EVENT_MODEL, grid, False), (PHENOLOGY_MODEL, host_grid, True)):
            family_folder = folder / model
            family_folder.mkdir()
            selection = select_grid(family_folder, data, accumulation, thresholds, weather,
                plan[outer.name], settings, identity, stage="inner_selection", phenology_only=host_only)
            training = subset_fields(data, outer.train)
            preprocessing = _training_record(family_folder / "outer_training_record.json", training,
                weather, dict(stage="outer_refit", outer_fold=outer.name, model=model), identity, [selection["selected"]])
            fitted = _fitted(selection["selected"], preprocessing)
            fitted["phenology_only"] = host_only
            path = family_folder / "frozen_outer_fit.json"
            write_frozen_json(path, dict(status="fit_complete", identity=identity, fitted=fitted,
                selected_candidate=selection["selected"]["name"], selection_sha256=sha(family_folder / "selection.json"),
                outer_test_outcomes_used=False, prior_outcome_exposure=True))
            frozen[outer.name][model] = dict(path=str(path.relative_to(destination)), sha256=sha(path))
            print(f"{outer.name} {model}: {fitted['name']}; inner bracket distance {selection['ranking'][0]['primary_distance_days']:.3f} d", flush=True)
    verify_dependencies(hashes)
    manifest = destination / "all_outer_fits_frozen_before_scoring.json"
    write_frozen_json(manifest, dict(outer_fits=frozen, identity=identity, outer_outcomes_used_to_select=False,
        validation_label=VALIDATION_LABEL, prior_historical_outcome_exposure=True))
    assessment_parts, bracket_parts, event_parts = [], [], []
    for outer in outers:
        validation = subset_fields(data, outer.test, redact_values=True)
        truth = subset_fields(data, outer.test)
        fields = np.sort(data.targets.iloc[outer.test].field_index.unique())
        for model, fit_model, host_only in ((EVENT_MODEL, EVENT_MODEL, False),
                (PHENOLOGY_MODEL, PHENOLOGY_MODEL, True), (HOST_MODEL, EVENT_MODEL, True)):
            fit_record = frozen[outer.name][fit_model]
            path = destination / fit_record["path"]
            if sha(path) != fit_record["sha256"]:
                raise ValueError("An outer fit changed after freezing.")
            fitted = json.loads(path.read_text())["fitted"]
            trajectory = predict_event_record(validation, accumulation[fields], thresholds, fitted, phenology_only=host_only)
            # Repeated redacted simulation proves predictions carry no disease
            # magnitude or observation-stage input through the interface.
            second = predict_event_record(replace(validation, targets=validation.targets.drop(columns="metric")),
                accumulation[fields], thresholds, fitted, phenology_only=host_only)
            np.testing.assert_array_equal(trajectory.symptom_day, second.symptom_day)
            assessment_parts.append(assessment_predictions(truth, trajectory, model, outer.name, outer.kind))
            bracket_parts.append(bracket_predictions(truth, trajectory, model, outer.name, outer.kind))
            daily, events = daily_and_events(truth, trajectory, model, outer.name, outer.kind)
            _save_frame(destination / "outer_folds" / outer.name / f"{model}_daily_top3.parquet", daily)
            event_parts.append(events)
    assessments = pd.concat(assessment_parts, ignore_index=True)
    brackets = pd.concat(bracket_parts, ignore_index=True)
    events = pd.concat(event_parts, ignore_index=True)
    _save_frame(destination / "out_of_fold_assessment_sign_predictions.parquet", assessments)
    _save_frame(destination / "out_of_fold_first_symptom_brackets.parquet", brackets)
    _save_frame(destination / "out_of_fold_daily_event_dates.parquet", events)
    v6_assessments, v6_brackets = _v6_descriptive(data, accumulation, thresholds)
    _save_frame(destination / "frozen_v6_descriptive_assessment_sign_predictions.parquet", v6_assessments)
    _save_frame(destination / "frozen_v6_descriptive_first_symptom_brackets.parquet", v6_brackets)
    combined_assessments = pd.concat([assessments, v6_assessments], ignore_index=True)
    combined_brackets = pd.concat([brackets, v6_brackets], ignore_index=True)
    detection, onset, occurrence = summarize(combined_assessments, combined_brackets)
    for name, frame in [("assessment_detection_metrics", detection), ("onset_bracket_metrics", onset),
                        ("observed_window_occurrence_metrics", occurrence)]:
        save_frozen_csv(destination / f"{name}.csv", frame)
        _save_frame(destination / f"{name}.parquet", frame)
    incremental = []
    for kind, group in brackets.loc[brackets.leaf_index.lt(3)].groupby("evaluation_kind"):
        event = group.loc[group.model.eq(EVENT_MODEL)]
        for comparator in (PHENOLOGY_MODEL, HOST_MODEL):
            host = group.loc[group.model.eq(comparator)]
            matched = event.merge(host[["outer_fold", "field_id", "endpoint_series", "onset_distance_days", "onset_compatible"]],
                on=["outer_fold", "field_id", "endpoint_series"], validate="one_to_one", suffixes=("", "_host"))
            for censoring, part in matched.groupby("censoring"):
                weight = equal_hierarchy_weights(part)
                incremental.append(dict(evaluation_kind=kind, comparator=comparator, censoring=censoring, n=len(part),
                    event_minus_phenology_distance_days=float(weight @ (part.onset_distance_days-part.onset_distance_days_host)),
                    event_minus_phenology_compatible_fraction=float(weight @ (part.onset_compatible.astype(float)-part.onset_compatible_host.astype(float)))))
    save_frozen_csv(destination / "matched_phenology_incremental_value.csv", pd.DataFrame(incremental))
    checks = independent_checks(combined_assessments, combined_brackets, detection)
    checks.update(exact_original_outer_membership=True, selected_and_predictions_positive_magnitude_invariant=True,
        all_outer_fits_frozen_before_scoring=True, fit_outcome_values_are_signs_only=True,
        physical_train_subsets=True, evaluation_prediction_values_redacted=True)
    write_frozen_json(destination / "independent_arithmetic_checks.json", checks)
    # Full-data selection occurs only after complete outer prediction/scoring.
    full_folder = destination / "full_data_refit"
    full_folder.mkdir()
    full_partitions = spatial_folds(data.targets, n_splits=3, seed=SEED)
    full_selection = select_grid(full_folder, data, accumulation, thresholds, weather,
        full_partitions, grid, identity, stage="post_evaluation_full_data_selection")
    preprocessing = _training_record(full_folder / "full_training_record.json", data, weather,
        dict(stage="post_evaluation_full_data_refit"), identity, [full_selection["selected"]])
    write_frozen_json(destination / "full_data_refit_model.json", dict(schema_version=1,
        model_id="effective_infection_event_v1_20261006", fitted=_fitted(full_selection["selected"], preprocessing),
        status="full_data_refit_after_retrospective_evaluation", evaluation_prediction=False,
        publication_deployment_replaced=False, directly_observed_infection_dates=False,
        conditional_on_inoculum_presence=True, calibrated_occurrence_probability=False,
        selection_sha256=sha(full_folder / "selection.json"), validation_label=VALIDATION_LABEL))
    verify_dependencies(hashes)
    for family_records in frozen.values():
        for record in family_records.values():
            if sha(destination / record["path"]) != record["sha256"]:
                raise ValueError("Frozen outer fits changed during scoring/refit.")
    write_frozen_json(destination / "receipt.json", dict(status="complete", validation_label=VALIDATION_LABEL,
        prior_historical_outcome_exposure=True, untouched_test_evaluated=False,
        directly_observed_infection_date_validation=False, outer_fold_count=len(outers),
        inner_fold_count_per_outer=3, candidate_count=len(grid),
        independently_selected_phenology_candidate_count=len(host_grid),
        out_of_fold_assessment_prediction_count=len(assessments), out_of_fold_bracket_prediction_count=len(brackets),
        frozen_manifest_sha256=sha(manifest), independent_checks_sha256=sha(destination / "independent_arithmetic_checks.json"),
        full_data_refit_saved=True, deployed_model_replaced=False,
        output_sha256={str(path.relative_to(destination)): sha(path) for path in sorted(destination.rglob("*"))
            if path.is_file() and "source_snapshot" not in path.parts and "__pycache__" not in path.parts
            and path.name not in ("run.py", "receipt.json")}))


if __name__ == "__main__":
    main()
