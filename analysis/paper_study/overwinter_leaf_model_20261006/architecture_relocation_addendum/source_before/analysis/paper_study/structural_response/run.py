"""Registered finite observation-response experiment on reused evidence."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from analysis.paper_study.calibrate_seasonal import sha
from analysis.paper_study.refine_seasonal_accuracy import score
from analysis.paper_study.run_empirical_benchmarks import endpoint, paired_intervals
from analysis.paper_study.run_structural_canopy import development_inputs
from analysis.paper_study.structural_evaluation.run import subset_fields, write_frozen_json
from model.seasonal_septoria.field_data import prepare_fields

from .response import (
    HOST_PARAMETERS, INITIATION_BOUNDS, MULTISTARTS, POWER_BOUNDS,
    fit_response, predict_response, response_onsets, same_date_gradients,
)


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
DEFAULT = PACKAGE / "experiment_20261006"
KEYS = ["field_id", "endpoint_series", "date"]
PATHS = dict(
    basf=ROOT / "analysis/paper_study/seasonal_calibration_v1/basf_source_assessments_snapshot.csv",
    corteva=ROOT / "data/paper_study/observations/corteva_external_assessments.csv",
    strict_units=ROOT / "data/paper_study/observations/corteva_location_disjoint_external_units.csv",
    weather=ROOT / "data/paper_study/field_weather_completed/daily_weather.parquet",
    calendars=ROOT / "data/paper_study/wheat_area/trial_point_calendar_scenarios.csv",
    phenology=ROOT / "process_model/parameters/calibration.json",
    protocol=ROOT / "analysis/paper_study/seasonal_calibration_v1/configuration_before_fitting.json",
    original_predictions=ROOT / "analysis/paper_study/structural_canopy_v4_20261006/frozen_evaluation_predictions.parquet",
    structural_selection=ROOT / "analysis/paper_study/structural_canopy_v4_20261006/calibration_only_selection.json",
)


def register_experiment(destination, site_fold, dependencies):
    """Freeze eight candidates, equal hazard bounds and dependency hashes."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    candidates = []
    for delay in [20., 30.]:
        for operator in ["daily_or", "duration_proxy"]:
            for link in ["identity", "nonlinear"]:
                candidates.append(dict(name=f"{operator}_delay{delay:g}_{link}",
                    latent_days=delay, weather_operator=operator, observation_link=link,
                    host_parameters=HOST_PARAMETERS,
                    initiation_bounds=list(INITIATION_BOUNDS),
                    observation_power_bounds=list(POWER_BOUNDS),
                    observation_power_fixed=1. if link == "identity" else None,
                    amplification_fixed=0.))
    config = dict(registered_utc=datetime.now(timezone.utc).isoformat(), candidates=candidates,
        calibration_years=[2017, 2018], site_fold=site_fold, final_loss_weight=.5,
        loss="equal coordinate-years, fields and leaf series; half final, half all assessments",
        selection="minimum original-three-site-fold pooled final-numeric-assessment RMSE; all ordinal leaves",
        deterministic_multistarts=[list(start) for start in MULTISTARTS],
        evaluation_models="selected candidate and identity control with its same delay and operator; original archived reference",
        secondary_endpoints=["all-assessment severity", "upper-three-leaf severity", "same-date adjacent-leaf gradients", "censored onset"],
        observation_response="O(D,b)=100*(1-exp(-(-log(1-D))**b)); O(0)=0, O(1)=100",
        response_interpretation="shared empirical proxy observation link, not identified leaf area or incidence",
        prior_structure_development=True, all_evaluation_sources_previously_examined=True,
        evaluation_label="reused retrospective development evidence", untouched_test=False,
        no_further_tuning_from_evaluation_outcomes=True,
        weather_assumptions=["daily meteorology only", "symmetric temperature cycle for duration proxy",
            "constant within-day vapour pressure", "synthetic RH constrained to supplied daily mean",
            "independent rain/humidity overlap", "rain intensity fitted from each training fold's warm rainy days"],
        source_and_input_sha256={str(Path(path).resolve()): sha(Path(path)) for path in dependencies})
    write_frozen_json(destination / "configuration_before_fitting.json", config)
    return config


def verify_registered_dependencies(configuration):
    for name, digest in configuration["source_and_input_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Registered source/input hash changed: {name}")


def original_calibration_membership(targets, site_fold):
    if not targets.season_year.isin([2017, 2018]).all():
        raise ValueError("Only BASF 2017 and 2018 calibration assessments are eligible.")
    assigned = targets.site_id.map(site_fold)
    if assigned.isna().any() or set(assigned.unique()) != {0, 1, 2}:
        raise ValueError("All three original calibration site folds are required.")
    for field, group in targets.assign(site_fold=assigned).groupby("field_id"):
        if group.site_fold.nunique() != 1:
            raise ValueError(f"A field-season crosses original site folds: {field}")
    for fold in range(3):
        if set(targets.loc[assigned.ne(fold), "site_id"]) & set(targets.loc[assigned.eq(fold), "site_id"]):
            raise ValueError("Original fold membership leaks physical locations.")
    return assigned.to_numpy(int)


def severity_summary(predictions):
    rows = []
    for (partition, model), group in predictions.groupby(["partition", "model"], sort=True):
        for final in [False, True]:
            for upper in [False, True]:
                frame = endpoint(group, final, upper)
                if frame.empty:
                    continue
                rows.append(dict(partition=partition, model=model,
                    endpoint="final_numeric_assessment" if final else "all_assessments",
                    leaf_scope="upper_three" if upper else "all_ordinal_leaves", **score(frame)))
    return pd.DataFrame(rows)


def _subset_development(data, accumulation, rows, redact=True):
    fields = np.sort(data.targets.iloc[rows].field_index.unique()).astype(int)
    return subset_fields(data, rows, redact_values=redact), accumulation[fields].copy()


def _scored_prediction(data, accumulation, thresholds, fitted, rows, model, partition, source):
    redacted, development = _subset_development(data, accumulation, rows)
    forecast, trajectory = predict_response(redacted, development, thresholds, fitted)
    # Restore labels only after forecast computation, for endpoint scoring.
    actual = subset_fields(data, rows)
    scored = actual.targets.copy()
    scored["model"], scored["partition"], scored["source"] = model, partition, source
    scored["predicted_percent"] = forecast
    actual.targets["partition"] = partition
    onsets = response_onsets(actual, trajectory.expressed, fitted["observation_power"], model)
    numerical = dict(model=model, partition=partition, source=source,
        mass_error=float(np.max(np.abs(trajectory.state.sum(axis=-1) - 1.))),
        minimum_state=float(trajectory.state.min()), maximum_state=float(trajectory.state.max()),
        minimum_score=float(forecast.min()), maximum_score=float(forecast.max()),
        raw_state_interpretation_unchanged=True)
    return scored, onsets, numerical


def _parquet(path, frame):
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object", "str"]).columns:
        nonmissing = frame[column].dropna()
        if nonmissing.map(type).nunique() > 1:
            frame[column] = frame[column].map(lambda value: None if pd.isna(value) else str(value)).astype("string")
    if Path(path).exists():
        raise FileExistsError("Frozen predictions already exist.")
    frame.to_parquet(path, index=False)


def _onset_summary(onsets):
    rows = []
    for labels, group in onsets.groupby(["partition", "model", "cutoff_percent", "censoring"], sort=True):
        # Equal coordinate-years, fields and leaf series within this censoring stratum.
        weights = (1 / group.groupby("coordinate_year").field_id.transform("nunique")
            / group.groupby("field_id").endpoint_series.transform("nunique"))
        available = group.delta_days.notna()
        rows.append(dict(partition=labels[0], model=labels[1], cutoff_percent=labels[2], censoring=labels[3],
            n=len(group), coordinate_years=group.coordinate_year.nunique(),
            weighted_compatibility=float(np.average(group.compatible, weights=weights)),
            weighted_missing_prediction_rate=float(np.average(group.missing_prediction, weights=weights)),
            weighted_absolute_interval_distance_days=None if not available.any()
                else float(np.average(group.loc[available, "delta_days"].abs(), weights=weights[available]))))
    return pd.DataFrame(rows)


def _gradient_summaries(predictions):
    pairs, metrics = [], []
    for (partition, model), group in predictions.groupby(["partition", "model"], sort=True):
        gradients = same_date_gradients(group)
        if gradients.empty:
            continue
        gradients["partition"], gradients["model"] = partition, model
        pairs.append(gradients)
        scored = gradients.rename(columns=dict(observed_gradient="value", predicted_gradient="predicted_percent"))
        metrics.append(dict(partition=partition, model=model, **score(scored)))
    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(), pd.DataFrame(metrics)


def _paired_summaries(predictions):
    rows = []
    rng = np.random.default_rng(20261006)
    for partition, group in predictions.groupby("partition", sort=True):
        for comparator in ["matched_identity_control", "reference_v1"]:
            baseline = group.loc[group.model.eq(comparator)]
            if baseline.empty:
                continue
            for final in [False, True]:
                for upper in [False, True]:
                    selected = endpoint(group.loc[group.model.eq("selected_response")], final, upper)
                    control = endpoint(baseline, final, upper)
                    paired = selected.merge(control[KEYS + ["predicted_percent"]], on=KEYS,
                        how="left", validate="one_to_one", suffixes=("", "_comparator"))
                    if len(paired) != len(control) or paired.predicted_percent_comparator.isna().any():
                        raise ValueError("Paired endpoint membership differs.")
                    paired["comparator_percent"] = paired.predicted_percent_comparator
                    clusters = np.array(sorted(paired.coordinate_year.unique()))
                    draws = rng.integers(0, len(clusters), size=(2000, len(clusters)))
                    rows.append(dict(partition=partition, comparator=comparator,
                        endpoint="final_numeric_assessment" if final else "all_assessments",
                        leaf_scope="upper_three" if upper else "all_ordinal_leaves",
                        bootstrap_unit="coordinate-year", bootstrap_draws=2000,
                        **paired_intervals(paired, clusters, draws)))
    return pd.DataFrame(rows)


def _report(destination, chosen, fitted, selection, severity, gradients, onset):
    control = next(row for row in selection if row["weather_operator"] == chosen["weather_operator"]
        and row["latent_days"] == chosen["latent_days"] and row["observation_link"] == "identity")
    lines = [
        "Final ordinal-leaf INFECT percentage at the last numeric assessment is the primary endpoint. "
        "All-assessment severity, upper-three-leaf severity, within-field same-date adjacent-leaf contrasts "
        "and interval-censored visible onset are secondary endpoints.",
        "The expressed normalized process state D is an effective score capacity. The shared observation "
        "response O(D,b) = 100[1 - exp{-[-log(1-D)]^b}] preserves zero and 100%; b = 1 is the identity. "
        "This empirical proxy link does not identify true affected leaf area, leaf incidence or a physical measurement denominator.",
        "The eight registered structures combine delay values of 20 or 30 thermal-reference days, "
        "daily-OR or duration-proxy weather exposure, and identity or nonlinear observation responses. "
        "Initiation has identical bounds [10^-6, 1] across all structures; nonlinear b has bounds [0.5, 4]. "
        "Amplification is zero. Host timing, spacing and unfolding parameters are fixed at 0.3, 120 and 100. "
        "The thermal queue, growth dilution, transferred winter phenology and calendar scenario are inherited assumptions.",
        "Calibration uses complete BASF 2017–2018 field-seasons and the original three physical-location folds. "
        "The training loss allocates half its mass to final assessments and half to all assessments, "
        "with equal coordinate-years, fields and leaf series at each endpoint. Duration-proxy rainfall intensity "
        "is the median precipitation/rain-hour ratio on positive-rain days above 2 °C within each training fold. "
        "The symmetric temperature cycle, constant within-day vapour pressure and independent rain/humidity "
        "overlap approximate daily duration; inferred exposure is not measured leaf wetness.",
        f"Calibration-only selection identified {chosen['name']}, with final-assessment out-of-fold RMSE "
        f"{chosen['rmse']:.4f} percentage points. Its identity control with the same delay, exposure and "
        f"initiation bounds has RMSE {control['rmse']:.4f}. The selected full-calibration initiation is "
        f"{fitted['parameters']['initiation']:.8g}, and b is {fitted['observation_power']:.8g}.",
        "Selected and matched-control fits are frozen before newly computing BASF 2019 and location-disjoint "
        "Corteva predictions. All these outcomes were previously examined during structure development. "
        "Their repeated scoring is retrospective reused development evidence; no untouched confirmatory test is present. "
        "Candidate choice uses calibration predictions only, with no further parameter or structural tuning from evaluation scores.",
    ]
    primary = severity.query("endpoint == 'final_numeric_assessment' and leaf_scope == 'all_ordinal_leaves'")
    for partition, group in primary.groupby("partition", sort=True):
        pieces = [f"{row.model}: RMSE {row.rmse:.4f}, MAE {row.mae:.4f}, bias {row.bias:.4f}, "
            f"weighted R² {row.weighted_r2:.4f}, n = {row.n}" for row in group.itertuples()]
        lines.append(f"{partition}. " + "; ".join(pieces) + ". Errors are in percentage points.")
    for row in gradients.itertuples():
        lines.append(f"Same-date adjacent-leaf contrasts in {row.partition}, {row.model}: "
            f"RMSE {row.rmse:.4f}, MAE {row.mae:.4f}, bias {row.bias:.4f}, weighted R² {row.weighted_r2:.4f}, "
            f"n = {row.n}. Contrast weights allocate equal coordinate-years, fields, assessment dates and available adjacent-leaf pairs.")
    informative = onset.loc[onset.censoring.eq("interval") & onset.cutoff_percent.eq(1.)]
    for row in informative.itertuples():
        lines.append(f"At the secondary 1% visibility threshold, informative onset-interval compatibility "
            f"in {row.partition}, {row.model}, is {100 * row.weighted_compatibility:.2f}% "
            f"across {row.n} leaf series. Threshold crossing uses the mapped daily response; raw process states retain their original meaning.")
    lines.append("A shared observation link can absorb scale and curvature mismatch in an unspecified-denominator "
        "source score. Its parameter remains confounded with effective initiation, exposure and host assumptions. "
        "Retrospective model selection and reused evaluation outcomes limit transport claims; independent, "
        "prospectively reserved observations with explicit percentage denominators and observed crop development "
        "are required to establish generalisation and biological interpretation.")
    (destination / "scientific_summary.txt").write_text("\n\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(PACKAGE):
        raise ValueError("Observation-response archives must remain under structural_response/.")
    protocol = json.loads(PATHS["protocol"].read_text())
    dependencies = [*PATHS.values(), *sorted(PACKAGE.glob("*.py")),
        ROOT / "tests/test_structural_response.py",
        ROOT / "analysis/paper_study/calibrate_seasonal.py",
        ROOT / "analysis/paper_study/refine_seasonal_accuracy.py",
        ROOT / "analysis/paper_study/run_empirical_benchmarks.py",
        ROOT / "analysis/paper_study/run_structural_canopy.py",
        ROOT / "analysis/paper_study/run_weather_canopy.py",
        *sorted((ROOT / "analysis/paper_study/structural_evaluation").glob("*.py")),
        *sorted((ROOT / "model/seasonal_septoria").glob("*.py")),
        *sorted((ROOT / "process_model").glob("*.py")),
        *sorted((ROOT / "process_model/_support").glob("*.py"))]
    configuration = register_experiment(destination, protocol["site_fold"], dependencies)
    # Copy executable sources immediately; the hashes remain authoritative.
    snapshot = destination / "source_snapshot"
    for path in dependencies:
        if path.suffix == ".py":
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    weather = pd.read_parquet(PATHS["weather"])
    calendars = pd.read_csv(PATHS["calendars"])
    phenology = json.loads(PATHS["phenology"].read_text())
    basf = pd.read_csv(PATHS["basf"])
    calibration = prepare_fields(basf.loc[basf.season_year.isin([2017, 2018])].copy(), calendars, weather)
    calibration.targets["partition"] = "calibration_oof"
    accumulation, thresholds = development_inputs(calibration, weather, phenology)
    folds = original_calibration_membership(calibration.targets, configuration["site_fold"])
    membership = calibration.targets[KEYS + ["field_index", "site_id", "season_year", "coordinate_year"]].copy()
    membership["global_target_index"], membership["site_fold"] = np.arange(len(membership)), folds
    membership.to_csv(destination / "calibration_membership_before_fitting.csv", index=False)
    all_rows = np.arange(len(calibration.targets))
    selection, oof_frames, oof_onsets, numerical_checks = [], {}, {}, []
    for candidate in configuration["candidates"]:
        pieces, onset_pieces = [], []
        for fold in range(3):
            training, heldout = all_rows[folds != fold], all_rows[folds == fold]
            fitted = fit_response(calibration, accumulation, thresholds, training, candidate, weather)
            fitted["candidate"], fitted["site_fold"] = candidate, fold
            write_frozen_json(destination / f"fit_{candidate['name']}_fold{fold}.json", fitted)
            scored, onsets, numerical = _scored_prediction(calibration, accumulation, thresholds,
                fitted, heldout, candidate["name"], "calibration_oof", "BASF")
            scored["site_fold"] = fold
            _parquet(destination / f"oof_{candidate['name']}_fold{fold}.parquet", scored)
            pieces.append(scored)
            onset_pieces.append(onsets)
            numerical_checks.append(numerical)
        pooled = pd.concat(pieces, ignore_index=True)
        result = dict(**candidate, **score(endpoint(pooled, True, False)),
            all_assessment_rmse=score(endpoint(pooled, False, False))["rmse"])
        selection.append(result)
        oof_frames[candidate["name"]] = pooled
        oof_onsets[candidate["name"]] = pd.concat(onset_pieces, ignore_index=True)
        print(f"CV {len(selection)}/8 {candidate['name']}: final RMSE {result['rmse']:.4f}", flush=True)
    write_frozen_json(destination / "calibration_only_selection.json", selection)
    candidate_metrics = severity_summary(pd.concat(oof_frames.values(), ignore_index=True))
    candidate_metrics.to_csv(destination / "all_candidate_calibration_metrics.csv", index=False)
    chosen = min(selection, key=lambda row: (row["rmse"], row["name"]))
    control = next(candidate for candidate in configuration["candidates"]
        if candidate["latent_days"] == chosen["latent_days"]
        and candidate["weather_operator"] == chosen["weather_operator"]
        and candidate["observation_link"] == "identity")
    fitted = fit_response(calibration, accumulation, thresholds, all_rows, chosen, weather)
    fitted["selected_candidate"], fitted["selection_cv_rmse"] = chosen["name"], chosen["rmse"]
    control_fit = fitted if chosen["name"] == control["name"] else fit_response(
        calibration, accumulation, thresholds, all_rows, control, weather)
    write_frozen_json(destination / "frozen_selected_fit.json", fitted)
    write_frozen_json(destination / "frozen_matched_identity_fit.json", control_fit)
    verify_registered_dependencies(configuration)
    freeze = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), selection=chosen,
        selected_fit_sha256=sha(destination / "frozen_selected_fit.json"),
        matched_identity_fit_sha256=sha(destination / "frozen_matched_identity_fit.json"),
        newly_scored_evaluation_disease_used_in_fitting=False,
        prior_structure_development=True, prior_evaluation_exposure=True,
        evaluation_label="reused retrospective development evidence", untouched_test=False)
    write_frozen_json(destination / "frozen_before_evaluation.json", freeze)
    print(f"FROZEN {chosen['name']}, b={fitted['observation_power']:.6g}", flush=True)
    predictions, onsets = [], []
    for name, candidate in [("selected_response", chosen), ("matched_identity_control", control)]:
        scored = oof_frames[candidate["name"]].copy()
        scored["model"] = name
        predictions.append(scored)
        onset = oof_onsets[candidate["name"]].copy()
        onset["model"] = name
        onsets.append(onset)
    # Evaluation label access and scoring occur only after both fits are frozen.
    reused_basf = prepare_fields(basf.loc[basf.season_year.eq(2019)].copy(), calendars, weather)
    reused_accumulation, _ = development_inputs(reused_basf, weather, phenology)
    external = pd.read_csv(PATHS["corteva"])
    strict = set(pd.read_csv(PATHS["strict_units"]).source_unit.astype(str))
    external = external.loc[external.assessment_eligible.eq(True)
        & external.physical_unit.astype(str).isin(strict)].copy()
    if external.empty or not set(external.site_id).isdisjoint(set(calibration.targets.site_id)):
        raise ValueError("The strict Corteva physical-location partition is empty or overlaps BASF calibration.")
    external["endpoint_series"] = external.physical_unit.astype(str) + "|" + external.organ.astype(str)
    transfer = prepare_fields(external, calendars, weather)
    transfer_accumulation, _ = development_inputs(transfer, weather, phenology)
    reference = pd.read_parquet(PATHS["original_predictions"])
    for source, data, development, partition in [
        ("BASF", reused_basf, reused_accumulation, "reused_development_2019"),
        ("Corteva", transfer, transfer_accumulation, "reused_external_strict")]:
        rows = np.arange(len(data.targets))
        for name, fit in [("selected_response", fitted), ("matched_identity_control", control_fit)]:
            scored, onset, numerical = _scored_prediction(data, development, thresholds,
                fit, rows, name, partition, source)
            predictions.append(scored)
            onsets.append(onset)
            numerical_checks.append(numerical)
        existing = reference.loc[reference.source.eq(source) & reference.model.eq("reference_v1")]
        baseline = data.targets.merge(existing[KEYS + ["predicted_percent"]],
            on=KEYS, how="left", validate="one_to_one")
        if baseline.predicted_percent.isna().any():
            raise ValueError("Archived original-reference membership differs from evaluation inputs.")
        baseline["source"], baseline["model"], baseline["partition"] = source, "reference_v1", partition
        predictions.append(baseline)
    predictions = pd.concat(predictions, ignore_index=True)
    onset = pd.concat(onsets, ignore_index=True)
    _parquet(destination / "frozen_predictions.parquet", predictions)
    severity = severity_summary(predictions)
    severity.to_csv(destination / "severity_metrics.csv", index=False)
    gradients, gradient_metrics = _gradient_summaries(predictions)
    gradients.to_csv(destination / "same_date_gradient_predictions.csv", index=False)
    gradient_metrics.to_csv(destination / "same_date_gradient_metrics.csv", index=False)
    onset.to_csv(destination / "onset_interval_predictions.csv", index=False)
    onset_metrics = _onset_summary(onset)
    onset_metrics.to_csv(destination / "onset_metrics.csv", index=False)
    _paired_summaries(predictions).to_csv(destination / "paired_improvement_intervals.csv", index=False)
    write_frozen_json(destination / "numerical_checks.json", numerical_checks)
    _report(destination, chosen, fitted, selection, severity, gradient_metrics, onset_metrics)
    verify_registered_dependencies(configuration)
    if (sha(destination / "frozen_selected_fit.json") != freeze["selected_fit_sha256"]
            or sha(destination / "frozen_matched_identity_fit.json") != freeze["matched_identity_fit_sha256"]):
        raise ValueError("Frozen observation-response fits changed during evaluation.")
    write_frozen_json(destination / "receipt.json", dict(status="complete",
        selected_candidate=chosen["name"], selection_cv_rmse=chosen["rmse"],
        selected_fit_sha256=freeze["selected_fit_sha256"],
        matched_identity_fit_sha256=freeze["matched_identity_fit_sha256"],
        prior_structure_development=True, evaluation_is_reused_development=True,
        untouched_test_evaluated=False, no_post_evaluation_tuning=True,
        model_and_nested_evaluation_sources_unchanged=True,
        output_sha256={path.name: sha(path) for path in sorted(destination.iterdir()) if path.is_file()}))
    print(severity.query("endpoint == 'final_numeric_assessment' and leaf_scope == 'all_ordinal_leaves'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
