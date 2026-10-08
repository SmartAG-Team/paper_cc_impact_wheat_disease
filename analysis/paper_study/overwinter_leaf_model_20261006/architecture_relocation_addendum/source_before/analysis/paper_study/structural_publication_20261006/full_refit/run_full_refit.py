"""Freeze a deployment refit after retrospective nested development validation.

Both canopy components use the existing 18-candidate grid, grouped three-fold
selection, optimizer, observation loss, and meteorology-only preprocessing.
All-data predictions are fit diagnostics and provide no independent test.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from analysis.paper_study.refine_seasonal_accuracy import score
from analysis.paper_study.run_empirical_benchmarks import endpoint
from analysis.paper_study.structural_evaluation.membership import spatial_folds, guard_outer_membership
from analysis.paper_study.structural_evaluation.run import (
    DEFAULT_PATHS, SEED, candidates, load_inputs, subset_fields,
    estimate_training_rain_rate, _exposure, predict_record, read_completed_fit,
    write_frozen_json, save_frozen_csv, _save_frame, sha, verify_dependencies,
)
from model.seasonal_septoria.publication_model import load_publication_model, predict_publication_model
from model.seasonal_septoria.structural import fit_canopy

DESTINATION = Path(__file__).resolve().parent
MODEL_ID = "canopy_score_ensemble_v6_20261006"
STAGE = "deployment_refit_after_retrospective_nested_evaluation"
DIAGNOSTICS = "full_data_fit_diagnostics"
OPERATORS = ("daily_or", "duration_proxy")
PROTOCOL_PATH = DESTINATION.parent / "ensemble_protocol_before_nested_scoring.json"
NESTED_ARCHIVES = {
    "daily_or": ROOT / "analysis/paper_study/nested_canopy_daily_20261006",
    "duration_proxy": ROOT / "analysis/paper_study/nested_canopy_duration_20261006",
}


def check_ensemble_protocol(protocol):
    if (protocol.get("weights") != {"daily_or": .5, "duration_proxy": .5}
            or protocol.get("weights_fitted_to_outer_outcomes") is not False
            or protocol.get("pooled_nested_scores_read_before_registration") is not False):
        raise ValueError("The registered equal component weights and prior score-reading status must be preserved.")


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Recovered frozen record differs: {path.name}")
    else:
        write_frozen_json(path, value)


def make_bundle(fitted_components, identity):
    if set(fitted_components) != set(OPERATORS):
        raise ValueError("Exactly one component for each saved weather operator is required.")
    for operator, fitted in fitted_components.items():
        saved = fitted.get("weather_operator", fitted.get("weather_preprocessing", {}).get("weather_operator"))
        if saved != operator:
            raise ValueError("The fitted component's saved weather operator does not match its bundle role.")
        if operator == "duration_proxy":
            rate = fitted.get("rain_rate_mm_hour", fitted.get("weather_preprocessing", {}).get("rain_rate_mm_hour"))
            if rate is None or not np.isfinite(rate) or rate <= 0:
                raise ValueError("The duration component requires a positive frozen rainfall intensity.")
    return load_publication_model(dict(schema_version=1, model_id=MODEL_ID,
        observation="infection_percent_unspecified_basis",
        components=[dict(weight=.5, fitted=fitted_components[operator]) for operator in OPERATORS],
        deployment_refit_after_retrospective_nested_evaluation=True,
        independent_validation=False, prior_historical_outcome_exposure=True,
        observation_is_calibrated_incidence_probability=False,
        component_spread_is_prediction_interval=False,
        identity=identity))


def diagnostic_predictions(targets, forecasts):
    if set(forecasts) != set(OPERATORS):
        raise ValueError("Both frozen component predictions are required for diagnostics.")
    forecasts = {name: np.asarray(values, float) for name, values in forecasts.items()}
    if any(values.shape != (len(targets),) or not np.isfinite(values).all() for values in forecasts.values()):
        raise ValueError("Complete finite component predictions are required.")
    all_predictions = dict(forecasts)
    all_predictions[MODEL_ID] = .5 * forecasts["daily_or"] + .5 * forecasts["duration_proxy"]
    rows = []
    for name, values in all_predictions.items():
        part = targets.drop(columns=["validation_label"], errors="ignore").copy()
        part["model"], part["predicted_percent"] = name, values
        part["diagnostic_label"] = DIAGNOSTICS
        part["independent_validation"] = False
        part["deployment_stage"] = STAGE
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def fit_component(path, training, accumulation, thresholds, weather, operator,
                  candidate, run_identity, stage, fold=None):
    preprocessing = dict(weather_operator=operator)
    if operator == "duration_proxy":
        preprocessing.update(estimate_training_rain_rate(training.metadata, weather))
    identity = dict(run_identity=run_identity, candidate=candidate, weather_operator=operator,
        global_training_target_indices=training.targets.global_target_index.tolist(),
        preprocessing=preprocessing, stage=stage, fold=fold)
    fitted = read_completed_fit(path, identity)
    if fitted is not None:
        return fitted
    start = time.perf_counter()
    fitted = fit_canopy(training, accumulation, thresholds, np.arange(len(training.targets)),
        candidate["host_parameters"], amplification_free=False,
        latent_days=candidate["latent_days"], final_weight=.5,
        initiation_max=.1 if operator == "daily_or" else 1.,
        environmental_response=_exposure(training, preprocessing))
    fitted["weather_preprocessing"] = preprocessing
    fitted["weather_operator"] = operator
    if operator == "duration_proxy":
        fitted["rain_rate_mm_hour"] = preprocessing["rain_rate_mm_hour"]
    fitted["deployment_stage"] = STAGE
    fitted["fit_role"] = stage
    fitted["independent_validation"] = False
    fitted["prior_historical_outcome_exposure"] = True
    write_frozen_json(path, dict(status="fit_complete", identity=identity, fitted=fitted,
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        runtime_seconds=time.perf_counter() - start,
        prior_retrospective_evaluation_targets_available=True,
        model_use=STAGE, independent_validation=False))
    return fitted


def select_component(operator, data, accumulation, thresholds, weather, folds, settings, identity):
    folder = DESTINATION / operator
    folder.mkdir(exist_ok=True)
    selection = []
    for number, candidate in enumerate(settings, 1):
        predictions = []
        for fold in folds:
            guard_outer_membership(data.targets, fold)
            training = subset_fields(data, fold.train)
            validation = subset_fields(data, fold.test, redact_values=True)
            fields = np.sort(data.targets.iloc[fold.train].field_index.unique())
            validation_fields = np.sort(data.targets.iloc[fold.test].field_index.unique())
            path = folder / f"{candidate['name']}_{fold.name}_fit.json"
            fitted = fit_component(path, training, accumulation[fields], thresholds, weather,
                operator, candidate, identity, "deployment_internal_selection", fold.name)
            values = predict_record("structural_canopy", validation, accumulation[validation_fields], thresholds, fitted)
            part = data.targets.iloc[fold.test].drop(columns=["validation_label"], errors="ignore").copy()
            part["model"], part["candidate"], part["fold"] = operator, candidate["name"], fold.name
            part["predicted_percent"] = values
            part["diagnostic_label"] = "deployment_internal_selection_after_nested_evaluation"
            part["independent_validation"] = False
            _save_frame(path.with_name(path.stem + "_predictions.parquet"), part)
            predictions.append(part)
        pooled = pd.concat(predictions, ignore_index=True)
        final_score = score(endpoint(pooled, True, False))
        selection.append(dict(candidate=candidate, final_leaf_rmse=final_score["rmse"],
            all_assessment_rmse=score(endpoint(pooled, False, False))["rmse"],
            prediction_count=len(pooled), grouped_location_folds=3))
        print(f"{operator} candidate {number}/{len(settings)} {candidate['name']}: internal final RMSE {final_score['rmse']:.6f}", flush=True)
    record = dict(weather_operator=operator, selection=selection,
        selected=min(selection, key=lambda row: row["final_leaf_rmse"]),
        criterion="pooled location-grouped final-assessment all-ordinal-leaf RMSE",
        deployment_stage=STAGE, independent_validation=False,
        prior_historical_outcome_exposure=True)
    freeze_json(folder / "selection.json", record)
    save_frozen_csv(folder / "selection_trace.csv", pd.DataFrame([
        dict(candidate=row["candidate"]["name"], final_leaf_rmse=row["final_leaf_rmse"],
            all_assessment_rmse=row["all_assessment_rmse"], prediction_count=row["prediction_count"])
        for row in selection]))
    return record


def diagnostic_metrics(predictions):
    rows = []
    scopes = [("all_sources", predictions)] + list(predictions.groupby("source"))
    for source, scope in scopes:
        for model, group in scope.groupby("model"):
            for final in (False, True):
                for upper in (False, True):
                    frame = endpoint(group, final, upper)
                    if not frame.empty:
                        rows.append(dict(source=source, model=model,
                            endpoint="final_numeric_assessment" if final else "all_assessments",
                            leaf_scope="upper_three" if upper else "all_ordinal_leaves",
                            diagnostic_label=DIAGNOSTICS, independent_validation=False, **score(frame)))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    if (DESTINATION / "receipt.json").exists():
        raise FileExistsError("The completed deployment refit is immutable.")
    contract = DESTINATION / "configuration_before_fitting.json"
    if contract.exists() and not args.resume:
        raise FileExistsError("Recover the registered refit with --resume.")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    check_ensemble_protocol(protocol)
    prerequisites = {}
    for operator, folder in NESTED_ARCHIVES.items():
        receipt = json.loads((folder / "receipt.json").read_text())
        if receipt.get("status") != "complete" or receipt.get("untouched_test_evaluated") is not False:
            raise ValueError("Both completed retrospective nested evaluation archives are required.")
        if (receipt.get("frozen_manifest_sha256") != sha(folder / "all_outer_fits_frozen_before_scoring.json")
                or receipt.get("configuration_sha256") != sha(folder / "configuration_before_fitting.json")):
            raise ValueError("A prerequisite nested evaluation manifest/configuration hash differs from its receipt.")
        prerequisites[operator] = dict(archive=str(folder.relative_to(ROOT)),
            receipt_sha256=sha(folder / "receipt.json"),
            outer_fit_manifest_sha256=sha(folder / "all_outer_fits_frozen_before_scoring.json"))
    dependencies = [*DEFAULT_PATHS.values(), PROTOCOL_PATH,
        *sorted(DESTINATION.glob("*.py")),
        *sorted((ROOT / "analysis/paper_study/structural_evaluation").glob("*.py")),
        ROOT / "analysis/paper_study/run_structural_canopy.py",
        ROOT / "analysis/paper_study/calibrate_seasonal.py",
        ROOT / "analysis/paper_study/refine_seasonal_accuracy.py",
        ROOT / "analysis/paper_study/run_empirical_benchmarks.py",
        *sorted((ROOT / "model/seasonal_septoria").glob("*.py")),
        *sorted((ROOT / "process_model").glob("*.py")),
        *sorted((ROOT / "process_model/_support").glob("*.py"))]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in dependencies}
    settings = candidates()["structural_canopy"]
    if len(settings) != 18 or any(item["amplification_free"] for item in settings):
        raise ValueError("The registered 18-candidate amplification-zero grid is required.")
    configuration = dict(model_id=MODEL_ID, schema_version=1, deployment_stage=STAGE,
        independent_validation=False, prior_historical_outcome_exposure=True,
        completed_nested_evaluations=prerequisites, ensemble_protocol_sha256=sha(PROTOCOL_PATH),
        component_weights={operator: .5 for operator in OPERATORS},
        observation="infection_percent_unspecified_basis", candidates=settings,
        random_seed=SEED, grouped_complete_site_history_folds=3,
        selection="minimum pooled final-assessment all-ordinal-leaf RMSE",
        final_loss_weight=.5, amplification_fixed_zero=True,
        initiation_bounds={"daily_or": [1e-6, .1], "duration_proxy": [1e-6, 1.]},
        duration_rain_intensity="median daily precipitation / observed positive rain hours from unique training location/dates with tmean > 2 C",
        fitted_data=dict(targets=2305, field_seasons=218, source_names=["BASF", "Corteva"]),
        in_sample_predictions_label=DIAGNOSTICS, input_and_code_sha256=hashes,
        runtime_versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    freeze_json(contract, configuration)
    snapshot = DESTINATION / "runtime_source_snapshot"
    for path in dependencies:
        if path.suffix != ".py" and path != DEFAULT_PATHS["phenology"] and path != PROTOCOL_PATH:
            continue
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != sha(path):
                raise ValueError("Recovered runtime source snapshot differs.")
        else:
            with target.open("xb") as stream:
                stream.write(path.read_bytes())
    data, weather, accumulation, thresholds = load_inputs(DEFAULT_PATHS)
    if len(data.targets) != 2305 or len(data.metadata) != 218:
        raise ValueError("The established combined-source target/field membership changed.")
    membership_columns = ["field_id", "endpoint_series", "date", "value", "site_id", "season_year", "source"]
    for folder in NESTED_ARCHIVES.values():
        archived = pd.read_parquet(folder / "all_source_target_membership.parquet")[membership_columns]
        pd.testing.assert_frame_equal(data.targets[membership_columns].reset_index(drop=True),
            archived.reset_index(drop=True), check_dtype=False)
    folds = spatial_folds(data.targets, n_splits=3, seed=SEED)
    target_membership = data.targets.drop(columns=["validation_label"], errors="ignore").copy()
    _save_frame(DESTINATION / "all_data_target_membership.parquet", target_membership)
    _save_frame(DESTINATION / "all_data_field_seasons.parquet", data.metadata)
    records = []
    for fold in folds:
        for role, indices in [("training", fold.train), ("internal_holdout", fold.test)]:
            part = data.targets.iloc[indices][["global_target_index", "field_id", "site_id", "season_year",
                "endpoint_series", "date"]].copy()
            part["fold"], part["role"] = fold.name, role
            part["deployment_stage"] = STAGE
            records.append(part)
    save_frozen_csv(DESTINATION / "selection_membership_before_fitting.csv", pd.concat(records, ignore_index=True))
    identity = dict(configuration_sha256=sha(contract),
        target_membership_sha256=sha(DESTINATION / "all_data_target_membership.parquet"),
        selection_membership_sha256=sha(DESTINATION / "selection_membership_before_fitting.csv"),
        ensemble_protocol_sha256=sha(PROTOCOL_PATH))
    verify_dependencies(hashes)
    if args.prepare_only:
        print("Prepared post-evaluation deployment refit: 2305 targets, 218 field-seasons, 3 location-history folds, 18 candidates per operator; no fits or diagnostics calculated.", flush=True)
        return
    fitted_components, selections = {}, {}
    for operator in OPERATORS:
        selection = select_component(operator, data, accumulation, thresholds, weather, folds, settings, identity)
        selections[operator] = selection["selected"]
        training = subset_fields(data, np.arange(len(data.targets)))
        fit_path = DESTINATION / f"{operator}_full_data_fit_record.json"
        fitted_components[operator] = fit_component(fit_path, training, accumulation, thresholds,
            weather, operator, selection["selected"]["candidate"],
            dict(identity, selection_sha256=sha(DESTINATION / operator / "selection.json")),
            "all_available_data_deployment_fit")
        freeze_json(DESTINATION / f"{operator}_fitted_component.json", fitted_components[operator])
        print(f"{operator} all-data deployment fit frozen: {fitted_components[operator]['parameters']}", flush=True)
    verify_dependencies(hashes)
    bundle = make_bundle(fitted_components, identity)
    bundle_path = DESTINATION / "publication_model_bundle.json"
    freeze_json(bundle_path, bundle)
    frozen_sha = sha(bundle_path)
    freeze_json(DESTINATION / "frozen_bundle_before_fit_diagnostics.json",
        dict(bundle_sha256=frozen_sha, selected_components=selections, deployment_stage=STAGE,
            independent_validation=False, diagnostic_label=DIAGNOSTICS))
    redacted = subset_fields(data, np.arange(len(data.targets)), redact_values=True)
    forecasts = {operator: predict_record("structural_canopy", redacted, accumulation, thresholds, fitted)
        for operator, fitted in fitted_components.items()}
    runtime = predict_publication_model(redacted, accumulation, thresholds, bundle_path)
    expected = .5 * forecasts["daily_or"] + .5 * forecasts["duration_proxy"]
    if not np.allclose(runtime["predicted_percent"], expected, rtol=0, atol=1e-10):
        raise ValueError("Saved-model runtime differs from the registered fixed component average.")
    diagnostics = diagnostic_predictions(data.targets, forecasts)
    _save_frame(DESTINATION / "full_data_fit_diagnostic_predictions.parquet", diagnostics)
    metrics = diagnostic_metrics(diagnostics)
    _save_frame(DESTINATION / "full_data_fit_diagnostic_metrics.parquet", metrics)
    save_frozen_csv(DESTINATION / "full_data_fit_diagnostic_metrics.csv", metrics)
    verify_dependencies(hashes)
    if sha(bundle_path) != frozen_sha:
        raise ValueError("The model bundle changed during fit diagnostics.")
    fit_records = list(DESTINATION.glob("*_full_data_fit_record.json")) + list(DESTINATION.glob("*/*_fit.json"))
    runtimes = [json.loads(path.read_text())["runtime_seconds"] for path in fit_records]
    outputs = [bundle_path, *DESTINATION.glob("*_fitted_component.json"),
        *DESTINATION.glob("*.parquet"), *DESTINATION.glob("*.csv"),
        *DESTINATION.glob("*/selection.json"), *DESTINATION.glob("*/selection_trace.csv")]
    write_frozen_json(DESTINATION / "receipt.json", dict(status="complete", model_id=MODEL_ID,
        deployment_stage=STAGE, independent_validation=False, prior_historical_outcome_exposure=True,
        target_count=len(data.targets), field_season_count=len(data.metadata), location_count=data.targets.site_id.nunique(),
        component_weights={operator: .5 for operator in OPERATORS}, selected_components=selections,
        bundle_sha256=frozen_sha, configuration_sha256=sha(contract),
        runtime_dispatch_verified=True, fit_record_count=len(fit_records),
        summed_optimizer_runtime_seconds=sum(runtimes), elapsed_this_process_seconds=time.perf_counter() - start,
        runtime_source_snapshot_sha256={str(path.relative_to(snapshot)): sha(path)
            for path in sorted(snapshot.rglob("*")) if path.is_file()},
        output_sha256={str(path.relative_to(DESTINATION)): sha(path) for path in outputs}))
    print(f"Deployment refit complete: {bundle_path}; SHA256 {frozen_sha}. Saved predictions are full-data fit diagnostics.", flush=True)


if __name__ == "__main__":
    main()
