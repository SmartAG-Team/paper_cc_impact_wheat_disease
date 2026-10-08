"""Retrospective nested development validation with frozen outer fit records.

Example: python -m analysis.paper_study.structural_evaluation.run \
    --output analysis/paper_study/nested_structural_20261006 --weather-operator daily_or
Historical outcomes have already informed model development. These folds do
not establish an untouched confirmatory validation set.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analysis.paper_study.calibrate_seasonal import sha
from analysis.paper_study.refine_seasonal_accuracy import score
from analysis.paper_study.run_empirical_benchmarks import endpoint, paired_intervals
from analysis.paper_study.run_structural_canopy import development_inputs
from model.seasonal_septoria.calibrate import fit as fit_seir, predict as predict_seir
from model.seasonal_septoria.core import Parameters
from model.seasonal_septoria.field_data import prepare_fields
from model.seasonal_septoria.refinement import calibration_weights
from model.seasonal_septoria.structural import fit_canopy, predict_canopy
from model.seasonal_septoria.wetness import duration_exposure

from .membership import (
    spatial_folds, chronological_folds, inner_folds, membership_records,
    guard_outer_membership, guard_nested_membership,
)

SEED = 20261006
VALIDATION_LABEL = "retrospective_development_validation"
KEYS = ["field_id", "endpoint_series", "date"]
DEFAULT_PATHS = dict(
    basf=ROOT / "analysis/paper_study/seasonal_calibration_v1/basf_source_assessments_snapshot.csv",
    corteva=ROOT / "data/paper_study/observations/corteva_external_assessments.csv",
    weather=ROOT / "data/paper_study/field_weather_completed/daily_weather.parquet",
    calendars=ROOT / "data/paper_study/wheat_area/trial_point_calendar_scenarios.csv",
    phenology=ROOT / "process_model/parameters/calibration.json",
)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_dependencies(hashes):
    for name, digest in hashes.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Source/input hash changed during the experiment: {name}")


def write_frozen_json(path, value):
    """Install a complete record atomically, refusing to overwrite an archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_completed_fit(path, identity):
    path = Path(path)
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    if cached.get("status") != "fit_complete":
        raise ValueError("A cached fit record is incomplete.")
    if cached.get("identity") != identity:
        raise ValueError("Cached fit identity hashes differ; use a new output directory.")
    return cached["fitted"]


def subset_fields(data, target_indices, redact_values=False):
    """Physically remove withheld fields and outcomes from optimizer inputs."""
    indices = np.asarray(target_indices, int)
    if (indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(data.targets))):
        raise ValueError("Unique nonempty field-season target indices are required.")
    target = data.targets.iloc[indices].copy()
    fields = np.sort(target.field_index.unique())
    complete = np.flatnonzero(data.targets.field_index.isin(fields))
    if not np.array_equal(np.sort(indices), complete):
        raise ValueError("A subset requires complete field-season assessment histories.")
    mapping = {old: new for new, old in enumerate(fields)}
    target["global_target_index"] = (target.global_target_index.to_numpy()
        if "global_target_index" in target else indices)
    target["field_index"] = target.field_index.map(mapping).astype(int)
    if redact_values:
        target["value"] = np.nan
    metadata = data.metadata.set_index("field_index").loc[fields].reset_index().copy()
    metadata["field_index"] = metadata.field_index.map(mapping).astype(int)
    selected = replace(data,
        temperature=data.temperature[fields].copy(), humidity=data.humidity[fields].copy(),
        rain=data.rain[fields].copy(), host_active=data.host_active[fields].copy(),
        host_renewal=data.host_renewal[fields].copy(), metadata=metadata,
        targets=target.reset_index(drop=True), excluded=pd.DataFrame())
    if hasattr(data, "maximum_temperature"):
        selected.maximum_temperature = data.maximum_temperature[fields].copy()
    return selected


def estimate_training_rain_rate(metadata, weather):
    """Rain/hour ratio from unique, positive-rain, warm training weather days."""
    forcing = weather.copy()
    forcing["date"] = pd.to_datetime(forcing.date)
    if forcing.duplicated(["location_id", "date"]).any():
        raise ValueError("Duplicate weather location/date records.")
    keys = []
    for meta in metadata.itertuples():
        keys.append(pd.DataFrame(dict(location_id=meta.site_id,
            date=pd.date_range(meta.first_forcing_date, meta.last_forcing_date))))
    if not keys:
        raise ValueError("Training field-season meteorology is empty.")
    keys = pd.concat(keys, ignore_index=True).drop_duplicates(["location_id", "date"])
    matched = keys.merge(forcing, on=["location_id", "date"], how="left", validate="one_to_one")
    required = ["tmean_c", "precipitation_mm", "rain_hours_gt_0_1mm"]
    if not np.isfinite(matched[required].to_numpy(float)).all():
        raise ValueError("Complete finite training rainfall-duration meteorology is required.")
    eligible = matched.loc[matched.rain_hours_gt_0_1mm.gt(0) & matched.tmean_c.gt(2)
        & matched.precipitation_mm.gt(0)]
    if eligible.empty:
        raise ValueError("No positive-rain warm training days identify effective rainfall intensity.")
    rate = float(np.median(eligible.precipitation_mm/eligible.rain_hours_gt_0_1mm))
    return dict(rain_rate_mm_hour=rate, unique_training_location_dates=len(matched),
        eligible_positive_rain_days=len(eligible),
        estimation="median daily precipitation / observed positive rain hours; tmean > 2 C",
        meteorology_only=True)


def freeze_membership_plan(destination, targets, outers):
    plan, records = {}, []
    for outer in outers:
        inners = inner_folds(targets, outer, seed=SEED)
        plan[outer.name] = inners
        records.append(membership_records(targets, outer, inners))
    destination = Path(destination)
    membership = pd.concat(records, ignore_index=True)
    path = destination / "nested_membership_before_fitting.csv"
    if path.exists():
        stored = pd.read_csv(path)
        expected = pd.read_csv(io.StringIO(membership.to_csv(index=False)))
        pd.testing.assert_frame_equal(stored, expected)
    else:
        membership.to_csv(path, index=False, mode="x")
    return plan


def candidates():
    canopy = [dict(name=f"flag{flag:g}_interval{interval:g}_delay{delay:g}",
        host_parameters=dict(flag_fraction=flag, leaf_interval=float(interval), expansion_units=100.),
        amplification_free=False, latent_days=float(delay))
        for flag in (.3, .6, .9) for interval in (80, 120, 160) for delay in (20, 30)]
    seir = [dict(name=f"delay{delay}", latent_days=float(delay), beta_free=True)
        for delay in (10, 20, 30)]
    return dict(structural_canopy=canopy, original_seir=seir)


def _exposure(data, preprocessing):
    if preprocessing["weather_operator"] == "daily_or":
        return None
    return duration_exposure(data.temperature, data.maximum_temperature, data.humidity,
        data.rain, rain_rate_mm_hour=preprocessing["rain_rate_mm_hour"])["exposure"]


def fit_record(destination, family, candidate, training, accumulation, thresholds,
               weather, weather_operator, run_identity, membership):
    preprocessing = dict(weather_operator=weather_operator if family == "structural_canopy" else "daily_or")
    if family == "structural_canopy" and weather_operator == "duration_proxy":
        preprocessing.update(estimate_training_rain_rate(training.metadata, weather))
    identity = dict(run_identity=run_identity, family=family, candidate=candidate,
        global_training_target_indices=training.targets.global_target_index.tolist(),
        membership=membership, preprocessing=preprocessing)
    fitted = read_completed_fit(destination, identity)
    if fitted is None:
        rows = np.arange(len(training.targets))
        if family == "structural_canopy":
            fitted = fit_canopy(training, accumulation, thresholds, rows,
                candidate["host_parameters"], amplification_free=False,
                latent_days=candidate["latent_days"], final_weight=.5,
                initiation_max=1. if weather_operator == "duration_proxy" else .1,
                environmental_response=_exposure(training, preprocessing))
        else:
            weighted = replace(training, targets=training.targets.copy())
            weighted.targets["weight"] = calibration_weights(weighted.targets, final_weight=.5)
            fitted = fit_seir(weighted, rows, latent_days=candidate["latent_days"], beta_free=True)
        fitted["weather_preprocessing"] = preprocessing
        fitted["validation_label"] = VALIDATION_LABEL
        write_frozen_json(destination, dict(status="fit_complete", identity=identity,
            fitted=fitted, frozen_utc=datetime.now(timezone.utc).isoformat(),
            evaluation_observations_used=False, prior_outcome_exposure=True))
    return fitted


def predict_record(family, data, accumulation, thresholds, fitted):
    if data.targets.value.notna().any():
        raise ValueError("Prediction inputs must have all disease values redacted.")
    if family == "structural_canopy":
        forecast, _ = predict_canopy(data, accumulation, thresholds, fitted,
            environmental_response=_exposure(data, fitted["weather_preprocessing"]))
    else:
        forecast, _ = predict_seir(data, Parameters(**fitted["parameters"]))
    if (forecast.shape != (len(data.targets),) or not np.isfinite(forecast).all()
            or np.any(forecast < -1e-8) or np.any(forecast > 100+1e-8)):
        raise ValueError("A prediction is missing or outside the supported percentage range.")
    return forecast


def _save_frame(path, frame):
    """Recover artifacts only when their values are identical, never overwrite."""
    path = Path(path)
    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["object", "str"]).columns:
        normalized[column] = normalized[column].map(lambda x: None if pd.isna(x) else str(x)).astype("string")
    if path.exists():
        stored = pd.read_parquet(path)
        pd.testing.assert_frame_equal(stored, normalized)
    else:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            normalized.to_parquet(temporary, index=False)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def save_frozen_csv(path, frame):
    path = Path(path)
    content = frame.to_csv(index=False)
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Recovered CSV results differ from their frozen values.")
    else:
        with path.open("x") as stream:
            stream.write(content)


def select_family(destination, family, settings, data, accumulation, thresholds,
                  weather, weather_operator, outer, inners, identity):
    selection = []
    for candidate in settings:
        predictions = []
        for inner in inners:
            guard_nested_membership(data.targets, outer, inner)
            training = subset_fields(data, inner.train)
            validation = subset_fields(data, inner.test, redact_values=True)
            train_fields = np.sort(data.targets.iloc[inner.train].field_index.unique())
            validation_fields = np.sort(data.targets.iloc[inner.test].field_index.unique())
            fit_path = destination / f"{family}_{candidate['name']}_{inner.name}_fit.json"
            fitted = fit_record(fit_path, family, candidate, training, accumulation[train_fields],
                thresholds, weather, weather_operator, identity,
                dict(outer=outer.name, inner=inner.name, stage="inner_training"))
            forecast = predict_record(family, validation, accumulation[validation_fields], thresholds, fitted)
            part = data.targets.iloc[inner.test].copy()
            part["global_target_index"] = inner.test
            part["model"], part["candidate"], part["inner_fold"] = family, candidate["name"], inner.name
            part["outer_fold"], part["validation_label"], part["predicted_percent"] = outer.name, VALIDATION_LABEL, forecast
            _save_frame(fit_path.with_name(fit_path.stem + "_predictions.parquet"), part)
            predictions.append(part)
        pooled = pd.concat(predictions, ignore_index=True)
        final = score(endpoint(pooled, True, False))
        selection.append(dict(candidate=candidate, final_leaf_rmse=final["rmse"],
            all_assessment_rmse=score(endpoint(pooled, False, False))["rmse"],
            prediction_count=len(pooled), inner_fit_count=len(inners)))
        print(f"{outer.name} {family} {candidate['name']}: inner final RMSE {final['rmse']:.4f}", flush=True)
    record = dict(family=family, outer_fold=outer.name, selection=selection,
        selected=min(selection, key=lambda x: x["final_leaf_rmse"]),
        criterion="pooled inner final-assessment all ordinal-leaf coordinate-year weighted RMSE",
        outer_test_outcomes_used=False, validation_label=VALIDATION_LABEL)
    path = destination / f"{family}_selection.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("Recovered inner selection differs from its frozen record.")
    else:
        write_frozen_json(path, record)
    return record


def summarize_predictions(predictions, bootstrap_draws=5000):
    metrics, differences = [], []
    generator = np.random.default_rng(SEED)
    # Spatial observations have exactly one prediction per family. Forward
    # observations occur once in their harvest-year test. Resample complete
    # location histories, retaining repeated years and cross-source units.
    for (kind, scope), group in predictions.groupby(["evaluation_kind", "report_scope"]):
        for final in (False, True):
            for upper in (False, True):
                name = "final_numeric_assessment" if final else "all_assessments"
                leaf_scope = "upper_three" if upper else "all_ordinal_leaves"
                selected = {family: endpoint(part, final, upper)
                    for family, part in group.groupby("model")}
                if any(part.empty for part in selected.values()):
                    continue
                for family, part in selected.items():
                    metrics.append(dict(evaluation_kind=kind, report_scope=scope, model=family,
                        endpoint=name, leaf_scope=leaf_scope, validation_label=VALIDATION_LABEL, **score(part)))
                canopy, seir = selected["structural_canopy"], selected["original_seir"]
                matched = canopy.merge(seir[KEYS + ["predicted_percent"]], on=KEYS,
                    validate="one_to_one", suffixes=("", "_comparator"))
                if len(matched) != len(canopy) or len(matched) != len(seir):
                    raise ValueError("Outer paired prediction membership differs.")
                matched["comparator_percent"] = matched.predicted_percent_comparator
                # paired_intervals retains its shared weighted-error calculation;
                # this column selects location rather than coordinate-year draws.
                matched["coordinate_year"] = matched.site_id.astype(str)
                clusters = np.array(sorted(matched.coordinate_year.unique()))
                draws = generator.integers(0, len(clusters), (bootstrap_draws, len(clusters)))
                differences.append(dict(evaluation_kind=kind, report_scope=scope, endpoint=name,
                    leaf_scope=leaf_scope, cluster_unit="complete_location_history",
                    cluster_count=len(clusters), bootstrap_draws=bootstrap_draws,
                    difference="structural_canopy minus original_seir", validation_label=VALIDATION_LABEL,
                    **paired_intervals(matched, clusters, draws)))
    return pd.DataFrame(metrics), pd.DataFrame(differences)


def load_inputs(paths):
    basf, corteva = pd.read_csv(paths["basf"]), pd.read_csv(paths["corteva"])
    basf["source"] = "BASF"
    corteva = corteva.loc[corteva.assessment_eligible.eq(True)].copy()
    corteva["source"] = "Corteva"
    corteva["endpoint_series"] = corteva.physical_unit.astype(str) + "|" + corteva.organ.astype(str)
    source = pd.concat([basf, corteva], ignore_index=True, sort=False)
    source["physical_unit"] = source.physical_unit.astype(str)
    weather, calendars = pd.read_parquet(paths["weather"]), pd.read_csv(paths["calendars"])
    data = prepare_fields(source, calendars, weather)
    if data.targets.duplicated(KEYS).any():
        raise ValueError("Source field/leaf/date endpoints are not unique.")
    if not data.targets.metric.eq("infection_percent_unspecified_basis").all():
        raise ValueError("The canopy score requires source INFECT percentages.")
    data.targets["global_target_index"] = np.arange(len(data.targets))
    data.targets["validation_label"] = VALIDATION_LABEL
    phenology = json.loads(Path(paths["phenology"]).read_text())
    accumulation, thresholds = development_inputs(data, weather, phenology)
    return data, weather, accumulation, thresholds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weather-operator", choices=("daily_or", "duration_proxy"), default="daily_or")
    parser.add_argument("--resume", action="store_true", help="Recover completed individual fits with identical hashes")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze inputs and nested membership, without fitting")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")
    destination = args.output.resolve()
    if destination.exists() and (not args.resume or (destination / "receipt.json").exists()):
        raise FileExistsError("Use a new immutable nested development archive.")
    destination.mkdir(parents=True, exist_ok=True)
    dependencies = [*DEFAULT_PATHS.values(), *sorted(Path(__file__).parent.glob("*.py")),
        ROOT / "analysis/paper_study/run_structural_canopy.py",
        ROOT / "analysis/paper_study/calibrate_seasonal.py",
        ROOT / "analysis/paper_study/refine_seasonal_accuracy.py",
        ROOT / "analysis/paper_study/run_empirical_benchmarks.py",
        *sorted((ROOT / "model/seasonal_septoria").glob("*.py")),
        *sorted((ROOT / "process_model").glob("*.py")),
        *sorted((ROOT / "process_model/_support").glob("*.py"))]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in dependencies}
    settings = candidates()
    configuration = dict(validation_label=VALIDATION_LABEL,
        prior_historical_outcome_exposure=True, untouched_test=False,
        conditional_on_model_family_development=True, random_seed=SEED,
        spatial_outer_folds=5, forward_evaluation_years=[2016, 2017, 2018, 2019],
        forward_test_location_rule="absent from every preceding training year in both sources",
        inner_location_folds=3, candidates=settings, final_loss_weight=.5,
        weather_operator=args.weather_operator, comparator_weather_operator="original_daily_or",
        canopy_initiation_upper_bound=1. if args.weather_operator == "duration_proxy" else .1,
        amplification_fixed_zero=True, initiation_is_effective_score_parameter=True,
        selection="minimum pooled inner final leaf RMSE", bootstrap_draws=args.bootstrap_draws,
        bootstrap_unit="complete_location_history", input_and_source_sha256=hashes)
    config_path = destination / "configuration_before_fitting.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError("Input/source hashes or experiment configuration changed.")
    else:
        write_frozen_json(config_path, configuration)
    snapshot = destination / "source_snapshot"
    for path in dependencies:
        if path.suffix != ".py" and path != DEFAULT_PATHS["phenology"]:
            continue
        copy = snapshot / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        if copy.exists():
            if sha(copy) != sha(path):
                raise ValueError("Frozen source snapshot differs.")
        else:
            copy.write_bytes(path.read_bytes())
    data, weather, accumulation, thresholds = load_inputs(DEFAULT_PATHS)
    outers = spatial_folds(data.targets, seed=SEED) + chronological_folds(data.targets)
    plan = freeze_membership_plan(destination, data.targets, outers)
    _save_frame(destination / "all_source_target_membership.parquet", data.targets)
    _save_frame(destination / "all_source_field_seasons.parquet", data.metadata)
    if not data.excluded.empty:
        _save_frame(destination / "excluded_source_assessments.parquet", data.excluded)
    identity = dict(configuration_sha256=sha(config_path),
        target_membership_sha256=sha(destination / "all_source_target_membership.parquet"),
        nested_membership_sha256=sha(destination / "nested_membership_before_fitting.csv"))
    if args.prepare_only:
        print(f"Frozen {len(data.targets)} targets, {len(data.metadata)} complete field-seasons, and {len(outers)} outer folds. No fits or outer scores calculated.", flush=True)
        return
    verify_dependencies(hashes)
    frozen = {}
    for outer in outers:
        guard_outer_membership(data.targets, outer)
        fold_destination = destination / "outer_folds" / outer.name
        fold_destination.mkdir(parents=True, exist_ok=True)
        frozen[outer.name] = {}
        for family, family_settings in settings.items():
            selection = select_family(fold_destination, family, family_settings, data,
                accumulation, thresholds, weather, args.weather_operator, outer, plan[outer.name], identity)
            chosen = selection["selected"]["candidate"]
            training = subset_fields(data, outer.train)
            fields = np.sort(data.targets.iloc[outer.train].field_index.unique())
            path = fold_destination / f"frozen_{family}_outer_fit.json"
            fitted = fit_record(path, family, chosen, training, accumulation[fields], thresholds,
                weather, args.weather_operator, identity,
                dict(outer=outer.name, stage="outer_refit", selection_sha256=sha(fold_destination / f"{family}_selection.json")))
            frozen[outer.name][family] = dict(path=str(path.relative_to(destination)), sha256=sha(path),
                selected_candidate=chosen, inner_selection_final_rmse=selection["selected"]["final_leaf_rmse"])
    manifest_path = destination / "all_outer_fits_frozen_before_scoring.json"
    verify_dependencies(hashes)
    manifest = dict(outer_fit_records=frozen, identity=identity, validation_label=VALIDATION_LABEL,
        outer_outcomes_used_to_fit_or_select=False, prior_historical_outcome_exposure=True)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Frozen outer fit manifest changed.")
    else:
        write_frozen_json(manifest_path, manifest)
    predictions = []
    for outer in outers:
        validation = subset_fields(data, outer.test, redact_values=True)
        fields = np.sort(data.targets.iloc[outer.test].field_index.unique())
        for family in settings:
            record = frozen[outer.name][family]
            path = destination / record["path"]
            if sha(path) != record["sha256"]:
                raise ValueError("An outer fit changed after freezing.")
            fitted = json.loads(path.read_text())["fitted"]
            forecast = predict_record(family, validation, accumulation[fields], thresholds, fitted)
            part = data.targets.iloc[outer.test].copy()
            part["outer_fold"], part["evaluation_kind"], part["model"] = outer.name, outer.kind, family
            part["predicted_percent"], part["selected_candidate"] = forecast, record["selected_candidate"]["name"]
            part["fit_sha256"] = record["sha256"]
            predictions.append(part)
    combined = pd.concat(predictions, ignore_index=True)
    _save_frame(destination / "all_out_of_fold_predictions.parquet", combined)
    reporting = combined.copy()
    reporting["report_scope"] = "pooled"
    by_source = combined.copy()
    by_source["report_scope"] = by_source.source
    by_fold = combined.copy()
    by_fold["report_scope"] = by_fold.outer_fold
    metrics, differences = summarize_predictions(pd.concat([reporting, by_source, by_fold]), args.bootstrap_draws)
    _save_frame(destination / "severity_metrics.parquet", metrics)
    _save_frame(destination / "paired_cluster_bootstrap_differences.parquet", differences)
    save_frozen_csv(destination / "severity_metrics.csv", metrics)
    save_frozen_csv(destination / "paired_cluster_bootstrap_differences.csv", differences)
    for family_records in frozen.values():
        for record in family_records.values():
            if sha(destination / record["path"]) != record["sha256"]:
                raise ValueError("Frozen outer fits changed during scoring.")
    # A completion receipt follows all fit, prediction, membership, and score
    # verification. Preparation and individual fits never emit this receipt.
    verify_dependencies(hashes)
    write_frozen_json(destination / "receipt.json", dict(status="complete", validation_label=VALIDATION_LABEL,
        untouched_test_evaluated=False, prior_historical_outcome_exposure=True,
        outer_fold_count=len(outers), prediction_count=len(combined),
        frozen_manifest_sha256=sha(manifest_path), configuration_sha256=sha(config_path),
        output_sha256={path.name: sha(path) for path in sorted(destination.glob("*.parquet"))}))


if __name__ == "__main__":
    main()
