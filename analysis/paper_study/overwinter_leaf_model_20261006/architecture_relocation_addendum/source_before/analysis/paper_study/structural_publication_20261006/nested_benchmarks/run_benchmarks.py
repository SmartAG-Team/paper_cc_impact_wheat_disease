"""Retrospective empirical baselines on the frozen structural nested folds.

The finite benchmark plan is registered after structural results are known.
Outer labels remain physically absent from each fitted baseline and selection.
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
import sklearn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from analysis.paper_study.refine_seasonal_accuracy import score
from analysis.paper_study.run_empirical_benchmarks import endpoint, paired_intervals
from analysis.paper_study.structural_evaluation.membership import (
    EvaluationFold, guard_outer_membership, guard_nested_membership,
)
from analysis.paper_study.structural_evaluation.run import (
    DEFAULT_PATHS, SEED, VALIDATION_LABEL, KEYS, load_inputs, subset_fields,
    write_frozen_json, save_frozen_csv, _save_frame, read_completed_fit,
    verify_dependencies, sha,
)
from model.seasonal_septoria.empirical import assessment_features, WeightedRidge
from model.seasonal_septoria.refinement import calibration_weights

DESTINATION = Path(__file__).resolve().parent
ARCHIVES = {
    "daily_or": ROOT / "analysis/paper_study/nested_canopy_daily_20261006",
    "duration_proxy": ROOT / "analysis/paper_study/nested_canopy_duration_20261006",
}
PROTOCOL = DESTINATION.parent / "ensemble_protocol_before_nested_scoring.json"
FAMILIES = ("weighted_training_mean", "leaf_rank_training_mean", "weather_development_ridge")
PENALTIES = (.1, 1., 10., 100., 1000.)
ENSEMBLE = "fixed_structural_ensemble"
BOOTSTRAP_DRAWS = 5000
FEATURES = (
    [f"leaf_rank_{i}" for i in range(1, 8)]
    + ["elapsed_days", "cumulative_tpv", "leaf_age_days", "leaf_active", "day_sin", "day_cos"]
    + [f"{name}_{window}d" for window in (7, 21, 60)
        for name in ("tmean", "rh", "rain_sum", "rain_days", "humid_days", "establishment")]
)
FEATURE_LIMITATIONS = [
    "Weather summaries use the existing fixed 7, 21, and 60 day windows through the assessment day; this is contemporaneous severity prediction rather than a specified advance-warning forecast.",
    "Leaf age and activity use the original fixed cohort/phenology assumptions, rather than the canopy component's optimized flag timing and leaf spacing.",
    "The establishment summaries inherit the original daily-OR humidity/rain response; daily mean humidity and rainfall do not identify actual leaf-wetness duration.",
    "The transferred winter/rainfed calendar and T-P-V parameters are crop scenarios where source sowing or wheat type is missing; observed stage is not an input.",
    "The ridge response is linear in standardized features before clipping; nonlinear interactions, alternative windows, forests, and additional penalty tuning are outside the finite plan.",
    "Cultivar, verified biological replication, and measurement denominators remain unavailable; all methods predict the recorded source INFECT percentage proxy.",
]


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Recovered frozen record differs: {path.name}")
    else:
        write_frozen_json(path, value)


def validate_feature_columns(columns):
    if len(set(columns)) != len(columns) or not set(columns) <= set(FEATURES):
        raise ValueError("Only declared weather/development/ordinal-leaf features are permitted.")


def safe_features(data, target_indices, phenology):
    isolated = subset_fields(data, target_indices, redact_values=True)
    if isolated.targets.value.notna().any():
        raise ValueError("Feature input disease values must be redacted.")
    frame = assessment_features(isolated, phenology)
    validate_feature_columns(frame.columns.tolist())
    if frame.columns.tolist() != FEATURES or not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError("Feature membership/order or finite weather/development values differ.")
    return frame


def fit_empirical(family, targets, features, penalty=None):
    if family not in FAMILIES:
        raise ValueError("Unsupported finite benchmark family.")
    values = targets.value.to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ValueError("Finite percentage training outcomes are required.")
    weights = calibration_weights(targets, final_weight=.5)
    pooled = float(np.average(values, weights=weights))
    fitted = dict(family=family, final_loss_weight=.5, clipping=[0., 100.],
        training_target_count=len(targets), validation_label=VALIDATION_LABEL,
        outer_outcomes_used_to_fit_or_select=False)
    if family == "weighted_training_mean":
        fitted["pooled_mean"] = pooled
    elif family == "leaf_rank_training_mean":
        frame = targets.assign(training_weight=weights)
        fitted["pooled_mean"] = pooled
        fitted["leaf_means"] = {str(int(rank)): float(np.average(part.value, weights=part.training_weight))
            for rank, part in frame.groupby("leaf_index")}
        fitted["unseen_leaf_rank_fallback"] = "weighted_pooled_training_mean"
    else:
        if penalty not in PENALTIES or features is None or len(features) != len(targets):
            raise ValueError("Ridge requires the finite declared penalty and complete training features.")
        validate_feature_columns(features.columns.tolist())
        if not np.isfinite(features.to_numpy(float)).all():
            raise ValueError("Finite training features are required.")
        model = WeightedRidge(float(penalty)).fit(features, values, weights)
        fitted.update(penalty=float(penalty), feature_names=features.columns.tolist(),
            mean=model.mean.tolist(), scale=model.scale.tolist(),
            coefficients=model.estimator.coef_.tolist(), intercept=float(model.estimator.intercept_),
            normalized_training_weights="weights * training_row_count / weights.sum()")
    return fitted


def predict_empirical(fitted, targets, features):
    if targets.value.notna().any():
        raise ValueError("All prediction disease values must be redacted.")
    family = fitted["family"]
    if family == "weighted_training_mean":
        values = np.full(len(targets), fitted["pooled_mean"])
    elif family == "leaf_rank_training_mean":
        values = np.array([fitted["leaf_means"].get(str(int(rank)), fitted["pooled_mean"])
            for rank in targets.leaf_index], float)
    elif family == "weather_development_ridge":
        if features is None or features.columns.tolist() != fitted["feature_names"]:
            raise ValueError("Ridge prediction feature membership/order differs.")
        validate_feature_columns(features.columns.tolist())
        x = features.to_numpy(float)
        values = ((x - np.asarray(fitted["mean"])) / np.asarray(fitted["scale"])) @ np.asarray(fitted["coefficients"]) + fitted["intercept"]
    else:
        raise ValueError("Unsupported saved benchmark family.")
    values = np.clip(values, 0., 100.)
    if values.shape != (len(targets),) or not np.isfinite(values).all():
        raise ValueError("Complete finite benchmark predictions are required.")
    return values


def reconstruct_memberships(targets, membership, require_nine=True):
    identity_columns = [name for name in ("field_id", "dataset_id", "site_id", "season_year",
        "endpoint_series", "date", "coordinate_year") if name in targets and name in membership]
    for row in membership[["target_index", *identity_columns]].drop_duplicates().itertuples(index=False, name=None):
        index, *expected = row
        if int(index) != index or not 0 <= index < len(targets):
            raise ValueError("Frozen membership has an invalid target index.")
        actual = targets.iloc[int(index)]
        for name, value in zip(identity_columns, expected):
            same = pd.Timestamp(actual[name]) == pd.Timestamp(value) if name == "date" else str(actual[name]) == str(value)
            if not same:
                raise ValueError(f"Frozen target identity/membership differs: row {index}, {name}.")
    outers, inners = [], {}
    for name in membership.outer_fold.drop_duplicates():
        selected = membership[membership.outer_fold.eq(name)]
        outer_rows = selected[selected.level.eq("outer")]
        def rows(frame, role):
            return frame.loc[frame.role.eq(role), "target_index"].to_numpy(int)
        kind = "forward" if str(name).startswith("forward_") else "spatial"
        year = int(str(name).split("_")[-1]) if kind == "forward" else None
        outer = EvaluationFold(name, kind, rows(outer_rows, "training"),
            rows(outer_rows, "validation"), rows(outer_rows, "excluded"), year)
        guard_outer_membership(targets, outer)
        outers.append(outer)
        inner_rows = selected[selected.level.eq("inner")]
        inners[name] = []
        for fold_name in inner_rows.fold.drop_duplicates():
            part = inner_rows[inner_rows.fold.eq(fold_name)]
            inner = EvaluationFold(fold_name, "inner", rows(part, "training"), rows(part, "validation"))
            guard_nested_membership(targets, outer, inner)
            inners[name].append(inner)
        if len(inners[name]) != 3:
            raise ValueError("Each frozen outer membership requires exactly three inner folds.")
    if require_nine and (len(outers) != 9 or sum(item.kind == "spatial" for item in outers) != 5):
        raise ValueError("The completed nine-outer-fold frozen memberships are required.")
    return outers, inners


def fit_record(path, family, targets, features, indices, identity, stage, penalty=None):
    record_identity = dict(run_identity=identity, family=family, penalty=penalty,
        global_training_target_indices=np.asarray(indices, int).tolist(), stage=stage)
    fitted = read_completed_fit(path, record_identity)
    if fitted is None:
        fitted = fit_empirical(family, targets, features, penalty)
        write_frozen_json(path, dict(status="fit_complete", identity=record_identity, fitted=fitted,
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            outer_outcomes_used_to_fit_or_select=False, prior_historical_outcome_exposure=True))
    return fitted


def select_and_refit(outer, family, data, phenology, inners, identity, feature_cache):
    folder = DESTINATION / "outer_folds" / outer.name
    folder.mkdir(parents=True, exist_ok=True)
    settings = PENALTIES if family == "weather_development_ridge" else (None,)
    selections = []
    def features(indices):
        key = tuple(np.asarray(indices, int).tolist())
        if key not in feature_cache:
            feature_cache[key] = safe_features(data, np.asarray(indices, int), phenology)
        return feature_cache[key]
    for penalty in settings:
        predictions = []
        for inner in inners:
            guard_nested_membership(data.targets, outer, inner)
            training = subset_fields(data, inner.train)
            heldout = subset_fields(data, inner.test, redact_values=True)
            x_train = features(inner.train) if penalty is not None else None
            x_test = features(inner.test) if penalty is not None else None
            setting_name = f"penalty{penalty:g}" if penalty is not None else "fixed"
            path = folder / f"{family}_{setting_name}_{inner.name}_fit.json"
            fitted = fit_record(path, family, training.targets, x_train, inner.train, identity,
                dict(outer=outer.name, inner=inner.name, role="inner_training"), penalty)
            forecast = predict_empirical(fitted, heldout.targets, x_test)
            part = data.targets.iloc[inner.test].copy()
            part["model"], part["inner_fold"], part["outer_fold"] = family, inner.name, outer.name
            part["predicted_percent"] = forecast
            part["ridge_penalty"] = penalty
            _save_frame(path.with_name(path.stem + "_predictions.parquet"), part)
            predictions.append(part)
        pooled = pd.concat(predictions, ignore_index=True)
        selections.append(dict(penalty=penalty,
            final_leaf_rmse=score(endpoint(pooled, True, False))["rmse"],
            all_assessment_rmse=score(endpoint(pooled, False, False))["rmse"],
            inner_prediction_count=len(pooled)))
    chosen = min(selections, key=lambda row: row["final_leaf_rmse"])
    record = dict(model=family, outer_fold=outer.name, candidates=selections, selected=chosen,
        selection_required=family == "weather_development_ridge",
        criterion="minimum pooled inner final-assessment all-ordinal-leaf RMSE",
        validation_label=VALIDATION_LABEL, outer_outcomes_used_to_fit_or_select=False)
    selection_path = folder / f"{family}_selection.json"
    freeze_json(selection_path, record)
    training = subset_fields(data, outer.train)
    path = folder / f"frozen_{family}_outer_fit.json"
    fitted = fit_record(path, family, training.targets,
        features(outer.train) if family == "weather_development_ridge" else None,
        outer.train, dict(identity, selection_sha256=sha(selection_path)),
        dict(outer=outer.name, role="outer_refit"), chosen["penalty"])
    print(f"{outer.name} {family} frozen; selected penalty {chosen['penalty']}; inner final RMSE {chosen['final_leaf_rmse']:.6f}", flush=True)
    return dict(path=str(path.relative_to(DESTINATION)), sha256=sha(path),
        selected=chosen, family=family)


def structural_ensemble():
    parts = []
    for archive in ARCHIVES.values():
        all_models = pd.read_parquet(archive / "all_out_of_fold_predictions.parquet")
        parts.append(all_models[all_models.model.eq("structural_canopy")].copy())
    left, right = parts
    keys = ["outer_fold", *KEYS]
    merged = left.merge(right[keys + ["predicted_percent", "value", "site_id", "source", "evaluation_kind"]],
        on=keys, how="outer", validate="one_to_one", suffixes=("", "_duration"), indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Frozen canopy component OOF memberships differ.")
    for name in ["value", "site_id", "source", "evaluation_kind"]:
        if not merged[name].eq(merged[f"{name}_duration"]).all():
            raise ValueError("Frozen canopy OOF outcome or location identity differs.")
    merged["predicted_percent"] = .5 * merged.predicted_percent + .5 * merged.predicted_percent_duration
    merged["model"] = ENSEMBLE
    return merged.drop(columns=[name for name in merged if name.endswith("_duration")] + ["_merge"])


def summarize(predictions):
    metrics, comparisons, cluster_membership, draw_arrays = [], [], [], {}
    random = np.random.default_rng(SEED)
    reporting = predictions.assign(report_scope="pooled")
    sources = predictions.assign(report_scope=predictions.source)
    for (kind, scope), group in pd.concat([reporting, sources], ignore_index=True).groupby(["evaluation_kind", "report_scope"]):
        for final in (False, True):
            for upper in (False, True):
                name = "final_numeric_assessment" if final else "all_assessments"
                leaf_scope = "upper_three" if upper else "all_ordinal_leaves"
                selected = {model: endpoint(part, final, upper) for model, part in group.groupby("model")}
                if any(part.empty for part in selected.values()):
                    continue
                for model, part in selected.items():
                    metrics.append(dict(evaluation_kind=kind, report_scope=scope, model=model,
                        endpoint=name, leaf_scope=leaf_scope, validation_label=VALIDATION_LABEL, **score(part)))
                canopy = selected[ENSEMBLE]
                clusters = np.array(sorted(canopy.site_id.unique()))
                draws = random.integers(0, len(clusters), (BOOTSTRAP_DRAWS, len(clusters)))
                key = f"{kind}__{scope}__{name}__{leaf_scope}"
                draw_arrays[key] = draws
                cluster_membership.extend(dict(draw_key=key, cluster_index=i, site_id=site) for i, site in enumerate(clusters))
                for family in FAMILIES:
                    baseline = selected[family]
                    matched = canopy.merge(baseline[["outer_fold", *KEYS, "predicted_percent"]],
                        on=["outer_fold", *KEYS], validate="one_to_one", suffixes=("", "_baseline"))
                    if len(matched) != len(canopy) or len(matched) != len(baseline):
                        raise ValueError("Paired outer benchmark membership differs.")
                    matched["comparator_percent"] = matched.predicted_percent_baseline
                    matched["coordinate_year"] = matched.site_id
                    comparisons.append(dict(evaluation_kind=kind, report_scope=scope, endpoint=name,
                        leaf_scope=leaf_scope, structural_model=ENSEMBLE, benchmark=family,
                        difference="fixed structural ensemble minus benchmark",
                        bootstrap_unit="complete_location_history", cluster_count=len(clusters),
                        bootstrap_draws=BOOTSTRAP_DRAWS, shared_draw_key=key,
                        validation_label=VALIDATION_LABEL, **paired_intervals(matched, clusters, draws)))
    return pd.DataFrame(metrics), pd.DataFrame(comparisons), pd.DataFrame(cluster_membership), draw_arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    contract = DESTINATION / "configuration_before_fitting.json"
    if (DESTINATION / "receipt.json").exists():
        raise FileExistsError("Completed nested benchmark archives are immutable.")
    if contract.exists() and not args.resume:
        raise FileExistsError("Recover the registered finite benchmark plan with --resume.")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["weights"] != {"daily_or": .5, "duration_proxy": .5}:
        raise ValueError("The prior registered equal canopy weights are required.")
    prior_paths = []
    for folder in ARCHIVES.values():
        receipt = json.loads((folder / "receipt.json").read_text())
        if (receipt["status"] != "complete"
                or receipt["frozen_manifest_sha256"] != sha(folder / "all_outer_fits_frozen_before_scoring.json")
                or receipt["output_sha256"]["all_out_of_fold_predictions.parquet"] != sha(folder / "all_out_of_fold_predictions.parquet")):
            raise ValueError("A completed structural nested archive no longer agrees with its frozen receipt.")
        prior_paths.extend(folder / name for name in ["receipt.json", "configuration_before_fitting.json",
            "nested_membership_before_fitting.csv", "all_source_target_membership.parquet",
            "all_out_of_fold_predictions.parquet", "all_outer_fits_frozen_before_scoring.json"])
    memberships = [pd.read_csv(folder / "nested_membership_before_fitting.csv") for folder in ARCHIVES.values()]
    pd.testing.assert_frame_equal(*memberships)
    dependencies = [*DEFAULT_PATHS.values(), PROTOCOL, *prior_paths,
        *sorted(DESTINATION.glob("*.py")),
        *sorted((ROOT / "analysis/paper_study/structural_evaluation").glob("*.py")),
        ROOT / "analysis/paper_study/run_structural_canopy.py",
        ROOT / "analysis/paper_study/run_empirical_benchmarks.py",
        ROOT / "analysis/paper_study/refine_seasonal_accuracy.py",
        ROOT / "analysis/paper_study/calibrate_seasonal.py",
        *sorted((ROOT / "model/seasonal_septoria").glob("*.py")),
        *sorted((ROOT / "process_model").glob("*.py")),
        *sorted((ROOT / "process_model/_support").glob("*.py"))]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in dependencies}
    configuration = dict(validation_label=VALIDATION_LABEL, untouched_test=False,
        prior_historical_outcome_exposure=True, benchmarks_added_after_structural_nested_scores_available=True,
        families=list(FAMILIES), ridge_penalties=list(PENALTIES), final_loss_weight=.5,
        prediction_clipping=[0., 100.], ridge_weight_standardization="normalized .5 final/all training weights only",
        selection="minimum pooled inner final-assessment all-ordinal-leaf RMSE",
        mean_selection="fixed pooled and leaf-rank training estimates; no tuning",
        outer_folds=9, inner_folds_per_outer=3, expected_fit_records=216,
        memberships="exact frozen structural nested row memberships in both archives",
        feature_names=FEATURES, feature_limitations=FEATURE_LIMITATIONS,
        feature_weather_known_through="assessment day, with subsequent forcing excluded from each feature",
        prohibited_features=["disease outcomes", "observed stage", "future weather", "site identity", "country identity"],
        structural_ensemble_weights={"daily_or": .5, "duration_proxy": .5},
        bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_unit="complete_location_history",
        random_seed=SEED, input_and_source_sha256=hashes,
        runtime_versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__))
    freeze_json(contract, configuration)
    save_frozen_csv(DESTINATION / "exact_nested_membership_before_fitting.csv", memberships[0])
    snapshot = DESTINATION / "source_snapshot"
    for path in dependencies:
        if path.suffix != ".py" and path not in (PROTOCOL, DEFAULT_PATHS["phenology"]):
            continue
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != sha(path):
                raise ValueError("Recovered source snapshot differs.")
        else:
            with target.open("xb") as stream:
                stream.write(path.read_bytes())
    data, _, _, _ = load_inputs(DEFAULT_PATHS)
    columns = ["field_id", "endpoint_series", "date", "value", "site_id", "season_year", "source"]
    for folder in ARCHIVES.values():
        archived = pd.read_parquet(folder / "all_source_target_membership.parquet")
        pd.testing.assert_frame_equal(data.targets[columns].reset_index(drop=True), archived[columns].reset_index(drop=True), check_dtype=False)
    outers, inners = reconstruct_memberships(data.targets, memberships[0])
    _save_frame(DESTINATION / "all_source_target_membership.parquet", data.targets)
    identity = dict(configuration_sha256=sha(contract),
        exact_membership_sha256=sha(DESTINATION / "exact_nested_membership_before_fitting.csv"),
        target_membership_sha256=sha(DESTINATION / "all_source_target_membership.parquet"))
    verify_dependencies(hashes)
    if args.prepare_only:
        print("Registered finite empirical benchmarks: exact 9 outer/27 inner memberships; 2305 targets; 216 fits; no fits or scores calculated.", flush=True)
        return
    phenology = json.loads(DEFAULT_PATHS["phenology"].read_text())
    frozen, feature_cache = {}, {}
    for outer in outers:
        frozen[outer.name] = {family: select_and_refit(outer, family, data, phenology,
            inners[outer.name], identity, feature_cache) for family in FAMILIES}
    verify_dependencies(hashes)
    manifest = DESTINATION / "all_outer_benchmarks_frozen_before_scoring.json"
    freeze_json(manifest, dict(outer_fits=frozen, identity=identity, validation_label=VALIDATION_LABEL,
        outer_outcomes_used_to_fit_or_select=False, prior_historical_outcome_exposure=True))
    predictions = []
    for outer in outers:
        heldout = subset_fields(data, outer.test, redact_values=True)
        x_test = safe_features(data, outer.test, phenology)
        for family in FAMILIES:
            record = frozen[outer.name][family]
            path = DESTINATION / record["path"]
            if sha(path) != record["sha256"]:
                raise ValueError("A frozen outer benchmark fit changed.")
            fitted = json.loads(path.read_text())["fitted"]
            forecast = predict_empirical(fitted, heldout.targets, x_test if family == "weather_development_ridge" else None)
            part = data.targets.iloc[outer.test].copy()
            part["outer_fold"], part["evaluation_kind"], part["model"] = outer.name, outer.kind, family
            part["predicted_percent"], part["fit_sha256"] = forecast, record["sha256"]
            predictions.append(part)
    ensemble = structural_ensemble()
    empirical = pd.concat(predictions, ignore_index=True)
    reference_keys = ["outer_fold", *KEYS]
    expected_membership = set(map(tuple, ensemble[reference_keys].to_numpy()))
    for family, frame in empirical.groupby("model"):
        if len(frame) != len(ensemble) or set(map(tuple, frame[reference_keys].to_numpy())) != expected_membership:
            raise ValueError(f"Exact canopy/benchmark outer membership differs for {family}.")
    combined = pd.concat([empirical, ensemble], ignore_index=True)
    _save_frame(DESTINATION / "all_out_of_fold_predictions.parquet", combined)
    metrics, comparisons, clusters, draws = summarize(combined)
    _save_frame(DESTINATION / "severity_metrics.parquet", metrics)
    _save_frame(DESTINATION / "paired_complete_location_history_differences.parquet", comparisons)
    save_frozen_csv(DESTINATION / "severity_metrics.csv", metrics)
    save_frozen_csv(DESTINATION / "paired_complete_location_history_differences.csv", comparisons)
    save_frozen_csv(DESTINATION / "bootstrap_cluster_membership.csv", clusters)
    draw_path = DESTINATION / "shared_cluster_bootstrap_draws.npz"
    if draw_path.exists():
        with np.load(draw_path) as prior:
            if set(prior.files) != set(draws) or any(not np.array_equal(prior[name], value) for name, value in draws.items()):
                raise ValueError("Recovered paired bootstrap draws differ.")
    else:
        np.savez_compressed(draw_path, **draws)
    for records in frozen.values():
        for record in records.values():
            if sha(DESTINATION / record["path"]) != record["sha256"]:
                raise ValueError("An outer benchmark record changed during scoring.")
    verify_dependencies(hashes)
    fit_count = len(list((DESTINATION / "outer_folds").glob("*/*_fit.json")))
    if fit_count != 216:
        raise ValueError("The completed finite benchmark fit count differs.")
    outputs = [*DESTINATION.glob("*.parquet"), *DESTINATION.glob("*.csv"), draw_path, manifest]
    write_frozen_json(DESTINATION / "receipt.json", dict(status="complete", validation_label=VALIDATION_LABEL,
        untouched_test=False, prior_historical_outcome_exposure=True,
        fit_count=fit_count, outer_fold_count=9, inner_fold_count=27,
        prediction_count=len(combined), model_prediction_count=len(ensemble), feature_count=len(FEATURES),
        exact_prior_memberships_verified=True, physical_holdout_disease_redaction=True,
        feature_limitations=FEATURE_LIMITATIONS, elapsed_seconds=time.perf_counter() - started,
        configuration_sha256=sha(contract), outer_fit_manifest_sha256=sha(manifest),
        output_sha256={path.name: sha(path) for path in outputs}))
    print("Retrospective nested empirical benchmarks complete; mean, leaf-rank mean, ridge, and fixed canopy ensemble share exact outer memberships.", flush=True)


if __name__ == "__main__":
    main()
