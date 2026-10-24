"""Fresh, fixed autonomous comparators on reused retrospective disease records.

Models receive trailing realized weather, calendar values and recorded leaf rank.
No field disease history or fitted crop features enter any candidate. Joint
earlier-year/held-site evaluation is computational qualification evidence only;
these reused outcomes cannot establish prospective or physiological validation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from paper.analyses.source_support.metrics import hierarchical_weights, observed_windows
from .provenance import auth_source, prepare_destination

MODELS = ["training_leaf_mean", "calendar_ridge", "weather_calendar_rf"]
FEATURES = [f"{name}_{'log1p_' if name == 'rain' else ''}mean_{width}d"
            for width in (7, 21, 42) for name in ("temperature", "humidity", "rain")]
FEATURES += ["elapsed_days_scaled", "calendar_sin", "calendar_cos"]
INPUT_SCHEMA = FEATURES + [f"crop_cdf_{stage}" for stage in (10, 31, 51, 87)]
TEST_YEARS = list(range(2015, 2020))
FOLDS = 5
SEED = 20261013
INPUT_DIR = "data/final_study/evidence/disease_joint_forecast/inputs"
CONFIG = dict(site_folds=FOLDS, test_years=TEST_YEARS, random_seed=SEED,
    calendar_ridge=dict(alpha=10.0, training_only_weighted_standardization=True),
    weather_calendar_rf=dict(n_estimators=96, min_samples_leaf=8, max_depth=8,
        max_features=1.0, n_jobs=1), outcome_tuning=False,
    weighting="unit coordinate-year mass, split equally across fields, leaves and records",
    unknown_leaf="same weighted pooled training mean for every candidate; flagged explicitly",
    forecast_information="realized weather through assessment date; assumed sowing calendar",
    uncertainty="paired site bootstrap conditional on fixed fitted predictions and common support")


def assessment_features(observations: pd.DataFrame, arrays) -> tuple[np.ndarray, np.ndarray]:
    """Select declared weather/calendar values at the one-based recorded day.

    The final four crop features are never indexed or checked for finiteness.
    Unavailable predictor rows remain NaN rather than being clipped to a day.
    """
    if list(arrays["feature_names"].astype(str)) != INPUT_SCHEMA:
        raise ValueError("Unexpected retained feature schema")
    features = arrays["features"]
    valid = arrays["valid"]
    if features.ndim != 3 or features.shape[-1] != len(INPUT_SCHEMA) or valid.shape != features.shape[:2]:
        raise ValueError("Invalid retained feature dimensions")
    field = observations.field_index.to_numpy(int)
    day = observations.day_index.to_numpy(int) - 1
    available = (field >= 0) & (field < len(features)) & (day >= 0) & (day < features.shape[1])
    selected = np.full((len(observations), len(FEATURES)), np.nan)
    index = np.flatnonzero(available)
    selected[index] = features[field[index], day[index], :len(FEATURES)]
    available[index] &= valid[field[index], day[index]] & (day[index] < arrays["horizons"][field[index]])
    available &= np.isfinite(selected).all(axis=1)
    selected[~available] = np.nan
    return selected, available


def site_fold(site: str) -> int:
    """Stable site assignment independent of dates, outcomes and Python hashing."""
    return int(hashlib.sha256(str(site).encode()).hexdigest(), 16) % FOLDS


def _validate_frame(frame: pd.DataFrame, features: np.ndarray) -> None:
    required = ["global_target_index", "site_id", "field_id", "coordinate_year", "season_year",
                "source", "leaf_index", "endpoint_series", "date", "observed_percent"]
    if frame[required].isna().any().any() or frame.global_target_index.duplicated().any():
        raise ValueError("Complete unique observation identities required")
    if features.shape != (len(frame), len(FEATURES)):
        raise ValueError("Assessment predictor dimensions changed")
    if not np.isfinite(frame.observed_percent).all() or not frame.observed_percent.between(0, 100).all():
        raise ValueError("Finite percentage observations required")
    if not frame.leaf_index.isin(range(7)).all():
        raise ValueError("Unknown retained leaf categories")
    if frame.groupby("coordinate_year").site_id.nunique().gt(1).any():
        raise ValueError("Coordinate-year hierarchy crosses sites")
    if frame.groupby("field_id").site_id.nunique().gt(1).any():
        raise ValueError("Field identity crosses sites")
    if not np.array_equal(pd.to_datetime(frame.date).dt.year, frame.season_year):
        raise ValueError("Actual observation date disagrees with declared assessment season")
    if frame.duplicated(["field_id", "endpoint_series", "leaf_index", "date"]).any():
        raise ValueError("Duplicate source-series observation dates")


def autonomous_predictions(frame: pd.DataFrame, features: np.ndarray):
    """Fit the fixed candidate set once in each earlier-year/held-site fold.

    Returned membership lists every training and test record, including dates,
    site identities and years. Train-only preprocessing and weighted losses are
    fixed in CONFIG. No outcomes select folds, predictors or hyperparameters.
    """
    frame = frame.reset_index(drop=True).copy()
    features = np.asarray(features, float)
    _validate_frame(frame, features)
    frame["site_fold"] = frame.site_id.map(site_fold)
    frame["predictors_available"] = np.isfinite(features).all(axis=1)
    frame["outer_fold"] = ""
    frame["unseen_leaf"] = False
    frame["prediction_status"] = "not_evaluated_training_year"
    for name in MODELS:
        frame[name] = np.nan
    folds, membership = [], []
    leaf_encoding = np.eye(7)[frame.leaf_index.to_numpy(int)]
    calendar = np.column_stack([features[:, 9:12], leaf_encoding])
    weather = np.column_stack([features, leaf_encoding])
    for year in TEST_YEARS:
        for fold in range(FOLDS):
            fold_id = f"year{year}_sitefold{fold}"
            test = frame.season_year.eq(year) & frame.site_fold.eq(fold)
            train = frame.season_year.lt(year) & frame.site_fold.ne(fold) & frame.predictors_available
            train_rows, test_rows = frame.loc[train], frame.loc[test]
            train_sites, test_sites = set(train_rows.site_id), set(test_rows.site_id)
            if train_sites & test_sites or (len(train_rows) and len(test_rows) and train_rows.season_year.max() >= year):
                raise ValueError("Temporal or held-site leakage in autonomous fold")
            record = dict(fold_id=fold_id, test_year=year, site_fold=fold,
                train_rows=len(train_rows), test_rows=len(test_rows),
                train_sites=len(train_sites), test_sites=len(test_sites),
                min_training_year=int(train_rows.season_year.min()) if len(train_rows) else None,
                max_training_year=int(train_rows.season_year.max()) if len(train_rows) else None,
                first_training_date=str(train_rows.date.min()) if len(train_rows) else None,
                last_training_date=str(train_rows.date.max()) if len(train_rows) else None,
                first_test_date=str(test_rows.date.min()) if len(test_rows) else None,
                last_test_date=str(test_rows.date.max()) if len(test_rows) else None,
                train_leaf_categories=json.dumps(sorted(train_rows.leaf_index.unique().astype(int).tolist())),
                excluded_same_fold_earlier_records=int((frame.season_year.lt(year) & frame.site_fold.eq(fold)).sum()),
                excluded_invalid_training_records=int((frame.season_year.lt(year) & frame.site_fold.ne(fold) & ~frame.predictors_available).sum()),
                provenance_passed=True,
                status="no_test_records" if not len(test_rows) else "no_training_records" if not len(train_rows) else "fitted")
            folds.append(record)
            if not len(test_rows):
                continue
            for role, rows in [("train", train_rows), ("test", test_rows)]:
                for row in rows[["global_target_index", "field_id", "site_id", "season_year", "date", "leaf_index"]].to_dict("records"):
                    membership.append(dict(fold_id=fold_id, role=role, **row))
            frame.loc[test, "outer_fold"] = fold_id
            frame.loc[test, "prediction_status"] = "no_training_records" if not len(train_rows) else "predictors_unavailable"
            if not len(train_rows):
                continue
            weights = hierarchical_weights(train_rows)
            fit_weights = weights * len(weights) / weights.sum()
            targets = train_rows.observed_percent.to_numpy(float)
            pooled = float(np.average(targets, weights=weights))
            leaves = set(train_rows.leaf_index)
            usable = test & frame.predictors_available
            unseen = usable & ~frame.leaf_index.isin(leaves)
            known = usable & frame.leaf_index.isin(leaves)
            frame.loc[unseen, "unseen_leaf"] = True
            frame.loc[unseen, "prediction_status"] = "unseen_leaf_pooled_fallback"
            frame.loc[unseen, MODELS] = pooled
            frame.loc[known, "prediction_status"] = "fitted_known_leaf"
            for leaf in leaves:
                support = train_rows.leaf_index.eq(leaf).to_numpy()
                frame.loc[known & frame.leaf_index.eq(leaf), MODELS[0]] = np.average(targets[support], weights=weights[support])
            if not known.any():
                continue
            with threadpool_limits(limits=1):
                scaler = StandardScaler().fit(calendar[train], sample_weight=fit_weights)
                ridge = Ridge(alpha=CONFIG["calendar_ridge"]["alpha"])
                ridge.fit(scaler.transform(calendar[train]), targets, sample_weight=fit_weights)
                frame.loc[known, MODELS[1]] = np.clip(ridge.predict(scaler.transform(calendar[known])), 0, 100)
                forest = RandomForestRegressor(random_state=SEED, **CONFIG["weather_calendar_rf"])
                forest.fit(weather[train], targets, sample_weight=fit_weights)
                frame.loc[known, MODELS[2]] = np.clip(forest.predict(weather[known]), 0, 100)
    test_frame = frame.loc[frame.season_year.isin(TEST_YEARS)].copy()
    for j, name in enumerate(FEATURES):
        test_frame[name] = features[frame.season_year.isin(TEST_YEARS), j]
    return test_frame.reset_index(drop=True), pd.DataFrame(folds), pd.DataFrame(membership)


def _score(frame: pd.DataFrame, model: str) -> dict:
    common = np.isfinite(frame[MODELS].to_numpy(float)).all(axis=1) if len(frame) else np.zeros(0, bool)
    available = frame.loc[common]
    result = dict(n_required=len(frame), n_available=len(available), n_unavailable=int((~common).sum()),
        fields=available.field_id.nunique(), sites=available.site_id.nunique(),
        coordinate_years=available.coordinate_year.nunique(),
        status="no_paired_support" if not len(available) else "incomplete_support" if not common.all() else "insufficient_sites" if available.site_id.nunique() < 2 else "ok",
        rmse=np.nan, mae=np.nan, bias=np.nan, r2=np.nan, observed_mean=np.nan, prediction_mean=np.nan)
    if not len(available):
        return result
    weight = hierarchical_weights(available)
    observed = available.observed_percent.to_numpy(float)
    predicted = available[model].to_numpy(float)
    error = predicted - observed
    mean = np.average(observed, weights=weight)
    variance = np.average((observed - mean) ** 2, weights=weight)
    mse = np.average(error ** 2, weights=weight)
    result.update(rmse=float(np.sqrt(mse)), mae=float(np.average(np.abs(error), weights=weight)),
        bias=float(np.average(error, weights=weight)),
        r2=float(1 - mse / variance) if variance > 0 else np.nan,
        observed_mean=float(mean), prediction_mean=float(np.average(predicted, weights=weight)))
    return result


def _endpoints(frame: pd.DataFrame):
    windows = observed_windows(frame, MODELS) if len(frame) else pd.DataFrame()
    if windows.empty:
        windows = pd.DataFrame(columns=list(frame.columns) + ["observed_audpc"] + [name + "_audpc" for name in MODELS])
    integral = windows.copy()
    integral["observed_percent"] = integral.observed_audpc
    for name in MODELS:
        integral[name] = integral[name + "_audpc"]
    return [("assessment", frame), ("window_mean", windows), ("window_integral", integral)]


def _populations(frame: pd.DataFrame):
    yield "pooled", 0, frame
    for source in sorted(frame.source.unique()):
        yield source, 0, frame.loc[frame.source.eq(source)]
    for year in TEST_YEARS:
        yield "pooled", year, frame.loc[frame.season_year.eq(year)]


def score_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Common complete-candidate support, with unavailable denominators visible."""
    rows = []
    for source, year, population in _populations(frame):
        for endpoint, data in _endpoints(population):
            for model in MODELS:
                rows.append(dict(source=source, test_year=year, endpoint=endpoint, model=model, **_score(data, model)))
    return pd.DataFrame(rows)


def paired_intervals(frame: pd.DataFrame, *, bootstrap_draws: int = 1000) -> pd.DataFrame:
    """Paired site resampling conditional on predictions, with no refitting.

    Cluster sufficient statistics preserve each original coordinate-year's unit
    mass. Duplicate sampled sites duplicate their complete site/year histories.
    Every candidate uses the identical complete-record support and site draw.
    """
    if bootstrap_draws < 1:
        raise ValueError("Positive bootstrap_draws required")
    rows = []
    for source, year, population in _populations(frame):
        for endpoint, data in _endpoints(population):
            common = np.isfinite(data[MODELS].to_numpy(float)).all(axis=1) if len(data) else np.zeros(0, bool)
            paired = data.loc[common].copy()
            sites = sorted(paired.site_id.unique())
            enough = len(sites) >= 2
            weights = hierarchical_weights(paired) if len(paired) else np.zeros(0)
            point = {name: _score(data, name)["rmse"] for name in MODELS}
            draws = {}
            if enough:
                mass = np.array([weights[paired.site_id.eq(site)].sum() for site in sites])
                chosen = np.random.default_rng(SEED).integers(0, len(sites), (bootstrap_draws, len(sites)))
                denominator = mass[chosen].sum(axis=1)
                for name in MODELS:
                    err2 = (paired[name].to_numpy(float) - paired.observed_percent.to_numpy(float)) ** 2
                    numerator = np.array([(weights * err2)[paired.site_id.eq(site)].sum() for site in sites])
                    draws[name] = np.sqrt(numerator[chosen].sum(axis=1) / denominator)
            for candidate, reference in itertools.combinations(MODELS, 2):
                # Candidate order follows increasing model complexity.
                candidate, reference = reference, candidate
                interval = np.quantile(draws[candidate] - draws[reference], [.025, .975]) if enough else [np.nan, np.nan]
                rows.append(dict(source=source, test_year=year, endpoint=endpoint,
                    candidate=candidate, reference=reference, metric="rmse_difference",
                    value=point[candidate] - point[reference], lower=float(interval[0]), upper=float(interval[1]),
                    sites=len(sites), n_required=len(data), n_available=len(paired),
                    requested_draws=bootstrap_draws, valid_draws=bootstrap_draws if enough else 0,
                    status="no_paired_support" if not len(paired) else "insufficient_sites" if not enough else "incomplete_support" if not common.all() else "ok",
                    conditional_on_fitted_predictions=True, model_refitting=False))
    return pd.DataFrame(rows)


def weather_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed weather bins; assessment diagnostics use common model support."""
    definitions = [(FEATURES[0], [-np.inf, 10, 15, np.inf], ["below_10_C", "10_to_below_15_C", "at_least_15_C"]),
        (FEATURES[1], [-np.inf, 70, 85, np.inf], ["below_70_percent", "70_to_below_85_percent", "at_least_85_percent"]),
        (FEATURES[2], [-np.inf, np.log1p(.1), np.log1p(2), np.inf], ["below_0.1_mm_per_day", "0.1_to_below_2_mm_per_day", "at_least_2_mm_per_day"])]
    rows = []
    for variable, bins, labels in definitions:
        strata = pd.cut(frame[variable], bins=bins, labels=labels, right=False)
        for label in labels + ["unavailable"]:
            selected = strata.isna() if label == "unavailable" else strata.eq(label)
            for name in MODELS:
                rows.append(dict(variable=variable, stratum=label, model=name,
                    endpoint="assessment", **_score(frame.loc[selected], name)))
    return pd.DataFrame(rows)


def _authenticated_inputs(root: Path):
    return [auth_source(root, f"{INPUT_DIR}/{filename}")
            for filename in ("features.npz", "observations.csv", "fields.csv")]


def run(output: Path, *, root: Path, bootstrap_draws: int = 1000) -> dict:
    """Write authenticated fresh benchmark results to a new output directory."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Benchmark output already exists: " + str(output))
    if bootstrap_draws < 1:
        raise ValueError("Positive bootstrap_draws required")
    authenticated = _authenticated_inputs(root)
    base = root / INPUT_DIR
    observations = pd.read_csv(base / "observations.csv", float_precision="round_trip")
    fields = pd.read_csv(base / "fields.csv", float_precision="round_trip")
    if fields.field_index.duplicated().any() or fields.field_id.duplicated().any():
        raise ValueError("Unique field identities required")
    observations = observations.merge(fields[["field_index", "field_id", "site_id", "season_year", "sowing_date"]],
        on=["field_index", "field_id", "site_id"], validate="many_to_one", how="left")
    if observations.season_year.isna().any():
        raise ValueError("Observation field identities do not match retained fields")
    observations["season_year"] = observations.season_year.astype(int)
    days = (pd.to_datetime(observations.date) - pd.to_datetime(observations.sowing_date)).dt.days + 1
    if not np.array_equal(days, observations.day_index):
        raise ValueError("Recorded date and one-based assessment day do not align")
    with np.load(base / "features.npz", allow_pickle=False) as arrays:
        ordered = fields.sort_values("field_index")
        if not np.array_equal(ordered.field_index, np.arange(len(ordered))) or not np.array_equal(ordered.field_id.to_numpy(str), arrays["field_ids"]):
            raise ValueError("NPZ field indices do not match retained field identities")
        features, available = assessment_features(observations, arrays)
    predictions, folds, membership = autonomous_predictions(observations, features)
    summary = score_table(predictions)
    intervals = paired_intervals(predictions, bootstrap_draws=bootstrap_draws)
    windows = observed_windows(predictions, MODELS)
    strata = weather_strata(predictions)
    complete = bool(np.isfinite(predictions[MODELS].to_numpy(float)).all())
    receipt = dict(stage="autonomous_benchmark", schema_version=1,
        authenticated_inputs=authenticated, configuration={**CONFIG, "bootstrap_draws": bootstrap_draws},
        predictors=FEATURES, forbidden_predictors_excluded=INPUT_SCHEMA[12:],
        input_records=len(observations), test_records=len(predictions),
        training_only_records=int((~observations.season_year.isin(TEST_YEARS)).sum()),
        observation_identity_closure=bool(set(predictions.global_target_index) ==
            set(observations.loc[observations.season_year.isin(TEST_YEARS), "global_target_index"])),
        crop_predictors_used=False,
        test_sites=int(predictions.site_id.nunique()), observed_windows=len(windows),
        incomplete_predictor_records=int((~available).sum()),
        pooled_fallback_records=int(predictions.unseen_leaf.sum()),
        provenance_passed=bool(folds.provenance_passed.all()), complete_prediction_support=complete,
        qualification=dict(climate_ready=False, prospective_validation=False,
            physiological_validation=False, reused_outcomes=True,
            evidence="fresh chronology/site-separated predictions on reused retrospective records",
            limits=["realized assessment-date weather rather than issued weather forecasts",
                "assumed sowing calendars rather than verified field sowing observations",
                "recorded leaf ranks require compatible projection leaf mapping",
                "earlier-year support begins with 2014 only",
                "fixed candidate comparisons do not establish physiological canopy or yield response",
                "site intervals condition on fitted models, fixed support and reused outcomes"]),
        output_files={})
    prepare_destination(output, root)
    for filename, table in [("predictions.csv", predictions), ("folds.csv", folds),
        ("fold_membership.csv", membership), ("summary.csv", summary),
        ("paired_intervals.csv", intervals), ("observed_windows.csv", windows),
        ("weather_strata.csv", strata)]:
        path = output / filename
        table.to_csv(path, index=False)
        receipt["output_files"][filename] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size, rows=len(table))
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt
