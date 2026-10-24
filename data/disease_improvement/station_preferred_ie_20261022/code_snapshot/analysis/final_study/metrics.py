"""Recompute frozen final-paper endpoints from retained predictions and outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/final_study/path_manifest.json"
STAGES = [10, 31, 51, 87]
GROUP_KEYS = ["coordinate_year", "field_id", "leaf_index"]
SERIES_KEYS = ["partition", "field_id", "coordinate_year", "source", "leaf_index", "endpoint_series"]
MODELS = ["baseline", "current_residual", "archived_tpv", "wheat_refit_tpv", "agc_native", "agc_calibrated", "agc_residual"]
REFERENCES = ("baseline", "archived_tpv", "wheat_refit_tpv", "current_residual")
PHENOLOGY_COUNTS = ["cycles", "stations", "observed_events", "common_resolved_events", "lost_reference_events", "unresolved_observed_events", "duration_pairs"]
PHENOLOGY_VALUES = ["stage_equal_crps", "stage_equal_mae", "terminal_mae", "terminal_bias", "duration_mae", "duration_bias"]
DISEASE_VALUES = ["rmse", "mae", "bias", "r2", "observed_mean", "prediction_mean"]


def median_days(cdf):
    cdf = np.asarray(cdf)
    return np.where(cdf[:, :, -1] >= .5, (cdf >= .5).argmax(2), np.nan)


def restricted_crps(cdf, targets):
    cdf, targets = np.asarray(cdf), np.asarray(targets)
    truth = np.arange(cdf.shape[2])[None, None, :] >= targets[:, :, None]
    values = np.square(cdf.astype(np.float64) - truth).sum(2)
    return np.where(targets >= 0, values, np.nan)


def _mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def phenology_metrics(cdfs, targets, cycles, references=REFERENCES):
    """Original fixed-reference date/duration support; missing candidate medians stay NaN."""
    targets = np.asarray(targets)
    if targets.shape != (len(cycles), 4) or set(references) - set(cdfs):
        raise ValueError("Complete observation/reference dimensions are required")
    if cycles.cycle_id.duplicated().any() or cycles.PEP_ID.isna().any() or not np.isfinite(targets).all():
        raise ValueError("Unique cycles and complete stations/targets are required")
    for cdf in cdfs.values():
        if (cdf.shape[:2] != targets.shape or not np.isfinite(cdf).all() or np.any(cdf < 0)
                or np.any(cdf > 1) or np.any(np.diff(cdf, axis=2) < -1e-6)
                or np.any(np.diff(cdf, axis=1) > 1e-6)):
            raise ValueError("Invalid stage CDF")
    medians = {name: median_days(cdf) for name, cdf in cdfs.items()}
    observed = targets >= 0
    support = observed & np.logical_and.reduce([np.isfinite(medians[name]) for name in references])
    pairs = support[:, 2] & support[:, 3]
    cohorts = {"all": np.ones(len(cycles), bool), "reserved_station": cycles.historical_station_group.eq("test").to_numpy()}
    rows = []
    for name, cdf in cdfs.items():
        error = medians[name] - targets
        duration = (medians[name][:, 3] - medians[name][:, 2]) - (targets[:, 3] - targets[:, 2])
        crps = restricted_crps(cdf, targets)
        for cohort, mask in cohorts.items():
            mae, bias, probability = [], [], []
            for stage in range(4):
                common, known = support[:, stage] & mask, observed[:, stage] & mask
                mae.append(_mean(np.abs(error[common, stage])))
                bias.append(_mean(error[common, stage]))
                probability.append(_mean(crps[known, stage]))
            use = pairs & mask
            rows.append(dict(cohort=cohort, model=name, cycles=int(mask.sum()), stations=int(cycles.loc[mask, "PEP_ID"].nunique()),
                observed_events=int((observed & mask[:, None]).sum()), common_resolved_events=int((support & mask[:, None]).sum()),
                lost_reference_events=int((support & mask[:, None] & ~np.isfinite(medians[name])).sum()),
                unresolved_observed_events=int((observed & mask[:, None] & ~np.isfinite(medians[name])).sum()),
                stage_equal_crps=float(np.mean(probability)), stage_equal_mae=float(np.mean(mae)), terminal_mae=mae[3],
                terminal_bias=bias[3], duration_pairs=int(use.sum()), duration_mae=_mean(np.abs(duration[use])),
                duration_bias=_mean(duration[use])))
    return pd.DataFrame(rows)


def _guard_score(cdf, targets, reference):
    targets = np.asarray(targets)
    prediction, original = median_days(cdf), median_days(reference)
    observed = targets >= 0
    support = observed & np.isfinite(original)
    missing = support & ~np.isfinite(prediction)
    error = prediction - targets
    mae = [_mean(np.abs(error[support[:, stage], stage])) for stage in range(4)]
    pairs = support[:, 2] & support[:, 3]
    duration = ((prediction[:, 3] - prediction[:, 2]) - (targets[:, 3] - targets[:, 2]))[pairs]
    crps = restricted_crps(cdf, targets)
    probability = [_mean(crps[observed[:, stage], stage]) for stage in range(4)]
    return dict(crps=float(np.mean(probability)), stage_equal_mae=float(np.mean(mae)), terminal_mae=mae[3],
        duration_mae=_mean(np.abs(duration)), unresolved_observed_events=int((observed & ~np.isfinite(prediction)).sum()),
        lost_baseline_resolved_events=int(missing.sum()),
        lost_baseline_duration_pairs=int((pairs & (~np.isfinite(prediction[:, 2]) | ~np.isfinite(prediction[:, 3]))).sum()),
        stage_event_counts=support.sum(0).tolist(), duration_pairs=int(pairs.sum()))


def promotion_guards(candidate, targets, reference, tolerance=1e-6):
    base, value = _guard_score(reference, targets, reference), _guard_score(candidate, targets, reference)
    finite = all(np.isfinite(value[key]) and np.isfinite(base[key]) for key in ("crps", "stage_equal_mae", "terminal_mae", "duration_mae"))
    return dict(finite_metrics=bool(finite),
        complete_scoring_support=all(n > 0 for n in value["stage_event_counts"]) and value["duration_pairs"] > 0,
        no_lost_baseline_events=value["lost_baseline_resolved_events"] == 0,
        no_lost_baseline_pairs=value["lost_baseline_duration_pairs"] == 0,
        no_more_unresolved=value["unresolved_observed_events"] <= base["unresolved_observed_events"],
        crps_not_worse=bool(finite and value["crps"] <= base["crps"] + tolerance),
        stage_equal_mae_not_worse=bool(finite and value["stage_equal_mae"] <= base["stage_equal_mae"] + tolerance),
        terminal_mae_lower=bool(finite and value["terminal_mae"] < base["terminal_mae"] - tolerance),
        duration_mae_lower=bool(finite and value["duration_mae"] < base["duration_mae"] - tolerance))


def hierarchical_weights(frame):
    """Unit coordinate-year mass split equally over fields, leaves and records."""
    if frame.empty or frame[GROUP_KEYS].isna().any().any():
        raise ValueError("Nonempty complete hierarchy required")
    fields = frame.groupby(GROUP_KEYS[0]).field_id.transform("nunique").to_numpy(float)
    leaves = frame.groupby(GROUP_KEYS[:2]).leaf_index.transform("nunique").to_numpy(float)
    records = frame.groupby(GROUP_KEYS).leaf_index.transform("size").to_numpy(float)
    return 1 / (fields * leaves * records)


def observed_windows(frame, prediction_columns):
    """Trapezoids between retained observation dates with at least two assessments."""
    if frame.duplicated(SERIES_KEYS + ["date"]).any():
        raise ValueError("Duplicate date within source leaf series")
    result = []
    for key, part in frame.groupby(SERIES_KEYS, sort=True):
        part = part.sort_values("date")
        days = (pd.to_datetime(part.date) - pd.to_datetime(part.date.iloc[0])).dt.total_seconds().to_numpy() / 86400
        if len(days) < 2 or days[-1] <= 0:
            continue
        record = dict(zip(SERIES_KEYS, key))
        for meta in ["site_id", "country", "season_year", "outer_fold"]:
            if meta in part:
                if part[meta].nunique() != 1:
                    raise ValueError("Inconsistent window metadata: " + meta)
                record[meta] = part[meta].iloc[0]
        record.update(span_days=float(days[-1]), n_assessments=len(part), window_start=part.date.iloc[0], window_end=part.date.iloc[-1])
        for column in ["observed_percent"] + list(prediction_columns):
            values = part[column].to_numpy(float)
            if column == "observed_percent" and not np.isfinite(values).all():
                raise ValueError("Nonfinite observed window score")
            integral = float(np.sum(np.diff(days) * (values[:-1] + values[1:]) / 2)) if np.isfinite(values).all() else np.nan
            record[column] = integral / days[-1]
            record["observed_audpc" if column == "observed_percent" else column + "_audpc"] = integral
        result.append(record)
    return pd.DataFrame(result)


def disease_scores(frame, prediction):
    weights = hierarchical_weights(frame)
    observed, predicted = frame.observed_percent.to_numpy(float), frame[prediction].to_numpy(float)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Finite paired scores required")
    error = predicted - observed
    mean = np.average(observed, weights=weights)
    variance = np.average((observed - mean) ** 2, weights=weights)
    mse = np.average(error ** 2, weights=weights)
    return dict(rmse=float(np.sqrt(mse)), mae=float(np.average(abs(error), weights=weights)),
        bias=float(np.average(error, weights=weights)), r2=float(1 - mse / variance) if variance > 0 else np.nan,
        observed_mean=float(mean), prediction_mean=float(np.average(predicted, weights=weights)))


class RetainedFiles:
    """Resolve and check only the retained files consumed by these metrics."""
    def __init__(self, path=MANIFEST):
        self.path = Path(path)
        content = json.loads(self.path.read_text())
        self.files = content.get("files", content)
        self.checked = set()

    def resolve(self, original):
        if original not in self.files:
            raise KeyError("Required retained metric input missing from manifest: " + original)
        record = self.files[original]
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        if original not in self.checked:
            if not path.is_file() or ("bytes" in record and path.stat().st_size != record["bytes"]):
                raise ValueError("Retained metric input missing or changed: " + str(path))
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != record["sha256"]:
                raise ValueError("Retained metric input changed: " + str(path))
            self.checked.add(original)
        return path

    def csv(self, original):
        return pd.read_csv(self.resolve(original), float_precision="round_trip")

    def json(self, original):
        return json.loads(self.resolve(original).read_text())


def compare_tables(actual, expected, keys, counts, values, tolerance):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("Duplicate metric identities")
    left, right = actual.set_index(keys), expected.set_index(keys)
    if set(left.index) != set(right.index):
        raise ValueError("Metric populations or models changed")
    right = right.loc[left.index]
    for key in counts:
        np.testing.assert_array_equal(left[key].to_numpy(), right[key].to_numpy(), err_msg="Changed metric denominator: " + key)
    maximum = 0.
    for key in values:
        a, b = left[key].to_numpy(float), right[key].to_numpy(float)
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=0, equal_nan=True, err_msg="Changed metric: " + key)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(a[finite] - b[finite]))))
    return maximum


def _summary_rows(temporal, source, expected):
    rows = []
    populations = {"all_future_records": temporal, "new_sites": temporal.loc[temporal.new_site], "source_transfer": source}
    for population, frame in populations.items():
        population_expected = expected.loc[expected.population.eq(population)]
        for year in sorted(population_expected.test_year.unique()):
            sample = frame if year == 0 else frame.loc[frame.test_year.eq(year)]
            models = population_expected.loc[population_expected.test_year.eq(year), "model"].unique().tolist()
            windows = observed_windows(sample, models)
            integral = windows.copy()
            integral["observed_percent"] = integral.observed_audpc
            for model in models:
                integral[model] = integral[model + "_audpc"]
            for endpoint, data in [("assessment", sample), ("window_mean", windows), ("window_integral", integral)]:
                groups = [("pooled", data)] + list(data.groupby("source", sort=True))
                for label, part in groups:
                    for model in models:
                        rows.append(dict(population=population, test_year=year, endpoint=endpoint, source=label, model=model,
                            n=len(part), fields=part.field_id.nunique(), sites=part.site_id.nunique(), **disease_scores(part, model)))
    return pd.DataFrame(rows)


def _disease_criteria(summary, intervals, ledger):
    criteria, maximum = [], 0.
    for row in ledger["selected_pipeline"]["criteria"]:
        identity = {key: row[key] for key in ["population", "endpoint", "source"]}
        if row["rule"] == "paired_upper_below_zero":
            evidence = intervals
            for key, value in {**identity, "candidate": "selected_pipeline", "reference": row["reference"], "metric": "rmse_difference"}.items():
                evidence = evidence.loc[evidence[key].eq(value)]
            if len(evidence) != 1:
                raise ValueError("Incomplete frozen paired criterion")
            value = float(evidence.upper.iloc[0])
            passed = value < 0
            # The frozen interval's point difference must match recomputed endpoints.
            part = summary.loc[summary.test_year.eq(0)]
            for key, item in identity.items():
                part = part.loc[part[key].eq(item)]
            part = part.set_index("model")
            point = float(part.loc["selected_pipeline", "rmse"] - part.loc[row["reference"], "rmse"])
            maximum = max(maximum, abs(point - float(evidence.value.iloc[0])))
            if abs(point - float(evidence.value.iloc[0])) > 1e-8:
                raise ValueError("Frozen paired interval point value differs")
        elif row["rule"] == "point_error_does_not_increase":
            part = summary.loc[summary.test_year.eq(0)]
            for key, item in identity.items():
                part = part.loc[part[key].eq(item)]
            part = part.set_index("model")
            value = float(part.loc["selected_pipeline", "rmse"] - part.loc[row["reference"], "rmse"])
            passed = value <= 0
        else:
            raise ValueError("Unrecognized frozen criterion")
        maximum = max(maximum, abs(value - row["value"]))
        if abs(value - row["value"]) > 1e-8 or passed != row["passed"]:
            raise ValueError("Original disease criterion changed")
        criteria.append({**row, "value": value, "passed": bool(passed)})
    if len(criteria) != 16 or all(row["passed"] for row in criteria) != ledger["selected_pipeline"]["magnitude_support"]:
        raise ValueError("Original sixteen-criterion status changed")
    return criteria, maximum


def verify(manifest_path=MANIFEST):
    files = RetainedFiles(manifest_path)
    prefix = "analysis/paper_study/agc_wheat_backbone_20261012/evaluation/"
    expected = files.csv(prefix + "metrics.csv")
    decisions = files.json(prefix + "decisions.json")
    crop_rows, guard_checks, primary_flags = [], 0, 0
    for population, size in [("later2017_2024", 4317), ("temporal2025", 442)]:
        cycles = files.csv(prefix + population + "_membership.csv")
        with np.load(files.resolve(prefix + population + "_predictions.npz"), allow_pickle=False) as saved:
            targets = saved["targets"]
            cdfs = {name: saved[name] for name in MODELS}
            np.testing.assert_array_equal(saved["cycle_id"], cycles.cycle_id.to_numpy(dtype=str))
        if len(cycles) != size:
            raise ValueError("Frozen phenology population changed")
        np.testing.assert_array_equal(cdfs["agc_residual"], cdfs["agc_calibrated"])
        crop_rows.append(phenology_metrics(cdfs, targets, cycles).assign(population=population))
        for cohort, mask in [("all", np.ones(len(cycles), bool)), ("reserved_station", cycles.historical_station_group.eq("test").to_numpy())]:
            for reference in ["baseline", "current_residual", "archived_tpv", "wheat_refit_tpv"]:
                for candidate in ["agc_native", "agc_calibrated", "agc_residual"]:
                    guards = promotion_guards(cdfs[candidate][mask], targets[mask], cdfs[reference][mask])
                    original = decisions[population][cohort][reference]["checks"][candidate]
                    if guards != original:
                        raise ValueError("Original crop guard ledger changed")
                    guard_checks += len(guards)
                    if candidate == "agc_calibrated" and reference == "baseline":
                        primary_flags += sum(guards.values())
    crop = pd.concat(crop_rows, ignore_index=True)
    crop_difference = compare_tables(crop, expected, ["population", "cohort", "model"], PHENOLOGY_COUNTS, PHENOLOGY_VALUES, 1e-10)
    if primary_flags != 29:
        raise ValueError("Original AGC29/36 status changed")
    disease_results = {}
    study_prefixes = {
        "current": "analysis/paper_study/disease_coupled_ml_20261012/results/",
        "agc_refit": "analysis/paper_study/disease_agc_ml_20261012/results/",
        "frozen_head_agc_sensitivity": "analysis/paper_study/agc_wheat_backbone_20261012/coupling/",
    }
    for name, base in study_prefixes.items():
        temporal, source = files.csv(base + "assessment_predictions.csv"), files.csv(base + "source_predictions.csv")
        if len(temporal) != 2018 or int(temporal.new_site.sum()) != 1272 or len(source) != 2287:
            raise ValueError("Frozen disease population changed")
        expected_summary = files.csv(base + "summary.csv")
        summary = _summary_rows(temporal, source, expected_summary)
        maximum = compare_tables(summary, expected_summary, ["population", "test_year", "endpoint", "source", "model"],
            ["n", "fields", "sites"], DISEASE_VALUES, 1e-8)
        criteria, criterion_difference = _disease_criteria(summary, files.csv(base + "paired_intervals.csv"), files.json(base + "decisions.json"))
        primary = summary.loc[summary.population.eq("all_future_records") & summary.test_year.eq(0)
            & summary.source.eq("pooled") & summary.model.eq("selected_pipeline")].set_index("endpoint")
        disease_results[name] = dict(assessment_rmse_pp=float(primary.loc["assessment", "rmse"]),
            observed_window_integral_rmse_pp_days=float(primary.loc["window_integral", "rmse"]),
            temporal_assessments=len(temporal), new_site_assessments=int(temporal.new_site.sum()),
            source_transfer_assessments=len(source), summary_rows=len(summary),
            maximum_summary_difference=maximum, maximum_criterion_difference=criterion_difference,
            original_criteria_passed=sum(row["passed"] for row in criteria), original_criteria_total=16)
    return dict(status="passed", fitting_performed=False, retained_inputs_checked=len(files.checked),
        phenology=dict(cohorts=4, models=len(MODELS), rows=len(crop), maximum_metric_difference=crop_difference,
            original_guard_booleans_checked=guard_checks, original_primary_guards_passed=primary_flags,
            original_primary_guards_total=36, agc_residual_equals_calibrated=True,
            candidate_missing_medians_remain_undefined=True, probability_residual_adopted=False),
        disease=disease_results, paired_uncertainty="Original saved paired intervals retained; point differences and criteria verified without resampling",
        independent_prospective_validation=False, climate_outputs_recomputed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.manifest)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
