"""Sampled weather/host attribution with uncalibrated canopy sensitivities.

All disease parameters and published inputs are immutable dependencies. Actual
elapsed days since sowing align the weather and host corners. These calculations
are diagnostic model responses, not physiological or grain-loss estimates.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from models.climate.kernel import FIELDS, simulate_draw_seasons
from models.climate.run import (
    DATA, cell_registry, check_file, digest, forcing_path, load_weather, read_manifest,
)
from models.seasonal_septoria.host import sowing_date_from_calendar
from models.seasonal_septoria.overwinter import OverwinterParameters, simulate_overwinter
from models.seasonal_septoria.wetness import duration_exposure
from .provenance import prepare_destination


@dataclass(frozen=True)
class Setting:
    name: str
    anthesis_scale: float = 1.
    terminal_scale: float = 1.
    reference_senescence: bool = False
    detection_gated: bool = True
    fixed_overlap: bool = False


def settings():
    """Predeclared diagnostic settings; scale bounds are not confidence limits."""
    return (
        Setting("baseline"),
        Setting("anthesis_early_5pct", anthesis_scale=.95),
        Setting("anthesis_late_5pct", anthesis_scale=1.05),
        Setting("terminal_early_5pct", terminal_scale=.95),
        Setting("terminal_late_5pct", terminal_scale=1.05),
        Setting("reference_linear_senescence", reference_senescence=True),
        Setting("field_ungated_damage", detection_gated=False),
        Setting("fixed_common_overlap", fixed_overlap=True),
    )


def setting_thresholds(thresholds, setting):
    result = {int(key): float(value) for key, value in thresholds.items()}
    result[65] *= setting.anthesis_scale
    result[85] *= setting.terminal_scale
    if not np.isfinite(list(result.values())).all() or np.any(np.diff(
            [result[key] for key in sorted(result)]) <= 0):
        raise ValueError("Sensitivity must preserve finite ordered crop thresholds")
    return result


def align_elapsed_days(source_dates, source_sowing, target_dates, target_sowing, values):
    """Map actual donor sowing + target elapsed age; absent donor days stay NaN."""
    source_dates, target_dates = pd.DatetimeIndex(source_dates), pd.DatetimeIndex(target_dates)
    source_sowing, target_sowing = [np.asarray(x, dtype="datetime64[D]")
                                  for x in (source_sowing, target_sowing)]
    values = np.asarray(values, float)
    if (values.ndim != 2 or values.shape[1] != len(source_dates)
            or source_sowing.shape != (len(values),) or target_sowing.shape != source_sowing.shape
            or np.isnat(source_sowing).any() or np.isnat(target_sowing).any()
            or not len(source_dates) or not len(target_dates)
            or not source_dates.equals(pd.date_range(source_dates[0], source_dates[-1]))
            or not target_dates.equals(pd.date_range(target_dates[0], target_dates[-1]))):
        raise ValueError("Consecutive dated weather and one finite sowing date per draw required")
    age = target_dates.to_numpy(dtype="datetime64[D]")[None, :] - target_sowing[:, None]
    donor = source_sowing[:, None] + age
    positions = (donor - source_dates[0].to_datetime64().astype("datetime64[D]")).astype(int)
    supported = (positions >= 0) & (positions < len(source_dates))
    gathered = values[np.arange(len(values))[:, None], np.clip(positions, 0, len(source_dates) - 1)]
    return np.where(supported, gathered, np.nan)


def fixed_overlap_windows(reference_dates, reference_sowing, reference_window,
                          future_dates, future_sowing, future_window):
    """Identical elapsed-age interval inside both reached host windows.

    The overlap does not extend a modeled host beyond its terminal event. Empty
    intersections remain unavailable rather than zero-length loss estimates.
    """
    windows, ages = [], []
    for dates, sowing, (start, end) in (
            (reference_dates, reference_sowing, reference_window),
            (future_dates, future_sowing, future_window)):
        dates = pd.DatetimeIndex(dates)
        sowing = np.asarray(sowing, dtype="datetime64[D]")
        start, end = np.asarray(start, int), np.asarray(end, int)
        first = (dates[0].to_datetime64().astype("datetime64[D]") - sowing).astype(int)
        ages.append((first + start, first + end, (start >= 0) & (end >= start)))
        windows.append(first)
    start_age = np.maximum(ages[0][0], ages[1][0])
    end_age = np.minimum(ages[0][1], ages[1][1])
    valid = ages[0][2] & ages[1][2] & (end_age >= start_age)
    return tuple((np.where(valid, start_age - first, -1),
                  np.where(valid, end_age - first, -1)) for first in windows)


@dataclass
class Corner:
    lost_had3: np.ndarray
    reference_had3: np.ndarray
    valid: np.ndarray
    status: np.ndarray
    window_days: np.ndarray
    symptom_day: np.ndarray
    mass_error: float


def simulate_corner(dates, weather, host_details, host_valid, fit, setting, *, window=None):
    """Frozen disease replay with supplied host; validate all active weather first."""
    dates = pd.DatetimeIndex(dates)
    raw = {key: np.asarray(weather[key], float) for key in FIELDS}
    mask = np.asarray(host_details["crop_mask"], bool).copy()
    host = host_details["host"]
    host_valid = np.asarray(host_valid, bool)
    if any(value.shape != mask.shape for value in raw.values()) or len(dates) != mask.shape[1]:
        raise ValueError("Corner weather must share the supplied daily host shape")
    finite = np.logical_and.reduce([np.isfinite(value) for value in raw.values()])
    finite &= (raw["tmax_c"] >= raw["tmean_c"] - .001)
    finite &= (raw["rh_mean_pct"] >= 0) & (raw["rh_mean_pct"] <= 100)
    finite &= raw["precipitation_mm"] >= 0
    finite &= (raw["tmean_c"] >= -80) & (raw["tmax_c"] <= 80)
    invalid_weather = (mask & ~finite).any(axis=1)
    unsupported = (mask & (raw["tmean_c"] > 40)).any(axis=1)
    start, end = (host.stage_day_index[65], host.stage_day_index[85]) if window is None else window
    start, end = np.asarray(start, int), np.asarray(end, int)
    valid_window = (start >= 0) & (end >= start) & (end < len(dates))
    valid = host_valid & valid_window & ~invalid_weather & ~unsupported
    status = np.where(~host_valid, "unavailable_host_season", np.where(~valid_window,
        "unavailable_integration_window", np.where(invalid_weather,
        "missing_or_invalid_active_weather", np.where(unsupported,
        "unsupported_active_temperature_above40", "complete"))))
    mask &= valid[:, None]
    # Inactive inputs are numerical placeholders; invalid active rows never
    # receive an estimate and never enter common-support attribution.
    safe = {key: np.where(mask, value, 0.) for key, value in raw.items()}
    exposure = duration_exposure(safe["tmean_c"], safe["tmax_c"], safe["rh_mean_pct"],
        safe["precipitation_mm"], fit["weather_preprocessing"]["rain_rate_mm_hour"])["exposure"]
    trajectory = simulate_overwinter(safe["tmean_c"], exposure, safe["precipitation_mm"],
        host.active, host.renewal, host.area, mask,
        OverwinterParameters(**fit["parameters"]), initial_local_source=1.,
        imported_pressure=np.full_like(safe["tmean_c"], fit["constant_imported_pressure"]),
        detection_fraction=.001)
    mass_error = float(np.max(np.abs(trajectory.state.sum(axis=-1) - 1.)))
    if mass_error > 1e-10:
        raise AssertionError("Frozen tissue mass conservation failed")
    loss = trajectory.damage[:, 1:, :3].copy()
    if setting.detection_gated:
        day = np.arange(1, len(dates) + 1)[None, :, None]
        symptoms = trajectory.symptom_day[:, :3]
        loss *= (symptoms[:, None, :] >= 0) & (day >= symptoms[:, None, :])
    loss *= mask[:, :, None]
    reference = host.area[:, :, :3].copy() / 3.
    if setting.reference_senescence:
        # Only reference-canopy accounting changes. No calibrated senescence
        # physiology or host feedback is implied by this linear 1-to-0 profile.
        anthesis = host.stage_day_index[65]
        terminal = host.stage_day_index[85]
        denominator = terminal - anthesis
        factor = 1. - np.clip(np.divide(np.arange(len(dates))[None, :] - anthesis[:, None],
            denominator[:, None], out=np.ones(mask.shape, float), where=denominator[:, None] > 0), 0., 1.)
        reference *= factor[:, :, None]
    interval = (np.arange(len(dates))[None, :] >= start[:, None]) & (
        np.arange(len(dates))[None, :] <= end[:, None]) & mask
    reference_had = (reference * interval[:, :, None]).sum(axis=(1, 2))
    lost_had = (reference * loss * interval[:, :, None]).sum(axis=(1, 2))
    valid &= reference_had > 0
    status[(status == "complete") & ~valid] = "unavailable_reference_canopy"
    if np.any(lost_had > reference_had + 1e-10) or np.any(lost_had < -1e-10):
        raise AssertionError("Canopy loss must lie inside reference-canopy mass")
    return Corner(np.where(valid, lost_had, np.nan), np.where(valid, reference_had, np.nan),
        valid, status, np.where(valid, end - start + 1., np.nan),
        trajectory.symptom_day[:, :3], mass_error)


def components(y00, y10, y01, y11):
    """Symmetric four-corner decomposition on identical finite support."""
    a, b, c, d = np.broadcast_arrays(*[np.asarray(x, float) for x in (y00, y10, y01, y11)])
    common = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
    values = dict(net=d-a, weather_alone=b-a, host_alone=c-a,
                  interaction=d-b-c+a, weather=.5*(b-a+d-c), host=.5*(c-a+d-b))
    return dict(common=common, **{key: np.where(common, value, np.nan) for key, value in values.items()})


def summarize_pairs(pairs, *, expected_models=None):
    """Design-weighted model ratios followed by strict equal-member means."""
    pairs = pairs.copy()
    expected_models = list(pairs.model.unique()) if expected_models is None else list(expected_models)
    if (not expected_models or len(set(expected_models)) != len(expected_models)
            or not set(pairs.model.unique()).issubset(expected_models)):
        raise ValueError("Unique declared climate-model membership required")
    identity = ["model", "reference_year", "future_year", "spatial_draw_id"]
    if pairs.duplicated(["setting", *identity]).any():
        raise ValueError("Duplicate setting/model/year/draw identities")
    parts = components(*[pairs[key].to_numpy() for key in ("y00", "y10", "y01", "y11")])
    for key, value in parts.items():
        pairs[key] = value
    setting_count = pairs.setting.nunique()
    support = pairs.groupby(identity, sort=False).common.agg(["all", "size"])
    support["common_across_settings"] = support["all"] & support["size"].eq(setting_count)
    pairs = pairs.merge(support[["common_across_settings"]], left_on=identity,
        right_index=True, how="left", validate="many_to_one")
    rows = []
    component_names = ("net", "weather", "host", "weather_alone", "host_alone", "interaction")
    window_names = ("reference_window_days", "future_window_days", "reference_host_window_days", "future_host_window_days")
    for setting, frame in pairs.groupby("setting", sort=False):
        groups = [(model, part) for model, part in frame.groupby("model", sort=False)]
        setting_rows = []
        for model, group in groups:
            total_weight = group.area_mean_weight.sum()
            if not np.isfinite(total_weight) or total_weight <= 0 or (group.area_mean_weight <= 0).any():
                raise ValueError("Finite positive design weights required")
            for label, column in (("within_setting", "common"), ("across_settings", "common_across_settings")):
                selected = group.loc[group[column]]
                common_weight = selected.area_mean_weight.sum()
                row = dict(setting=setting, model=model, support=label, requested_pairs=len(group),
                    valid_pairs=len(selected), unique_valid_cells=selected.cell_id.nunique(),
                    common_area_time_fraction=float(common_weight / total_weight),
                    requested_models=1, available_model_estimates=int(common_weight > 0),
                    complete_model_ensemble=bool(common_weight > 0))
                for component in component_names:
                    row[f"{component}_had_days"] = float(np.average(selected[component],
                        weights=selected.area_mean_weight)) if common_weight > 0 else np.nan
                weather, host = row["weather_had_days"], row["host_had_days"]
                row["offset_percent"] = -100*host/weather if np.isfinite(weather) and abs(weather) > 1e-12 else np.nan
                for window in window_names:
                    if window in group:
                        row[window] = float(np.average(selected[window], weights=selected.area_mean_weight)) if common_weight > 0 else np.nan
                rows.append(row)
                setting_rows.append(row)
        for label, column in (("within_setting", "common"), ("across_settings", "common_across_settings")):
            members = [row for row in setting_rows if row["support"] == label]
            available = [row for row in members if np.isfinite(
                [row[f"{component}_had_days"] for component in component_names]).all()]
            complete = len(available) == len(expected_models)
            selected = frame.loc[frame[column]]
            total_weight, common_weight = frame.area_mean_weight.sum(), selected.area_mean_weight.sum()
            row = dict(setting=setting, model="ensemble", support=label,
                requested_pairs=len(frame), valid_pairs=len(selected),
                unique_valid_cells=selected.cell_id.nunique(),
                common_area_time_fraction=float(common_weight / total_weight),
                requested_models=len(expected_models), available_model_estimates=len(available),
                complete_model_ensemble=complete)
            for component in component_names:
                key = f"{component}_had_days"
                row[key] = float(np.mean([member[key] for member in available])) if complete else np.nan
            weather, host = row["weather_had_days"], row["host_had_days"]
            row["offset_percent"] = -100*host/weather if np.isfinite(weather) and abs(weather) > 1e-12 else np.nan
            for window in window_names:
                if window in frame:
                    row[window] = float(np.mean([member[window] for member in available])) if complete else np.nan
            rows.append(row)
    return pd.DataFrame(rows), pairs


def _authenticated_inputs(root, manifest, models, reference_years):
    paths = {str(DATA / name) for name in ("cells.parquet", "calendars.parquet", "spatial_draws.csv",
        "phenology_parameters.json", "disease_parameters.json", "stage_thresholds.json")}
    for model in models:
        paths.add(str(DATA / "alignment" / model / "ssp585" / "coefficients.parquet"))
        paths.add(str(DATA / "alignment" / model / "ssp585" / "receipt.json"))
    records = [row for row in manifest["inputs"] if row["path"] in paths]
    missing = paths - {row["path"] for row in records}
    if missing:
        raise ValueError("Required inputs not registered: " + ", ".join(sorted(missing)))
    records += [*manifest["core_sources"], *manifest.get("implementation_sources", [])]
    forcing = {row["path"]: row for row in manifest["forcing"]}
    for model in models:
        for reference_year in reference_years:
            for harvest_year in (reference_year, reference_year + 80):
                for year in (harvest_year - 1, harvest_year):
                    record = forcing[forcing_path(model, "ssp585", year)]
                    records.append(record)
                    records.append(dict(path=record["receipt"], sha256=record["receipt_sha256"]))
    by_path = {row["path"]: row for row in records}
    result = []
    for path, row in sorted(by_path.items()):
        check_file(root / path, row, verify_hash=True)
        result.append(dict(path=path, sha256=row["sha256"], bytes=(root / path).stat().st_size))
    result.append(dict(path=str(DATA / "scientific_manifest.json"),
        sha256=digest(root / DATA / "scientific_manifest.json"),
        bytes=(root / DATA / "scientific_manifest.json").stat().st_size))
    implementation = Path(__file__).resolve()
    result.append(dict(path=str(implementation.relative_to(root)), sha256=digest(implementation),
        bytes=implementation.stat().st_size))
    return result


def run(output: Path, *, root: Path, models: list[str], reference_years: list[int]) -> dict:
    """Run selected sampled season pairs; all thirty reference years are allowed."""
    started = time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = read_manifest(root)
    if (not models or len(set(models)) != len(models) or not set(models).issubset(manifest["models"])
            or not reference_years or len(set(reference_years)) != len(reference_years)
            or any(not isinstance(year, int) or year < 1991 or year > 2020 for year in reference_years)):
        raise ValueError("Unique registered climate models and reference harvest years 1991–2020 required")
    # Registration precedes loading, replay and scoring. Existing destinations
    # are refused rather than silently replacing earlier diagnostic runs.
    output = prepare_destination(output, root)
    registration_path = output / "attribution_registration.json"
    inputs = _authenticated_inputs(root, manifest, models, reference_years)
    configuration = dict(models=models, scenario="ssp585", reference_years=reference_years,
        future_years=[year+80 for year in reference_years], pairing="reference harvest year + 80",
        settings=[asdict(setting) for setting in settings()], endpoint="GS65_85_lost_had3",
        units="normalized top-three-leaf HAD days", settings_uncalibrated=True,
        physiological_validation=False, disease_parameters_unchanged=True,
        fixed_window="common overlap in actual elapsed sowing days, no host extrapolation",
        senescence="reference-only linear decline from 1 at anthesis to 0 at terminal; host tissue unchanged",
        weighting="retained area_mean_weight; duplicate draws retained; each model's conditional area-time ratio followed by strict equal-model means; offset from component means",
        period_estimate=len(reference_years) == 30, inputs=inputs)
    registration_path.write_text(json.dumps(configuration, indent=2) + "\n")
    draws = pd.read_csv(root / DATA / "spatial_draws.csv")
    cells = cell_registry(root)
    unique_cells = cells.loc[cells.cell_id.isin(draws.cell_id)].reset_index(drop=True)
    indexes = pd.Index(unique_cells.cell_id).get_indexer(draws.cell_id)
    if len(draws) != 64 or len(unique_cells) != 62 or np.any(indexes < 0):
        raise ValueError("Expected exactly 64 retained spatial draws and 62 registered unique cells")
    draw_cells = unique_cells.iloc[indexes].reset_index(drop=True)
    phenology = json.loads((root / DATA / "phenology_parameters.json").read_text())
    fit = json.loads((root / DATA / "disease_parameters.json").read_text())["fitted"]
    thresholds = {int(k): v for k, v in json.loads((root / DATA / "stage_thresholds.json").read_text())["all_stage_thresholds"].items()}
    all_pairs, verification, maximum_mass_error = [], [], 0.
    for model in models:
        for reference_year in reference_years:
            hosts, weather_by_year, dates_by_year, sowing_by_year = {}, [], [], []
            for year in (reference_year, reference_year+80):
                dates, weather = load_weather(model, "ssp585", year, unique_cells, root)
                weather = {key: value[indexes] for key, value in weather.items()}
                sowing = np.array([sowing_date_from_calendar(year, row.planting_doy, row.maturity_doy)
                    .to_datetime64().astype("datetime64[D]") for row in draw_cells.itertuples()])
                dates_by_year.append(dates)
                weather_by_year.append(weather)
                sowing_by_year.append(sowing)
            for setting in settings():
                setting_hosts = []
                threshold_key = (setting.anthesis_scale, setting.terminal_scale)
                for period in (0, 1):
                    key = (threshold_key, period)
                    if key not in hosts:
                        metric, detail = simulate_draw_seasons(dates_by_year[period], weather_by_year[period],
                            sowing_by_year[period], draw_cells.latitude.to_numpy(), phenology, fit,
                            setting_thresholds(thresholds, setting), return_daily=True)
                        metric.loc[~draw_cells.calendar_valid.to_numpy(), "valid_complete_season"] = False
                        metric.loc[~draw_cells.calendar_valid.to_numpy(), "status"] = "missing_calendar"
                        hosts[key] = metric, detail
                    setting_hosts.append(hosts[key])
                windows = [(part[1]["host"].stage_day_index[65], part[1]["host"].stage_day_index[85])
                           for part in setting_hosts]
                if setting.fixed_overlap:
                    windows = fixed_overlap_windows(dates_by_year[0], sowing_by_year[0], windows[0],
                        dates_by_year[1], sowing_by_year[1], windows[1])
                corners = {}
                for weather_period, host_period in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    metric, detail = setting_hosts[host_period]
                    corner_weather = {key: align_elapsed_days(dates_by_year[weather_period],
                        sowing_by_year[weather_period], dates_by_year[host_period], sowing_by_year[host_period],
                        weather_by_year[weather_period][key]) for key in FIELDS}
                    corner = simulate_corner(dates_by_year[host_period], corner_weather, detail,
                        metric.valid_complete_season.to_numpy(), fit, setting, window=windows[host_period])
                    maximum_mass_error = max(maximum_mass_error, corner.mass_error)
                    name = f"y{weather_period}{host_period}"
                    corners[name] = corner
                    if setting.name == "baseline" and weather_period == host_period:
                        expected = metric.GS65_85_lost_had3.to_numpy()
                        np.testing.assert_allclose(corner.lost_had3, expected, rtol=0, atol=1e-10, equal_nan=True)
                        difference = np.abs(corner.lost_had3 - expected)
                        finite = difference[np.isfinite(difference)]
                        verification.append(dict(model=model, harvest_year=reference_year+80*host_period,
                            maximum_difference=float(finite.max()) if finite.size else None,
                            valid_draws=int(corner.valid.sum()), matches_frozen_diagonal=True))
                pair = draws[["spatial_draw_id", "cell_id", "area_mean_weight"]].copy()
                pair["setting"], pair["model"] = setting.name, model
                pair["reference_year"], pair["future_year"] = reference_year, reference_year+80
                for name, corner in corners.items():
                    pair[name] = corner.lost_had3
                    pair[f"{name}_reference_had3"] = corner.reference_had3
                    pair[f"{name}_status"] = corner.status
                for period, label in ((0, "reference"), (1, "future")):
                    pair[f"{label}_window_days"] = corners[f"y{period}{period}"].window_days
                    host = setting_hosts[period][1]["host"]
                    pair[f"{label}_host_status"] = setting_hosts[period][0].status.to_numpy()
                    duration = host.stage_day_index[85] - host.stage_day_index[65] + 1.
                    pair[f"{label}_host_window_days"] = np.where(setting_hosts[period][0].valid_complete_season, duration, np.nan)
                all_pairs.append(pair)
    summary, pairs = summarize_pairs(pd.concat(all_pairs, ignore_index=True), expected_models=models)
    pairs.to_parquet(output / "attribution_pairs.parquet", index=False)
    summary.to_csv(output / "attribution_summary.csv", index=False)
    pd.DataFrame(verification).to_csv(output / "attribution_baseline_replay.csv", index=False)
    status_rows = []
    for name in ("y00", "y10", "y01", "y11"):
        counted = pairs.groupby(["setting", "model", f"{name}_status"], sort=False).agg(
            pairs=("spatial_draw_id", "size"), area_time_weight=("area_mean_weight", "sum")).reset_index()
        counted["corner"] = name
        counted = counted.rename(columns={f"{name}_status": "status"})
        status_rows.append(counted)
    pd.concat(status_rows, ignore_index=True).to_csv(output / "attribution_status.csv", index=False)
    elapsed = time.monotonic() - started
    result = dict(status="complete", models=models, reference_years=reference_years,
        reference_future_pairs=len(models)*len(reference_years), settings=len(settings()),
        spatial_draws=64, unique_cells=62, pair_records=len(pairs),
        baseline_diagonal_replays=len(verification), maximum_mass_error=maximum_mass_error,
        diagnostic_only=True, model_qualified=False, physiological_validation=False,
        period_estimate=len(reference_years) == 30, elapsed_seconds=elapsed,
        estimated_full_30_year_seconds=elapsed*30/len(reference_years),
        input_files=len(inputs), input_hashes=inputs,
        registration_sha256=digest(registration_path),
        output_hashes={path.name: digest(path) for path in output.iterdir() if path.is_file()},
        summary=summary.loc[(summary.model == "ensemble") & (summary.support == "across_settings")]
            .replace({np.nan: None}).to_dict("records"))
    (output / "attribution_receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
