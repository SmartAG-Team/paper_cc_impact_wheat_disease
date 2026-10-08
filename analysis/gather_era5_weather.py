#!/usr/bin/env python3
"""Acquire actual ERA5 hourly weather, preserve provenance, and validate coverage.

Dependencies: pandas, numpy, requests, pyarrow. Existing raw responses are reused.
The fixed September–September interval is coverage, not an inferred sowing season.
Run from any directory; paths are resolved relative to this script's repository.
Additional verified public sites may be passed as --external-sites PATH (CSV with
dataset_id,site_id,latitude,longitude,season_year,country,coordinate_source).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "era5"
ANALYSIS = ROOT / "analysis" / "era5"
API = "https://archive-api.open-meteo.com/v1/archive"
DOCS = "https://open-meteo.com/en/docs/historical-weather-api"
VARIABLES = {
    "temperature_2m": ("temperature_2m_c", "°C", -90, 65),
    "relative_humidity_2m": ("relative_humidity_2m_pct", "%", 0, 100),
    "dew_point_2m": ("dew_point_2m_c", "°C", -100, 65),
    "precipitation": ("precipitation_mm", "mm", 0, 500),
    "rain": ("rain_mm", "mm", 0, 500),
    "wind_speed_10m": ("wind_speed_10m_m_s", "m/s", 0, 100),
    "shortwave_radiation": ("shortwave_radiation_w_m2", "W/m²", 0, 1600),
}
GENERATED = ".generated-by-gather-era5-weather"


def location_id(latitude: float, longitude: float) -> str:
    text = f"{latitude:.6f},{longitude:.6f}"
    return "era5loc_" + hashlib.sha256(text.encode()).hexdigest()[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_directories() -> None:
    for folder in (DEST, ANALYSIS):
        if folder.exists() and not (folder / GENERATED).exists() and any(folder.iterdir()):
            raise FileExistsError(f"Refusing to overwrite unowned directory: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / GENERATED).write_text("analysis/gather_era5_weather.py\n", encoding="utf-8")
    (DEST / "raw").mkdir(exist_ok=True)


def target_tables(external_sites: Path | None):
    basf = pd.read_csv(ROOT / "data" / "basf-wheat-diseases.txt", sep="\t")
    basf["Date"] = pd.to_datetime(basf["Date"], errors="raise")
    trials = []
    for trial_id, group in basf.groupby("TrialId"):
        metadata = group[["Country", "Latitude", "Longitude"]].drop_duplicates()
        if len(metadata) != 1 or group["Date"].dt.year.nunique() != 1:
            raise ValueError(f"Trial {trial_id} is not one coordinate-year")
        row = metadata.iloc[0]
        trials.append({
            "TrialId": int(trial_id),
            "location_id": location_id(row.Latitude, row.Longitude),
            "season_year": int(group["Date"].dt.year.iloc[0]),
            "requested_latitude": row.Latitude,
            "requested_longitude": row.Longitude,
            "country_source": row.Country,
            "first_observation_date": group["Date"].min().date().isoformat(),
            "last_observation_date": group["Date"].max().date().isoformat(),
            "coordinate_source": "data/basf-wheat-diseases.txt; coordinates rounded to 0.1 degrees",
            "dataset_id": "basf-wheat-diseases",
        })
    trials = pd.DataFrame(trials)
    swiss = pd.DataFrame([{
        "dataset_id": "ethz-2025-septoria-field",
        "site_id": "Lindau-Eschikon",
        "latitude": 47.449,
        "longitude": 8.682,
        "season_year": year,
        "country": "SWITZERLAND",
        "coordinate_source": "https://doi.org/10.3929/ethz-b-000739194",
    } for year in (2016, 2023, 2024)])
    external = swiss
    if external_sites is not None:
        added = pd.read_csv(external_sites)
        required = set(swiss.columns)
        if not required.issubset(added.columns):
            raise ValueError(f"External sites CSV needs columns {sorted(required)}")
        external = pd.concat([swiss, added[swiss.columns]], ignore_index=True).drop_duplicates()
    external["location_id"] = [location_id(a, b) for a, b in zip(external.latitude, external.longitude)]
    external = external.rename(columns={"latitude": "requested_latitude", "longitude": "requested_longitude", "country": "country_source"})
    coordinates = pd.concat([trials, external], ignore_index=True)
    seasons = []
    for (loc_id, year), group in coordinates.groupby(["location_id", "season_year"], sort=True):
        row = group.iloc[0]
        seasons.append({
            "location_id": loc_id,
            "season_year": int(year),
            "requested_latitude": row.requested_latitude,
            "requested_longitude": row.requested_longitude,
            "country_source": "|".join(sorted(group.country_source.unique())),
            "dataset_ids": "|".join(sorted(group.dataset_id.unique())),
            "coordinate_sources": "|".join(sorted(group.coordinate_source.unique())),
            "window_start": f"{int(year)-1}-09-01",
            "window_end": f"{int(year)}-09-30",
            "basf_trial_count": int(group.TrialId.notna().sum()) if "TrialId" in group else 0,
        })
    return trials, external, pd.DataFrame(seasons)


def params_for(season: dict) -> dict:
    return {
        "latitude": float(season["requested_latitude"]),
        "longitude": float(season["requested_longitude"]),
        "start_date": season["window_start"],
        "end_date": season["window_end"],
        "hourly": ",".join(VARIABLES),
        "models": "era5",
        "timezone": "UTC",
        "elevation": "nan",
        "cell_selection": "nearest",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }


def validate_response(payload: dict, season: dict) -> pd.DataFrame:
    if payload.get("error"):
        raise ValueError(payload.get("reason", "API returned error"))
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Response is not UTC")
    hourly = payload.get("hourly", {})
    units = payload.get("hourly_units", {})
    expected_times = pd.date_range(season["window_start"], pd.Timestamp(season["window_end"]) + pd.Timedelta(hours=23), freq="h", tz="UTC")
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    if not times.equals(expected_times):
        raise ValueError(f"Hourly coverage mismatch: expected {len(expected_times)}, got {len(times)}")
    output = pd.DataFrame({"time_utc": times})
    for source, (column, unit, lower, upper) in VARIABLES.items():
        if units.get(source) != unit:
            raise ValueError(f"Unit mismatch for {source}: {units.get(source)}")
        vals = np.asarray(hourly[source], dtype=float)
        if len(vals) != len(times) or not np.isfinite(vals).all():
            raise ValueError(f"Missing/invalid observations in {source}")
        if np.any(vals < lower) or np.any(vals > upper):
            raise ValueError(f"Range outside [{lower},{upper}] for {source}")
        output[column] = vals
    # API output uses 0.1-mm rounding, so allow one rounding interval.
    if (output.rain_mm > output.precipitation_mm + 0.10001).any():
        raise ValueError("Liquid precipitation exceeds total precipitation")
    if (output.dew_point_2m_c > output.temperature_2m_c + 0.2).any():
        raise ValueError("Dew point exceeds air temperature beyond rounding")
    output.insert(0, "location_id", season["location_id"])
    return output


def retrieve(session: requests.Session, season: dict, minimum_spacing: float) -> tuple[dict, dict, pd.DataFrame]:
    stem = f"{season['location_id']}_{season['season_year']}"
    raw_path = DEST / "raw" / f"{stem}.json.gz"
    provenance_path = DEST / "raw" / f"{stem}.provenance.json"
    params = params_for(season)
    if raw_path.exists() and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance["request_parameters"] != params:
            raise ValueError(f"Cached request differs: {stem}")
        raw_bytes = gzip.decompress(raw_path.read_bytes())
        if hashlib.sha256(raw_bytes).hexdigest() != provenance["response_sha256"]:
            raise ValueError(f"Cached response SHA256 mismatch: {stem}")
        payload = json.loads(raw_bytes)
        return payload, provenance, validate_response(payload, season)
    errors = []
    for attempt in range(1, 5):
        started = time.monotonic()
        try:
            response = session.get(API, params=params, timeout=(15, 60))
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = min(float(response.headers.get("Retry-After", 10 * attempt)), 45)
                errors.append(f"HTTP {response.status_code}")
                print(f"retry {stem}: HTTP {response.status_code}; wait {retry_after}s", flush=True)
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            raw_bytes = response.content
            payload = response.json()
            hourly = validate_response(payload, season)
            provenance = {
                "source_model": "ERA5",
                "source_provider": "Open-Meteo Historical Weather API",
                "request_url": response.url,
                "request_parameters": params,
                "retrieved_utc": utc_now(),
                "response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "response_bytes_uncompressed": len(raw_bytes),
                "response_content_type": response.headers.get("Content-Type"),
                "request_attempts": attempt,
                "previous_attempt_errors": errors,
                "requested_latitude": season["requested_latitude"],
                "requested_longitude": season["requested_longitude"],
                "grid_latitude": payload["latitude"],
                "grid_longitude": payload["longitude"],
                "grid_elevation_m": payload.get("elevation"),
                "hourly_units": payload["hourly_units"],
                "documentation_url": DOCS,
                "attribution": "Contains modified Copernicus Climate Change Service information; ERA5 supplied through Open-Meteo (CC BY 4.0).",
                "downscaling": "disabled (elevation=nan)",
                "cell_selection": "nearest",
            }
            raw_path.write_bytes(gzip.compress(raw_bytes, mtime=0))
            write_json(provenance_path, provenance)
            return payload, provenance, hourly
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            print(f"retry {stem}: {errors[-1]}", flush=True)
            if attempt == 4:
                raise RuntimeError("; ".join(errors)) from exc
            time.sleep(min(5 * attempt, 20))
        finally:
            remaining = minimum_spacing - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    raise RuntimeError("; ".join(errors))


def save_table(table: pd.DataFrame, basename: str, *, compressed: bool = False) -> None:
    extension = ".csv.gz" if compressed else ".csv"
    table.to_csv(DEST / (basename + extension), index=False, float_format="%.6f", compression="gzip" if compressed else None)
    table.to_parquet(DEST / (basename + ".parquet"), index=False, compression="zstd")


def daily_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    x = hourly.copy()
    x["date"] = x.time_utc.dt.date.astype(str)
    x["rain_hours_gt_0_1mm"] = (x.rain_mm > 0.1).astype(int)
    x["rh_hours_ge_90pct"] = (x.relative_humidity_2m_pct >= 90).astype(int)
    x["shortwave_mj_m2"] = x.shortwave_radiation_w_m2 * 3600 / 1_000_000
    return x.groupby(["location_id", "date"], as_index=False).agg(
        tmean_c=("temperature_2m_c", "mean"),
        tmin_c=("temperature_2m_c", "min"),
        tmax_c=("temperature_2m_c", "max"),
        rh_mean_pct=("relative_humidity_2m_pct", "mean"),
        dewpoint_mean_c=("dew_point_2m_c", "mean"),
        precipitation_mm=("precipitation_mm", "sum"),
        rain_mm=("rain_mm", "sum"),
        rain_hours_gt_0_1mm=("rain_hours_gt_0_1mm", "sum"),
        rh_hours_ge_90pct=("rh_hours_ge_90pct", "sum"),
        wind_mean_m_s=("wind_speed_10m_m_s", "mean"),
        shortwave_mj_m2=("shortwave_mj_m2", "sum"),
        hour_count=("time_utc", "size"),
    )


def summarize_daily(daily: pd.DataFrame) -> dict:
    return {
        "day_count": len(daily),
        "hour_count": int(daily.hour_count.sum()),
        "tmean_c": float(daily.tmean_c.mean()),
        "tmin_c": float(daily.tmin_c.min()),
        "tmax_c": float(daily.tmax_c.max()),
        "rh_mean_pct": float(daily.rh_mean_pct.mean()),
        "dewpoint_mean_c": float(daily.dewpoint_mean_c.mean()),
        "precipitation_mm": float(daily.precipitation_mm.sum()),
        "rain_mm": float(daily.rain_mm.sum()),
        "rain_hours_gt_0_1mm": int(daily.rain_hours_gt_0_1mm.sum()),
        "rh_hours_ge_90pct": int(daily.rh_hours_ge_90pct.sum()),
        "wind_mean_m_s": float(daily.wind_mean_m_s.mean()),
        "shortwave_mj_m2": float(daily.shortwave_mj_m2.sum()),
        "rain_days_gt_1mm": int((daily.rain_mm > 1).sum()),
        "tmean_days_10_to_20c": int(daily.tmean_c.between(10, 20).sum()),
    }


def write_data_dictionary() -> None:
    rows = []
    for source, (name, unit, _, _) in VARIABLES.items():
        temporal = "preceding-hour sum" if source in ("rain", "precipitation") else "preceding-hour mean" if source == "shortwave_radiation" else "instantaneous hourly value"
        rows.append({"table": "hourly_weather", "field": name, "unit": unit, "definition": source + "; " + temporal})
    fields = {
        "location_id": ("identifier", "SHA256-based identifier of six-decimal requested latitude,longitude; not an independent grid cell"),
        "date": ("UTC date", "Calendar date of returned hourly timestamps"),
        "tmean_c": ("°C", "Mean of 24 hourly air temperatures at 2m"),
        "tmin_c": ("°C", "Minimum of 24 hourly air temperatures; monthly/seasonal value is minimum of daily minima"),
        "tmax_c": ("°C", "Maximum of 24 hourly air temperatures; monthly/seasonal value is maximum of daily maxima"),
        "rh_mean_pct": ("%", "Mean of 24 instantaneous hourly relative humidities at 2m"),
        "dewpoint_mean_c": ("°C", "Mean of 24 hourly dew-point temperatures at 2m"),
        "precipitation_mm": ("mm", "Sum of 24 returned hourly total precipitation values, including snow water equivalent"),
        "rain_mm": ("mm", "Sum of 24 returned hourly liquid precipitation values"),
        "rain_hours_gt_0_1mm": ("h", "Count of returned hourly rain values strictly greater than 0.1mm"),
        "rh_hours_ge_90pct": ("h", "Count of hourly 2m relative humidity >=90%; atmospheric humidity exposure, not measured leaf wetness"),
        "wind_mean_m_s": ("m/s", "Mean of 24 hourly wind speeds at 10m"),
        "shortwave_mj_m2": ("MJ/m²", "Sum(hourly shortwave W/m² * 3600/1e6)"),
        "hour_count": ("h", "Number of hourly records; daily value must equal 24"),
    }
    for name, (unit, definition) in fields.items():
        rows.append({"table": "daily_weather", "field": name, "unit": unit, "definition": definition})
    rows.extend([
        {"table":"seasonal_weather/monthly_weather","field":"rain_days_gt_1mm","unit":"days","definition":"Count of days with total liquid precipitation strictly greater than 1 mm"},
        {"table":"seasonal_weather/monthly_weather","field":"tmean_days_10_to_20c","unit":"days","definition":"Count of days with daily mean temperature between 10 and 20°C inclusive; descriptive temperature exposure, not a calibrated infection threshold"},
        {"table":"seasonal_weather/monthly_weather","field":"mean/sum aggregation","unit":"field-dependent","definition":"Temperature/RH/dewpoint/wind averaged over daily means; precipitation/rain/humidity hours/rain hours/radiation summed; minimum and maximum are period extrema"},
    ])
    pd.DataFrame(rows).to_csv(DEST / "data_dictionary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-sites", type=Path)
    parser.add_argument("--minimum-spacing", type=float, default=3.0, help="Seconds between seasonal API requests; conservative rate limiting.")
    args = parser.parse_args()
    prepare_directories()
    write_data_dictionary()
    external_path = args.external_sites
    if external_path is None and (DEST / "verified_external_sites.csv").exists():
        external_path = DEST / "verified_external_sites.csv"
    trials, external, seasons = target_tables(external_path)
    trials.to_csv(DEST / "basf_trial_weather_links.csv", index=False)
    external.to_csv(DEST / "external_site_weather_links.csv", index=False)
    seasons.to_csv(DEST / "coordinate_seasons.csv", index=False)
    inventory, frames, grid_rows = [], [], []
    session = requests.Session()
    session.headers["User-Agent"] = "NWAFU-public-Septoria-research-weather/1.0"
    for i, season in enumerate(seasons.to_dict("records"), 1):
        entry = dict(season)
        try:
            payload, provenance, hourly = retrieve(session, season, args.minimum_spacing)
            entry.update({
                "status": "success", "hourly_records": len(hourly),
                "day_count": len(hourly) // 24,
                "retrieved_utc": provenance["retrieved_utc"],
                "response_sha256": provenance["response_sha256"],
                "grid_latitude": payload["latitude"], "grid_longitude": payload["longitude"],
                "grid_elevation_m": payload.get("elevation"),
                "raw_path": f"data/era5/raw/{season['location_id']}_{season['season_year']}.json.gz",
                "error": "",
            })
            frames.append(hourly)
            grid_rows.append({"location_id": season["location_id"], "grid_latitude": payload["latitude"], "grid_longitude": payload["longitude"], "grid_elevation_m": payload.get("elevation")})
        except Exception as exc:
            entry.update({"status": "failed", "hourly_records": 0, "error": f"{type(exc).__name__}: {exc}"})
        inventory.append(entry)
        pd.DataFrame(inventory).to_csv(DEST / "retrieval_inventory.csv", index=False)
        print(f"[{i}/{len(seasons)}] {season['location_id']} {season['season_year']} {entry['status']} {entry.get('hourly_records',0)} hours", flush=True)
    if not frames:
        raise RuntimeError("No seasons downloaded successfully")
    combined = pd.concat(frames, ignore_index=True)
    value_cols = list(VARIABLES[k][0] for k in VARIABLES)
    duplicates = combined.duplicated(["location_id", "time_utc"], keep=False)
    overlapping = combined.loc[duplicates]
    if len(overlapping):
        inconsistent = overlapping.groupby(["location_id", "time_utc"])[value_cols].nunique().gt(1).any(axis=1)
        if inconsistent.any():
            raise ValueError("Overlapping seasonal responses conflict")
    hourly = combined.drop_duplicates(["location_id", "time_utc"]).sort_values(["location_id", "time_utc"]).reset_index(drop=True)
    daily = daily_from_hourly(hourly)
    assert not hourly.duplicated(["location_id", "time_utc"]).any()
    assert not daily.duplicated(["location_id", "date"]).any()
    assert daily.hour_count.eq(24).all()
    assert daily.tmin_c.le(daily.tmean_c).all() and daily.tmean_c.le(daily.tmax_c).all()
    assert daily.rh_hours_ge_90pct.between(0,24).all()
    assert daily.rain_hours_gt_0_1mm.between(0,24).all()
    save_table(hourly, "hourly_weather", compressed=True)
    save_table(daily, "daily_weather")
    successful = pd.DataFrame(inventory).query("status == 'success'")
    summary_rows, monthly_rows = [], []
    for season in successful.to_dict("records"):
        subset = daily.loc[(daily.location_id == season["location_id"]) & daily.date.between(season["window_start"],season["window_end"])]
        expected_days = (pd.Timestamp(season["window_end"])-pd.Timestamp(season["window_start"])).days+1
        assert len(subset) == expected_days
        summary_rows.append({k: season[k] for k in ("location_id", "season_year", "window_start", "window_end", "requested_latitude", "requested_longitude", "grid_latitude", "grid_longitude", "country_source", "dataset_ids", "basf_trial_count")} | summarize_daily(subset))
        for month, part in subset.groupby(subset.date.str[:7]):
            monthly_rows.append({"location_id": season["location_id"], "season_year": season["season_year"], "month": month, **summarize_daily(part)})
    seasonal = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(monthly_rows)
    save_table(seasonal, "seasonal_weather")
    save_table(monthly, "monthly_weather")
    registry = seasons.groupby("location_id", as_index=False).agg(
        requested_latitude=("requested_latitude", "first"), requested_longitude=("requested_longitude", "first"),
        country_source=("country_source", "first"), dataset_ids=("dataset_ids", lambda v: "|".join(sorted(set("|".join(v).split("|"))))),
        coordinate_sources=("coordinate_sources", "first"), requested_season_count=("season_year", "size"),
    )
    grids = pd.DataFrame(grid_rows).drop_duplicates()
    if grids.location_id.duplicated().any():
        raise ValueError("A requested location maps to different grids")
    registry = registry.merge(grids, how="left", on="location_id", validate="one_to_one")
    registry["grid_cell_id"] = [f"era5grid_{a:.2f}_{b:.2f}" if pd.notna(a) else None for a,b in zip(registry.grid_latitude,registry.grid_longitude)]
    lat1 = np.radians(registry.requested_latitude)
    lat2 = np.radians(registry.grid_latitude)
    delta_lat = lat2 - lat1
    delta_lon = np.radians(registry.grid_longitude-registry.requested_longitude)
    haversine = np.sin(delta_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2)**2
    registry["requested_to_grid_distance_km"] = 6371.0088*2*np.arcsin(np.sqrt(haversine.clip(0,1)))
    save_table(registry,"location_registry")
    failure_count = int(pd.DataFrame(inventory).status.eq("failed").sum())
    checks = {
        "completed_utc": utc_now(), "requested_coordinate_seasons": len(seasons),
        "successful_coordinate_seasons": len(successful), "failed_coordinate_seasons": failure_count,
        "requested_locations": seasons.location_id.nunique(), "retrieved_locations": hourly.location_id.nunique(),
        "unique_grid_cells": registry.grid_cell_id.nunique(), "basf_trials": len(trials),
        "basf_coordinate_seasons": len(seasons.loc[seasons.basf_trial_count.gt(0)]),
        "unique_hourly_rows": len(hourly), "unique_daily_rows": len(daily),
        "seasonal_summary_rows": len(seasonal), "monthly_summary_rows": len(monthly),
        "overlapping_hourly_rows_removed": len(combined)-len(hourly),
        "all_requested_hourly_timestamps_complete": failure_count == 0,
        "all_daily_hour_counts_24": bool(daily.hour_count.eq(24).all()),
        "duplicate_hourly_keys": int(hourly.duplicated(["location_id", "time_utc"]).sum()),
        "duplicate_daily_keys": int(daily.duplicated(["location_id", "date"]).sum()),
        "missing_weather_cells": int(hourly[value_cols].isna().sum().sum()),
        "overlapping_responses_agree": True,
        "units_and_physical_range_checks_pass": True,
        "humidity_exposure_is_measured_leaf_wetness": False,
        "hourly_numeric_ranges": {c: {"min":float(hourly[c].min()),"max":float(hourly[c].max())} for c in value_cols},
    }
    write_json(ANALYSIS / "validation.json", checks)
    outputs = []
    for path in sorted(DEST.iterdir()):
        if path.is_file() and path.name != GENERATED and path.name != "manifest.json":
            outputs.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(DEST / "manifest.json", {
        "created_utc": utc_now(), "source_model": "ERA5", "model_request": "models=era5",
        "api_endpoint": API, "documentation_url": DOCS,
        "license_url": "https://open-meteo.com/en/licence", "terms_url": "https://open-meteo.com/en/terms",
        "source_license": "CC BY 4.0 (Open-Meteo data); Copernicus ERA5 attribution applies",
        "attribution": "Contains modified Copernicus Climate Change Service information; ERA5 supplied through Open-Meteo.",
        "spatial_resolution_degrees": 0.25, "time_zone": "UTC", "cell_selection": "nearest",
        "downscaling": "disabled by elevation=nan", "hourly_variables": list(VARIABLES),
        "coverage_window_definition": "September 1 of the preceding year through September 30 of observation year, inclusive; fixed coverage window, not observed sowing/harvest dates",
        "hourly_key": ["location_id","time_utc"], "daily_key": ["location_id","date"],
        "seasonal_key": ["location_id","season_year"],
        "aggregation_notes": "UTC date of returned hourly timestamps; precipitation/rain are preceding-hour sums, shortwave is preceding-hour mean; radiation converted W m-2 to MJ m-2 with 3600/1e6. Consecutive windows overlap in September; matching raw values are deduplicated in daily/hourly tables.",
        "humidity_note": "Hours with 2m relative humidity >=90% describe atmospheric humidity exposure and are not measurements of canopy or leaf wetness.",
        "independence_note": "Duplicate BASF trials at one coordinate-year share weather; rounded source coordinates can also map to one ERA5 grid cell. Weather samples are not independent at trial level.",
        "validation": checks, "table_artifacts": outputs,
    })
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
