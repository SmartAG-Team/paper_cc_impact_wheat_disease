"""Final forcing verification and unchanged frozen crop/disease replay.

Only explicit scientific dependencies enter the manifest. Original registration
documents remain provenance; old archive paths and unrelated code hashes are
not executable dependencies.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from model.seasonal_septoria.climate_alignment import apply_monthly_alignment
from model.seasonal_septoria.host import sowing_date_from_calendar
from .kernel import FIELDS, simulate_draw_seasons

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("data/final_climate")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(root=ROOT):
    return json.loads((Path(root) / DATA / "scientific_manifest.json").read_text())


def forcing_path(model, scenario, year):
    experiment = "historical" if year <= 2014 else scenario
    return f"data/paper_study/climate/nasa_v2/daily/{model}/{experiment}/{year}.parquet"


def required_forcing(manifest):
    return {
        forcing_path(model, scenario, year)
        for model in manifest["models"]
        for scenario in manifest["scenarios"]
        for harvest_year in manifest["harvest_years"]
        for year in (harvest_year - 1, harvest_year)
    }


def check_file(path, record, *, verify_hash):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in record and path.stat().st_size != record["bytes"]:
        raise ValueError(f"Input size changed: {path}")
    if verify_hash and digest(path) != record["sha256"]:
        raise ValueError(f"Input checksum changed: {path}")


def check_inputs(root=ROOT, *, verify_forcing_hashes=False):
    root = Path(root)
    manifest = read_manifest(root)
    supplied = {row["path"] for row in manifest["forcing"]}
    if supplied != required_forcing(manifest) or len(supplied) != 687:
        raise ValueError("Final forcing support must be exactly the 687 registered files")
    for row in [*manifest["inputs"], *manifest["reference_monthly_statistics"],
                *manifest["core_sources"], *manifest.get("implementation_sources", []),
                *manifest.get("reference_source_receipts", [])]:
        check_file(root / row["path"], row, verify_hash=True)
    if "environment" in manifest:
        check_file(root / manifest["environment"]["path"], manifest["environment"], verify_hash=True)
    for row in manifest["forcing"]:
        check_file(root / row["path"], row, verify_hash=verify_forcing_hashes)
        receipt = root / row["receipt"]
        check_file(receipt, {"sha256": row["receipt_sha256"]}, verify_hash=True)
        if json.loads(receipt.read_text())["parquet_sha256"] != row["sha256"]:
            raise ValueError(f"Forcing source identity changed: {receipt}")
    cells = cell_registry(root)
    if len(cells) != 14941 or int(cells.calendar_valid.sum()) != 14932:
        raise ValueError("Final land-use or crop-calendar population changed")
    return dict(status="verified", forcing_files=687, cells=14941,
                calendar_eligible_cells=14932,
                alignment_tables=len(manifest["models"]) * len(manifest["scenarios"]),
                reference_monthly_statistics=len(manifest["reference_monthly_statistics"]),
                reference_source_receipts=len(manifest.get("reference_source_receipts", [])),
                forcing_content_hashes_verified=verify_forcing_hashes,
                forcing_receipts_and_sizes_verified=True,
                core_source_files=len(manifest["core_sources"]))


def cell_registry(root=ROOT):
    root = Path(root)
    cells = pd.read_parquet(root / DATA / "cells.parquet")
    calendars = pd.read_parquet(root / DATA / "calendars.parquet")
    calendar = calendars.loc[calendars.crop_season.eq("winter_wheat")
                             & calendars.water_system.eq("rainfed")]
    cells = cells.merge(calendar[["cell_id", "planting_doy", "maturity_doy", "calendar_valid"]],
                        on="cell_id", how="left", sort=False, validate="one_to_one")
    cells["calendar_valid"] = cells.calendar_valid.fillna(False).astype(bool)
    cells["calendar_valid"] &= cells.planting_doy.notna() & cells.maturity_doy.notna()
    return cells


def sample_registry(root=ROOT):
    cells = cell_registry(root)
    draws = pd.read_csv(Path(root) / DATA / "spatial_draws.csv")
    return cells.loc[cells.cell_id.isin(draws.cell_id)].reset_index(drop=True)


def load_weather(model, scenario, year, cells, root=ROOT):
    root = Path(root)
    manifest = read_manifest(root)
    records = {row["path"]: row for row in manifest["forcing"]}
    start, end = pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year, 12, 31)
    dates = pd.date_range(start, end)
    arrays = {key: np.empty((len(cells), len(dates)), np.float32) for key in FIELDS}
    seen = np.zeros(len(dates), bool)
    ids = cells.cell_id.to_numpy()
    for forcing_year in (year - 1, year):
        relative = forcing_path(model, scenario, forcing_year)
        if relative not in records:
            raise ValueError(f"Year/model/scenario outside final registered support: {relative}")
        record = records[relative]
        check_file(root / relative, record, verify_hash=True)
        check_file(root / record["receipt"],
                   {"sha256": record["receipt_sha256"]}, verify_hash=True)
        frame = pq.read_table(root / relative, columns=["date", "cell_id", *FIELDS],
                              filters=[("cell_id", "in", ids.tolist())]).to_pandas()
        unique_dates = pd.DatetimeIndex(pd.to_datetime(frame.date.unique()))
        if len(frame) != len(cells) * len(unique_dates):
            raise ValueError("Incomplete forcing cell/day coverage")
        if not np.all(frame.cell_id.to_numpy().reshape(len(unique_dates), len(cells)) == ids[None, :]):
            raise ValueError("Forcing cell order differs from final registry")
        positions = dates.get_indexer(unique_dates)
        if np.any(positions < 0) or seen[positions].any():
            raise ValueError("Conflicting forcing dates")
        seen[positions] = True
        for key in FIELDS:
            arrays[key][:, positions] = frame[key].to_numpy().reshape(len(unique_dates), len(cells)).T
    if not seen.all():
        raise ValueError("Incomplete final forcing dates")
    coefficients = pd.read_parquet(root / DATA / "alignment" / model / scenario / "coefficients.parquet")
    if coefficients.duplicated(["cell_id", "month"]).any():
        raise ValueError("Duplicate alignment cell/month")
    joined = coefficients.set_index(["cell_id", "month"]).reindex(
        pd.MultiIndex.from_product([ids, range(1, 13)], names=["cell_id", "month"]))
    keys = ["temperature_offset", "humidity_logit_offset", "precipitation_ratio"]
    if not np.isfinite(joined[keys].to_numpy()).all():
        raise ValueError("Incomplete alignment inputs")
    aligned = apply_monthly_alignment(arrays, dates.month,
        {key: joined[key].to_numpy().reshape(len(cells), 12) for key in keys})
    return dates, aligned


def simulate_year(model, scenario, year, *, cells=None, root=ROOT, chunk_size=256):
    if chunk_size < 1:
        raise ValueError("Positive chunk size required")
    root = Path(root)
    manifest = read_manifest(root)
    relevant = {str(DATA / name) for name in ("cells.parquet", "calendars.parquet",
                "phenology_parameters.json", "disease_parameters.json", "stage_thresholds.json")}
    relevant.add(str(DATA / "alignment" / model / scenario / "coefficients.parquet"))
    for record in [*manifest["core_sources"], *manifest.get("implementation_sources", []),
                   *[row for row in manifest["inputs"] if row["path"] in relevant]]:
        check_file(root / record["path"], record, verify_hash=True)
    cells = cell_registry(root) if cells is None else cells
    dates, weather = load_weather(model, scenario, year, cells, root)
    phenology = json.loads((root / DATA / "phenology_parameters.json").read_text())
    fit = json.loads((root / DATA / "disease_parameters.json").read_text())["fitted"]
    thresholds = {int(k): v for k, v in json.loads(
        (root / DATA / "stage_thresholds.json").read_text())["all_stage_thresholds"].items()}
    eligible = np.flatnonzero(cells.calendar_valid.to_numpy(bool))
    outputs, maximum_mass_error = [], 0.0
    for first in range(0, len(eligible), chunk_size):
        indexes = eligible[first:first + chunk_size]
        part = cells.iloc[indexes]
        sowing = np.array([sowing_date_from_calendar(year, row.planting_doy, row.maturity_doy)
                           .to_datetime64().astype("datetime64[D]") for row in part.itertuples()])
        result, checks = simulate_draw_seasons(dates,
            {key: weather[key][indexes] for key in FIELDS}, sowing,
            part.latitude.to_numpy(), phenology, fit, thresholds)
        result["cell_id"] = part.cell_id.to_numpy()
        outputs.append(result)
        maximum_mass_error = max(maximum_mass_error, checks["mass_error"])
    if not outputs:
        raise ValueError("No calendar-eligible cells")
    metrics = pd.concat(outputs, ignore_index=True)
    columns = ["cell_id", "latitude", "longitude", "harvested_total_ha"]
    if "dominant_source_country" in cells:
        columns.append("dominant_source_country")
    result = cells[columns].merge(metrics, on="cell_id", how="left", sort=False, validate="one_to_one")
    result["status"] = result.status.fillna("missing_calendar")
    result["valid_complete_season"] = result.valid_complete_season.fillna(False).astype(bool)
    result["provider"], result["model"], result["scenario"], result["harvest_year"] = "nasa", model, scenario, year
    return result, dict(cells=len(cells), calendar_eligible_cells=len(eligible),
                        maximum_mass_error=maximum_mass_error)


def replay_sample(root=ROOT):
    root = Path(root)
    manifest = read_manifest(root)
    selected = manifest["sample_expectation"]
    expected = pd.read_parquet(root / selected["path"]).set_index("cell_id")
    cells = sample_registry(root)
    actual, diagnostics = simulate_year(selected["model"], selected["scenario"],
                                        selected["year"], cells=cells, root=root)
    actual = actual.set_index("cell_id").loc[expected.index, expected.columns]
    for column in expected:
        if pd.api.types.is_datetime64_any_dtype(expected[column]):
            actual[column] = actual[column].astype(expected[column].dtype)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-10)
    numeric = expected.select_dtypes(include=["number"]).columns
    difference = np.abs(actual[numeric].to_numpy(float) - expected[numeric].to_numpy(float))
    finite = difference[np.isfinite(difference)]
    return dict(status="verified", model=selected["model"], scenario=selected["scenario"],
                year=selected["year"], matches_frozen_final_census=True,
                maximum_numeric_difference=float(finite.max()) if finite.size else 0.0,
                **diagnostics)


def run_year(model, scenario, year, output, *, root=ROOT, chunk_size=256):
    output = Path(output).resolve()
    destination = output / model / scenario / f"{year}.parquet"
    receipt = destination.with_suffix(".json")
    manifest_sha = digest(Path(root) / DATA / "scientific_manifest.json")
    if destination.exists() or receipt.exists():
        if not destination.is_file() or not receipt.is_file():
            raise ValueError(f"Incomplete checkpoint: {destination}")
        record = json.loads(receipt.read_text())
        if record["scientific_manifest_sha256"] != manifest_sha or record["output_sha256"] != digest(destination):
            raise ValueError(f"Changed checkpoint: {destination}")
        return record
    result, diagnostics = simulate_year(model, scenario, year, root=root, chunk_size=chunk_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False, compression="zstd")
    record = dict(status="complete", model=model, scenario=scenario, harvest_year=year,
                  output=str(destination), output_sha256=digest(destination),
                  scientific_manifest_sha256=manifest_sha,
                  original_parameters_unchanged=True, **diagnostics)
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-inputs", action="store_true")
    action.add_argument("--replay-sample", action="store_true")
    action.add_argument("--year", type=int)
    action.add_argument("--all", action="store_true", help="Replay all 810 final census year/model/scenario jobs")
    action.add_argument("--rebuild-alignment", action="store_true", help="Reconstruct nine historical-only alignment tables")
    action.add_argument("--summarize", type=Path, metavar="ANNUAL_DIRECTORY",
                        help="Aggregate all 810 regenerated annual files with the original formulas")
    parser.add_argument("--verify-forcing-hashes", action="store_true")
    parser.add_argument("--model", default="ACCESS-CM2")
    parser.add_argument("--scenario", default="ssp585")
    parser.add_argument("--output", type=Path, help="Separate destination for regenerated annual checkpoints")
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.check_inputs:
        result = check_inputs(args.root, verify_forcing_hashes=args.verify_forcing_hashes)
    elif args.replay_sample:
        result = replay_sample(args.root)
    elif args.rebuild_alignment:
        if args.output is None:
            parser.error("--output is required for alignment reconstruction")
        from .alignment import rebuild_alignment
        result = rebuild_alignment(args.output, args.root)
    elif args.summarize:
        if args.output is None:
            parser.error("--output is required for aggregation")
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Aggregation requires an empty output directory")
        from .summary import summarize
        receipt = summarize(args.summarize, args.output, root=args.root)
        result = {key: receipt[key] for key in ("status", "annual_jobs", "landuse_cells",
                  "calendar_eligible_cells", "model_grid_seasons", "model_parameters_changed")}
        result["summary_directory"] = str(args.output.resolve())
    else:
        if args.output is None:
            parser.error("--output is required for annual reruns")
        manifest = read_manifest(args.root)
        if args.chunk_size < 1:
            parser.error("--chunk-size must be positive")
        if args.all:
            check_inputs(args.root)
            jobs = [(model, scenario, year) for model in manifest["models"]
                    for scenario in manifest["scenarios"] for year in manifest["harvest_years"]]
        else:
            if (args.model not in manifest["models"] or args.scenario not in manifest["scenarios"]
                    or args.year not in manifest["harvest_years"]):
                parser.error("Year/model/scenario must belong to final registered support")
            jobs = [(args.model, args.scenario, args.year)]
        for model, scenario, year in jobs:
            result = run_year(model, scenario, year, args.output,
                              root=args.root, chunk_size=args.chunk_size)
            print(json.dumps(result), flush=True)
        if args.all:
            from .summary import summarize
            result = summarize(args.output, args.output / "summary", root=args.root)
            print(json.dumps({"status": result["status"], "annual_jobs": result["annual_jobs"],
                              "summary_directory": str(args.output / "summary")}), flush=True)
        return
    print(json.dumps(result, indent=2))
