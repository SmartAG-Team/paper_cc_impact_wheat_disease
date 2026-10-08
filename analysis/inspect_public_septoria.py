"""Profile downloaded Septoria sources and export traceable observation tables.

Run with pandas, openpyxl and xlrd. Source files are never modified.
"""
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public_septoria"
OUT = ROOT / "analysis" / "public_septoria"
OUT.mkdir(exist_ok=True)
profiles = {}
flags = []


def add_flag(dataset, field, issue, count, detail):
    flags.append(dict(dataset_id=dataset, field=field, issue=issue,
                      affected_count=int(count), detail=detail))


# Controlled inoculations: preserve each source cell and its original row number.
# A first visible pycnidium is a proxy for reproductive development, not a
# measurement of spore release or the exact time of stomatal penetration.
path = DATA / "hafeez-2025-infection" / "Hafeez et al. Stb15, Nature Plants - pathology data.xlsx"
sheet_isolates = {
    "IPO323_results_AUDPC": "IPO323",
    "IPOO88004_raw": "IPO88004",  # Original sheet name contains an extra O.
    "IPO90012_scores_raw": "IPO90012",
}
parts = []
sheet_profiles = []
for sheet, isolate in sheet_isolates.items():
    frame = pd.read_excel(path, sheet_name=sheet)
    score_cols = [c for c in frame if re.fullmatch(r"[pd]\d+", str(c))]
    frame["source_excel_row"] = range(2, len(frame) + 2)
    metadata_cols = [c for c in ["Index", "Batch", "Rep", "Scorer", "Tray", "Box", "Block", "Plot", "Acc", "Name", "Line", "GRU", "Type"] if c in frame]
    long = frame.melt(id_vars=["source_excel_row"] + metadata_cols,
                      value_vars=score_cols, var_name="source_column", value_name="raw_value")
    number = pd.to_numeric(long["raw_value"], errors="coerce")
    is_missing = long["raw_value"].isna() | long["raw_value"].astype(str).str.strip().isin(["", "-", "NA"])
    invalid = number.notna() & ~number.between(0, 100)
    ambiguous = number.isna() & ~is_missing
    long["quality_status"] = "valid"
    long.loc[is_missing, "quality_status"] = "missing"
    long.loc[invalid, "quality_status"] = "outside_0_100"
    long.loc[ambiguous, "quality_status"] = "unresolved_text"
    long["value_percent"] = number.where(number.between(0, 100))
    long["days_post_inoculation"] = long["source_column"].str[1:].astype(int)
    long["measurement"] = long["source_column"].str[0].map({"p": "pycnidial_coverage", "d": "leaf_damage_necrosis_and_chlorosis"})
    long["source_sheet"] = sheet
    long["isolate"] = isolate
    long["source_file"] = str(path.relative_to(ROOT))
    long["source_record_id"] = sheet + ":" + long["source_excel_row"].astype(str)
    parts.append(long)
    sheet_profiles.append(dict(sheet=sheet, isolate=isolate, seedling_records=len(frame),
                               scheduled_days=sorted({int(c[1:]) for c in score_cols}),
                               valid_scores=int((long["quality_status"] == "valid").sum()),
                               out_of_range=int(invalid.sum()), unresolved_text=int(ambiguous.sum())))
    if invalid.any():
        add_flag("hafeez-2025-infection", sheet, "percentage_out_of_range", invalid.sum(),
                 "One original damage score is 200%; the normalized value is missing. The original cell is preserved.")
    if ambiguous.any():
        add_flag("hafeez-2025-infection", sheet, "unresolved_score_text", ambiguous.sum(),
                 "Tokens such as 5?, ,70 and sn remain unresolved; no replacement values are inferred.")
scores = pd.concat(parts, ignore_index=True)
assert not scores.duplicated(["source_sheet", "source_excel_row", "source_column"]).any()
scores.to_csv(OUT / "hafeez_infection_scores.csv", index=False)
profiles["hafeez-2025-infection"] = {"sheets": sheet_profiles, "source_records": sum(p["seedling_records"] for p in sheet_profiles),
    "source_score_cells": len(scores), "valid_score_cells": int((scores["quality_status"] == "valid").sum()),
    "design": "300 Watkins landraces plus controls; source protocol uses five replicates per line, 18 C day / 12 C night and conidial inoculation.",
    "limitations": "One thermal regime; source row is the experimental record. Pycnidia coverage is a proxy for reproduction, not emitted spore counts."}

# Field sporulation assessments: comments and semicolon separator are explicit.
path = DATA / "orellana-torrejon-2022-field" / "F1_Field_disease_severity_rawdata.csv"
frame = pd.read_csv(path, sep=";", comment="#")
frame["date_iso"] = pd.to_datetime(frame["date"], dayfirst=True).dt.strftime("%Y-%m-%d")
value_cols = [c for c in frame if c.startswith("%spor_")]
long = frame.melt(id_vars=[c for c in frame if c not in value_cols], value_vars=value_cols,
                  var_name="source_leaf_category", value_name="sporulating_area_percent")
long["raw_value"] = long["sporulating_area_percent"]
tokens = long["raw_value"].astype(str).str.strip()
numeric = pd.to_numeric(long["raw_value"], errors="coerce")
long["quality_status"] = "valid"
long.loc[long["raw_value"].isna() | tokens.eq("NA"), "quality_status"] = "not_assessed"
long.loc[tokens.eq("S"), "quality_status"] = "senescent"
long["sporulating_area_percent"] = numeric
assert numeric.dropna().between(0, 100).all()
add_flag("orellana-torrejon-2022-field", "%spor_F*", "senescence_marker", tokens.eq("S").sum(),
         "The source analysis explicitly defines S as senescent. It is a leaf-state marker, not zero disease or 100% sporulation.")
long.to_csv(OUT / "inrae_field_sporulation_scores.csv", index=False)
profiles["orellana-torrejon-2022-field"] = dict(plant_assessment_records=len(frame), leaf_assessments=int(numeric.notna().sum()),
    senescent_leaf_markers=int(tokens.eq("S").sum()),
    distinct_dates=frame["date_iso"].nunique(), date_counts=frame["date_iso"].value_counts().sort_index().to_dict(),
    cultivars=sorted(frame["var_origin"].unique()), mixtures=sorted(frame["mixture"].unique()), blocks=sorted(frame["rep"].unique()),
    limitations="Distinct dates are pooled across treatments, not 19 measurements for every plot. Plant labels and leaf categories do not establish the identity of the same physical leaf across dates. No airborne inoculum series is present in the downloaded F1 data.")

# Large field collection: keep two nonidentical observations sharing a leaf label.
path = DATA / "karisto-2018-field" / "Output_c1_c3.csv"
frame = pd.read_csv(path)
ident = frame["Picture"].str.extract(r"^(c\d+)_sn(\d+)_(\d+)$")
assert not ident.isna().any().any()
frame["source_row"] = range(2, len(frame) + 2)
frame["collection"] = ident[0]
frame["sowing_number"] = ident[1].astype(int)
frame["date_from_repository_notes"] = frame["collection"].map({"c1": "2016-05-25", "c3": "2016-07-04"})
frame["duplicate_leaf_label"] = frame.duplicated("Picture", keep=False)
cultivars = pd.read_csv(path.parent / "cultivars_sowingnumbers.csv")
assert cultivars["Sow_Nr"].is_unique
frame = frame.merge(cultivars, how="left", left_on="sowing_number", right_on="Sow_Nr", validate="many_to_one")
frame.to_csv(OUT / "karisto_field_leaf_scores.csv", index=False)
profiles["karisto-2018-field"] = dict(leaf_records=len(frame), collection_counts=frame["collection"].value_counts().to_dict(),
    plots=frame["sowing_number"].nunique(), unmapped_cultivars=int(frame["Gen"].isna().sum()),
    duplicate_label_rows=int(frame["duplicate_leaf_label"].sum()),
    undefined_pycnidia_measurements=int(frame["meanPycnidiaArea"].isna().sum()),
    limitations="Two destructive leaf collections. Repository notes supply nominal collection dates. Day-zero visual score is assumed in the separate AUDPC workbook, not observed.")
add_flag("karisto-2018-field", "Picture", "duplicate_leaf_label", frame["duplicate_leaf_label"].sum(),
         "c1_sn471_1 identifies two different measurements; retain both source rows and do not deduplicate by label.")

# Durum trial files contain end-of-season leaf measurements and daily weather.
archive = DATA / "durum-mixtures-2020" / "Durum_mixtures_Data_out.zip"
weather_parts = []
with zipfile.ZipFile(archive) as bundle:
    frame = pd.read_excel(BytesIO(bundle.read("out/Cultivar_Mix_severity_relative_leaf_data_all.xls")))
    frame.to_csv(OUT / "durum_field_leaf_scores.csv", index=False)
    profiles["durum-mixtures-2020"] = {"leaf_records": len(frame), "year_counts": {str(k): int(v) for k, v in frame["year"].value_counts().sort_index().items()},
        "limitations": "Late-season assessments do not resolve infection onset. The repository abstract describes 3074 scanned leaves, while deposited leaf-level tables contain 3004 rows; the reason for the difference is undocumented in inspected files."}
    for season in [2018, 2019]:
        source_name = f"out/{season}_Tunisia_cultivar_mixtures_weather.xlsx"
        weather = pd.read_excel(BytesIO(bundle.read(source_name)))
        weather["source_season_label"] = season
        weather["source_archive_member"] = source_name
        weather["date_valid_for_season"] = season == 2018
        weather_parts.append(weather)
weather = pd.concat(weather_parts, ignore_index=True)
weather.to_csv(OUT / "durum_daily_weather.csv", index=False)
profiles["durum-mixtures-2020"]["weather_rows"] = len(weather)
add_flag("durum-mixtures-2020", "2019 weather Date", "calendar_mismatch", len(weather_parts[1]),
         "The 2019 workbook repeats 2017-11-01 to 2018-05-31 dates despite different weather values; date correction requires source clarification. Original dates are retained.")

# Process experiments: observed traits and numerical candidate fits are separate.
path = DATA / "chaloner-2019-weather-response" / "rstb20180266_si_002.xlsx"
workbook = pd.ExcelFile(path)
sheet_profiles = []
for sheet in workbook.sheet_names:
    frame = pd.read_excel(path, sheet_name=sheet)
    sheet_profiles.append(dict(sheet=sheet, rows=len(frame), columns=list(frame.columns.astype(str))))
    if sheet == "1h Thermal death":
        numeric = pd.to_numeric(frame["percentage_dead_1h"], errors="coerce")
        add_flag("chaloner-2019-weather-response", sheet, "negative_corrected_mortality", (numeric < 0).sum(),
                 "Baseline-corrected mortality contains negative estimates; these are not valid negative death probabilities and require a constrained observation model.")
profiles["chaloner-2019-weather-response"] = {"sheets": sheet_profiles,
    "limitations": "Germination and in vitro growth constrain establishment processes; they do not directly estimate in planta latent period or secondary transmission. Brute-force fit sheets are model outputs, not new biological replicates."}

# Nordic-Baltic data contain mixed leaf-blotch management outcomes, not a
# Septoria-specific longitudinal epidemic series. Retain packed weather as raw.
path = DATA / "nordic-baltic-2012-2016" / "Yield.xlsx"
frame = pd.read_excel(path, header=3)
year = pd.to_numeric(frame["Year"], errors="coerce")
keep = year.between(1900, 2100)
yield_data = frame.loc[keep].copy()
profiles["nordic-baltic-2012-2016"] = {"yield_rows": len(yield_data), "yield_years": sorted(year[keep].astype(int).unique().tolist()),
    "limitations": "Multiple leaf-blotch pathogens, fungicide and yield data. Weather uses a custom packed format and xx missing markers. Development-stage workbook uses 2022 display dates as day-of-year templates; these must not be read as trial-year dates."}

# Independent integrity audit of every downloaded source file.
manifests = json.loads((DATA / "download-manifest.json").read_text())
integrity = []
for record in manifests:
    for item in record["files"]:
        payload = (ROOT / item["project_relative_path"]).read_bytes()
        passed = (len(payload) == item["bytes"] and hashlib.sha256(payload).hexdigest() == item["sha256"]
                  and hashlib.md5(payload).hexdigest() == item["md5"] and item["repository_checksum_verified"])
        assert passed, item["project_relative_path"]
        integrity.append(dict(dataset_id=record["dataset_id"], filename=item["filename"], bytes=len(payload),
                              repository_md5_verified=True, local_sha256_verified=True))
pd.DataFrame(integrity).to_csv(OUT / "file_integrity.csv", index=False)
pd.DataFrame(flags).to_csv(OUT / "quality_flags.csv", index=False)
(OUT / "profiles.json").write_text(json.dumps(profiles, indent=2, ensure_ascii=False) + "\n")
summary = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "source_collections_with_files": len({r["dataset_id"] for r in manifests}),
    "downloaded_files": len(integrity), "downloaded_bytes": sum(r["bytes"] for r in integrity),
    "repository_md5_checks_passed": len(integrity), "quality_flags": len(flags), "profiles": profiles}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "profiles"}, indent=2))
