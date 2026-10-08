"""Independent archive integrity, linkage, and aggregation checks."""

from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "era5"
ast.parse((ROOT / "analysis" / "gather_era5_weather.py").read_text())
manifest = json.loads((DATA / "manifest.json").read_text())
inventory = pd.read_csv(DATA / "retrieval_inventory.csv")
hourly = pd.read_parquet(DATA / "hourly_weather.parquet")
daily = pd.read_parquet(DATA / "daily_weather.parquet")
seasonal = pd.read_parquet(DATA / "seasonal_weather.parquet")
registry = pd.read_csv(DATA / "location_registry.csv")
links = pd.read_csv(DATA / "basf_trial_weather_links.csv")
external = pd.read_csv(DATA / "external_site_weather_links.csv")
basf = pd.read_csv(ROOT / "data" / "basf-wheat-diseases.txt", sep="\t")

assert inventory.status.eq("success").all()
assert len(inventory) == 72 and len(seasonal) == 72
assert len(hourly) == 676128 and len(daily) == 28172
assert not hourly.duplicated(["location_id", "time_utc"]).any()
assert not daily.duplicated(["location_id", "date"]).any()
assert daily.hour_count.eq(24).all()
assert len(links) == basf.TrialId.nunique() == 76
assert set(links.TrialId) == set(basf.TrialId)
joined = links.merge(seasonal[["location_id", "season_year"]], how="left", on=["location_id", "season_year"], indicator=True, validate="many_to_one")
assert joined._merge.eq("both").all()
assert set(external.dataset_id) == {"ethz-2025-septoria-field", "karisto-2018-field", "durum-mixtures-2020", "orellana-torrejon-2022-field"}
assert len(external) == 8
external_join = external.merge(seasonal[["location_id", "season_year"]], how="left", on=["location_id", "season_year"], indicator=True, validate="many_to_one")
assert external_join._merge.eq("both").all()
assert registry.location_id.nunique() == 59 and registry.grid_cell_id.nunique() == 54
assert registry.requested_to_grid_distance_km.le(20).all()

responses_checked = 0
for row in inventory.to_dict("records"):
    raw = ROOT / row["raw_path"]
    payload_bytes = gzip.decompress(raw.read_bytes())
    assert hashlib.sha256(payload_bytes).hexdigest() == row["response_sha256"]
    provenance = json.loads(raw.with_name(raw.name.replace(".json.gz", ".provenance.json")).read_text())
    assert provenance["response_sha256"] == row["response_sha256"]
    assert provenance["request_parameters"]["models"] == "era5"
    assert provenance["request_parameters"]["elevation"] == "nan"
    assert provenance["request_parameters"]["cell_selection"] == "nearest"
    x = json.loads(payload_bytes)
    expected = ((pd.Timestamp(row["window_end"]) - pd.Timestamp(row["window_start"])).days+1)*24
    assert len(x["hourly"]["time"]) == row["hourly_records"] == expected
    assert all(len(v) == expected for v in x["hourly"].values())
    assert x["utc_offset_seconds"] == 0
    responses_checked += 1

artifacts_checked = 0
for artifact in manifest["table_artifacts"]:
    path = ROOT / artifact["path"]
    assert path.stat().st_size == artifact["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    artifacts_checked += 1

# Conservation across all hours/days checks unit handling and loss during aggregation.
for name in ("rain_mm", "precipitation_mm"):
    assert np.isclose(hourly[name].sum(), daily[name].sum(), rtol=1e-12)
assert np.isclose((hourly.shortwave_radiation_w_m2*0.0036).sum(), daily.shortwave_mj_m2.sum(), rtol=1e-12)
assert int((hourly.relative_humidity_2m_pct >= 90).sum()) == int(daily.rh_hours_ge_90pct.sum())
assert int((hourly.rain_mm > 0.1).sum()) == int(daily.rain_hours_gt_0_1mm.sum())

# Direct scalar arithmetic on each location's first available day does not reuse the pipeline aggregator.
sample_days_checked = 0
for row in daily.groupby("location_id", sort=True).first().reset_index().to_dict("records"):
    loc = hourly.loc[hourly.location_id == row["location_id"]]
    h = loc.loc[loc.time_utc.dt.strftime("%Y-%m-%d") == row["date"]]
    assert len(h) == 24
    assert math.isclose(math.fsum(h.temperature_2m_c.tolist())/24, row["tmean_c"], abs_tol=1e-10)
    assert min(h.temperature_2m_c) == row["tmin_c"]
    assert max(h.temperature_2m_c) == row["tmax_c"]
    assert math.isclose(math.fsum(h.shortwave_radiation_w_m2.tolist())*0.0036, row["shortwave_mj_m2"], abs_tol=1e-10)
    assert sum(v >= 90 for v in h.relative_humidity_2m_pct.tolist()) == row["rh_hours_ge_90pct"]
    sample_days_checked += 1

result = {
    "checked_utc": datetime.now(timezone.utc).isoformat(),
    "result": "pass", "raw_response_sha256_verified": responses_checked,
    "manifest_artifacts_sha256_verified": artifacts_checked,
    "direct_daily_aggregation_samples_verified": sample_days_checked,
    "global_hourly_to_daily_conservation": "pass",
    "all_basf_trials_join_to_exactly_one_weather_season": True,
    "all_external_site_years_join_to_weather": True,
    "script_syntax": "pass",
    "requested_to_grid_distance_max_km": float(registry.requested_to_grid_distance_km.max()),
}
(ROOT / "analysis" / "era5" / "independent_validation.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
