"""Frozen prediction equivalence and operational input contracts."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

DATA = Path(__file__).resolve().parents[2] / "data/final_models"


def runtime(name):
    spec = importlib.util.find_spec("analysis.final_models." + name)
    assert spec is not None, "Portable frozen runtime is missing"
    return importlib.import_module(spec.name)


def test_current_phenology_reproduces_saved_cdfs_and_shortened_prefix():
    model = runtime("phenology")
    with np.load(DATA / "verification/crop_queries.npz") as z:
        kwargs = dict(known_sowing_dates=z["sowing_dates"], latitudes=z["latitudes"],
                      longitudes=z["longitudes"])
        with threadpool_limits(limits=1):
            full = model.selected_cdf(z["baseline"], z["features"], z["progress"], z["increments"], **kwargs)
            short = model.selected_cdf(z["baseline"][:, :, :215], z["features"][:, :215],
                                       z["progress"][:, :215], z["increments"][:, :215], **kwargs)
        np.testing.assert_array_equal(full, z["current"])
        np.testing.assert_array_equal(short, full[:, :, :215])
        with pytest.raises(ValueError, match="sowing"):
            model.selected_cdf(z["baseline"], z["features"], z["progress"], z["increments"],
                               latitudes=z["latitudes"], longitudes=z["longitudes"])


def test_agc_native_and_calibrated_reproduce_saved_cdfs_and_prefix():
    model = runtime("agc")
    with np.load(DATA / "verification/crop_queries.npz") as z:
        kwargs = dict(known_sowing_dates=z["sowing_dates"], latitudes=z["latitudes"],
                      longitudes=z["longitudes"], progress=z["progress"], increments=z["increments"])
        for native, label in [(True, "agc_native"), (False, "agc_calibrated")]:
            full = model.predict_cdf(z["weather"], native=native, **kwargs)
            np.testing.assert_allclose(full, z[label], atol=5e-6, rtol=0)
            short_kwargs = {**kwargs, "progress": z["progress"][:, :215], "increments": z["increments"][:, :215]}
            short = model.predict_cdf(z["weather"][:, :215], native=native, **short_kwargs)
            np.testing.assert_allclose(short, full[:, :, :215], atol=5e-6, rtol=0)


def test_current_weather_to_cdf_matches_original_source_with_identical_batch():
    model = runtime("phenology")
    with np.load(DATA / "verification/crop_queries.npz") as z:
        full = model.predict_cdf(z["weather"], z["sowing_dates"], z["latitudes"], z["longitudes"])
        np.testing.assert_array_equal(full, z["current_weather_reference"])


def test_disease_reproduces_all_saved_queries_and_strictly_earlier_links():
    model = runtime("disease")
    queries = pd.read_csv(DATA / "verification/disease_queries.csv")
    with np.load(DATA / "verification/disease_queries.npz") as z:
        with threadpool_limits(limits=1):
            values, links = model.predict_queries(z["features"], queries, queries)
        np.testing.assert_array_equal(values, z["predictions"])
        assert sum(map(len, links.prior_targets)) == int(z["earlier_links"])
        for link in links.itertuples(index=False):
            dates = queries.set_index("global_target_index").loc[link.prior_targets, "date"]
            assert (pd.to_datetime(dates) < pd.Timestamp(link.date)).all()


def test_disease_rejects_fractional_identifiers_and_conflicting_metadata():
    model = runtime("disease")
    queries = pd.read_csv(DATA / "verification/disease_queries.csv")
    with np.load(DATA / "verification/disease_queries.npz") as z:
        fractional = queries.iloc[:1].copy()
        fractional["global_target_index"] = 0.5
        with pytest.raises(ValueError, match="integer"):
            model.predict_queries(z["features"][:1], fractional, queries)
        conflict = queries.iloc[:1].copy()
        conflict["site_id"] = "conflicting_site"
        with pytest.raises(ValueError, match="conflicts"):
            model.predict_queries(z["features"][:1], conflict, queries)
        invalid_leaf = queries.iloc[:1].copy()
        invalid_leaf["leaf_index"] = 0.5
        with pytest.raises(ValueError, match="integers"):
            model.predict_queries(z["features"][:1], invalid_leaf, queries)


def test_same_day_earlier_timestamp_is_excluded_like_empty_history():
    model = runtime("disease")
    queries = pd.read_csv(DATA / "verification/disease_queries.csv").iloc[:1].copy()
    history = queries.copy()
    history["date"] = history.date + "T00:01:00"
    queries["date"] = queries.date + "T23:59:00"
    with np.load(DATA / "verification/disease_queries.npz") as z:
        actual, links = model.predict_queries(z["features"][:1], queries, history)
        empty, _ = model.predict_queries(z["features"][:1], queries, history.iloc[:0])
    np.testing.assert_array_equal(actual, empty)
    assert links.prior_targets.iloc[0] == []


def test_compact_integrity_detects_changed_checkpoint(tmp_path):
    integrity = runtime("integrity")
    import json
    manifest = json.loads((DATA / "manifest.json").read_text())
    item = next(row for row in manifest["files"] if row["path"].endswith("current/disease.joblib"))
    changed = tmp_path / "changed.joblib"
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        integrity.verify_record({**item, "path": str(changed)})
