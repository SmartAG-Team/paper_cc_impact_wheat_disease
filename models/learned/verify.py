"""Verify frozen equivalence without importing any historical analysis archive."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from . import agc, disease, phenology
from .integrity import DATA, digest, verify, verify_record


def run(output=None):
    contract = verify()
    for record in contract["fixtures"]:
        verify_record(record)
    torch.set_num_threads(1)
    differences = {}
    with threadpool_limits(limits=1), np.load(DATA / "verification/crop_queries.npz") as z:
        kwargs = dict(known_sowing_dates=z["sowing_dates"], latitudes=z["latitudes"], longitudes=z["longitudes"])
        current = phenology.selected_cdf(z["baseline"], z["features"], z["progress"], z["increments"], **kwargs)
        np.testing.assert_array_equal(current, z["current"])
        differences["current_saved_CDF"] = float(np.max(np.abs(current - z["current"])))
        shortened = phenology.selected_cdf(z["baseline"][:, :, :215], z["features"][:, :215],
            z["progress"][:, :215], z["increments"][:, :215], **kwargs)
        np.testing.assert_array_equal(shortened, current[:, :, :215])
        differences["current_prefix"] = float(np.max(np.abs(shortened - current[:, :, :215])))
        weather_current = phenology.predict_cdf(z["weather"], **kwargs)
        np.testing.assert_array_equal(weather_current, z["current_weather_reference"])
        differences["current_original_same_batch_weather"] = float(np.max(np.abs(weather_current - z["current_weather_reference"])))
        for native, label in [(True, "agc_native"), (False, "agc_calibrated")]:
            predicted = agc.predict_cdf(z["weather"], progress=z["progress"], increments=z["increments"], native=native, **kwargs)
            np.testing.assert_allclose(predicted, z[label], atol=5e-6, rtol=0)
            differences[label + "_saved_CDF"] = float(np.max(np.abs(predicted - z[label])))
            short = agc.predict_cdf(z["weather"][:, :215], progress=z["progress"][:, :215],
                                   increments=z["increments"][:, :215], native=native, **kwargs)
            np.testing.assert_allclose(short, predicted[:, :, :215], atol=5e-6, rtol=0)
            differences[label + "_prefix"] = float(np.max(np.abs(short - predicted[:, :, :215])))
    queries = pd.read_csv(DATA / "verification/disease_queries.csv")
    with threadpool_limits(limits=1), np.load(DATA / "verification/disease_queries.npz") as z:
        values, links = disease.predict_queries(z["features"], queries, queries)
        np.testing.assert_array_equal(values, z["predictions"])
        differences["current_disease_percent"] = float(np.max(np.abs(values - z["predictions"])))
        link_count = sum(map(len, links.prior_targets))
        assert link_count == int(z["earlier_links"]) == 2610
    result = dict(status="passed", original_scientific_outputs_unchanged=True,
        portable_integrity_files=len(contract["files"]), source_nodes=len(contract["extracted_source_nodes"]),
        retained_crop_queries=6, current_disease_queries=len(queries), strictly_earlier_links=link_count,
        maximum_absolute_differences=differences, AGC_tolerance=5e-6,
        original_archive_batch_scope=contract["weather_reference_scope"],
        manifest_sha256=digest(DATA / "manifest.json"),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__))
    if output is not None:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
