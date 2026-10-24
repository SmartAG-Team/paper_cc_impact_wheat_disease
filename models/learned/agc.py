"""Experimental frozen native/calibrated AGC crop CDFs on supplied weather."""
import json

import numpy as np
import pandas as pd
import torch

from .integrity import DATA, verify
from . import agc_math
from .phenology_math import build_features


def predict_cdf(weather, known_sowing_dates, latitudes, longitudes, *,
                progress=None, increments=None, native=False, batch_size=64):
    """Use the three frozen fits; no probability-residual head was adopted."""
    verify("agc")
    weather = np.asarray(weather)
    n = len(weather)
    cycles = pd.DataFrame(dict(SOWING_DATE=known_sowing_dates, lat=latitudes, lon=longitudes))
    if len(cycles) != n:
        raise ValueError("One known sowing/coordinate record per weather sequence required")
    if (progress is None) != (increments is None):
        raise ValueError("Both frozen TPV arrays required together")
    if progress is None:
        metadata = json.loads((DATA / "current/model_metadata.json").read_text())
        p, inc = [], []
        for i, row in enumerate(cycles.itertuples(index=False)):
            dates = pd.date_range(row.SOWING_DATE, periods=weather.shape[1])
            _, value, step = build_features(weather[i], dates, row.lat, metadata["gates"])
            p.append(value)
            inc.append(step)
        progress, increments = np.asarray(p), np.asarray(inc)
    fitting = json.loads((DATA / "agc/preprocessing.json").read_text())
    selection = json.loads((DATA / "agc/selection.json").read_text())
    batch = agc_math.input_batch(weather, progress, increments, cycles, np.arange(n), fitting)
    members = []
    for seed in selection["seeds"]:
        saved = torch.load(DATA / "agc" / f"seed{seed}.pt", map_location="cpu", weights_only=False)
        model = agc_math.make_model(saved["settings"])
        model.load_state_dict(saved["state_dict"])
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                output = model({key: value[start:start + batch_size] for key, value in batch.items()})
                values.append(output["corrected_progress"].numpy())
        members.append(np.concatenate(values))
    recipe = {} if native else selection["recipe"]
    return agc_math.cdf_from_members(np.asarray(members), progress, fitting["process_thresholds"],
                                     fitting["residual_temperature"], **recipe)
