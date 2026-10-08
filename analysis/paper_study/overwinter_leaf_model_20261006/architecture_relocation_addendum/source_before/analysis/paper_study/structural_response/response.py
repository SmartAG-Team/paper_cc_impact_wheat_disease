"""Shared monotone observation response with physically isolated calibration.

The normalized expressed process state is an effective score capacity. The
response maps that state to the source INFECT percentage proxy; neither its
shape nor its fitted hazard identifies true affected leaf area or incidence.
"""

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analysis.paper_study.run_structural_canopy import onset_predictions
from analysis.paper_study.run_weather_canopy import training_rain_rate
from analysis.paper_study.structural_evaluation.run import subset_fields
from model.seasonal_septoria.refinement import calibration_weights
from model.seasonal_septoria.structural import (
    CanopyParameters, _target_coordinates, canopy_host, simulate_canopy,
)
from model.seasonal_septoria.wetness import duration_exposure


INITIATION_BOUNDS = (1e-6, 1.)
POWER_BOUNDS = (.5, 4.)
HOST_PARAMETERS = dict(flag_fraction=.3, leaf_interval=120., expansion_units=100.)
MULTISTARTS = ((-2.5, 1.), (-1.5, 1.), (-3.5, 1.),
              (-2.5, .6), (-2.5, 2.), (-1.5, 3.5))


def observation_response(fraction, power=1.):
    """Return 100 * (1 - exp(-(-log(1-D))**b)), preserving both endpoints."""
    values = np.asarray(fraction, float)
    if (not np.isfinite(power) or not POWER_BOUNDS[0] <= power <= POWER_BOUNDS[1]
            or not np.isfinite(values).all() or np.any(values < -1e-12)
            or np.any(values > 1 + 1e-12)):
        raise ValueError("Finite normalized fractions and bounded observation power required.")
    values = np.clip(values, 0., 1.)
    if power == 1.:
        return 100 * values
    # Explicit endpoint assignment avoids log(0) and preserves exact zeros.
    result = np.zeros(values.shape, float)
    result[values == 1.] = 100.
    interior = (values > 0.) & (values < 1.)
    dose = -np.log1p(-values[interior])
    result[interior] = -100 * np.expm1(-(dose ** power))
    return result


def _driver(data, operator, rain_rate):
    if operator == "daily_or":
        return None
    if operator != "duration_proxy" or rain_rate is None:
        raise ValueError("A declared weather operator and its fitted rain rate are required.")
    if not hasattr(data, "maximum_temperature"):
        raise ValueError("Duration exposure requires maximum-temperature forcing.")
    return duration_exposure(data.temperature, data.maximum_temperature,
        data.humidity, data.rain, rain_rate)["exposure"]


def fit_response(data, accumulation, thresholds, target_indices, candidate, weather):
    """Fit complete selected field-seasons after removing all withheld targets."""
    raw = np.asarray(target_indices)
    if (raw.ndim != 1 or raw.dtype.kind == "b"
            or not np.isfinite(raw.astype(float)).all()
            or np.any(raw.astype(float) != np.floor(raw.astype(float)))):
        raise ValueError("Calibration row indices must be finite integers.")
    indices = raw.astype(int)
    selected = subset_fields(data, indices)
    original_fields = np.sort(data.targets.iloc[indices].field_index.unique()).astype(int)
    development = np.asarray(accumulation, float)[original_fields].copy()
    target = selected.targets
    idx = _target_coordinates(selected, target)
    if not target.metric.eq("infection_percent_unspecified_basis").all():
        raise ValueError("The shared response requires unspecified-denominator INFECT proxies.")
    observed = target.value.to_numpy(float)
    if not np.isfinite(observed).all() or np.any((observed < 0.) | (observed > 100.)):
        raise ValueError("Finite calibration percentage proxies are required.")
    operator = candidate["weather_operator"]
    rate = training_rain_rate(selected, weather,
        selected.metadata.field_index.to_numpy(int)) if operator == "duration_proxy" else None
    exposure = _driver(selected, operator, rate)
    host = dict(candidate["host_parameters"])
    if host != HOST_PARAMETERS or candidate["latent_days"] not in [20., 30.]:
        raise ValueError("Only the registered fixed host and delay candidates are supported.")
    active, renewal, _, _ = canopy_host(development, selected.temperature, thresholds, **host)
    forcing = (selected.temperature, selected.humidity, selected.rain, active, renewal)
    weights = calibration_weights(target, .5)
    nonlinear = candidate["observation_link"] == "nonlinear"
    if candidate["observation_link"] not in ["identity", "nonlinear"]:
        raise ValueError("Unsupported observation response.")

    def unpack(x):
        return CanopyParameters(initiation=float(10 ** x[0]), amplification=0.,
            latent_days=float(candidate["latent_days"])), float(x[1]) if nonlinear else 1.

    def residual(x):
        parameters, power = unpack(x)
        trajectory = simulate_canopy(*forcing, parameters, environmental_response=exposure)
        return np.sqrt(weights) * (observation_response(trajectory.expressed[idx], power) - observed)

    lower, upper = ([-6., POWER_BOUNDS[0]], [0., POWER_BOUNDS[1]]) if nonlinear else ([-6.], [0.])
    records = []
    starts = MULTISTARTS if nonlinear else MULTISTARTS[:3]
    for log_initial, power_initial in starts:
        start = [log_initial, power_initial] if nonlinear else [log_initial]
        fitted = least_squares(residual, start, bounds=(lower, upper), max_nfev=150,
            ftol=1e-7, xtol=1e-7, gtol=1e-7)
        parameters, power = unpack(fitted.x)
        records.append(dict(start=start, parameters=asdict(parameters), observation_power=power,
            weighted_sse=float(np.square(fitted.fun).sum()), success=bool(fitted.success),
            evaluations=int(fitted.nfev), termination=str(fitted.message)))
    converged = [i for i, row in enumerate(records) if row["success"] and np.isfinite(row["weighted_sse"])]
    if not converged:
        raise RuntimeError("No deterministic observation-response calibration start converged.")
    winner = min(converged, key=lambda i: records[i]["weighted_sse"])
    best = records[winner]
    return dict(parameters=best["parameters"], observation_power=best["observation_power"],
        host_parameters=host, weather_operator=operator, rain_rate_mm_hour=rate,
        observation_link=candidate["observation_link"], final_weight=.5,
        initiation_bounds=list(INITIATION_BOUNDS), observation_power_bounds=list(POWER_BOUNDS),
        power_fixed=not nonlinear, amplification_fixed=0.,
        target_indices=indices.tolist(), calibration_field_ids=sorted(target.field_id.unique().tolist()),
        calibration_global_target_indices=target.global_target_index.astype(int).tolist(),
        training_weighted_sse=best["weighted_sse"],
        training_rmse=float(np.sqrt(best["weighted_sse"] / weights.sum())),
        multistart_records=records, winning_start=winner,
        calibration_only=True, withheld_fields_physically_removed=True,
        observed_disease_assimilated=False,
        observation="source_INFECT_percentage_proxy_unspecified_denominator",
        response_interpretation="empirical observation link; leaf area and incidence not identified")


def predict_response(data, accumulation, thresholds, fitted):
    """Map a trajectory while retaining its unchanged raw state interpretation."""
    if data.targets.value.notna().any():
        raise ValueError("Prediction inputs must contain redacted disease values.")
    idx = _target_coordinates(data, data.targets)
    active, renewal, _, _ = canopy_host(accumulation, data.temperature, thresholds,
        **fitted["host_parameters"])
    exposure = _driver(data, fitted["weather_operator"], fitted.get("rain_rate_mm_hour"))
    trajectory = simulate_canopy(data.temperature, data.humidity, data.rain, active, renewal,
        CanopyParameters(**fitted["parameters"]), environmental_response=exposure)
    return observation_response(trajectory.expressed[idx], fitted["observation_power"]), trajectory


def response_onsets(data, expressed, power, model):
    """Evaluate secondary onset from mapped scores with the existing censoring API."""
    mapped = observation_response(expressed, power) / 100.
    return onset_predictions(data, mapped, model)


def same_date_gradients(frame):
    """Pair adjacent ordinal leaves within each field and exact assessment date."""
    keys = ["field_id", "date", "leaf_index"]
    if frame.duplicated(keys).any():
        raise ValueError("Same-date leaf gradients require unique field/date/leaf assessments.")
    rows = []
    for (field, date), group in frame.groupby(["field_id", "date"]):
        leaves = group.set_index("leaf_index")
        for upper in sorted(leaves.index):
            lower = upper + 1
            if lower not in leaves.index:
                continue
            observed = float(leaves.loc[lower, "value"] - leaves.loc[upper, "value"])
            predicted = float(leaves.loc[lower, "predicted_percent"] - leaves.loc[upper, "predicted_percent"])
            rows.append(dict(field_id=field, date=date,
                coordinate_year=leaves.iloc[0].coordinate_year,
                upper_leaf_index=int(upper), lower_leaf_index=int(lower),
                observed_gradient=observed, predicted_gradient=predicted,
                gradient_error=predicted - observed))
    result = pd.DataFrame(rows, columns=["field_id", "date", "coordinate_year",
        "upper_leaf_index", "lower_leaf_index", "observed_gradient", "predicted_gradient", "gradient_error"])
    if not result.empty:
        result["weight"] = (1 / result.groupby("coordinate_year").field_id.transform("nunique")
            / result.groupby("field_id").date.transform("nunique")
            / result.groupby(["field_id", "date"]).upper_leaf_index.transform("size"))
    return result
