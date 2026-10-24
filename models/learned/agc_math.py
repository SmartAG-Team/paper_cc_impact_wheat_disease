"""Original canonical preprocessing and calibrated CDF definitions."""
from dataclasses import asdict, is_dataclass
import numpy as np
import pandas as pd
import torch
from scipy.special import log_expit
from .integrity import canonical_identity
from . import agc_network

def source_manifest():
    return canonical_identity()

def activate_source():
    return agc_network


HORIZON = 330


STAGES = [10, 31, 51, 87]


PROTOCOL = "canonical-agc-frozen-wheat-tpv-sowing-onward-no-ndvi-v1"


FEATURE_NAMES = ["day_of_year_sine", "day_of_year_cosine", "daylight_fraction",
                 "elapsed_days_over_365.25", "t_mean_scaled", "t_max_scaled",
                 "t_min_scaled", "p_sum_scaled", "t_mean_finite", "t_max_finite",
                 "t_min_finite", "p_sum_finite", "tpv_cumulative_scaled", "tpv_increment_scaled"]


def _indices(indices, size, *, unique=False):
    values = np.asarray(indices)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
        raise ValueError("Nonempty integer cycle indices are required")
    if (values < 0).any() or (values >= size).any() or (unique and len(np.unique(values)) != len(values)):
        raise ValueError("Cycle indices are out of bounds or duplicated")
    return values.astype(np.int64, copy=False)


def _thresholds(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or values[0] <= 0 or (np.diff(values) <= 0).any():
        raise ValueError("Four finite positive increasing wheat thresholds are required")
    return values


def _select_inputs(weather, progress, increments, cycles, indices, *, unique=False):
    weather = np.asarray(weather)
    progress, increments = np.asarray(progress), np.asarray(increments)
    if weather.ndim != 3 or weather.shape[-1] != 4 or not 0 < weather.shape[1] <= HORIZON:
        raise ValueError("Weather must have shape [N,H,4] with 1 <= H <= 330")
    if progress.shape != weather.shape[:2] or increments.shape != progress.shape:
        raise ValueError("TPV progress and increments must have shape [N,H]")
    records = cycles if isinstance(cycles, pd.DataFrame) else pd.DataFrame(cycles)
    if len(records) != len(weather) or not {"SOWING_DATE", "lat", "lon"}.issubset(records.columns):
        raise ValueError("One known sowing date and latitude/longitude per cycle are required")
    selected = _indices(indices, len(records), unique=unique)
    records = records.iloc[selected].copy()
    try:
        sowing = pd.DatetimeIndex(pd.to_datetime(records["SOWING_DATE"], errors="raise"))
        coordinates = records[["lon", "lat"]].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Known sowing dates and coordinates must be valid") from error
    if sowing.isna().any() or sowing.tz is not None or not (sowing == sowing.normalize()).all():
        raise ValueError("Known sowing dates must be complete timezone-free calendar dates")
    if not np.isfinite(coordinates).all() or (np.abs(coordinates) > [180, 90]).any():
        raise ValueError("Known longitude/latitude must be finite and within their ranges")
    horizon = weather.shape[1]
    lengths = records["input_length"].to_numpy() if "input_length" in records else np.full(len(records), horizon)
    try:
        finite_lengths = np.asarray(lengths, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Input lengths must be finite positive integers") from error
    if (not np.isfinite(finite_lengths).all() or (finite_lengths != np.floor(finite_lengths)).any()
            or (finite_lengths < 1).any() or (finite_lengths > horizon).any()):
        raise ValueError("Input lengths must be nonempty valid prefixes within H")
    lengths = finite_lengths.astype(np.int64)
    valid = np.arange(horizon)[None] < lengths[:, None]
    weather = np.asarray(weather[selected], dtype=np.float64)
    progress = np.asarray(progress[selected], dtype=np.float64)
    increments = np.asarray(increments[selected], dtype=np.float64)
    raw_weather = weather[valid]
    if not np.isfinite(raw_weather).all() or (raw_weather[:, 3] < 0).any():
        raise ValueError("Valid daily weather must be finite with nonnegative precipitation")
    if ((raw_weather[:, 2] > raw_weather[:, 0]) | (raw_weather[:, 0] > raw_weather[:, 1])).any():
        raise ValueError("Daily temperatures must satisfy minimum <= mean <= maximum")
    if (not np.isfinite(progress[valid]).all() or not np.isfinite(increments[valid]).all()
            or (progress[valid] < 0).any() or (increments[valid] < 0).any()):
        raise ValueError("Valid raw TPV must be finite and nonnegative")
    previous = np.concatenate([np.zeros((len(records), 1)), progress[:, :-1]], axis=1)
    tolerance = 64*np.finfo(np.float64).eps*np.maximum(np.maximum(np.abs(progress), np.abs(previous)), 1)
    if ((np.abs(progress-previous-increments) > tolerance) & valid).any():
        raise ValueError("TPV cumulative differences must equal daily increments from sowing")
    if ((np.diff(progress, axis=1) < 0) & valid[:, 1:]).any():
        raise ValueError("TPV progress must be monotone")
    # Padding carries no information; invalid raw padding may contain NaN/junk.
    weather = np.where(valid[:, :, None], weather, 0)
    progress = np.where(valid, progress, 0)
    increments = np.where(valid, increments, 0)
    return weather, progress, increments, records, selected, sowing, coordinates, valid, lengths


def _transform(values, fit, width):
    try:
        imputation, mean, scale = (np.asarray(fit[key], dtype=np.float64) for key in ("imputation", "mean", "scale"))
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("Training preprocessing is incomplete") from error
    if (any(value.shape != (width,) for value in (imputation, mean, scale))
            or not all(np.isfinite(value).all() for value in (imputation, mean, scale)) or (scale <= 0).any()):
        raise ValueError("Training feature scales must be finite positive vectors")
    return ((np.where(np.isfinite(values), values, imputation)-mean)/scale).astype(np.float32)


def _validate_preprocessing(preprocessing):
    manifest = source_manifest()
    if (preprocessing.get("protocol") != PROTOCOL or preprocessing.get("source_sha256") != manifest["sha256"]
            or preprocessing.get("source_path") != manifest["path"]):
        raise ValueError("Training preprocessing canonical source/protocol differs")
    if preprocessing.get("stages") != STAGES or preprocessing.get("weather_order") != ["t_mean", "t_max", "t_min", "p_sum"]:
        raise ValueError("Training wheat stage or weather order differs")
    if preprocessing.get("coordinate_order") != ["lon", "lat"] or preprocessing.get("feature_order") != FEATURE_NAMES:
        raise ValueError("Training coordinate or canonical daily feature order differs")
    try:
        fit_date = pd.Timestamp(preprocessing["fit_max_date"])
        temperature = float(preprocessing["residual_temperature"])
        prior = np.asarray(preprocessing["rate_prior"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Training preprocessing metadata are incomplete") from error
    if pd.isna(fit_date) or fit_date > pd.Timestamp("2015-12-31"):
        raise ValueError("Training preprocessing dates must end in historical 2015")
    if not np.isfinite(temperature) or temperature <= 0 or prior.shape != (4,) or not np.isfinite(prior).all() or (prior <= 0).any():
        raise ValueError("Training residual temperature and rate priors must be positive")
    if preprocessing.get("strategy") not in {"process_residual", "baseline"}:
        raise ValueError("Unknown no-NDVI AGC strategy")
    return _thresholds(preprocessing["process_thresholds"]), temperature, prior


def _supervision(targets, indices, population, horizon, lengths):
    keys = {"event_kind", "event_lower", "event_upper", "event_limit"}
    if not isinstance(targets, dict) or not keys.issubset(targets):
        raise ValueError("Supervision requires explicit canonical event metadata")
    result = {}
    for key in keys:
        values = np.asarray(targets[key])
        if values.shape != (population, 4) or values.dtype.kind not in "iu":
            raise ValueError("Event metadata must be integer arrays [N,4]")
        result[key] = torch.from_numpy(values[indices].astype(np.int64, copy=True))
    kind, lower, upper, limit = (result[key].numpy() for key in ("event_kind", "event_lower", "event_upper", "event_limit"))
    interval, censor = kind == 1, kind == 2
    if ((kind < 0) | (kind > 2)).any():
        raise ValueError("Event kinds are missing=0, interval=1, explicit right censor=2")
    if (interval & ((lower < 0) | (upper <= lower) | (upper > limit) | (limit > lengths[:, None]))).any():
        raise ValueError("Observed events must obey 0 <= lower < upper <= limit <= valid inputs")
    if (censor & ((limit < 0) | (limit > lengths[:, None]))).any():
        raise ValueError("Explicit censor limits must lie within valid inputs")
    weights = np.asarray(targets.get("cycle_weight", np.ones(population)))
    if weights.shape != (population,) or not np.isfinite(weights[indices]).all() or (weights[indices] < 0).any():
        raise ValueError("Cycle weights must be finite nonnegative [N]")
    observed = np.asarray(targets.get("observed_state", np.full((population, horizon+1), -1, dtype=np.int64)))
    if observed.shape != (population, horizon+1) or observed.dtype.kind not in "iu" or ((observed[indices] < -1) | (observed[indices] > 4)).any():
        raise ValueError("Observed states must be integer [N,H+1] and values -1 or 0..4")
    result["cycle_weight"] = torch.from_numpy(weights[indices].astype(np.float32, copy=True))
    result["observed_state"] = torch.from_numpy(observed[indices].astype(np.int64, copy=True))
    return result


def input_batch(weather, progress, increments, cycles, indices, preprocessing, targets=None):
    """Return CPU canonical tensors; optional targets are full-population arrays.

    ``cycles`` uses only SOWING_DATE, lat, lon and optional input_length for
    predictor construction. No station identity, END_DATE or stage date enters
    predictors. Missing labels remain absent evidence. No inference placeholder
    labels are emitted when targets=None.
    """
    theta, temperature, prior = _validate_preprocessing(preprocessing)
    w, p, inc, records, indices, sowing, xy, valid, lengths = _select_inputs(
        weather, progress, increments, cycles, indices)
    horizon = w.shape[1]
    dates = sowing.to_numpy(dtype="datetime64[D]")[:, None] + np.arange(horizon).astype("timedelta64[D]")[None]
    doy = pd.DatetimeIndex(dates.ravel()).dayofyear.to_numpy().reshape(dates.shape)
    phase = 2*np.pi*(doy-1)/365.25
    declination = .4093*np.sin(2*np.pi*(doy-81)/365)
    daylight = np.arccos(np.clip(-np.tan(np.radians(xy[:, 1:2]))*np.tan(declination), -1, 1))/np.pi
    calendar = np.stack([np.sin(phase), np.cos(phase), daylight,
                         np.broadcast_to(np.arange(1, horizon+1)/365.25, doy.shape)], axis=-1)
    features = [calendar, _transform(w, preprocessing["weather"], 4), np.isfinite(w).astype(np.float32)]
    if preprocessing["strategy"] == "process_residual":
        features.append(_transform(np.stack([p, inc], axis=-1), preprocessing["process"], 2))
    known = np.where(valid[:, :, None], np.concatenate(features, axis=-1), 0).astype(np.float32)
    b = dict(static=torch.from_numpy(_transform(xy, preprocessing["coordinates"], 2)),
             known_future=torch.from_numpy(known), future_valid=torch.from_numpy(valid.copy()),
             rate_prior=torch.from_numpy(np.broadcast_to(prior, (len(indices), 4)).copy()),
             tpv_cumulative=torch.from_numpy(p.copy()), tpv_increment=torch.from_numpy(inc.copy()),
             process_thresholds=torch.from_numpy(np.broadcast_to(theta, (len(indices), 4)).copy()),
             residual_temperature=torch.full((len(indices),), temperature, dtype=torch.float64))
    if targets is not None:
        b.update(_supervision(targets, indices, len(weather), horizon, lengths))
    return b


def make_model(settings=None):
    """Initialize fresh canonical residual weights, or a native baseline arm."""
    canonical = activate_source()
    if settings is None:
        options = {}
    elif is_dataclass(settings) and not isinstance(settings, type):
        options = asdict(settings)
        options["strategy"] = "baseline" if options.get("future_dim") == 12 else "process_residual"
    else:
        options = dict(settings)
    strategy = options.pop("strategy", "process_residual")
    if strategy not in {"process_residual", "baseline"} or options.pop("use_ndvi", False) is not False:
        raise ValueError("Only no-NDVI residual and baseline AGC arms are supported")
    expected = 14 if strategy == "process_residual" else 12
    if options.pop("static_dim", 2) != 2 or options.pop("future_dim", expected) != expected:
        raise ValueError("The declared AGC arm has incompatible feature dimensions")
    if options.pop("device", "cpu") != "cpu":
        raise ValueError("The canonical float64 TPV residual adapter requires CPU")
    if set(options) - {"d_model", "n_heads", "layers", "dropout", "encoder"}:
        raise ValueError("Unknown canonical AGC model setting")
    config = canonical.AblationConfig(static_dim=2, future_dim=expected, **options)
    return canonical.ProcessResidualPhenformer(config) if strategy == "process_residual" else canonical.SeasonalPhenformer(config)


def cdf_from_members(members, raw_progress, thresholds, temperature, *, strength=1.,
                     temperature_scale=1., temperature_mode='absolute'):
    members=np.asarray(members,np.float64);raw=np.asarray(raw_progress,np.float64)
    theta=np.asarray(thresholds,np.float64)
    if (members.ndim!=3 or members.shape[1:]!=raw.shape or theta.shape!=(4,)
            or not np.isfinite(members).all() or not np.isfinite(raw).all()
            or np.any(members<0) or np.any(raw<0) or np.any(np.diff(members,axis=2)<-1e-10)
            or np.any(np.diff(raw,axis=1)<0) or not np.isfinite(theta).all()
            or theta[0]<=0 or np.any(np.diff(theta)<=0)):
        raise ValueError('Finite monotone member/raw progress and four ordered thresholds required')
    if not np.isfinite(strength) or strength<0 or strength>1:
        raise ValueError('Residual shrinkage must lie within0–1')
    if not np.isfinite(temperature*temperature_scale) or temperature<=0 or temperature_scale<=0:
        raise ValueError('Positive finite calibration temperature required')
    tau=np.full(4,temperature*temperature_scale)
    if temperature_mode=='relative_threshold':tau*=theta/theta[0]
    elif temperature_mode!='absolute':raise ValueError('Unknown dispersion calibration mode')
    total=np.zeros((len(raw),4,raw.shape[1]),np.float64)
    for progress in members:
        corrected=raw+strength*(progress-raw)
        ratio=log_expit((theta[None,:,None]-corrected[:,None,:])/tau[None,:,None])-log_expit(theta/tau)[None,:,None]
        total+=np.clip(-np.expm1(ratio),0.,1.)
    return total/len(members)


