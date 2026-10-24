"""Frozen canonical AGC fitting preprocessing and native CDF conversion."""
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from analysis.final_models.agc_math import (
    HORIZON, STAGES, PROTOCOL, FEATURE_NAMES, _select_inputs, _thresholds,
    source_manifest, input_batch, make_model,
)
from .objective import ablation_loss

def activate_source():
    return SimpleNamespace(ablation_loss=ablation_loss)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _fit(values):
    mean, scale = values.mean(0), values.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    return dict(imputation=np.median(values, axis=0).tolist(), mean=mean.tolist(),
                scale=scale.tolist(), observed_count=np.isfinite(values).sum(0).tolist())


def fit_preprocessing(weather, progress, increments, cycles, train_indices, thresholds):
    """Fit no-NDVI scaling and process priors using historical training only.

    Required fit metadata are partition='train' and integer harvest_year<=2015.
    Optional input_length describes weather availability, never observed-stage
    dates. Actual historical fitting dates are derived from sowing and inputs.
    """
    w, p, inc, records, indices, sowing, xy, valid, lengths = _select_inputs(
        weather, progress, increments, cycles, train_indices, unique=True)
    theta = _thresholds(thresholds)
    if not {"partition", "harvest_year"}.issubset(records.columns):
        raise ValueError("Historical training partition and harvest year are required")
    years = pd.to_numeric(records["harvest_year"], errors="coerce").to_numpy(dtype=float)
    last_dates = sowing + pd.to_timedelta(lengths-1, unit="D")
    if (not records["partition"].eq("train").all() or not np.isfinite(years).all()
            or (years != np.floor(years)).any() or (years > 2015).any()
            or (last_dates > pd.Timestamp("2015-12-31")).any()):
        raise ValueError("Preprocessing fitting is restricted to historical training through 2015")
    positive = inc[valid & (inc > 0)]
    if not len(positive):
        raise ValueError("Historical training requires at least one positive TPV increment")
    gaps = np.diff(np.r_[0., theta])
    floor = .1*np.quantile(positive, .1)/gaps
    manifest = source_manifest()
    result = dict(protocol=PROTOCOL, strategy="process_residual", stages=STAGES,
                  feature_order=FEATURE_NAMES, coordinate_order=["lon", "lat"],
                  weather_order=["t_mean", "t_max", "t_min", "p_sum"],
                  source_path=manifest["path"], source_sha256=manifest["sha256"],
                  fit_cycle_count=len(indices), fit_indices_sha256=_identity(indices.tolist()),
                  fit_max_date=last_dates.max().strftime("%Y-%m-%d"),
                  weather=_fit(w[valid]), coordinates=_fit(xy),
                  process=_fit(np.stack([p[valid], inc[valid]], axis=1)),
                  process_thresholds=theta.tolist(),
                  rate_prior=np.maximum(inc[valid].mean()/gaps, floor).tolist(),
                  residual_temperature=max(float(np.median(positive)), 1e-3))
    if "PEP_ID" in records:
        result["fit_station_ids"] = sorted(records["PEP_ID"].astype(int).unique().tolist())
    return result


def model_loss(outputs, batch):
    """Use the unchanged canonical truncated-logistic composite event loss."""
    return activate_source().ablation_loss(outputs, batch)


def model_outputs_cdf(output):
    """Map native outputs to end-of-day stage CDFs with shape [B,4,H].

    For residual outputs, F(c)=1-S(c)/S(0) conditions a logistic threshold on
    positive process progress. Its zero boundary is exact, and no mass accrues
    when frozen TPV increments are zero. The baseline pure-birth arm retains
    its native time-driven dynamics and is a separate comparison.
    """
    if output.get("objective") == "process_residual_truncated_logistic_event_nll":
        progress = output["corrected_progress"].double()
        thresholds = output["process_thresholds"].double()
        temperature = output["residual_temperature"].double()
        if progress.ndim != 2 or thresholds.shape != (len(progress), 4) or temperature.shape != (len(progress),):
            raise ValueError("Residual model outputs have incompatible process shapes")
        if (not torch.isfinite(progress).all() or (progress < 0).any() or (progress.diff(dim=1) < 0).any()
                or not torch.isfinite(thresholds).all() or (thresholds[:, 0] <= 0).any()
                or (thresholds.diff(dim=1) <= 0).any() or not torch.isfinite(temperature).all() or (temperature <= 0).any()):
            raise ValueError("Residual outputs require monotone progress and positive ordered thresholds/temperature")
        theta = thresholds[:, None]
        tau = temperature[:, None, None]
        log_survival_ratio = F.logsigmoid((theta-progress[:, :, None])/tau)-F.logsigmoid(theta/tau)
        return (-torch.expm1(log_survival_ratio)).clamp(0, 1).transpose(1, 2)
    if "stage_cdf" in output:
        cdf = output["stage_cdf"]
        if cdf.ndim != 3 or cdf.shape[-1] != 4 or cdf.shape[1] < 2:
            raise ValueError("Native baseline CDF must have shape [B,H+1,4]")
        return cdf[:, 1:].transpose(1, 2)
    raise ValueError("Unsupported canonical AGC model output")

