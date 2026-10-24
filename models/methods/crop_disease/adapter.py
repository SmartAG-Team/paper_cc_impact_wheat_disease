"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_disease_coupling_20261010')
import numpy as np
from scipy.special import logit

def development_coordinate(cdf, anchors, *, quantile=0.5):
    """Invert each day's ordered stage CDFs without accessing future days.

    Coordinates retain the supplied anchor units. They are not measured
    thermal accumulation. Clipping is confined to the interpolation, while
    original probabilities determine exact stage crossings.
    """
    p, a = (np.asarray(cdf, float), np.asarray(anchors, float))
    epsilon = 1e-07
    if p.ndim != 3 or not p.size or a.ndim != 1 or (len(a) < 2) or (p.shape[1] != len(a)) or (not np.isfinite(p).all()) or np.any((p < 0) | (p > 1)) or np.any(np.diff(p, axis=1) > 0) or np.any(np.diff(p, axis=2) < 0) or (not np.isfinite(a).all()) or np.any(a <= 0) or np.any(np.diff(a) <= 0) or (np.ndim(quantile) != 0) or (not np.isfinite(quantile)) or (not epsilon <= quantile <= 1 - epsilon):
        raise ValueError('Ordered finite CDFs, positive increasing anchors and an interior quantile required')
    daily = np.moveaxis(p, 1, -1)
    crossed = (daily >= quantile).sum(axis=-1)
    out = np.zeros(crossed.shape)
    out[crossed == len(a)] = a[-1]
    middle = (crossed > 0) & (crossed < len(a))
    left = crossed[middle] - 1
    selected = daily[middle]
    rows = np.arange(len(selected))
    lo = logit(np.clip(selected[rows, left], epsilon, 1 - epsilon))
    hi = logit(np.clip(selected[rows, left + 1], epsilon, 1 - epsilon))
    weight = np.divide(lo - logit(quantile), lo - hi, out=np.zeros_like(lo), where=lo > hi)
    value = a[left] + np.clip(weight, 0.0, 1.0) * (a[left + 1] - a[left])
    out[middle] = np.clip(value, a[left], np.nextafter(a[left + 1], -np.inf))
    return np.maximum.accumulate(out, axis=1)

def extend_completed(coordinate, *, horizon, terminal_anchor):
    """Carry a completed terminal coordinate forward; never infer new stages."""
    c = np.asarray(coordinate, float)
    if c.ndim != 2 or not c.size or (not np.isfinite(c).all()) or np.any(c < 0) or np.any(np.diff(c, axis=1) < 0) or (np.ndim(terminal_anchor) != 0) or (not np.isfinite(terminal_anchor)) or (terminal_anchor <= 0) or np.any(c > terminal_anchor) or isinstance(horizon, (bool, np.bool_)) or (not isinstance(horizon, (int, np.integer))) or (horizon < 1):
        raise ValueError('Valid monotone coordinates, terminal anchor and positive integer horizon required')
    if horizon <= c.shape[1]:
        return c[:, :horizon].copy()
    if np.any(c[:, -1] != terminal_anchor):
        raise ValueError('Terminal attainment is required before carry-forward')
    return np.pad(c, ((0, 0), (0, horizon - c.shape[1])), mode='edge')
