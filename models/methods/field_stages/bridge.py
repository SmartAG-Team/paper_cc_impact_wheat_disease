"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/field_stage_extension_20261010')
import numpy as np
import pandas as pd
from scipy.special import expit, logit
EVENTS = (37, 39, 65)
GRID = np.arange(1, 200, dtype=float) / 200

def interpolate(earlier, later, fraction):
    """Daily interpolation only; no future predicted event date enters."""
    a, b = (np.asarray(earlier, float), np.asarray(later, float))
    out = expit((1 - fraction) * logit(np.clip(a, 1e-07, 1 - 1e-07)) + fraction * logit(np.clip(b, 1e-07, 1 - 1e-07)))
    return np.minimum(a, np.maximum(b, out))

def extend_cdf(parent, fractions):
    parent = np.asarray(parent)
    fractions = np.asarray(fractions, float)
    if parent.ndim != 3 or parent.shape[1] != 4 or (not parent.shape[2]):
        raise ValueError('Expected field by four parent stages by daily prefix')
    if not np.isfinite(parent).all() or np.any((parent < 0) | (parent > 1)) or np.any(np.diff(parent, axis=-1) < 0) or np.any(np.diff(parent, axis=1) > 0):
        raise ValueError('Parent probabilities must be ordered in stage and time')
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or np.any((fractions <= 0) | (fractions >= 1)) or (fractions[0] >= fractions[1]):
        raise ValueError('Three interior fractions with fraction37 < fraction39 required')
    p = [parent[:, j] for j in range(4)]
    return np.stack([p[0], p[1], interpolate(p[1], p[2], fractions[0]), interpolate(p[1], p[2], fractions[1]), p[2], interpolate(p[2], p[3], fractions[2]), p[3]], axis=1)

def median_indices(cdf):
    reached = np.asarray(cdf) >= 0.5
    return np.where(reached.any(axis=-1), reached.argmax(axis=-1), np.nan)

def ordered_fit(cost):
    cost = np.asarray(cost, float)
    if cost.ndim != 2 or not np.isfinite(cost).all() or (not cost.shape[0]) or (cost.shape[1] < cost.shape[0]):
        raise ValueError('Finite stage by grid costs with enough ordered candidates required')
    stages, points = cost.shape
    forward, backward = (np.full_like(cost, np.inf), np.full_like(cost, np.inf))
    forward[0], backward[-1] = (cost[0], cost[-1])
    for s in range(1, stages):
        forward[s, 1:] = cost[s, 1:] + np.minimum.accumulate(forward[s - 1])[:-1]
    for s in range(stages - 2, -1, -1):
        backward[s, :-1] = cost[s, :-1] + np.minimum.accumulate(backward[s + 1, ::-1])[::-1][1:]
    profile = forward + backward - cost
    minimum = forward[-1].min()
    centres = [np.median(np.flatnonzero(np.isclose(p, minimum, rtol=0, atol=1e-12))) for p in profile]
    secondary = (np.arange(points) - centres[0]) ** 2
    paths = [(k,) for k in range(points)]
    for s in range(1, stages):
        next_secondary = np.full(points, np.inf)
        next_paths = [None] * points
        for k in range(s, points):
            eligible = np.flatnonzero(np.isclose(forward[s - 1, :k], forward[s, k] - cost[s, k], rtol=0, atol=1e-12))
            j = min(eligible, key=lambda j: (secondary[j], paths[j]))
            next_secondary[k] = secondary[j] + (k - centres[s]) ** 2
            next_paths[k] = paths[j] + (k,)
        secondary, paths = (next_secondary, next_paths)
    winners = np.flatnonzero(np.isclose(forward[-1], minimum, rtol=0, atol=1e-12))
    selected = min(winners, key=lambda j: (secondary[j], paths[j]))
    return (np.asarray(paths[selected]), profile)

def interval_loss(prediction, lower, upper, horizon):
    p, lo, hi, end = np.broadcast_arrays(*[np.asarray(x, float) for x in (prediction, lower, upper, horizon)])
    if np.isinf(p).any() or np.isinf(lo).any() or np.isinf(hi).any() or (not np.isfinite(end).all()) or np.any(end < 0) or np.any(np.isnan(lo) & np.isnan(hi)) or np.any(lo >= hi):
        raise ValueError('Finite horizons and consistent event bounds required')
    early = np.where(np.isfinite(lo), np.maximum(lo + 1 - p, 0), 0)
    late = np.where(np.isfinite(hi), np.maximum(p - hi, 0), 0)
    missing_loss = np.where(np.isfinite(hi), np.maximum(end + 1 - hi, 0), 0)
    return np.where(np.isnan(p), missing_loss, early + late)

def calendar_offsets(registry):
    candidates = pd.date_range('2001-01-01', periods=365)
    result = []
    for row in registry.itertuples(index=False):
        sowing = pd.Timestamp(row.sowing_date)
        dates = pd.DatetimeIndex([pd.Timestamp(year=int(row.season_year), month=d.month, day=d.day) for d in candidates])
        result.append((dates - sowing).days.to_numpy())
    return np.asarray(result, float)
