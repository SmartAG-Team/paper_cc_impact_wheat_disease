"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/fip_heading_adaptation_20261011')
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from analysis.methods.crop_canopy.models import interpolate
DEVELOPMENT_YEARS = (2016, 2017, 2018)
CANDIDATES = ('frozen', 'bridge', 'odds')
KINDS = CANDIDATES + ('calendar_das', 'calendar_doy')
FRACTIONS = np.arange(1, 200, dtype=float) / 200
OFFSETS = np.arange(-100, 101, dtype=float) / 20
FROZEN_FRACTION = 0.175

def calibrate_odds(cdf, offset):
    cdf = np.asarray(cdf, float)
    if not np.isfinite(cdf).all() or np.any(cdf < 0) or np.any(cdf > 1) or (not np.isfinite(offset)):
        raise ValueError('Finite probabilities and offset required')
    return expit(logit(cdf) + offset)

def validate_parent(registry, parent):
    parent = np.asarray(parent, float)
    if parent.ndim != 3 or parent.shape[:2] != (len(registry), 4) or registry.year.duplicated().any() or (not registry.horizon_days.eq(parent.shape[-1]).all()):
        raise ValueError('One ordered four-stage parent cycle per year required')
    if not np.isfinite(parent).all() or np.any(parent < 0) or np.any(parent > 1) or np.any(np.diff(parent, axis=1) > 1e-07) or np.any(np.diff(parent, axis=2) < -1e-07):
        raise ValueError('Finite time- and stage-ordered parent probabilities required')
    return parent

def label_weights(labels):
    if labels.empty or labels.plot_id.isna().any() or labels.plot_id.duplicated().any():
        raise ValueError('Unique nonempty plot labels required')
    if labels.year.isna().any() or not np.isfinite(labels.elapsed_day).all() or (labels.elapsed_day < 0).any() or (not np.equal(labels.elapsed_day, np.floor(labels.elapsed_day)).all()):
        raise ValueError('Finite nonnegative integer event days required')
    return 1 / (labels.year.nunique() * labels.groupby('year').plot_id.transform('size').to_numpy(float))

def per_record_crps(labels, registry, cdf):
    label_weights(labels)
    cdf = np.asarray(cdf, float)
    if cdf.ndim != 2 or cdf.shape[0] != len(registry) or registry.year.duplicated().any() or (not labels.year.isin(registry.year).all()) or (not np.isfinite(cdf).all()) or np.any(cdf < 0) or np.any(cdf > 1) or np.any(np.diff(cdf, axis=1) < -1e-07):
        raise ValueError('Matched finite monotone prediction curves required')
    index = pd.Series(np.arange(len(registry)), index=registry.year)
    predicted = cdf[labels.year.map(index).to_numpy(int)]
    observed = np.arange(cdf.shape[1])[None, :] >= labels.elapsed_day.to_numpy()[:, None]
    return np.sum((predicted - observed) ** 2, axis=1)

def restricted_crps(labels, registry, cdf):
    return float(label_weights(labels) @ per_record_crps(labels, registry, cdf))

def choose_grid(values, losses, reference):
    values, losses = (np.asarray(values, float), np.asarray(losses, float))
    if values.shape != losses.shape or not len(values) or (not np.isfinite(losses).all()):
        raise ValueError('Finite grid losses required')
    tied = np.flatnonzero(losses <= losses.min() + 1e-12)
    return int(tied[np.lexsort((values[tied], np.abs(values[tied] - reference)))[0]])

def predict(fitted, registry, parent):
    parent = validate_parent(registry, parent)
    kind = fitted['model']
    if kind in CANDIDATES:
        fraction = fitted.get('fraction', FROZEN_FRACTION)
        curve = interpolate(parent[:, 2], parent[:, 3], fraction)
        return calibrate_odds(curve, fitted['offset']) if kind == 'odds' else curve
    if kind not in KINDS:
        raise ValueError('Unknown heading model')
    values = np.asarray(fitted['event_offsets'], float)
    weights = np.asarray(fitted['event_weights'], float)
    if not np.isclose(weights.sum(), 1, atol=1e-12) or np.any(weights <= 0):
        raise ValueError('Normalized empirical event weights required')
    rows = []
    for row in registry.itertuples():
        grid = np.arange(parent.shape[-1])
        if kind == 'calendar_doy':
            grid = grid + (pd.Timestamp(row.sowing_date) - pd.Timestamp(int(row.year), 1, 1)).days
        rows.append((grid[:, None] >= values[None, :]) @ weights)
    return np.clip(np.asarray(rows), 0, 1)

def fit(kind, labels, registry, parent):
    weights = label_weights(labels)
    if not labels.year.isin(DEVELOPMENT_YEARS).all():
        raise ValueError('Fitting is restricted to development years')
    parent = validate_parent(registry, parent)
    if kind not in KINDS or not labels.year.isin(registry.year).all():
        raise ValueError('Known model and matched development cycles required')
    result = dict(model=kind, training_years=sorted((int(y) for y in labels.year.unique())), training_labels=len(labels), fit_horizon_days=parent.shape[-1], parameter_fitted=kind != 'frozen')
    if kind in ('calendar_das', 'calendar_doy'):
        values = labels.elapsed_day.to_numpy(int)
        if kind == 'calendar_doy':
            values = np.array([(pd.Timestamp(date) - pd.Timestamp(int(year), 1, 1)).days for year, date in zip(labels.year, labels.heading_date)])
        result.update(event_offsets=values.tolist(), event_weights=weights.tolist())
    elif kind == 'frozen':
        result['fraction'] = FROZEN_FRACTION
    else:
        grid = FRACTIONS if kind == 'bridge' else OFFSETS
        losses = np.zeros(len(grid))
        for year, part in labels.groupby('year', sort=True):
            i = int(np.flatnonzero(registry.year.to_numpy() == year)[0])
            target = (np.arange(parent.shape[-1])[None, :] >= part.elapsed_day.to_numpy()[:, None]).mean(axis=0)
            if kind == 'bridge':
                curves = np.array([interpolate(parent[i, 2], parent[i, 3], value) for value in grid])
            else:
                frozen = interpolate(parent[i, 2], parent[i, 3], FROZEN_FRACTION)
                curves = np.array([calibrate_odds(frozen, value) for value in grid])
            losses += np.sum(curves ** 2 - 2 * curves * target + target, axis=1) / labels.year.nunique()
        best = choose_grid(grid, losses, FROZEN_FRACTION if kind == 'bridge' else 0.0)
        result['fraction' if kind == 'bridge' else 'offset'] = float(grid[best])
        result.update(grid_boundary=best in (0, len(grid) - 1), grid_profile=[{'parameter': float(x), 'crps': float(y)} for x, y in zip(grid, losses)])
    result['training_crps'] = restricted_crps(labels, registry, predict(result, registry, parent))
    return result

def cross_validate(labels, registry, parent):
    if set(labels.year) != set(DEVELOPMENT_YEARS):
        raise ValueError('Exactly the three development years required')
    rows, fitted = ([], [])
    for year in DEVELOPMENT_YEARS:
        training = labels[labels.year.ne(year)]
        held = labels[labels.year.eq(year)]
        for kind in KINDS:
            item = fit(kind, training, registry, parent)
            value = restricted_crps(held, registry, predict(item, registry, parent))
            rows.append(dict(held_year=year, model=kind, observations=len(held), crps=value))
            fitted.append(dict(held_year=year, fit=item))
    frame = pd.DataFrame(rows)
    means = frame.groupby('model').crps.mean()
    best = min((means[k] for k in CANDIDATES))
    selected = next((kind for kind in CANDIDATES if means[kind] <= best + 1e-12))
    selection = dict(selected_model=selected, cv_crps=float(means[selected]), training_years=list(DEVELOPMENT_YEARS), criterion='Leave-one-development-year-out restricted CRPS', evaluation_outcomes_used=False)
    return (selection, frame, fitted)
