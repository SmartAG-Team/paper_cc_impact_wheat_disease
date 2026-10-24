"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_canopy_coupling_20261011')
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, logit
from analysis.methods.crop_canopy.inputs import weather_prefix
KINDS = ('training_mean', 'calendar', 'median', 'marginal')
FRACTIONS = np.arange(1, 200, dtype=float) / 200
BOUNDS = {'calendar': ((100.0, 1.0), (330.0, 100.0)), 'median': ((1.0, 1.0), (3000.0, 1000.0)), 'marginal': ((1.0, 1.0), (3000.0, 1000.0))}

def weights(frame):
    if frame.empty or frame[['year', 'plot_id']].isna().any().any():
        raise ValueError('Nonempty year/plot identities required')
    plots = frame.groupby('year').plot_id.transform('nunique').to_numpy(float)
    visits = frame.groupby(['year', 'plot_id']).plot_id.transform('size').to_numpy(float)
    return 1 / (frame.year.nunique() * plots * visits)

def interpolate(a, b, fraction):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    if not 0 <= fraction <= 1 or np.any(b > a + 1e-07):
        raise ValueError('Stage probabilities must be ordered')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0) or np.any(a > 1) or np.any(b > 1):
        raise ValueError('Finite probabilities required')
    curve = expit((1 - fraction) * logit(np.clip(a, 1e-07, 1 - 1e-07)) + fraction * logit(np.clip(b, 1e-07, 1 - 1e-07)))
    return np.maximum(b, np.minimum(a, curve))

def fit_bridge(labels, parent_cdf, registry):
    if labels.empty or labels.plot_id.duplicated().any():
        raise ValueError('Unique historical source-stage labels required')
    if not labels.year.le(2017).all():
        raise ValueError('Bridge fitting is restricted to historical years')
    loss = np.zeros(len(FRACTIONS))
    for year, part in labels.groupby('year', sort=True):
        ids = np.flatnonzero(registry.year.to_numpy() == year)
        if len(ids) != 1:
            raise ValueError('One parent cycle per year and policy required')
        i = ids[0]
        days = parent_cdf.shape[-1]
        observed = (pd.to_datetime(part.heading_date) - pd.Timestamp(registry.iloc[i].sowing_date)).dt.days.to_numpy()
        if np.any(observed < 0) or np.any(observed >= days):
            raise ValueError('Heading outside restricted scoring horizon')
        target = (np.arange(days)[None, :] >= observed[:, None]).mean(axis=0)
        for j, fraction in enumerate(FRACTIONS):
            curve = interpolate(parent_cdf[i, 2], parent_cdf[i, 3], fraction)
            loss[j] += np.sum(curve ** 2 - 2 * curve * target + target) / labels.year.nunique()
    best = int(np.argmin(loss))
    fit = dict(fraction=float(FRACTIONS[best]), crps=float(loss[best]), training_years=sorted((int(y) for y in labels.year.unique())), training_labels=len(labels), boundary=best in (0, len(FRACTIONS) - 1))
    return (fit, pd.DataFrame(dict(fraction=FRACTIONS, crps=loss)))

def thermal_basis(cdf, temperature, assessment, kind):
    """Sum positive temperature strictly after the event; retain all tail mass."""
    if assessment < 0 or assessment >= len(cdf) or assessment >= len(temperature):
        raise ValueError('Assessment outside daily prefix')
    c = np.asarray(cdf, float)[:assessment + 1]
    temp = np.asarray(temperature, float)[:assessment + 1]
    if not np.isfinite(c).all() or not np.isfinite(temp).all() or np.any(np.diff(c) < -1e-07):
        raise ValueError('Finite monotone daily prefixes required')
    if np.any(c < 0) or np.any(c > 1):
        raise ValueError('CDF probabilities outside unit interval')
    c = np.maximum.accumulate(c)
    cumulative = np.cumsum(np.maximum(0.0, temp))
    if kind == 'median':
        hits = np.flatnonzero(c >= 0.5)
        return (np.array([cumulative[-1] - cumulative[hits[0]] if len(hits) else 0.0]), np.ones(1))
    if kind != 'marginal':
        raise ValueError('Unknown thermal coupling')
    exposure = np.r_[cumulative[-1] - cumulative, 0.0]
    mass = np.r_[np.diff(np.r_[0.0, c]), 1 - c[-1]]
    return (exposure, mass)

@dataclass
class Basis:
    kind: str
    exposure: np.ndarray
    mass: np.ndarray
    inverse: np.ndarray

def canopy_basis(frame, registry, parent_cdf, weather, fraction, kind):
    keys = frame[['year', 'date']].drop_duplicates().sort_values(['year', 'date']).reset_index(drop=True)
    lookup = {(row.year, row.date): i for i, row in enumerate(keys.itertuples())}
    inverse = np.array([lookup[row.year, row.date] for row in frame.itertuples()])
    exposures, masses = ([], [])
    for row in keys.itertuples():
        indexes = np.flatnonzero(registry.year.to_numpy() == row.year)
        if len(indexes) != 1:
            raise ValueError('One parent cycle per year and policy required')
        i = indexes[0]
        start = registry.iloc[i].sowing_date
        day = (pd.Timestamp(row.date) - pd.Timestamp(start)).days
        if day < 0 or day >= parent_cdf.shape[-1]:
            raise ValueError('Assessment outside fixed sowing horizon')
        if kind in ('calendar', 'training_mean'):
            x, m = (np.array([float(day)]), np.ones(1))
        else:
            curve = interpolate(parent_cdf[i, 2, :day + 1], parent_cdf[i, 3, :day + 1], fraction)
            temperature = weather_prefix(weather, start, day + 1)[:, 0]
            x, m = thermal_basis(curve, temperature, day, kind)
        exposures.append(x)
        masses.append(m)
    length = max(map(len, exposures))
    pad = lambda array: np.pad(array, (0, length - len(array)))
    return Basis(kind, np.stack([pad(x) for x in exposures]), np.stack([pad(m) for m in masses]), inverse)

def collapse(frame, basis):
    w = weights(frame)
    y = frame.green_fraction.to_numpy(float)
    n = len(basis.exposure)
    grouped_weight = np.bincount(basis.inverse, weights=w, minlength=n)
    if np.any(grouped_weight <= 0):
        raise ValueError('Every predictor group must have positive weight')
    target = np.bincount(basis.inverse, weights=w * y, minlength=n) / grouped_weight
    constant = float(w @ (y - target[basis.inverse]) ** 2)
    return (target, grouped_weight, constant)

def unique_prediction(basis, midpoint, scale):
    return np.sum(basis.mass * expit((midpoint - basis.exposure) / scale), axis=1)

def prediction_jacobian(basis, theta):
    midpoint, scale = np.exp(theta)
    z = (midpoint - basis.exposure) / scale
    p = expit(z)
    derivative = basis.mass * p * (1 - p)
    return np.column_stack([np.sum(derivative * (midpoint / scale), axis=1), np.sum(derivative * -z, axis=1)])

def predict_canopy(basis, parameters):
    if parameters['model'] == 'training_mean':
        return np.full(len(basis.inverse), parameters['mean'])
    return unique_prediction(basis, parameters['midpoint'], parameters['scale'])[basis.inverse]

def fit_canopy(frame, basis, kind):
    y = frame.green_fraction.to_numpy(float)
    w = weights(frame)
    if not np.isfinite(y).all() or not np.all((y >= 0) & (y <= 1)):
        raise ValueError('Finite unit-interval observations required')
    common = dict(model=kind, training_years=sorted((int(v) for v in frame.year.unique())), training_records=len(frame), training_plots=frame.plot_id.nunique(), unique_predictors=len(basis.exposure))
    if kind == 'training_mean':
        mean = float(w @ y)
        return dict(common, mean=mean, training_mse=float(w @ (mean - y) ** 2))
    target, gw, constant = collapse(frame, basis)
    low, high = map(np.asarray, BOUNDS[kind])
    expected = np.sum(basis.exposure * basis.mass, axis=1)
    starts = (10.0, 30.0) if kind == 'calendar' else (150.0, 450.0)
    attempts = []
    for fraction in (0.25, 0.5, 0.75):
        for width in starts:
            initial = np.clip([expected.min() + fraction * np.ptp(expected), width], low + 1e-06, high - 1e-06)

            def residual(theta):
                return np.sqrt(gw) * (unique_prediction(basis, *np.exp(theta)) - target)

            def jacobian(theta):
                return np.sqrt(gw)[:, None] * prediction_jacobian(basis, theta)
            fit = least_squares(residual, np.log(initial), bounds=(np.log(low), np.log(high)), method='dogbox', jac=jacobian, x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=2000)
            if fit.success:
                attempts.append(fit)
    if not attempts:
        raise ValueError('No converged canopy fit')
    best = min(attempts, key=lambda x: x.cost)
    midpoint, scale = np.exp(best.x)
    return dict(common, midpoint=float(midpoint), scale=float(scale), training_mse=float(2 * best.cost + constant), within_group_variance=constant, converged_starts=len(attempts), active_bounds=best.active_mask.tolist())
