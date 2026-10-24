"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/canopy_senescence_20261010')
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
FEATURES = {'calendar_logistic': 'elapsed_days', 'thermal_logistic': 'degree_days'}
BOUNDS = {'calendar_logistic': ((1.0, 1.0), (150.0, 100.0)), 'thermal_logistic': ((1.0, 1.0), (3000.0, 1000.0))}

def weights(frame):
    if frame.empty or frame[['year', 'plot_id']].isna().any().any():
        raise ValueError('Nonempty year/plot identities required')
    plots = frame.groupby('year').plot_id.transform('nunique').to_numpy(float)
    visits = frame.groupby(['year', 'plot_id']).plot_id.transform('size').to_numpy(float)
    return 1 / (frame.year.nunique() * plots * visits)

def predict(frame, parameters):
    if parameters['model'] == 'training_mean':
        return np.full(len(frame), parameters['mean'])
    feature = FEATURES[parameters['model']]
    return expit((parameters['midpoint'] - frame[feature].to_numpy(float)) / parameters['scale'])

def fit(frame, kind):
    y = frame.green_fraction.to_numpy(float)
    w = weights(frame)
    if not np.isfinite(y).all() or not np.all((y >= 0) & (y <= 1)):
        raise ValueError('Finite green fractions required for fitting')
    common = dict(model=kind, training_years=sorted((int(v) for v in frame.year.unique())), training_records=len(frame), training_plots=frame.plot_id.nunique())
    if kind == 'training_mean':
        return dict(common, mean=float(w @ y))
    x = frame[FEATURES[kind]].to_numpy(float)
    if not np.isfinite(x).all() or np.ptp(x) <= 0:
        raise ValueError('Variable finite predictor required')
    lower, upper = (np.asarray(BOUNDS[kind][0]), np.asarray(BOUNDS[kind][1]))
    scales = (10.0, 30.0) if kind == 'calendar_logistic' else (150.0, 450.0)
    attempts = []
    for fraction in (0.25, 0.5, 0.75):
        for scale in scales:
            initial = np.clip([x.min() + fraction * np.ptp(x), scale], lower + 1e-06, upper - 1e-06)

            def residual(theta):
                midpoint, width = np.exp(theta)
                return np.sqrt(w) * (expit((midpoint - x) / width) - y)
            solution = least_squares(residual, np.log(initial), bounds=(np.log(lower), np.log(upper)), ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=2000)
            if solution.success:
                attempts.append(solution)
    if not attempts:
        raise ValueError('No converged canopy fit')
    best = min(attempts, key=lambda s: s.cost)
    midpoint, scale = np.exp(best.x)
    return dict(common, midpoint=float(midpoint), scale=float(scale), training_mse=float(2 * best.cost), training_feature_min=float(x.min()), training_feature_max=float(x.max()), converged_starts=len(attempts), active_bounds=best.active_mask.tolist())

def scores(frame, prediction):
    y = frame.green_fraction.to_numpy(float)
    p = np.asarray(prediction, float)
    if len(p) != len(y) or not np.isfinite(p).all() or (not np.isfinite(y).all()):
        raise ValueError('Finite predictions on the same observations required')
    w = weights(frame)
    error = p - y
    integrals = []
    for (year, plot), part in frame.assign(prediction=p).groupby(['year', 'plot_id']):
        part = part.sort_values('date')
        if len(part) < 2:
            continue
        t = (pd.to_datetime(part.date) - pd.to_datetime(part.date.iloc[0])).dt.days.to_numpy(float)
        if np.any(np.diff(t) <= 0):
            raise ValueError('Distinct ordered observation dates required')
        gap = np.diff(t)

        def integrate(v):
            return np.sum(gap * (v[:-1] + v[1:]) / 2)
        difference = integrate(part.prediction.to_numpy()) - integrate(part.green_fraction.to_numpy())
        integrals.append(dict(year=year, plot_id=plot, absolute=abs(difference), signed=difference))
    windows = pd.DataFrame(integrals)
    mean_integral = float(windows.groupby('year').absolute.mean().mean()) if len(windows) else np.nan
    return dict(records=len(frame), plots=frame.plot_id.nunique(), years=frame.year.nunique(), mse=float(w @ error ** 2), rmse=float(np.sqrt(w @ error ** 2)), mae=float(w @ np.abs(error)), bias=float(w @ error), integral_plots=len(windows), integral_mae_green_fraction_days=mean_integral)
