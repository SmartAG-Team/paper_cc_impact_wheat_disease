"""Bounded mean-severity calibration from honest historical OOB predictions."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression


def fit_calibration(prediction, target, weights, counts, kind):
    p, y, w, count = [np.asarray(a) for a in [prediction, target, weights, counts]]
    if (p.ndim != 1 or any(a.shape != p.shape for a in [y, w, count]) or not len(p)
            or kind not in ['logit', 'isotonic']):
        raise ValueError('Aligned calibration vectors and declared method required')
    p, y, w = p.astype(float), y.astype(float), w.astype(float)
    if (not np.isfinite(np.r_[y, w, count]).all() or (w <= 0).any()
            or ((y < 0) | (y > 100)).any() or (count < 0).any() or (count != np.rint(count)).any()):
        raise ValueError('Finite bounded targets, positive scientific weights and integer OOB counts required')
    covered = count > 0
    if not np.isnan(p[~covered]).all():
        raise ValueError('Every zero-count prediction must stay NaN; fitted fallback is forbidden')
    if not np.isfinite(p[covered]).all():
        raise ValueError('Every positive-count OOB prediction must be finite')
    if not covered.any():
        return dict(kind='identity', requested_kind=kind, fit_status='no_honest_calibration_support',
            calibration_rows=np.array([], int), normalized_weight=np.array([], float),
            covered_count=0, missing_count=len(p), converged=None, objective=None, initial_objective=None)
    if ((p[covered] < 0) | (p[covered] > 100)).any():
        raise ValueError('Covered bounded OOB forecasts required')
    p, y, mass = p[covered], y[covered], w[covered]/w[covered].sum()
    result = dict(kind=kind, calibration_rows=np.flatnonzero(covered), normalized_weight=mass,
        covered_count=int(covered.sum()), missing_count=int((~covered).sum()),
        loss='Original hierarchy-weighted severity MSE; no assumed binomial denominator')
    if kind == 'isotonic':
        model = IsotonicRegression(y_min=0., y_max=100., out_of_bounds='clip')
        model.fit(p, y, sample_weight=mass)
        values = model.predict(p)
        result.update(x=model.X_thresholds_, y=model.y_thresholds_, converged=True,
            objective=float(np.sum(mass*((values-y)/100.)**2)),
            initial_objective=float(np.sum(mass*((p-y)/100.)**2)))
    else:
        x = logit(np.clip(p/100., .005, .995))
        trace = []

        def objective(theta):
            fitted = expit(theta[0]+np.exp(theta[1])*x)
            value = float(np.sum(mass*(fitted-y/100.)**2)+1e-4*np.sum(theta**2))
            trace.append(value)
            return value

        initial = objective(np.zeros(2))
        optimized = minimize(objective, np.zeros(2), method='L-BFGS-B',
            bounds=[(-5., 5.), (np.log(.1), np.log(5.))], options=dict(maxiter=100, ftol=1e-12, gtol=1e-9))
        if not np.isfinite(optimized.fun) or not np.isfinite(optimized.x).all():
            raise RuntimeError('Nonfinite bounded-severity calibration')
        result.update(parameters=optimized.x, endpoint_clip=.005, ridge=1e-4,
            converged=bool(optimized.success), message=str(optimized.message), iterations=optimized.nit,
            objective=float(optimized.fun), initial_objective=initial, trace=trace)
    return result


def predict_calibration(fitted, prediction):
    p = np.asarray(prediction, float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 100)).any():
        raise ValueError('Finite percentage forecasts required; query outcomes are not an input')
    if fitted['kind'] == 'identity':
        return p.copy()
    if fitted['kind'] == 'isotonic':
        return np.interp(p, fitted['x'], fitted['y'])
    theta = fitted['parameters']; eps = fitted['endpoint_clip']
    return 100*expit(theta[0]+np.exp(theta[1])*logit(np.clip(p/100., eps, 1-eps)))
