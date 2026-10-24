"""Source-native random intercept/slope covariance from honest forecast errors."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit


def _basis(frame, clock):
    category = np.eye(7)[frame.leaf_index.to_numpy(int)]
    return np.column_stack([category, category*(np.asarray(clock, float)-2.)[:, None]])


def _link(value, link):
    value = np.asarray(value, float)/100
    return logit(np.clip(value, .005, .995)) if link == 'logit' else value


def _posterior(z, residual, weights, covariance, noise):
    precision = np.linalg.inv(covariance) + (z.T*weights) @ z/noise
    variance = np.linalg.inv(precision)
    mean = variance @ (z.T @ (weights*residual))/noise
    return mean, (variance+variance.T)/2


def fit_coefficients(frame, baseline, clock, weights, link='logit', iterations=30):
    if link not in ['linear', 'logit']:
        raise ValueError('Registered random-coefficient link required')
    frame = frame.reset_index(drop=True)
    weights = np.asarray(weights, float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Positive finite scientific weights required')
    z = _basis(frame, clock)
    residual = _link(frame.observed_percent, link)-_link(baseline, link)
    fitted = dict(link=link, sources={}, supported={}, iterations=iterations,
        training_ids=frame.global_target_index.astype(int).tolist(), training_weight=weights,
        training_baseline=np.asarray(baseline, float))
    for source, group in frame.groupby('source', sort=False):
        groups = [f.index.to_numpy() for _, f in group.groupby('field_id', sort=False)]
        field_mass = np.array([weights[ix].sum() for ix in groups])
        field_mass /= field_mass.sum()
        # Nonstationary covariance is source-specific; native-code positions
        # have no asserted cross-source anatomical equivalence.
        initial = np.diag(np.r_[np.full(7, 1. if link == 'logit' else .2),
            np.full(7, 2. if link == 'logit' else .5)]**2)
        covariance = initial.copy()
        noise = (.7 if link == 'logit' else .1)**2
        trace = []
        for iteration in range(iterations):
            second_moment = np.zeros((14, 14))
            error_mass, observation_mass = 0., 0.
            for mass, ix in zip(field_mass, groups):
                w = min(4, frame.date.iloc[ix].nunique()) * weights[ix]/weights[ix].sum()
                mean, variance = _posterior(z[ix], residual[ix], w, covariance, noise)
                second_moment += mass*(variance+np.outer(mean, mean))
                error = (residual[ix]-z[ix]@mean)**2 + np.sum((z[ix]@variance)*z[ix], axis=1)
                error_mass += mass*float(w@error)
                observation_mass += mass*w.sum()
            # Shrinkage makes this a regularized EM-like empirical-Bayes fit,
            # rather than an exact unpenalized likelihood maximizer.
            covariance = .8*second_moment + .2*np.diag(np.diag(second_moment)) + .01*initial
            covariance = (covariance+covariance.T)/2
            noise = max((.5 if link == 'logit' else .04)**2, error_mass/observation_mass)
            trace.append(dict(iteration=iteration+1, noise_sd=float(np.sqrt(noise)),
                covariance_minimum_eigenvalue=float(np.linalg.eigvalsh(covariance).min())))
        fitted['sources'][str(source)] = dict(covariance=covariance, noise=noise, trace=trace)
        fitted['supported'][str(source)] = sorted(group.leaf_index.unique().astype(int).tolist())
    return fitted


def predict_coefficients(fitted, frame, baseline, clock):
    frame = frame.reset_index(drop=True)
    baseline = np.asarray(baseline, float)
    result = baseline.copy()
    linked_base = _link(baseline, fitted['link'])
    residual = _link(frame.observed_percent, fitted['link'])-linked_base
    z = _basis(frame, clock)
    dates = pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
    nodes, quadrature = np.polynomial.hermite.hermgauss(9)
    quadrature = quadrature/np.sqrt(np.pi)
    for _, group in frame.groupby('field_id', sort=False):
        ix = group.index.to_numpy()
        source = str(group.source.iloc[0])
        if source not in fitted['sources']:
            continue
        parameters = fitted['sources'][source]
        known = np.isin(frame.leaf_index.iloc[ix].to_numpy(), fitted['supported'][source])
        for now in np.unique(dates[ix]):
            query = ix[(dates[ix] == now) & known]
            past = ix[(dates[ix] < now) & known]
            if not len(past):
                continue
            days, counts = np.unique(dates[past], return_counts=True)
            count = dict(zip(days, counts))
            w = np.array([1./count[d] for d in dates[past]]) * min(4, len(days))/len(days)
            mean, variance = _posterior(z[past], residual[past], w,
                parameters['covariance'], parameters['noise'])
            location = linked_base[query]+z[query]@mean
            if fitted['link'] == 'logit':
                sd = np.sqrt(np.maximum(0, np.sum((z[query]@variance)*z[query], axis=1)))
                result[query] = 100*(expit(location[:, None]+np.sqrt(2)*sd[:, None]*nodes)@quadrature)
            else:
                result[query] = 100*np.clip(location, 0, 1)
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite random-coefficient prediction')
    return result
