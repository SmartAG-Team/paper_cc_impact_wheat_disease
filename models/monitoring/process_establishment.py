"""Shared physical disease equations with an assumed introduction mixture.

Nominal observation operators are statistical mixtures, not recovered leaf
anatomy, spore counts, cultivar resistance or calibrated healthy-area loss.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize

from model.seasonal_septoria.overwinter import _advance_day, OverwinterParameters


@njit(cache=True)
def _establishment_batch(t, exposure, rain, active, renewal, area, valid, coefficients,
                         ready_fraction, stages, threshold, pressure):
    fields, days, leaves = active.shape
    state = np.zeros((fields, days+1, leaves, stages+4)); state[..., 0] = 1.
    for field in range(fields):
        current = state[field, 0].copy()
        residue = np.array([1-ready_fraction, ready_fraction])
        accumulated = 0.
        local_coefficients = coefficients.copy()
        for day in range(days):
            if valid[field, day]:
                # Only primary hazard is gated. Residue maturation/decay and
                # secondary transfer retain the actual current-day forcing.
                local_coefficients[0] = coefficients[0] if accumulated >= threshold else 0.
                current, residue, _ = _advance_day(current, residue, t[field, day], exposure[field, day],
                    rain[field, day], active[field, day], renewal[field, day], area[field, day],
                    pressure, local_coefficients, stages, 4)
                accumulated += exposure[field, day]
            state[field, day+1] = current
    return state


def simulate_establishment(t, exposure, rain, active, renewal, area, valid, parameters, threshold, pressure=.1):
    t, exposure, rain, renewal, area = [np.asarray(a, float) for a in [t, exposure, rain, renewal, area]]
    active, valid = np.asarray(active), np.asarray(valid)
    if (t.ndim != 2 or exposure.shape != t.shape or rain.shape != t.shape or valid.shape != t.shape
            or active.shape[:2] != t.shape or active.ndim != 3 or active.shape[2] == 0
            or renewal.shape != active.shape or area.shape != active.shape
            or not all(np.isfinite(a).all() for a in [t, exposure, rain, renewal, area])
            or ((exposure < 0)|(exposure > 1)).any() or (rain < 0).any()
            or ((active != 0)&(active != 1)).any() or ((valid != 0)&(valid != 1)).any()
            or ((renewal < 0)|(renewal > 1)).any() or ((area < 0)|(area > 1)).any()
            or not isinstance(parameters, OverwinterParameters) or not np.isfinite([threshold, pressure]).all()
            or threshold < 0 or pressure < 0):
        raise ValueError('Valid daily process forcing, host, parameters and introduction threshold required')
    return _establishment_batch(t, exposure, rain, active.astype(bool), renewal, area, valid.astype(bool),
        parameters.vector(), parameters.initial_ready_fraction, parameters.latent_stages, float(threshold), float(pressure))


def _simplex_fit(x, y, mass, center, ridge):
    hessian = (x.T*mass)@x+ridge*np.eye(x.shape[1])
    rhs = x.T@(mass*y)+ridge*center
    result = minimize(lambda w: float(w@hessian@w-2*w@rhs), center,
        jac=lambda w: 2*(hessian@w-rhs), method='SLSQP', bounds=[(0., 1.)]*x.shape[1],
        constraints=[dict(type='eq', fun=lambda w: w.sum()-1., jac=lambda w: np.ones_like(w))],
        options=dict(maxiter=80, ftol=1e-11))
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError('Nonfinite convex observation mixture')
    weight = np.maximum(result.x, 0.); weight /= weight.sum()
    return weight, bool(result.success)


def _category(frame):
    return (frame.source.astype(str)+':'+frame.leaf_index.astype(str)).to_numpy()


def _operators(states, y, mass, category, arrival):
    effective = np.einsum('nkl,k->nl', states, arrival[:3])
    pooled, success = _simplex_fit(effective, y, mass, np.full(8, 1/8), 1e-4)
    native, diagnostics = {}, [success]
    for key in np.unique(category):
        if key == 'BASF:0':
            native[key] = np.eye(8)[0]
        else:
            use = category == key
            native[key], ok = _simplex_fit(effective[use], y[use], mass[use]/mass[use].sum(), pooled, .003)
            diagnostics.append(ok)
    return pooled, native, diagnostics


def _nominal_weights(category, pooled, native):
    return np.stack([np.eye(8)[0] if key == 'BASF:0' else native.get(key, pooled) for key in category])


def fit_observation(states, frame, weights, mixture):
    states, mass = np.asarray(states, float), np.asarray(weights, float)
    y = frame.observed_percent.to_numpy(float)/100.
    if (states.shape != (len(frame), 3, 8) or mass.shape != (len(frame),)
            or not np.isfinite(states).all() or ((states < -1e-10)|(states > 1+1e-10)).any()
            or not np.isfinite(np.r_[y, mass]).all() or ((y < 0)|(y > 1)).any() or (mass <= 0).any()):
        raise ValueError('Three bounded introduction states, targets and positive scientific mass required')
    states = np.clip(states, 0, 1); mass = mass/mass.sum(); category = _category(frame)
    arrival = np.array([1., 0., 0., 0.]); success = []
    for _ in range(5 if mixture else 1):
        pooled, native, ok = _operators(states, y, mass, category, arrival); success.extend(ok)
        if mixture:
            operators = _nominal_weights(category, pooled, native)
            x = np.column_stack([np.einsum('nkl,nl->nk', states, operators), np.zeros(len(y))])
            arrival, ok = _simplex_fit(x, y, mass, np.array([1., 0., 0., 0.]), .001)
            success.append(ok)
    if mixture:
        pooled, native, ok = _operators(states, y, mass, category, arrival); success.extend(ok)
    fitted = dict(arrival_weight=arrival, pooled_operator=pooled, native_operators=native,
        mixture=bool(mixture), optimizer_success=all(success), optimizer_calls=len(success),
        native_anatomy_recovered=False, flag_mapping='BASF:0 only; all other endpoint operators statistical')
    predicted = predict_observation(fitted, states, frame.drop(columns='observed_percent'))/100.
    error = float(np.sum(mass*(predicted-y)**2))
    regularizer = 1e-4*np.sum((pooled-1/8)**2)+.001*np.sum((arrival-np.array([1., 0., 0., 0.]))**2)
    for key, value in native.items():
        if key != 'BASF:0':
            regularizer += .003*mass[category == key].sum()*np.sum((value-pooled)**2)
    fitted.update(training_mse=error, regularizer=float(regularizer), objective=error+float(regularizer))
    return fitted


def predict_observation(fitted, states, queries):
    if 'observed_percent' in queries:
        raise ValueError('Query outcomes must be absent')
    states = np.asarray(states, float)
    if states.shape != (len(queries), 3, 8) or not np.isfinite(states).all():
        raise ValueError('Aligned process states required')
    operator = _nominal_weights(_category(queries), fitted['pooled_operator'], fitted['native_operators'])
    effective = np.einsum('nkl,k->nl', states, fitted['arrival_weight'][:3])
    return 100*np.clip(np.sum(effective*operator, axis=1), 0, 1)
