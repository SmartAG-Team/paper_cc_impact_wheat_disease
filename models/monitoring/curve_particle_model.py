"""Historical sigmoid curve particles with updates from earlier assessment days."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, softmax


def _fit_curve(clock, target, weight, prior):
    weight = np.asarray(weight, float)
    weight = weight / weight.mean()
    prior = np.asarray(prior, float)

    def residual(parameters):
        errors = (expit(parameters[0] + parameters[1] * (clock - 2.)) - target)
        return np.r_[np.sqrt(weight) * errors, .05 * (parameters - prior) / [6., 10.]]

    result = least_squares(residual, np.clip(prior, [-25., 0.], [25., 50.]),
        bounds=([-25., 0.], [25., 50.]), max_nfev=100)
    return result.x


def fit_curve_bank(clock, frame, weights):
    """Fit parameters using only the outer historical training fields."""
    frame = frame.reset_index(drop=True)
    clock = np.asarray(clock, float)
    weights = np.asarray(weights, float)
    y = frame.observed_percent.to_numpy(float) / 100
    if not (len(clock) == len(weights) == len(frame)):
        raise ValueError('Clock, weights and training rows must align')
    if not np.isfinite(clock).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Finite clocks and positive scientific weights required')
    fitted = dict(banks={}, phase_shifts=[-.2, -.1, 0., .1, .2],
        slope_multipliers=[.5, 1., 2.], sigma_pp=12., clock_center=2.,
        pooled_fraction=float(np.average(y, weights=weights)),
        training_ids=frame.global_target_index.astype(int).tolist(), training_weight=weights)
    codes = frame.leaf_index.to_numpy(int)
    if not np.isin(codes, np.arange(7)).all():
        raise ValueError('Only retained native codes 0 through 6 are supported')
    for source, source_ix in frame.groupby('source', sort=False).groups.items():
        source_ix = np.asarray(source_ix)
        supported = np.unique(codes[source_ix])
        pooled = np.zeros((7, 2))
        for code in supported:
            ix = source_ix[codes[source_ix] == code]
            average = np.clip(np.average(y[ix], weights=weights[ix]), .005, .995)
            intercept = np.log(average / (1 - average)) - 8 * (np.average(clock[ix], weights=weights[ix]) - 2.)
            pooled[code] = _fit_curve(clock[ix], y[ix], weights[ix], [intercept, 8.])
        parameters, latitudes, longitudes, mass, field_ids, observed = [], [], [], [], [], []
        for field, group in frame.iloc[source_ix].groupby('field_id', sort=False):
            ix = group.index.to_numpy()
            field_parameters = pooled.copy()
            mask = np.zeros(7, bool)
            for code in np.unique(codes[ix]):
                ci = ix[codes[ix] == code]
                field_parameters[code] = _fit_curve(clock[ci], y[ci], weights[ci], pooled[code])
                mask[code] = True
            # Missing endpoints are latent curve proposals, not additional observations.
            shift = np.average((field_parameters - pooled)[mask, 0])
            ratio = np.clip(np.median((field_parameters[mask, 1] + 1.) / (pooled[mask, 1] + 1.)), .25, 4.)
            for code in supported:
                if not mask[code]:
                    field_parameters[code] = [np.clip(pooled[code, 0] + shift, -25, 25),
                        np.clip(pooled[code, 1] * ratio, 0, 50)]
            parameters.append(field_parameters)
            latitudes.append(float(group.latitude.iloc[0]))
            longitudes.append(float(group.longitude.iloc[0]))
            mass.append(float(weights[ix].sum()))
            field_ids.append(str(field))
            observed.append(mask)
        fitted['banks'][str(source)] = dict(parameters=np.asarray(parameters),
            latitude=np.asarray(latitudes), longitude=np.asarray(longitudes),
            weight=np.asarray(mass), field_ids=field_ids, observed_codes=np.asarray(observed),
            supported_codes=supported.tolist(), pooled_parameters=pooled)
    return fitted


def _distance(latitude, longitude, bank):
    lat, lon = np.radians(latitude), np.radians(longitude)
    bl, bo = np.radians(bank['latitude']), np.radians(bank['longitude'])
    value = np.sin((bl - lat) / 2) ** 2 + np.cos(lat) * np.cos(bl) * np.sin((bo - lon) / 2) ** 2
    return 6371. * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def predict_curve_bank(fitted, clock, frame, baseline):
    """Update curve probabilities with strict earlier-day observations only."""
    frame = frame.reset_index(drop=True)
    clock, baseline = np.asarray(clock, float), np.asarray(baseline, float)
    result = dict(same=baseline.copy(), field=baseline.copy(),
        free=np.full(len(frame), 100 * fitted['pooled_fraction']), ess=np.zeros(len(frame)))
    dates = pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
    codes = frame.leaf_index.to_numpy(int)
    y = frame.observed_percent.to_numpy(float)
    for _, group in frame.groupby('field_id', sort=False):
        ix = group.index.to_numpy()
        bank = fitted['banks'].get(str(group.source.iloc[0]))
        if bank is None:
            continue
        phase, slope = np.meshgrid(fitted['phase_shifts'], fitted['slope_multipliers'], indexing='ij')
        repeats = phase.size
        parameters = np.repeat(bank['parameters'], repeats, axis=0)
        phase = np.tile(phase.ravel(), len(bank['weight']))
        slope = np.tile(slope.ravel(), len(bank['weight']))
        mass = np.asarray(bank['weight']) * np.exp(-.5 * (_distance(
            group.latitude.iloc[0], group.longitude.iloc[0], bank) / 300.) ** 2)
        mass = np.maximum(mass, np.finfo(float).tiny)
        prior = np.repeat(mass / mass.sum() / repeats, repeats)
        supported = np.isin(codes[ix], bank['supported_codes'])
        curves = 100 * expit(parameters[:, codes[ix], 0].T +
            parameters[:, codes[ix], 1].T * slope[None, :] *
            (clock[ix, None] - fitted['clock_center'] - phase[None, :]))
        result['free'][ix[supported]] = (curves @ prior)[supported]
        for now in np.unique(dates[ix]):
            query = np.flatnonzero((dates[ix] == now) & supported)
            past = np.flatnonzero((dates[ix] < now) & supported)
            if not len(past):
                result['ess'][ix[query]] = 1. / np.sum(prior ** 2)
                continue
            day, counts = np.unique(dates[ix[past]], return_counts=True)
            count = dict(zip(day, counts))
            weight = np.array([1. / count[d] for d in dates[ix[past]]])
            density = -.5 * ((curves[past] - y[ix[past], None]) / fitted['sigma_pp']) ** 2
            likelihood = min(4, len(day)) * np.average(density, axis=0, weights=weight)
            posterior = .99 * softmax(np.log(prior) + likelihood) + .01 * prior
            result['ess'][ix[query]] = 1. / np.sum(posterior ** 2)
            for q in query:
                same = past[codes[ix[past]] == codes[ix[q]]]
                if len(same):
                    latest = same[dates[ix[same]] == dates[ix[same]].max()]
                    anchor = np.mean(y[ix[latest]])
                    increment = (curves[q] - curves[latest].mean(axis=0)) @ posterior
                    result['same'][ix[q]] = np.clip(anchor + increment, 0, 100)
                    result['field'][ix[q]] = result['same'][ix[q]]
                else:
                    result['field'][ix[q]] = curves[q] @ posterior
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError('Nonfinite curve-particle prediction')
    return result
