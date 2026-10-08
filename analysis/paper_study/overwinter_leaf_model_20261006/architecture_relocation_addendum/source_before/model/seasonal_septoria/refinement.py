"""Calibration refinements for source-compatible seasonal severity prediction.

The power link applies only to infection percentages with an unspecified
denominator. Named necrotic-area and pycnidial-area measurements retain their
direct observation operators. The link describes measurement, not transmission.
"""

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .core import Parameters, simulate_season
from .calibrate import predict


def map_observations(fraction, metric, power=1.):
    """Map normalized disease fractions while preserving zero and one."""
    values, labels = np.broadcast_arrays(np.asarray(fraction, float),
                                         np.asarray(metric, str))
    if (not np.isfinite(power) or power <= 0
            or not np.isfinite(values).all()
            or np.any(values < -1e-12) or np.any(values > 1+1e-12)):
        raise ValueError('Finite disease fractions in [0,1] and positive power required.')
    values = np.clip(values, 0., 1.)
    return np.where(labels == 'infection_percent_unspecified_basis',
                    values**power, values)


def calibration_weights(target, final_weight=0.):
    """Mix assessment and final-assessment losses, retaining cluster weights."""
    if target.empty or not np.isfinite(final_weight) or not 0 <= final_weight <= 1:
        raise ValueError('Nonempty targets and final weight in [0,1] required.')
    frame = target.reset_index(drop=True).copy()
    fields = frame.groupby('coordinate_year').field_id.transform('nunique')
    series = frame.groupby('field_id').endpoint_series.transform('nunique')
    dates = frame.groupby(['field_id', 'endpoint_series']).field_id.transform('size')
    all_dates = (1/fields/series/dates).to_numpy(float)
    last = frame.sort_values('date', kind='stable').groupby(
        ['field_id', 'endpoint_series']).tail(1).index.to_numpy()
    final = np.zeros(len(frame))
    final[last] = (1/fields/series).to_numpy(float)[last]
    return (1-final_weight)*all_dates+final_weight*final


def fit_refinement(data, target_indices, *, latent_free=False, power_free=False,
                   final_weight=0., latent_days=30.):
    """Fit only complete selected field-seasons with deterministic starts.

    Secondary pressure is fixed to zero because the original point estimate
    was effectively zero and its separation from external pressure was weakly
    identified. All parameter bounds and loss variants precede evaluation.
    """
    indices = np.asarray(target_indices, int)
    if (indices.ndim != 1 or indices.size == 0
            or np.unique(indices).size != indices.size
            or np.any(indices < 0) or np.any(indices >= len(data.targets))):
        raise ValueError('Unique nonempty calibration target indices required.')
    target = data.targets.iloc[indices]
    selected_fields = np.unique(target.field_index.to_numpy(int))
    membership = np.flatnonzero(data.targets.field_index.isin(selected_fields))
    if not np.array_equal(np.sort(indices), membership):
        raise ValueError('Calibration cannot split assessments within a field-season.')
    if not np.isfinite(latent_days) or latent_days <= 0:
        raise ValueError('Positive reference latent duration required.')
    weight = calibration_weights(target, final_weight)
    observed = target.value.to_numpy(float)
    if not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 100)):
        raise ValueError('Finite percentage observations required for calibration.')
    if power_free and not target.metric.eq('infection_percent_unspecified_basis').any():
        raise ValueError('Power fitting requires unspecified-basis infection percentages.')
    field_map = {field: position for position, field in enumerate(selected_fields)}
    field = target.field_index.map(field_map).to_numpy(int)
    day, leaf = target.day_index.to_numpy(int), target.leaf_index.to_numpy(int)
    forcing = tuple(x[selected_fields] for x in (data.temperature, data.humidity,
        data.rain, data.host_active, data.host_renewal))
    is_pycnidia = target.observation_operator.eq('pycnidia').to_numpy()
    metric = target.metric.to_numpy(str)
    lower, upper = [-8.], [-1.]
    if latent_free:
        lower.append(10./30.); upper.append(60./30.)
    if power_free:
        lower.append(np.log(.25)); upper.append(np.log(4.))

    def unpack(x):
        position = 1
        duration = float(x[position]*30) if latent_free else float(latent_days)
        position += int(latent_free)
        power = float(np.exp(x[position])) if power_free else 1.
        return Parameters(alpha=float(10**x[0]), beta=0., latent_days=duration), power

    def residual(x):
        parameters, power = unpack(x)
        trajectory = simulate_season(*forcing, parameters)
        fraction = np.where(is_pycnidia, trajectory.pycnidia[field, day, leaf],
                            trajectory.damage[field, day, leaf])
        forecast = 100*map_observations(fraction, metric, power)
        return np.sqrt(weight)*(forecast-observed)

    records = []
    for log_alpha, duration, power in [(-2., 20., 1.), (-1.3, 30., 2.), (-2.5, 45., .5)]:
        initial = [log_alpha]
        if latent_free:
            initial.append(duration/30.)
        if power_free:
            initial.append(np.log(power))
        solution = least_squares(residual, initial, bounds=(lower, upper),
            max_nfev=150, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        parameters, fitted_power = unpack(solution.x)
        records.append(dict(start=initial, parameters=asdict(parameters),
            observation_power=fitted_power, weighted_sse=float(np.square(solution.fun).sum()),
            success=bool(solution.success), evaluations=int(solution.nfev),
            termination=str(solution.message)))
    successful = [i for i, row in enumerate(records) if row['success']
                  and np.isfinite(row['weighted_sse'])]
    if not successful:
        raise RuntimeError('No calibration start converged.')
    winner = min(successful, key=lambda i: records[i]['weighted_sse'])
    best = records[winner]
    return dict(parameters=best['parameters'], observation_power=best['observation_power'],
        training_rmse=float(np.sqrt(best['weighted_sse']/weight.sum())),
        target_indices=indices.tolist(), field_indices=selected_fields.tolist(),
        calibration_field_ids=sorted(target.field_id.unique().tolist()),
        latent_free=bool(latent_free), power_free=bool(power_free),
        final_weight=float(final_weight), target_count=len(indices),
        field_count=len(selected_fields), winning_start=winner, multistart_records=records,
        calibration_only=True, initial_disease_assimilated=False,
        observation_mapping_scope='infection_percent_unspecified_basis')


def predict_refinement(data, fitted):
    """Predict mapped assessments and retain raw biological trajectories."""
    raw, trajectory = predict(data, Parameters(**fitted['parameters']))
    values = 100*map_observations(raw/100., data.targets.metric.to_numpy(str),
                                 fitted['observation_power'])
    return values, trajectory


def blend_severity(process_percent, empirical_percent, *, empirical_weight,
                   host_active):
    """Blend bounded severity estimates, enforcing available host cohorts.

    This assessment predictor does not define a disease-onset event. Timing
    diagnostics remain attached to the separately identified process model.
    """
    process, empirical = np.asarray(process_percent, float), np.asarray(empirical_percent, float)
    active = np.asarray(host_active)
    if (process.shape != empirical.shape or active.shape != process.shape
            or not np.isfinite(process).all() or not np.isfinite(empirical).all()
            or np.any((process < 0) | (process > 100))
            or np.any((empirical < 0) | (empirical > 100))
            or np.any((active != 0) & (active != 1))
            or not np.isfinite(empirical_weight) or not 0 <= empirical_weight <= 1):
        raise ValueError('Compatible bounded severity estimates, active host and blend weight required.')
    return np.where(active, (1-empirical_weight)*process+empirical_weight*empirical, 0.)
