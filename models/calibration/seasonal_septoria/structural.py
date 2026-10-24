"""Observation mapping and calibration for the parameterized canopy runtime."""

from dataclasses import asdict
import numpy as np
from scipy.optimize import least_squares
from model.seasonal_septoria.structural import CanopyParameters, canopy_host, simulate_canopy
from .refinement import calibration_weights


def _target_coordinates(data, target):
    names = ('field_index', 'day_index', 'leaf_index')
    raw = [target[name].to_numpy(float) for name in names]
    if any(not np.isfinite(value).all() or np.any(value != np.floor(value)) for value in raw):
        raise ValueError('Assessment indices must be finite integers.')
    field, day, leaf = [value.astype(int) for value in raw]
    if (np.any((field < 0) | (field >= len(data.temperature)))
            or np.any((day < 0) | (day > data.temperature.shape[1]))
            or np.any((leaf < 0) | (leaf >= 7))):
        raise ValueError('Assessment indices are outside field, forcing-day or ordinal-leaf bounds.')
    if not data.metadata.empty and {'field_index', 'forcing_days'}.issubset(data.metadata):
        lengths = data.metadata.set_index('field_index').forcing_days.to_dict()
        if any(int(f) not in lengths or int(d) > lengths[int(f)] for f, d in zip(field, day)):
            raise ValueError('Assessment indices exceed the field-specific forcing window.')
    return field, day, leaf


def predict_canopy(data, accumulation, thresholds, fitted, environmental_response=None):
    idx = _target_coordinates(data, data.targets)
    operator = fitted.get('weather_operator', fitted.get('weather_preprocessing', {}).get('weather_operator', 'daily_or'))
    if environmental_response is None and operator == 'duration_proxy':
        from model.seasonal_septoria.wetness import duration_exposure
        if not hasattr(data, 'maximum_temperature'):
            raise ValueError('Maximum temperature is required for the saved duration operator.')
        rate = fitted.get('rain_rate_mm_hour', fitted.get('weather_preprocessing', {}).get('rain_rate_mm_hour'))
        if rate is None:
            raise ValueError('The saved duration operator requires its calibration rainfall intensity.')
        environmental_response = duration_exposure(data.temperature, data.maximum_temperature,
            data.humidity, data.rain, rate)['exposure']
    elif environmental_response is None and operator != 'daily_or':
        raise ValueError('Unsupported saved weather operator.')
    active, renewal, _, _ = canopy_host(accumulation, data.temperature, thresholds,
                                       **fitted['host_parameters'])
    trajectory = simulate_canopy(data.temperature, data.humidity, data.rain, active,
        renewal, CanopyParameters(**fitted['parameters']), environmental_response=environmental_response)
    return 100*trajectory.expressed[idx], trajectory


def fit_canopy(data, accumulation, thresholds, target_indices, host_parameters,
               amplification_free=True, latent_days=30., final_weight=.5,
               environmental_response=None, initiation_max=.1):
    """Fit selected complete field-seasons without reading withheld scores."""
    raw_indices = np.asarray(target_indices)
    if (raw_indices.ndim != 1 or raw_indices.dtype.kind == 'b'
            or not np.isfinite(raw_indices.astype(float)).all()
            or np.any(raw_indices.astype(float) != np.floor(raw_indices.astype(float)))):
        raise ValueError('Calibration row indices must be finite integers.')
    indices = raw_indices.astype(int)
    if not np.isfinite(initiation_max) or initiation_max <= 1e-6:
        raise ValueError('Effective initiation upper bound must exceed its positive lower bound.')
    if (indices.ndim != 1 or not indices.size or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(data.targets))):
        raise ValueError('Unique nonempty calibration indices required.')
    target = data.targets.iloc[indices]
    _target_coordinates(data, target)
    fields = np.unique(target.field_index.to_numpy(int))
    membership = np.flatnonzero(data.targets.field_index.isin(fields))
    if not np.array_equal(np.sort(indices), membership):
        raise ValueError('Calibration cannot split a field-season.')
    if not target.metric.eq('infection_percent_unspecified_basis').all():
        raise ValueError('Structural INFECT calibration requires source percentage proxies.')
    observed = target.value.to_numpy(float)
    if not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 100)):
        raise ValueError('Finite percentage calibration observations required.')
    mapped = {field:index for index, field in enumerate(fields)}
    field = target.field_index.map(mapped).to_numpy(int)
    day, leaf = target.day_index.to_numpy(int), target.leaf_index.to_numpy(int)
    active, renewal, _, _ = canopy_host(accumulation[fields], data.temperature[fields],
                                       thresholds, **host_parameters)
    forcing = (data.temperature[fields], data.humidity[fields], data.rain[fields], active, renewal)
    weight = calibration_weights(target, final_weight)
    exposure = None if environmental_response is None else np.asarray(environmental_response)[fields]

    def unpack(x):
        return CanopyParameters(initiation=float(10**x[0]),
            amplification=float(10**x[1]) if amplification_free else 0., latent_days=latent_days)

    def residual(x):
        result = simulate_canopy(*forcing, unpack(x), environmental_response=exposure)
        return np.sqrt(weight)*(100*result.expressed[field, day, leaf]-observed)

    lower, upper = ([-6., -5.], [np.log10(initiation_max), np.log10(.3)]) if amplification_free else ([-6.], [np.log10(initiation_max)])
    records = []
    for log_initial, log_expansion in [(-2.5, -1.3), (-1.5, -2.), (-3.5, -1.)]:
        start = [log_initial, log_expansion] if amplification_free else [log_initial]
        start = np.clip(start, np.asarray(lower)+1e-10, np.asarray(upper)-1e-10)
        result = least_squares(residual, start, bounds=(lower, upper), max_nfev=150,
                               ftol=1e-7, xtol=1e-7, gtol=1e-7)
        records.append(dict(start=start.tolist(), parameters=asdict(unpack(result.x)),
            weighted_sse=float(np.square(result.fun).sum()), success=bool(result.success),
            evaluations=int(result.nfev), termination=str(result.message)))
    success = [i for i, row in enumerate(records) if row['success'] and np.isfinite(row['weighted_sse'])]
    if not success:
        raise RuntimeError('No structural calibration start converged.')
    winner = min(success, key=lambda i: records[i]['weighted_sse'])
    return dict(parameters=records[winner]['parameters'], host_parameters=host_parameters,
        amplification_free=bool(amplification_free), final_weight=final_weight,
        target_indices=indices.tolist(), field_indices=fields.tolist(),
        calibration_field_ids=sorted(target.field_id.unique().tolist()),
        training_rmse=float(np.sqrt(records[winner]['weighted_sse']/weight.sum())),
        multistart_records=records, winning_start=winner, calibration_only=True,
        observed_disease_assimilated=False, observation='source_INFECT_percentage_proxy',
        initiation_max=float(initiation_max))
