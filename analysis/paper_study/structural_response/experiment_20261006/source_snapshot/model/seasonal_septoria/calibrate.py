"""Calibration-only parameter fitting of seasonal disease simulations."""

from dataclasses import asdict

import numpy as np
from scipy.optimize import least_squares

from .core import Parameters, simulate_season


def predict(data, parameters):
    """Return source-compatible predictions without using target severity."""
    trajectory = simulate_season(data.temperature,data.humidity,data.rain,
                                  data.host_active,data.host_renewal,parameters)
    target = data.targets
    idx = (target.field_index.to_numpy(int),target.day_index.to_numpy(int),target.leaf_index.to_numpy(int))
    values = np.where(target.observation_operator.eq('pycnidia'),trajectory.pycnidia[idx],trajectory.damage[idx])*100
    return values,trajectory


def fit(data, target_indices, latent_days=20., beta_free=True):
    """Fit complete calibration field seasons with three deterministic starts."""
    indices = np.asarray(target_indices,int)
    if (indices.ndim != 1 or indices.size == 0 or np.unique(indices).size != indices.size
            or np.any(indices < 0) or np.any(indices >= len(data.targets))):
        raise ValueError('Unique nonempty calibration target indices required.')
    target = data.targets.iloc[indices]
    selected_fields = np.unique(target.field_index)
    full_membership = np.flatnonzero(data.targets.field_index.isin(selected_fields))
    if not np.array_equal(np.sort(indices),full_membership):
        raise ValueError('Calibration cannot split assessments within a field-season.')
    mapped = {value:i for i,value in enumerate(selected_fields)}
    field = target.field_index.map(mapped).to_numpy(int)
    day,leaf = target.day_index.to_numpy(int),target.leaf_index.to_numpy(int)
    observed,weight = target.value.to_numpy(float),target.weight.to_numpy(float)
    if (not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 100))
            or not np.isfinite(weight).all() or np.any(weight <= 0)):
        raise ValueError('Finite percentage observations and positive weights required.')
    weather = (data.temperature[selected_fields],data.humidity[selected_fields],data.rain[selected_fields],
               data.host_active[selected_fields],data.host_renewal[selected_fields])
    def parameters_at(x):
        return Parameters(alpha=float(10**x[0]),beta=float(x[1]) if beta_free else 0.,latent_days=latent_days)
    def residual(x):
        trajectory = simulate_season(*weather,parameters_at(x))
        damage = trajectory.damage[field,day,leaf]
        pycnidia = trajectory.pycnidia[field,day,leaf]
        forecast = np.where(target.observation_operator.eq('pycnidia'),pycnidia,damage)*100
        return np.sqrt(weight)*(forecast-observed)
    starts = [(-4.,.3),(-3.,1.),(-2.3,3.)]
    records,solutions = [],[]
    for log_alpha,beta in starts:
        x0 = [log_alpha,beta] if beta_free else [log_alpha]
        bounds = ([-8.,0.],[-1.,10.]) if beta_free else ([-8.],[-1.])
        solution = least_squares(residual,x0,bounds=bounds,max_nfev=150,
                                 ftol=1e-8,xtol=1e-8,gtol=1e-8)
        records.append(dict(start=x0,parameters=asdict(parameters_at(solution.x)),
            weighted_sse=float(np.square(solution.fun).sum()),success=bool(solution.success),
            evaluations=int(solution.nfev),termination=str(solution.message)))
        solutions.append(solution)
    winner = int(np.argmin([x['weighted_sse'] for x in records]))
    return dict(parameters=records[winner]['parameters'],
        training_rmse=float(np.sqrt(records[winner]['weighted_sse']/weight.sum())),
        target_indices=indices.tolist(),field_indices=selected_fields.astype(int).tolist(),
        target_count=len(indices),field_count=len(selected_fields),
        winning_start=winner,multistart_records=records,
        calibration_only=True,initial_disease_assimilated=False)
