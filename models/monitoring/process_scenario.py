"""Causal four-weather process replay with an explicit resistance scenario."""
import numpy as np
import pandas as pd

from .process_establishment import simulate_establishment
from .process_establishment_data import _forcing, INTRODUCTION
from .variety_resistance import resistance_parameters


def simulate_queries(weather, fields, queries, development_prior, parameters, susceptibility_multiplier=1.):
    if fields.field_id.duplicated().any():
        raise ValueError('Unique field sowing identities required')
    sowing = pd.to_datetime(fields.set_index('field_id').estimated_sowing_date, errors='coerce')
    query_dates = pd.to_datetime(queries.date, errors='coerce')
    query_sowing = queries.field_id.map(sowing)
    if queries.empty or query_dates.isna().any() or query_sowing.isna().any() or not query_dates.gt(query_sowing).all():
        raise ValueError('Every query must follow its field estimated sowing date')
    forcing = _forcing(weather, fields, queries, development_prior)
    parameters = resistance_parameters(parameters, susceptibility_multiplier)
    arguments = [forcing[k] for k in ['temperature', 'exposure', 'rain', 'active', 'renewal', 'area', 'valid']]
    damage = np.zeros((len(queries), len(INTRODUCTION), 8))
    mass_error = 0.; smallest = 1.
    for k, threshold in enumerate(INTRODUCTION):
        states = simulate_establishment(*arguments, parameters, threshold)
        mass_error = max(mass_error, float(np.max(np.abs(states.sum(-1)-1.))))
        smallest = min(smallest, float(states.min()))
        damage[:, k] = states[forcing['field_index'], forcing['day_index'], :, -3:].sum(-1)
    if mass_error > 1e-10 or smallest < -1e-12 or not np.isfinite(damage).all():
        raise RuntimeError('Invalid climate-scenario process states')
    return dict(damage=damage, largest_mass_error=mass_error, smallest_state=smallest,
        area=forcing['area'], active=forcing['active'], provenance=forcing['provenance'])
