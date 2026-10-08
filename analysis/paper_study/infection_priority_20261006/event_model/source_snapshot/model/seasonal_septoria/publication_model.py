"""Reproducible runtime for declared structural canopy-score ensembles.

The output targets the recorded INFECT percentage proxy. Component spread
describes structural alternatives, not a calibrated prediction interval.
"""

import json
from pathlib import Path

import numpy as np

from .structural import predict_canopy

DEFAULT_MODEL = Path(__file__).with_name('publication_model.json')


def load_publication_model(model=None):
    if model is None:
        model = DEFAULT_MODEL
    if isinstance(model, (str, Path)):
        model = json.loads(Path(model).read_text())
    if not isinstance(model, dict) or model.get('schema_version') != 1:
        raise ValueError('A schema-1 structural canopy ensemble is required.')
    if model.get('observation') != 'infection_percent_unspecified_basis':
        raise ValueError('The model measurement definition must be declared as the source INFECT proxy.')
    components = model.get('components', [])
    weights = np.asarray([component.get('weight', np.nan) for component in components], float)
    if (not len(weights) or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('Finite nonnegative component weights must sum to one.')
    if any(not isinstance(component.get('fitted'), dict) for component in components):
        raise ValueError('Each component requires its own frozen fitted assumptions.')
    return model


def predict_publication_model(data, accumulation, thresholds, model=None, *, return_daily=False):
    """Predict with frozen component operators without reading disease values.

    `data.maximum_temperature` is required by duration-proxy components. The
    supplied development index and thresholds identify the declared crop
    scenario. Metric labels keep predictions in the intended score domain.
    """
    model = load_publication_model(model)
    if 'metric' not in data.targets or not data.targets.metric.eq(model['observation']).all():
        raise ValueError('Prediction measurement labels must match the frozen INFECT proxy definition.')
    weights = np.asarray([component['weight'] for component in model['components']], float)
    component_values, daily = [], None
    for weight, component in zip(weights, model['components']):
        forecast, trajectory = predict_canopy(data, accumulation, thresholds, component['fitted'])
        component_values.append(forecast)
        if return_daily:
            if daily is None:
                daily = np.zeros_like(trajectory.expressed)
            daily += weight*100*trajectory.expressed
    values = np.stack(component_values)
    return dict(predicted_percent=np.sum(weights[:, None]*values, axis=0),
        component_percent=values, component_weights=weights,
        structural_range_percent=np.stack([values.min(axis=0), values.max(axis=0)]),
        daily_percent=daily, model_id=model.get('model_id'),
        observation=model['observation'], structural_range_is_prediction_interval=False)
