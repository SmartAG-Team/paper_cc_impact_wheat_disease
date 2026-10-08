"""Publication runtime preserves source semantics and component assumptions."""

import numpy as np
import pandas as pd
import pytest

from model.seasonal_septoria.field_data import FieldData
from model.seasonal_septoria.structural import predict_canopy
from model.seasonal_septoria.publication_model import predict_publication_model


def example():
    t = np.full((1, 60), 18.)
    active = np.ones((1, 60, 8), bool)
    target = pd.DataFrame(dict(field_index=[0, 0], day_index=[40, 60], leaf_index=[0, 1],
                              metric=['infection_percent_unspecified_basis']*2))
    data = FieldData(t, np.full_like(t, 85.), np.ones_like(t), active,
                    np.zeros(active.shape), pd.DataFrame(), target, pd.DataFrame())
    data.maximum_temperature = t+6.
    accumulation = np.arange(1., 61.)[None, :]*18
    thresholds = {10:10., 31:30., 51:100., 85:2000.}
    host = dict(flag_fraction=.3, leaf_interval=120., expansion_units=100.)
    one = dict(parameters=dict(initiation=.02, amplification=0., latent_days=20.),
               host_parameters=host, weather_operator='daily_or')
    two = dict(parameters=dict(initiation=.1, amplification=0., latent_days=30.),
               host_parameters=host, weather_operator='duration_proxy', rain_rate_mm_hour=.45)
    bundle = dict(schema_version=1, model_id='example', observation='infection_percent_unspecified_basis',
                  components=[dict(weight=.5, fitted=one), dict(weight=.5, fitted=two)])
    return data, accumulation, thresholds, bundle


def test_publication_prediction_is_declared_component_average():
    data, accumulation, thresholds, bundle = example()
    first, _ = predict_canopy(data, accumulation, thresholds, bundle['components'][0]['fitted'])
    second, _ = predict_canopy(data, accumulation, thresholds, bundle['components'][1]['fitted'])
    result = predict_publication_model(data, accumulation, thresholds, bundle, return_daily=True)
    np.testing.assert_allclose(result['predicted_percent'], .5*(first+second), atol=1e-12)
    assert result['daily_percent'].shape == (1, 61, 8)
    assert result['observation'] == 'infection_percent_unspecified_basis'


def test_publication_model_rejects_a_different_measurement_definition():
    data, accumulation, thresholds, bundle = example()
    data.targets['metric'] = 'pycnidial_coverage_percent'
    with pytest.raises(ValueError, match='measurement'):
        predict_publication_model(data, accumulation, thresholds, bundle)


def test_component_weights_cannot_silently_rescale_percentage_predictions():
    data, accumulation, thresholds, bundle = example()
    bundle['components'][0]['weight'] = .8
    with pytest.raises(ValueError, match='weights'):
        predict_publication_model(data, accumulation, thresholds, bundle)
