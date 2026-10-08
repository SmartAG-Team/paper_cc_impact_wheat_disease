"""Accuracy refinements must preserve observation definitions and partitions."""

import numpy as np
import pandas as pd
import pytest

from model.seasonal_septoria.core import Parameters, simulate_season
from calibration.seasonal_septoria.field_data import FieldData
from calibration.seasonal_septoria.refinement import map_observations, calibration_weights, fit_refinement, predict_refinement, blend_severity


def example_data():
    days = 100
    temperature = np.full((3, days), 18.)
    humidity = np.full_like(temperature, 90.)
    rain = np.tile(np.array([0., 2., 4.])[:, None], (1, days))
    active = np.ones((3, days, 1), bool)
    renewal = np.zeros(active.shape)
    trajectory = simulate_season(temperature, humidity, rain, active, renewal,
        Parameters(alpha=.035, beta=0., latent_days=25.))
    rows = []
    for field in range(3):
        for day in [20, 40, 60, 80, 100]:
            rows.append(dict(field_index=field, field_id=f'field{field}',
                site_id=f'site{field}', coordinate_year=f'site{field}|2018',
                endpoint_series=f'field{field}|F1', date=pd.Timestamp('2018-01-01')
                + pd.Timedelta(days=day-1), day_index=day, leaf_index=0,
                observation_operator='damage_proxy',
                metric='infection_percent_unspecified_basis',
                value=100*trajectory.damage[field, day, 0]**1.8, weight=.2))
    return FieldData(temperature, humidity, rain, active, renewal,
                     pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame())


def test_mapping_preserves_zero_and_full_severity_and_named_area_measures():
    values = np.array([0., .25, 1., .25, .25])
    metric = np.array(['infection_percent_unspecified_basis']*3
        + ['necrotic_leaf_area_percent', 'pycnidial_coverage_percent'])
    np.testing.assert_allclose(map_observations(values, metric, power=2.),
                               [0., .0625, 1., .25, .25])
    with pytest.raises(ValueError):
        map_observations(values, metric, power=0.)


def test_final_assessment_loss_keeps_equal_coordinate_year_totals():
    target = pd.DataFrame(dict(field_index=[0, 0, 1, 1, 2, 2],
        field_id=['a', 'a', 'b', 'b', 'c', 'c'],
        coordinate_year=['shared']*4+['other']*2,
        endpoint_series=['a|F1']*2+['b|F1']*2+['c|F1']*2,
        date=pd.to_datetime(['2018-05-01', '2018-06-01']*3)))
    np.testing.assert_allclose(calibration_weights(target, final_weight=.5),
                               [.125, .375, .125, .375, .25, .75])
    np.testing.assert_allclose(calibration_weights(target, final_weight=1.),
                               [0., .5, 0., .5, 0., 1.])


def test_refinement_learns_observation_mapping_without_reading_withheld_values():
    data = example_data()
    selected = np.arange(10)
    first = fit_refinement(data, selected, latent_free=True, power_free=True)
    predicted, _ = predict_refinement(data, first)
    assert np.sqrt(np.mean((predicted[:10]-data.targets.value[:10])**2)) < .05
    assert first['observation_power'] > 1.5
    data.targets.loc[10:, 'value'] = np.nan
    data.temperature[2] = -10.
    second = fit_refinement(data, selected, latent_free=True, power_free=True)
    assert first['parameters'] == second['parameters']
    assert first['observation_power'] == second['observation_power']


def test_refinement_rejects_incomplete_field_seasons():
    data = example_data()
    with pytest.raises(ValueError, match='field-season'):
        fit_refinement(data, [0, 1])


def test_blended_severity_preserves_inactive_cohorts_and_percentage_bounds():
    actual = blend_severity([20., 40., 0.], [60., 80., 50.],
                            empirical_weight=.25, host_active=[True, True, False])
    np.testing.assert_allclose(actual, [30., 50., 0.])
    with pytest.raises(ValueError):
        blend_severity([20.], [60.], empirical_weight=1.1, host_active=[True])
