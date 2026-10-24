"""Scientific identities and sampling units for the fixed-forecast displays."""
import importlib

import numpy as np
import pandas as pd
import pytest


def example():
    # Unequal weights and biased forecasts distinguish predictive R² from corr².
    rows = []
    for evaluation, shift in [('temporal', 0.), ('spatial_50km', 3.)]:
        for i, (observed, prediction, weight, site) in enumerate([
            (0., 20., 1., 's1'), (20., 40., 2., 's1'),
            (40., 50., 1., 's2'), (60., 75., 4., 's3'),
            (80., 90., 2., 's4'), (100., 95., 1., 's4'),
        ]):
            rows.append(dict(evaluation=evaluation, global_target_index=i,
                field_id='f'+site[1:], site_id=site, date=f'2017-06-{i + 1:02}',
                observed_percent=observed, prediction_percent=prediction + shift,
                scientific_weight=weight))
    points = pd.DataFrame(rows)
    observations = points[points.evaluation.eq('temporal')].drop(columns=[
        'evaluation', 'prediction_percent', 'scientific_weight']).copy()
    observations['latitude'] = observations.site_id.map(dict(s1=50., s2=51., s3=52., s4=53.))
    observations['longitude'] = 5.
    observations['coordinate_year'] = observations.site_id + '|2017'
    return points, observations


def module():
    return importlib.import_module('paper.scripts.figure_validation_v2')


def test_preserves_fixed_points_and_weighted_r2_identity():
    points, observations = example()
    data = module().prepare_validation(points, observations)
    pd.testing.assert_frame_equal(data['points'][points.columns], points)
    frame = points[points.evaluation.eq('temporal')]
    score = data['scores'].set_index('evaluation').loc['temporal']
    audit = data['r2_audit'].set_index('evaluation').loc['temporal']
    w = frame.scientific_weight.to_numpy()
    y = frame.observed_percent.to_numpy()
    p = frame.prediction_percent.to_numpy()
    mse = np.average((p-y)**2, weights=w)
    variance = np.average((y-np.average(y, weights=w))**2, weights=w)
    assert score.r2 == pytest.approx(1-mse/variance)
    assert score.rmse**2 == pytest.approx(mse)
    assert audit.sse/audit.sst == pytest.approx(mse/variance)
    assert abs(score.r2-audit.weighted_correlation_squared) > .05


def test_density_preserves_raw_counts_and_original_scientific_mass():
    points, observations = example()
    data = module().prepare_validation(points, observations)
    for evaluation, frame in points.groupby('evaluation'):
        density = data['density'].query('evaluation == @evaluation')
        assert len(density) == 40 * 40
        assert density.assessments.sum() == len(frame)
        assert density.scientific_weight_mass.sum() == pytest.approx(frame.scientific_weight.sum())
        assert density.scientific_mass_percent.sum() == pytest.approx(100.)
        first = density.query('observed_lower == 0 and predicted_lower == 20')
        if evaluation == 'temporal':
            assert first.scientific_mass_percent.item() == pytest.approx(100/11)


def test_site_bootstrap_recomputes_mean_and_preserves_original_relative_weights():
    points, observations = example()
    data = module().prepare_validation(points, observations)
    ids = sorted(points.site_id.unique())
    multiplicities = np.random.default_rng(20261024).multinomial(
        len(ids), np.full(len(ids), 1/len(ids)), size=2000)
    for evaluation in ['temporal', 'spatial_50km']:
        frame = points[points.evaluation.eq(evaluation)]
        for replicate in [0, 1, 37, 1999]:
            weights = frame.scientific_weight.to_numpy() * frame.site_id.map(
                dict(zip(ids, multiplicities[replicate]))).to_numpy()
            observed = frame.observed_percent.to_numpy()
            predicted = frame.prediction_percent.to_numpy()
            expected_mean = np.average(observed, weights=weights)
            expected_variance = np.average((observed-expected_mean)**2, weights=weights)
            expected_mse = np.average((predicted-observed)**2, weights=weights)
            result = data['metric_resamples'].query(
                'evaluation == @evaluation and replicate == @replicate').iloc[0]
            assert result.observed_mean == pytest.approx(expected_mean)
            assert result.rmse == pytest.approx(np.sqrt(expected_mse))
            assert result.r2 == pytest.approx(1-expected_mse/expected_variance)


def test_conflicting_coordinates_fail_instead_of_averaging_sites():
    points, observations = example()
    observations.loc[1, 'latitude'] = 54.
    with pytest.raises(ValueError, match='coordinates'):
        module().prepare_validation(points, observations)
