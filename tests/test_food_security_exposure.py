import numpy as np
import pandas as pd
import pytest

from paper.analyses.food_security.exposure import summarize_exposure, country_domains


def inputs():
    production = pd.DataFrame({
        'cell_id': ['a', 'b', 'c'], 'environment_region': ['Atlantic', 'Atlantic', 'Boreal'],
        'production_total_tonnes': [100., 200., 300.],
        'production_rainfed_tonnes': [80., 200., 250.],
    })
    changes = pd.DataFrame([
        dict(cell_id=cell, model=model, scenario='ssp585', period='2071-2100',
             change=value, valid_year_pairs=30 if np.isfinite(value) else 0)
        for model, values in [('m1', [2., -1., 3.]), ('m2', [1., -2., np.nan])]
        for cell, value in zip(['a', 'b', 'c'], values)
    ])
    return changes, production


def europe(frame, **filters):
    q = frame.loc[frame.domain_type.eq('Europe') & frame.weight_basis.eq('total')]
    for key, value in filters.items():
        q = q.loc[q[key].eq(value)]
    assert len(q) == 1
    return q.iloc[0]


def test_exposure_keeps_missing_production_in_full_denominator():
    changes, production = inputs()
    result = summarize_exposure(changes, production, models=['m1', 'm2'], thresholds=[0.])
    row = europe(result, members='m1|m2', threshold_days=0.)
    assert row.baseline_production_tonnes == 600
    assert row.unanimous_increase_tonnes == 100
    assert row.unanimous_increase_share_percent == pytest.approx(100 / 6)
    assert row.unavailable_tonnes == 300
    assert row.unanimous_decrease_tonnes == 200
    assert row.mass_balance_error_tonnes == pytest.approx(0)


def test_threshold_is_strict_and_equal_values_remain_nonunanimous():
    changes, production = inputs()
    result = summarize_exposure(changes, production, models=['m1', 'm2'], thresholds=[1.])
    row = europe(result, members='m1|m2', threshold_days=1.)
    assert row.unanimous_increase_tonnes == 0
    assert row.ensemble_increase_tonnes == 100
    assert row.mixed_or_zero_tonnes == 300


def test_omitted_model_uses_original_complete_population():
    changes, production = inputs()
    result = summarize_exposure(changes, production, models=['m1', 'm2'], thresholds=[0.])
    row = europe(result, members='m1', threshold_days=0.)
    assert row.unanimous_increase_tonnes == 100
    assert row.unavailable_tonnes == 300


def test_missing_member_and_duplicate_identity_are_rejected():
    changes, production = inputs()
    with pytest.raises(ValueError, match='model'):
        summarize_exposure(changes, production, models=['m1', 'missing'])
    with pytest.raises(ValueError, match='Duplicate'):
        summarize_exposure(pd.concat([changes, changes.iloc[:1]]), production, models=['m1', 'm2'])


def test_fractional_country_weights_conserve_each_cells_production():
    _, production = inputs()
    fractions = pd.DataFrame({
        'cell_id': ['a', 'a', 'b', 'c'], 'country': ['X', 'Y', 'X', 'Y'],
        'production_total_tonnes': [40., 60., 200., 300.],
        'production_rainfed_tonnes': [30., 50., 200., 250.],
    })
    domains = country_domains(production, fractions)
    assert domains[('country', 'X')].production_total_tonnes.sum() == 240
    fractions.loc[0, 'production_total_tonnes'] = 41
    with pytest.raises(ValueError, match='conserve'):
        country_domains(production, fractions)


def test_coverage_threshold_does_not_remove_denominator():
    changes, production = inputs()
    changes.loc[changes.cell_id.eq('a'), 'valid_year_pairs'] = 26
    result = summarize_exposure(changes, production, models=['m1', 'm2'], thresholds=[0.], min_pairs=27)
    row = europe(result, members='m1|m2', threshold_days=0.)
    assert row.baseline_production_tonnes == 600
    assert row.unavailable_tonnes == 400


def test_infinite_response_is_unavailable_with_its_production_retained():
    changes, production = inputs()
    changes.loc[changes.cell_id.eq('a') & changes.model.eq('m1'), 'change'] = np.inf
    result = summarize_exposure(changes, production, models=['m1', 'm2'], thresholds=[0.])
    row = europe(result, members='m1|m2', threshold_days=0.)
    assert row.unanimous_increase_tonnes == 0
    assert row.unavailable_tonnes == 400
    assert np.isfinite(row.production_weighted_change_days)

