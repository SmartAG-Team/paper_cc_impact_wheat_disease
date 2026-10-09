"""Safeguards for field-data grain, missing controls and grouped transfer tests."""
import numpy as np
from .run import prepare, grouped_predictions, fit_predict


def test_leaf_assessments_share_one_yield_outcome():
    plots, contrasts, quality = prepare()
    assert len(plots) == 82 and not plots.plot_season_id.duplicated().any()
    assert len(contrasts) == 40 and contrasts.relative_yield_gap_pct.lt(0).sum() == 1
    assert quality['combined_assessment_records'] == 423
    assert quality['combined_clean_assessment_records'] == 412
    assert quality['combined_clean_yield_units'] == 263


def test_missing_protected_severity_is_not_imputed():
    _, contrasts, quality = prepare()
    assert 'protected_severity_flag' not in contrasts
    assert 'delta_severity_flag' not in contrasts
    assert quality['protected_severity_observations'] == 0
    assert contrasts[['severity_flag', 'severity_lower']].notna().all().all()


def test_held_out_year_yields_cannot_change_predictions():
    plots, _, _ = prepare()
    a = grouped_predictions(plots, 'year', 'training_mean', [], 'yield_t_ha')
    changed = plots.copy()
    changed.loc[changed.year.eq(2019), 'yield_t_ha'] += 100
    b = grouped_predictions(changed, 'year', 'training_mean', [], 'yield_t_ha')
    np.testing.assert_allclose(a.loc[a.year.eq(2019), 'prediction'], b.loc[b.year.eq(2019), 'prediction'])
    for row in a.itertuples():
        assert str(row.year) not in row.training_groups.split('|')


def test_sign_constrained_yield_prediction_decreases_with_severity():
    plots, _, _ = prepare()
    train = plots[plots.year.eq(2018)]
    test = train.iloc[:2].copy()
    test.loc[:, 'severity_flag'] = [0., 1.]
    prediction, _ = fit_predict(train, test, 'nonnegative_damage', ['severity_flag'], 'yield_t_ha', 'year')
    assert prediction[1] <= prediction[0]
