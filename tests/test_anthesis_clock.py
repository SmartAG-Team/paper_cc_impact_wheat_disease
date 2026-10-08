"""Independent flowering clocks preserve censored stage information."""

import numpy as np
import pandas as pd
import pytest

from analysis.paper_study.infection_priority_20261006.anthesis_clock import run as clock


def stage_rows():
    rows = []
    for unit, date, lower, upper in [
        ('a', '2018-05-01', np.nan, 59.),
        ('a', '2018-05-01', np.nan, 59.),
        ('a', '2018-05-10', 65., np.nan),
        ('b', '2018-05-01', 80., 70.),
        ('c', '2018-05-01', 55., 60.),
        ('c', '2018-05-01', 65., 70.),
        ('d', '2018-05-01', 59., np.nan),
        ('d', '2018-05-05', np.nan, 57.),
    ]:
        rows.append(dict(dataset_id='basf-wheat-diseases', physical_unit=unit,
            season_year=2018, site_id=f'site_{unit}', date=date, stage_from=lower,
            stage_to=upper, country='GERMANY', latitude=50., longitude=8., value=42.))
    return pd.DataFrame(rows)


def metadata():
    return pd.DataFrame([dict(field_id='a', field_index=0, forcing_days=3,
        sowing_date='2018-01-01', last_forcing_date='2018-01-03')])


def test_clock_first_crossing_is_inclusive_and_keeps_missing_events():
    events = clock.predict_clock(np.array([[100., 200., 300.]]), metadata(),
        {51: 100., 65: 200., 85: 600.})
    indexed = events.set_index('event')
    assert indexed.loc[51, 'predicted_date'] == pd.Timestamp('2018-01-01')
    assert indexed.loc[65, 'predicted_date'] == pd.Timestamp('2018-01-02')
    assert pd.isna(indexed.loc[85, 'predicted_date'])
    assert indexed.loc[85, 'prediction_status'] == 'not_reached_by_forcing_end'


def test_stage_boundary_drops_disease_values_and_collapses_partial_leaf_duplicates():
    first = clock.build_stage_constraints(stage_rows(), 'BASF')
    changed = stage_rows()
    changed['value'] = [np.nan, -999., 1000., np.nan, -1., 1., 0., 999.]
    changed['assessment_eligible'] = False
    second = clock.build_stage_constraints(changed, 'BASF')
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)
        assert 'value' not in left
    normalized, rejected, constraints = first
    series = normalized[normalized.physical_unit.eq('a')]
    assert len(series) == 2
    assert series.iloc[0].source_row_count == 2
    bound = constraints[constraints.physical_unit.eq('a')].iloc[0]
    assert bound.lower_exclusive == pd.Timestamp('2018-05-01')
    assert bound.upper_inclusive == pd.Timestamp('2018-05-10')
    assert bound.genuinely_bracketed


def test_impossible_stage_rows_and_chronological_order_never_enter_loss():
    _, rejected, constraints = clock.build_stage_constraints(stage_rows(), 'BASF')
    assert set(rejected.reason) >= {
        'reversed_stage_interval', 'conflicting_date_stage_ranges',
        'temporally_inconsistent_stage_order'}
    conflict = constraints[constraints.physical_unit.eq('d')].iloc[0]
    assert not conflict.eligible_constraint
    assert conflict.reason == 'temporally_inconsistent_stage_order'


def test_missing_flowering_event_has_one_sided_horizon_distance():
    constraints = pd.DataFrame([dict(field_id='a', event=65, eligible_constraint=True,
        lower_exclusive=pd.Timestamp('2018-01-01'),
        upper_inclusive=pd.Timestamp('2018-01-02'))])
    predicted = clock.predict_clock(np.array([[100., 200., 300.]]), metadata(), {65: 600.})
    scored = clock.score_clock(constraints, predicted)
    assert scored.iloc[0].distance_days == 2.
    assert scored.iloc[0].distance_is_lower_bound
    assert not scored.iloc[0].compatible
    constraints['upper_inclusive'] = pd.NaT
    scored = clock.score_clock(constraints, predicted)
    assert scored.iloc[0].distance_days == 0.
    assert scored.iloc[0].compatible


def test_threshold_grid_uses_only_calibration_stage_bounds_and_reports_plateau():
    constraints = pd.DataFrame([dict(field_id='a', source='BASF', season_year=2018,
        event=65, eligible_constraint=True, genuinely_bracketed=True,
        lower_exclusive=pd.Timestamp('2018-01-01'),
        upper_inclusive=pd.Timestamp('2018-01-02'))])
    selected, curve, losses = clock.fit_threshold_grid(constraints,
        np.array([[105., 115., 135.]]), metadata(), 100., 200.,
        fractions=np.array([.02, .10, .20, .60]))
    assert selected['fraction'] == .10
    assert selected['threshold_65'] == 110.
    assert selected['minimum_loss_days'] == 0.
    np.testing.assert_allclose(curve.loss_days, [1., 0., 1., 2.])
    np.testing.assert_allclose(losses['a'], [1., 0., 1., 2.])
    constraints['season_year'] = 2019
    with pytest.raises(ValueError, match='2017.*2018'):
        clock.fit_threshold_grid(constraints, np.array([[105., 115., 135.]]),
            metadata(), 100., 200., fractions=np.array([.1]))
