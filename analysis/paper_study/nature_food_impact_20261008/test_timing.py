import numpy as np
import pandas as pd
import pytest
from .timing import common_pair, elapsed_dates


def frame(year=2020):
    sow = pd.Timestamp(year-1, 10, 1)
    flower = pd.Timestamp(year, 6, 1)
    symptom = pd.Timestamp(year, 5, 25)
    return pd.DataFrame(dict(cell_id=['a', 'b'], valid_complete_season=[True, True],
        calendar_sowing_date=[sow]*2, F1_symptom_date=[symptom]*2,
        BBCH65_date=[flower]*2, BBCH85_date=[flower+pd.Timedelta(days=30)]*2,
        F1_symptom_day_after_sowing=[(symptom-sow).days]*2,
        F1_symptom_relative_anthesis_days=[-7]*2))


def test_elapsed_calendar_days_include_leap_year_and_match_reordered_cells():
    past, future = frame(2019), frame(2020).iloc[::-1]
    a, b, valid = common_pair(past, future)
    assert valid.all()
    assert (b.symptom_days-a.symptom_days).eq(1).all()
    assert (b.symptom_relative_flowering_days-a.symptom_relative_flowering_days).eq(0).all()


def test_symptom_free_pair_is_excluded_from_every_event():
    past, future = frame(), frame(2021)
    future.loc[0, 'F1_symptom_date'] = pd.NaT
    future.loc[0, ['F1_symptom_day_after_sowing', 'F1_symptom_relative_anthesis_days']] = np.nan
    _, _, valid = common_pair(past, future)
    assert valid.tolist() == [False, True]


def test_duplicate_or_mismatched_identities_fail():
    bad = frame(); bad.cell_id = ['a', 'a']
    with pytest.raises(ValueError, match='Duplicate'):
        elapsed_dates(bad)
    bad = frame(); bad.cell_id = ['a', 'c']
    with pytest.raises(ValueError, match='identities differ'):
        common_pair(frame(), bad)


def test_recorded_days_must_equal_date_arithmetic():
    bad = frame(); bad.loc[0, 'F1_symptom_day_after_sowing'] += 1
    with pytest.raises(AssertionError):
        elapsed_dates(bad)
