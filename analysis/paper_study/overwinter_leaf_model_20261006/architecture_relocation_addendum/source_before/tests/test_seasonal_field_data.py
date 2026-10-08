import numpy as np
import pandas as pd
import pytest

from model.seasonal_septoria.field_data import prepare_fields


def source():
    assessments = pd.DataFrame(dict(dataset_id=['example']*2, physical_unit=['a']*2,
        site_id=['loc']*2, season_year=[2019]*2, country=['Germany']*2,
        latitude=[50.]*2, longitude=[10.]*2, organ=['LEAF, 1ST / FLAG LEAF','LEAF, 2ND'],
        date=['2019-01-09','2019-01-10'], value=[20.,40.],
        metric=['infection_percent_unspecified_basis']*2,unit=['percent']*2,
        endpoint_series=['a|1','a|2']))
    calendars = pd.DataFrame(dict(dataset_id=['example'],point_id=['loc'],
        crop_season=['winter_wheat'],water_system=['rainfed'],calendar_valid=[True],
        planting_doy=[1.],maturity_doy=[200.]))
    weather = pd.DataFrame(dict(location_id=['loc']*10,date=pd.date_range('2019-01-01',periods=10),
        tmean_c=[18.]*10,tmax_c=[23.]*10,rh_mean_pct=[80.]*10,precipitation_mm=[1.]*10))
    return assessments,calendars,weather


def test_observed_disease_does_not_change_season_start_simulation_inputs():
    assessments,calendars,weather = source()
    original = prepare_fields(assessments,calendars,weather)
    assessments['value'] = [99.,99.]
    changed = prepare_fields(assessments,calendars,weather)
    for key in ('temperature','humidity','rain','host_active','host_renewal'):
        np.testing.assert_array_equal(getattr(original,key),getattr(changed,key))
    assert original.targets.day_index.tolist() == [9,10]
    assert original.targets.leaf_index.tolist() == [0,1]
    assert original.metadata.sowing_basis.tolist() == ['calendar_scenario_not_observed']


def test_missing_weather_day_does_not_become_a_zero_forcing_day():
    assessments,calendars,weather = source()
    with pytest.raises(ValueError,match='weather'):
        prepare_fields(assessments,calendars,weather.drop(index=4))


def test_unknown_leaf_definition_is_retained_as_an_exclusion():
    assessments,calendars,weather = source()
    assessments.loc[1,'organ'] = 'PLANT,TOTAL'
    result = prepare_fields(assessments,calendars,weather)
    assert len(result.targets) == 1
    assert result.excluded.reason.tolist() == ['unsupported_organ_definition']


def test_observed_sowing_takes_precedence_over_calendar_scenario():
    assessments,calendars,weather = source()
    assessments['sowing_date'] = '2019-01-02'
    result = prepare_fields(assessments,calendars,weather)
    assert result.targets.day_index.tolist() == [8,9]
    assert result.metadata.sowing_date.tolist() == ['2019-01-02']
    assert result.metadata.sowing_basis.tolist() == ['observed_source_sowing']


def test_ordinal_leaf_count_cannot_be_interpreted_as_a_disease_percentage():
    assessments,calendars,weather = source()
    assessments.loc[1,['metric','unit']] = ['pycnidia_count','count']
    result = prepare_fields(assessments,calendars,weather)
    assert len(result.targets) == 1
    assert result.excluded.reason.tolist() == ['unsupported_metric_or_unit']


def test_two_digit_leaf_number_cannot_be_truncated_to_leaf_one():
    assessments,calendars,weather = source()
    assessments.loc[1,'organ'] = 'LEAF, 14TH'
    result = prepare_fields(assessments,calendars,weather)
    assert len(result.targets) == 1
    assert result.excluded.reason.tolist() == ['unsupported_organ_definition']
