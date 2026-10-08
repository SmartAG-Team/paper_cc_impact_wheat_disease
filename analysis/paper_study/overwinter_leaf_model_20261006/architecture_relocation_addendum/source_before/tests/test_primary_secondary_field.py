"""Initial-condition chronology and explicit rejection of malformed inputs."""
import importlib
import pandas as pd
import numpy as np
import pytest


def field():
    try:
        return importlib.import_module('model.primary_secondary.field_data')
    except ModuleNotFoundError:
        pytest.fail('Chronological field preparation is not implemented')


def records():
    return pd.DataFrame({'TrialId':[1,1,1,1,2], 'leaf_rank':[1,2,2,3,1],
        'Date':pd.to_datetime(['2017-05-01','2017-04-28','2017-05-06','2017-05-07','2017-05-01']),
        'Value':[10.,30.,90.,80.,100.]})


def test_future_assessments_do_not_change_canopy_snapshot():
    f=field()
    d=records()
    visible,active=f.canopy_snapshot(d,1,pd.Timestamp('2017-05-01'),np.arange(1,5))
    mutated=d.copy()
    mutated.loc[mutated.Date.gt(pd.Timestamp('2017-05-01')),'Value']=0
    other,other_active=f.canopy_snapshot(mutated,1,pd.Timestamp('2017-05-01'),np.arange(1,5))
    assert np.array_equal(visible,np.array([.1,.3,0.,0.]))
    assert np.array_equal(active,np.array([True,True,False,False]))
    assert np.array_equal(visible,other)
    assert np.array_equal(active,other_active)


def test_current_and_latest_past_canopy_assessments_are_used():
    f=field()
    visible,active=f.canopy_snapshot(records(),1,pd.Timestamp('2017-05-06'),np.arange(1,4))
    assert np.array_equal(visible,np.array([.1,.9,0.]))
    assert np.array_equal(active,np.array([True,True,False]))


def test_duplicate_canopy_date_rank_is_rejected():
    f=field()
    d=pd.concat([records(),records().iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match='[Dd]uplicate'):
        f.canopy_snapshot(d,1,pd.Timestamp('2017-05-01'),np.arange(1,4))


def test_missing_daily_forcing_is_rejected():
    f=field()
    w=pd.DataFrame({'location_id':['x'], 'date':pd.to_datetime(['2017-05-01']),
        'tmean_c':[18.], 'rh_hours_ge_90pct':[24.], 'rain_mm':[3.], 'hour_count':[24]})
    with pytest.raises(ValueError,match='[Mm]issing|[Ii]ncomplete'):
        f.weather_window(w,'x',pd.Timestamp('2017-05-01'),pd.Timestamp('2017-05-03'))


def test_assessment_day_weather_is_excluded():
    f=field()
    w=pd.DataFrame({'location_id':['x','x'], 'date':pd.to_datetime(['2017-05-01','2017-05-02']),
        'tmean_c':[18.,99.], 'rh_hours_ge_90pct':[24.,0.], 'rain_mm':[3.,0.], 'hour_count':[24,24]})
    out=f.weather_window(w,'x',pd.Timestamp('2017-05-01'),pd.Timestamp('2017-05-02'))
    assert len(out)==1
    assert out.tmean_c.iloc[0]==18.


def test_canopy_ages_measure_elapsed_days_since_available_assessment():
    f=field()
    visible,active,ages=f.canopy_snapshot(records(),1,pd.Timestamp('2017-05-01'),np.arange(1,5),return_age=True)
    assert np.array_equal(ages,np.array([0.,3.,0.,0.]))
    assert np.array_equal(visible,np.array([.1,.3,0.,0.]))
