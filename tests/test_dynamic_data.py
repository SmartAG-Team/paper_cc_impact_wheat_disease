import numpy as np
import pandas as pd
import pytest


def fixture():
    fields = pd.DataFrame([dict(field_id='a', site_id='s', source='BASF', latitude=50., longitude=5., sowing_date='2020-01-01')])
    weather = pd.DataFrame(dict(location_id=['s']*5, date=pd.date_range('2020-01-01', periods=5),
        tmean_c=[10.]*5, rh_mean_pct=[90.]*5, rain_mm=[2.]*5))
    obs = pd.DataFrame([dict(global_target_index=i, field_id='a', site_id='s', source='BASF',
        leaf_index=l, date=d, observed_percent=y) for i,l,d,y in
        [(1,0,'2020-01-02',20.),(2,1,'2020-01-02',40.),(3,0,'2020-01-05',60.)]])
    return weather, fields, obs


def test_sequence_alignment_daily_masks_and_query_label_rejection():
    from models.monitoring.dynamic_data import build_sequences
    w,f,o = fixture(); q=o.drop(columns='observed_percent')
    b=build_sequences(w,f,o,q)
    np.testing.assert_array_equal(b['query_day'], [1,1,4])
    np.testing.assert_array_equal(b['query_leaf'], [0,1,0])
    assert b['weather'].shape == (1,5,6)
    assert b['observed_mask'][0,1,:2].all()
    assert not b['observed_mask'][0,4].any()
    assert b['valid_weather'][0,:4].all() and not b['valid_weather'][0,4]
    with pytest.raises(ValueError, match='label'):
        build_sequences(w,f,o,o)


def test_sequence_rejects_missing_weather_duplicates_and_source_conflict():
    from models.monitoring.dynamic_data import build_sequences
    w,f,o=fixture(); q=o.drop(columns='observed_percent')
    for bad_w,bad_f,bad_o in [(w.drop(index=2),f,o),(w,f,pd.concat([o,o.iloc[:1]])),
            (w,f.assign(source='Corteva'),o)]:
        with pytest.raises(ValueError):
            build_sequences(bad_w,bad_f,bad_o,q)


def test_later_weather_cannot_enter_earlier_query_sequence():
    from models.monitoring.dynamic_data import build_sequences
    w,f,o=fixture(); q=o.iloc[:2].drop(columns='observed_percent')
    first=build_sequences(w,f,o,q)
    w.loc[1:,'rain_mm']=999.
    second=build_sequences(w,f,o,q)
    np.testing.assert_array_equal(first['weather'],second['weather'])
    assert first['history_ids'] == [[]]
