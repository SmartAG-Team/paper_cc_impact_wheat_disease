import numpy as np
import pandas as pd


def hourly_fixture():
    hour=pd.date_range('2020-01-01',periods=72,freq='h')
    rain=np.zeros(72);rain[46:52]=1.
    return pd.DataFrame(dict(location_id='s',hour_start=hour,precipitation_mm=rain,
        temperature_c=18.,humidity_pct=50.))


def test_contiguous_rain_spell_crosses_midnight_and_missing_hours_remain_unknown():
    from models.monitoring.hourly_event_features import daily_station_events
    raw=hourly_fixture();daily=daily_station_events(raw).set_index('date')
    row=daily.loc[pd.Timestamp('2020-01-03')]
    assert row.precip_run_hours==6 and row.precip_hours==4
    assert row.precip_max_6h==6
    missing=raw.copy();missing.loc[47,'precipitation_mm']=np.nan
    changed=daily_station_events(missing).set_index('date').loc[pd.Timestamp('2020-01-03')]
    assert pd.isna(changed.precip_run_hours)
    assert not changed.precip_timing_complete


def test_hourly_features_ignore_query_day_and_later_weather():
    from models.monitoring.hourly_event_features import daily_station_events,build_hourly_features
    raw=hourly_fixture()
    fields=pd.DataFrame(dict(field_id=['f'],sowing_date=['2020-01-01']))
    queries=pd.DataFrame(dict(global_target_index=[7],field_id=['f'],site_id=['s'],date=['2020-01-03']))
    first,names,ledger=build_hourly_features(daily_station_events(raw),fields,queries)
    altered=raw.copy();altered.loc[48:,'precipitation_mm']=100.
    altered.loc[48:,'humidity_pct']=100.
    second,_,_=build_hourly_features(daily_station_events(altered),fields,queries)
    np.testing.assert_array_equal(first,second)
    assert ledger.latest_weather_day.iloc[0]=='2020-01-02'
