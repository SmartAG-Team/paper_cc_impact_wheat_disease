import numpy as np
import pandas as pd


def test_precipitation_is_assigned_by_ending_hour_not_temperature_timestamp():
    from models.monitoring.coordinate_augmentation import hourly_to_daily
    times=pd.date_range('2018-06-01',periods=49,freq='h')
    n=len(times)
    payload={'hourly':dict(time=times.strftime('%Y-%m-%dT%H:%M').tolist(),
        temperature_2m=[10.]*n,relative_humidity_2m=[90.]*n,dew_point_2m=[8.]*n,
        precipitation=[0.]*24+[7.]+[0.]*(n-25),rain=[0.]*24+[5.]+[0.]*(n-25),
        wind_speed_10m=[2.]*n,shortwave_radiation=[100.]*n)}
    daily=hourly_to_daily(payload,'site')
    first=daily.loc[daily.date.eq(pd.Timestamp('2018-06-01'))].iloc[0]
    assert first.precipitation_mm==7 and first.rain_mm==5
    assert first.rain_hours_gt_0_1mm==1 and first.rh_hours_ge_90pct==24
    assert first.hour_count==24
    assert np.isclose(first.shortwave_mj_m2,8.64)
    assert daily.date.max()==pd.Timestamp('2018-06-02')
