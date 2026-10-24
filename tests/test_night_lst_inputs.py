import numpy as np
import pandas as pd
from models.monitoring.night_lst_inputs import qualified_night_pixels,build_night_features,temperature_content_columns


def test_night_quality_checks_detailed_flags_including_other_quality():
    # 65: produced(other), data good, emissivity<=.01, temperature error<=2K.
    qc=np.array([0,1,65,129,2,3,5,97,81,193])
    accepted=qualified_night_pixels(np.full(10,14000),qc,np.full(10,220),np.full(10,65))
    assert accepted.tolist()==[True,True,True,False,False,False,False,False,True,False]
    assert not qualified_night_pixels(np.array([0,14000]),np.array([0,0]),np.array([220,241]),np.array([65,65])).any()


def fixture():
    weather=pd.DataFrame(dict(location_id=['s']*40,date=pd.date_range('2015-05-01',periods=40),
        tmin_c=10.,dewpoint_mean_c=12.))
    rows=[]
    for sensor in ['terra','aqua']:
        for day in ['2015-05-10','2015-05-20','2015-05-30']:
            rows.append(dict(site_id='s',sensor=sensor,period_start=day,
                period_end=str((pd.Timestamp(day)+pd.Timedelta(days=1)).date()),
                night_lst_c_mean=9.,night_lst_c_stdDev=1.,view_hour_mean=22.,view_angle_abs_mean=10.,
                valid_mean=.5,night_lst_c_count=40,valid_count=80))
    queries=pd.DataFrame(dict(global_target_index=[1,2],site_id=['s','s'],date=['2015-05-22','2015-06-01']))
    return weather,pd.DataFrame(rows),queries


def test_night_features_withhold_incomplete_periods_and_retain_missingness_control():
    weather,rows,queries=fixture()
    x,names,available,meta=build_night_features(rows,weather,queries)
    assert available.tolist()==[True,True]
    assert np.isfinite(x).all()
    # May20 period endsMay21, with two-day lag only availableMay23.
    changed=rows.copy();changed.loc[changed.period_start.ge('2015-05-20'),'night_lst_c_mean']=90.
    again,_,_,_=build_night_features(changed,weather,queries)
    np.testing.assert_array_equal(x[0],again[0])
    assert not np.array_equal(x[1],again[1])
    control=[i for i,name in enumerate(names) if 'coverage' in name or 'age_' in name or 'available' in name]
    np.testing.assert_array_equal(x[:,control],again[:,control])
    assert pd.Timestamp(meta.iloc[0].latest_period_end)<pd.Timestamp(queries.iloc[0].date)


def test_night_prefix_order_and_low_coverage_fallback():
    weather,rows,queries=fixture()
    x,names,available,_=build_night_features(rows,weather,queries)
    prefix,_,_,_=build_night_features(rows,weather,queries.iloc[:1])
    np.testing.assert_array_equal(x[:1],prefix)
    shuffled,_,_,_=build_night_features(rows.iloc[::-1],weather,queries.iloc[::-1])
    np.testing.assert_array_equal(x[::-1],shuffled)
    rows['night_lst_c_count']=19
    masked,_,available,_=build_night_features(rows,weather,queries)
    assert not available.any()
    assert np.isfinite(masked).all()
    assert not masked[:,[i for i,n in enumerate(names) if 'lst_c' in n]].any()


def test_provider_gap_is_distinguished_from_a_cloud_rejected_image():
    weather,rows,queries=fixture()
    rows['night_lst_c_count']=0
    cloudy,names,_,_=build_night_features(rows,weather,queries.iloc[:1])
    missing,_,_,_=build_night_features(rows.loc[rows.period_start.ne('2015-05-10')],weather,queries.iloc[:1])
    index=names.index('terra_provider_coverage_14d')
    assert cloudy[0,index]==1/14 and missing[0,index]==0
    np.testing.assert_array_equal(cloudy[:,[i for i,n in enumerate(names) if 'night_coverage' in n]],
                                  missing[:,[i for i,n in enumerate(names) if 'night_coverage' in n]])


def test_matched_control_keeps_air_values_but_removes_surface_temperature_information():
    weather,rows,queries=fixture()
    x,names,_,_=build_night_features(rows,weather,queries)
    content=temperature_content_columns(names);control=x.copy();control[:,content]=0
    air=names.index('terra_latest_air_tmin_c')
    assert control[0,air]==.5
    changed=rows.copy();changed['night_lst_c_mean']+=7
    z,_,_,_=build_night_features(changed,weather,queries);z=z.copy();z[:,content]=0
    np.testing.assert_array_equal(z,control)
    warmer=weather.copy();warmer['tmin_c']+=4
    z,_,_,_=build_night_features(rows,warmer,queries);z=z.copy();z[:,content]=0
    assert z[0,air]==.7 and not np.array_equal(z,control)
