"""Information boundaries for recovered crop-stage predictors."""
import numpy as np
import pandas as pd


def example():
    fields=pd.DataFrame(dict(field_id=['f'],site_id=['s'],sowing_date=['2018-04-01'],latitude=[50.],longitude=[10.]))
    weather=pd.DataFrame(dict(location_id=['s']*40,date=pd.date_range('2018-04-01',periods=40),
        tmean_c=[10.]*40,tmax_c=[15.]*40,tmin_c=[5.]*40))
    query=pd.DataFrame(dict(global_target_index=[42],field_id=['f'],date=['2018-04-20'],leaf_index=[4]))
    calendar=pd.DataFrame(dict(field_id=['f']*4,date=['2018-03-20','2018-04-10','2018-04-20','2018-05-01'],
        stage_from=[10.,31.,65.,87.],stage_to=[12.,33.,69.,89.],
        stage_midpoint=[11.,32.,67.,88.],usable=[True]*4,source_rows=['a','b','c','d']))
    return weather,fields,query,calendar


def test_prior_stage_ignores_same_day_future_and_previous_season():
    from models.monitoring.stage_features import build_stage_features
    weather,fields,query,calendar=example()
    x,names,p=build_stage_features(weather,fields,query,calendar)
    assert x[0,names.index('prior_stage_midpoint_scaled')]==.32
    assert x[0,names.index('prior_stage_available')]==1
    assert x[0,names.index('prior_stage_age_30d')]==10/30
    assert p.iloc[0].latest_stage_date=='2018-04-10'
    assert p.iloc[0].weather_end=='2018-04-19'
    changed=calendar.copy()
    changed.loc[changed.date.ge('2018-04-20'),['stage_from','stage_to','stage_midpoint']]=99.
    altered=weather.copy()
    altered.loc[altered.date.ge('2018-04-20'),['tmean_c','tmax_c','tmin_c']]=40.
    y,_,_=build_stage_features(altered,fields,query,changed)
    np.testing.assert_array_equal(x,y)


def test_missing_conflicting_stage_has_flag_and_no_fabricated_value():
    from models.monitoring.stage_features import build_stage_features
    weather,fields,query,calendar=example()
    calendar.loc[calendar.date.eq('2018-04-10'),'usable']=False
    x,names,p=build_stage_features(weather,fields,query,calendar)
    assert x[0,names.index('prior_stage_available')]==0
    assert x[0,names.index('prior_stage_midpoint_scaled')]==0
    assert pd.isna(p.iloc[0].latest_stage_date)
    assert x[0,names.index('clock_positive_degree_days_1000')]==.19


def test_same_day_stage_is_separate_conditional_information():
    from models.monitoring.stage_features import contemporaneous_features
    _,_,query,calendar=example()
    x,names=contemporaneous_features(query,calendar)
    assert x[0,names.index('current_stage_available')]==1
    assert x[0,names.index('current_stage_midpoint_scaled')]==.67
    no_current=calendar.loc[~calendar.date.eq('2018-04-20')]
    y,_=contemporaneous_features(query,no_current)
    assert y[0,names.index('current_stage_available')]==0
