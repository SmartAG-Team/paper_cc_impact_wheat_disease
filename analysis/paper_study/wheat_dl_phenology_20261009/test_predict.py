import numpy as np
import pandas as pd
import pytest
from model.wheat_phenology_dl.predict import weather_features


def test_prediction_preprocessing_requires_observed_daily_mean():
    f=pd.DataFrame({'date':['2020-10-01'],'tmin_c':[5.],'tmax_c':[15.]})
    with pytest.raises(ValueError,match='tmean'):weather_features(f,'2020-10-01',50.)


def test_prediction_preprocessing_is_causal_and_stops_at_missing_weather():
    dates=pd.date_range('2020-10-01',periods=25)
    f=pd.DataFrame({'date':dates,'tmean_c':10.,'tmin_c':5.,'tmax_c':15.})
    x,g,n=weather_features(f,'2020-10-01',50.)
    future=f.copy();future.loc[15:,'tmean_c']=20.;future.loc[15:,'tmax_c']=25.
    xx,gg,nn=weather_features(future,'2020-10-01',50.)
    np.testing.assert_array_equal(x[:15],xx[:15]);np.testing.assert_array_equal(g[:15],gg[:15])
    assert n==nn==25
    _,_,broken=weather_features(f.drop(index=10),'2020-10-01',50.)
    assert broken==10
