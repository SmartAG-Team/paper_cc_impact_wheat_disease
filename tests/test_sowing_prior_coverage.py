import numpy as np
import pandas as pd


def test_future_weather_gaps_cannot_change_earlier_sowing_prior_coverage():
    from models.monitoring.sowing_distribution_data import prior_weather_coverage
    weather = pd.DataFrame(np.ones((12,4)), index=pd.date_range('2014-10-01',periods=12),
        columns=['tmean_c','tmax_c','tmin_c','precipitation_mm'])
    starts = [pd.Timestamp('2014-10-01'),pd.Timestamp('2014-10-03')]
    queries = pd.to_datetime(['2014-10-08','2014-10-13'])
    assert prior_weather_coverage(weather,starts,queries).tolist() == [True,True]
    altered = weather.copy(); altered.loc['2014-10-10','tmean_c'] = np.nan
    assert prior_weather_coverage(altered,starts,queries).tolist() == [True,False]
    extended = queries.append(pd.DatetimeIndex(['2014-10-20']))
    assert prior_weather_coverage(weather,starts,extended).tolist() == [True,True,False]
