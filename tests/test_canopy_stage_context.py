import numpy as np
import pandas as pd
import pytest


def _canopy():
    return pd.DataFrame(dict(period_end=pd.to_datetime(['2010-04-01','2010-04-05','2010-04-09']),
        lai_mean=[2.,3.,8.],fpar_mean=[.4,.6,.9],lai_stdDev=.2,fpar_stdDev=.05,
        lai_uncertainty_mean=.3,fpar_uncertainty_mean=.04,valid_mean=.8,lai_count=200.,crop_fraction=.7))


def test_stage_context_aligns_weather_to_previous_day_and_ignores_future():
    from models.monitoring.canopy_stage_context import stage_context
    dates=pd.date_range('2010-01-01','2010-05-01')
    weather=np.column_stack([np.arange(len(dates)),np.arange(len(dates))+2,
        np.arange(len(dates))-2,np.ones(len(dates))])
    query=pd.Timestamp('2010-04-14')
    x,names,available=stage_context(weather,dates,query,52.,_canopy())
    assert available and len(x)==len(names)==85
    index=dates.get_loc(query)
    assert x[names.index('stage_weather_tmean_mean_14d_c')]==weather[index-14:index,0].mean()
    assert x[names.index('stage_weather_precip_sum_84d_mm')]==84
    assert x[names.index('canopy_latest_lai')]==3.
    altered=weather.copy();altered[index:]=9999
    canopy=_canopy();canopy.loc[2,'lai_mean']=9999
    y,other_names,other_available=stage_context(altered,dates,query,52.,canopy)
    np.testing.assert_array_equal(x,y)
    assert names==other_names and available==other_available
    assert not any('sow' in name or 'season_year' in name for name in names)


def test_stage_context_rejects_incomplete_or_nonphysical_weather():
    from models.monitoring.canopy_stage_context import stage_context
    dates=pd.date_range('2010-01-01','2010-05-01')
    weather=np.tile([10.,12.,8.,1.],(len(dates),1))
    query=pd.Timestamp('2010-04-14')
    with pytest.raises(ValueError,match='84 complete'):
        stage_context(weather[40:],dates[40:],query,52.,_canopy())
    missing=weather.copy();missing[dates.get_loc(query)-1,0]=np.nan
    with pytest.raises(ValueError,match='finite physical'):
        stage_context(missing,dates,query,52.,_canopy())
    invalid=weather.copy();invalid[dates.get_loc(query)-1,3]=-1
    with pytest.raises(ValueError,match='finite physical'):
        stage_context(invalid,dates,query,52.,_canopy())


def test_calendar_stage_control_uses_only_station_weighted_training_events():
    from models.monitoring.canopy_stage_context import fit_calendar_cdf,predict_calendar_cdf
    records=[dict(PEP_ID=1,true_dates=[None,'2005-04-01','2005-05-01','2005-06-01']),
        dict(PEP_ID=1,true_dates=[None,'2006-04-11','2006-05-11','2006-06-11']),
        dict(PEP_ID=2,true_dates=[None,'2005-04-21','2005-05-21','2005-06-21']),
        dict(PEP_ID=3,true_dates=[None,'2005-01-01','2005-02-01','2005-03-01'])]
    prior=fit_calendar_cdf(records,{1,2})
    dates=pd.to_datetime(['2010-04-16','2010-05-16','2010-06-16'])
    pred=predict_calendar_cdf(prior,dates)
    np.testing.assert_allclose(pred,[[.5,0,0],[1,.5,0],[1,1,.5]])
    records[-1]['true_dates']=[None,'2005-09-01','2005-10-01','2005-11-01']
    np.testing.assert_array_equal(pred,predict_calendar_cdf(fit_calendar_cdf(records,{1,2}),dates))
    # Repeating every cycle at one station preserves that station's total mass.
    repeated=records+records[:2]
    np.testing.assert_array_equal(pred,predict_calendar_cdf(fit_calendar_cdf(repeated,{1,2}),dates))
