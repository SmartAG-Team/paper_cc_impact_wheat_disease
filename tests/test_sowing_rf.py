import numpy as np
import pandas as pd
import pytest


def test_full_calendar_temperature_uses_days_in_month_and_requires_complete_year():
    from models.monitoring.sowing_rf import annual_temperature_from_months
    dates=pd.date_range('2012-01-01',periods=12,freq='MS')
    kelvin=np.full(12,273.15);kelvin[1]+=28
    assert annual_temperature_from_months(dates,kelvin)==pytest.approx(28*29/366)
    with pytest.raises(ValueError,match='12'):
        annual_temperature_from_months(dates[:-1],kelvin[:-1])
    with pytest.raises(ValueError,match='finite'):
        annual_temperature_from_months(dates,np.full(12,np.nan))


def test_sowing_fit_excludes_future_labels_and_prediction_preserves_calendar_year():
    from models.monitoring.sowing_rf import fit_sowing_rf,predict_sowing_rf
    n=160
    years=np.tile(np.arange(1987,2007),8)
    table=pd.DataFrame(dict(station_id=np.repeat(np.arange(8),20),latitude=np.linspace(48,54,n),
        annual_mean_temperature_c=np.linspace(6,12,n),annual_temperature_year=years-1,
        sowing_date=pd.to_datetime([str(y)+'-10-01' for y in years])))
    future=table.iloc[[0]].assign(sowing_date=pd.Timestamp('2014-08-01'),annual_temperature_year=2013)
    fitted=fit_sowing_rf(pd.concat([table,future],ignore_index=True),trees=8)
    assert fitted['training_rows']==160
    query=pd.DataFrame(dict(field_id=['a'],season_year=[2017],latitude=[52.],
        annual_mean_temperature_c=[9.],annual_temperature_year=[2015]))
    result=predict_sowing_rf(fitted,query)
    assert result.iloc[0].estimated_sowing_date=='2016-10-01'
    assert result.iloc[0].sowing_basis=='rf_estimated_not_observed'
    assert fitted['estimator'].n_features_in_==2
    with pytest.raises(ValueError,match='labels'):
        predict_sowing_rf(fitted,query.assign(observed_percent=90.))
    with pytest.raises(ValueError,match='preceding'):
        predict_sowing_rf(fitted,query.assign(annual_temperature_year=2016))
