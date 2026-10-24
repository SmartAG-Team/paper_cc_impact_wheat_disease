"""Observed-sowing random forest with explicit calendar and availability rules."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES=['annual_mean_temperature_c','latitude']


def annual_temperature_from_months(months,kelvin):
    dates=pd.DatetimeIndex(pd.to_datetime(months))
    values=np.asarray(kelvin,float)
    if (len(dates)!=12 or values.shape!=(12,) or dates.year.nunique()!=1
            or set(dates.month)!=set(range(1,13))):
        raise ValueError('12 unique months of one complete calendar year required')
    if not np.isfinite(values).all():
        raise ValueError('All monthly temperatures must be finite')
    return float(np.average(values-273.15,weights=dates.days_in_month))


def calendar_day(dates):
    parsed=pd.to_datetime(dates,errors='raise')
    return np.asarray([pd.Timestamp(2001,d.month,d.day).dayofyear for d in parsed],float)


def fit_sowing_rf(training,*,cutoff='2012-12-31',trees=400):
    source=training.copy()
    source['sowing_date']=pd.to_datetime(source.sowing_date,errors='raise')
    source=source.loc[source.sowing_date.le(pd.Timestamp(cutoff))].copy()
    valid=np.isfinite(source[FEATURES].to_numpy(float)).all(axis=1)
    source=source.loc[valid]
    if not len(source) or source.duplicated(['station_id','sowing_date']).any():
        raise ValueError('Nonempty unique observed sowing records required')
    if not source.annual_temperature_year.eq(source.sowing_date.dt.year-1).all():
        raise ValueError('Temperature must represent the preceding calendar year')
    x=source[FEATURES].to_numpy(float)
    y=calendar_day(source.sowing_date)
    weight=1/source.groupby('station_id').station_id.transform('size').to_numpy(float)
    model=RandomForestRegressor(n_estimators=trees,max_depth=12,min_samples_leaf=20,
        max_features=1.,random_state=20261022,n_jobs=4)
    model.fit(x,y,sample_weight=weight/weight.mean())
    order=np.argsort(y,kind='stable');cumulative=np.cumsum(weight[order])
    median=float(y[order[np.searchsorted(cumulative,.5*cumulative[-1])]])
    return dict(estimator=model,feature_names=FEATURES.copy(),training_rows=len(source),
        training_station_ids=sorted(source.station_id.unique().astype(int).tolist()),
        training_dates=source.sowing_date.dt.strftime('%Y-%m-%d').tolist(),
        maximum_training_date=str(source.sowing_date.max().date()),cutoff=cutoff,
        lower=x.min(axis=0).tolist(),upper=x.max(axis=0).tolist(),
        median_baseline_day=median,target='Sowing day on nonleap2001calendar; actual date remains estimated')


def predict_sowing_rf(fitted,queries):
    if any(name in queries for name in ['observed_percent','observed_sowing_date','sowing_date']):
        raise ValueError('Query labels must be withheld')
    if not queries.annual_temperature_year.eq(queries.season_year-2).all():
        raise ValueError('Temperature must represent the complete calendar year preceding autumn sowing')
    x=queries[FEATURES].to_numpy(float)
    if not np.isfinite(x).all() or not queries.field_id.is_unique:
        raise ValueError('Complete finite predictors and unique field identities required')
    year=queries.season_year.to_numpy(int)-1
    if not np.all(year>pd.Timestamp(fitted['maximum_training_date']).year):
        raise ValueError('Application sowing must follow every fitted sowing observation')
    day=fitted['estimator'].predict(x)
    rounded=np.clip(np.rint(day).astype(int),1,365)
    canonical=pd.Timestamp('2001-01-01')+pd.to_timedelta(rounded-1,unit='D')
    dates=[pd.Timestamp(int(y),d.month,d.day) for y,d in zip(year,canonical)]
    result=queries.copy()
    result['estimated_sowing_day_nonleap']=day
    result['estimated_sowing_date']=[str(d.date()) for d in dates]
    result['sowing_basis']='rf_estimated_not_observed'
    result['outside_temperature_support']=(x[:,0]<fitted['lower'][0])|(x[:,0]>fitted['upper'][0])
    result['outside_latitude_support']=(x[:,1]<fitted['lower'][1])|(x[:,1]>fitted['upper'][1])
    return result
