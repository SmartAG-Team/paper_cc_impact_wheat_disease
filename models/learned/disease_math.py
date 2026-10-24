"""Frozen disease feature and validated monitoring definitions."""
import math
from numbers import Integral, Real
import numpy as np
import pandas as pd


NAMES=['prior_assessment_available','latest_severity_fraction','days_since_latest_scaled',
       'prior_assessment_count_scaled','first_severity_fraction','prior_mean_severity_fraction',
       'previous_severity_change_per_30d','days_since_first_scaled']


KEYS=['field_id','endpoint_series']


def design_for_queries(context,queries):
    if (context.global_target_index.duplicated().any() or context.duplicated(KEYS+['date']).any()
            or queries.global_target_index.duplicated().any()):raise ValueError('Unique assessment and query identities required')
    context=context.reset_index(drop=True).copy();context['_date']=pd.to_datetime(context.date,errors='coerce').dt.normalize()
    dates=pd.to_datetime(queries.date,errors='coerce').dt.normalize()
    y=context.observed_percent.to_numpy(float)/100
    if (context._date.isna().any() or dates.isna().any() or not np.isfinite(y).all() or np.any((y<0)|(y>1))):
        raise ValueError('Valid dates and observed history percentages required')
    groups={}
    for identity,part in context.groupby(KEYS,sort=True):
        if any(part[name].nunique()!=1 for name in ['source','leaf_index','site_id']):raise ValueError('Source-series metadata is inconsistent')
        groups[identity]=part.sort_values('_date')
    values=np.zeros((len(queries),len(NAMES)));provenance=[]
    for position,item in enumerate(queries.itertuples(index=False)):
        part=groups.get((item.field_id,item.endpoint_series));prior_targets=[];latest_date=None
        if part is not None:
            if part.source.iloc[0]!=item.source or part.leaf_index.iloc[0]!=item.leaf_index:raise ValueError('Query/source-series mismatch')
            past=part.loc[part._date.lt(dates.iloc[position])]
            if len(past):
                severity=past.observed_percent.to_numpy(float)/100
                elapsed=(dates.iloc[position]-past._date.iloc[-1]).total_seconds()/86400
                since_first=(dates.iloc[position]-past._date.iloc[0]).total_seconds()/86400
                rate=0.
                if len(past)>=2:
                    gap=(past._date.iloc[-1]-past._date.iloc[-2]).total_seconds()/86400
                    rate=30*(severity[-1]-severity[-2])/gap
                values[position]=[1,severity[-1],elapsed/30,len(past)/10,severity[0],severity.mean(),rate,since_first/30]
                prior_targets=past.global_target_index.astype(int).tolist();latest_date=str(past.date.iloc[-1])
        provenance.append(dict(global_target_index=int(item.global_target_index),field_id=item.field_id,
            endpoint_series=item.endpoint_series,date=str(item.date),prior_targets=prior_targets,latest_prior_date=latest_date))
    if not np.isfinite(values).all():raise ValueError('Nonfinite history features')
    return values,pd.DataFrame(provenance)


def state(fitted,h,leaf):
    prior=fitted['baseline'][np.asarray(leaf,int)]
    return np.where(h[:,0]>0,h[:,1],prior) if fitted['history_conditioned'] else prior


def predict(fitted,x,h,leaf,strength):
    base=state(fitted,h,leaf)
    if fitted['kind']=='baseline':value=base
    else:
        xx=np.column_stack([x,h]) if fitted['history_conditioned'] else x
        value=base+strength*fitted['estimator'].predict(xx)
    if not np.isfinite(value).all():raise ValueError('Nonfinite rolling prediction')
    return 100*np.clip(value,0,1)


def daily_dates(frame):
    result=frame.copy();dates=pd.to_datetime(result.date,errors='coerce',format='mixed')
    if dates.isna().any():raise ValueError('Valid local calendar dates required')
    result['date']=dates.dt.strftime('%Y-%m-%d')
    return result


def history_for_queries(observed_history,queries):
    return design_for_queries(daily_dates(observed_history),daily_dates(queries))


def calibrated_predict(fitted,weather_crop_features,queries,observed_history,*,strength=None):
    if strength is None:strength=fitted.get('selected_strength')
    if strength is None or not np.isfinite(strength) or strength<0 or strength>1:
        raise ValueError('An explicit validated residual strength or registered selected_strength is required')
    h,provenance=history_for_queries(observed_history,queries)
    x=np.asarray(weather_crop_features,float)
    if x.ndim!=2 or len(x)!=len(queries) or not np.isfinite(x).all():raise ValueError('Complete query-aligned weather/crop features required')
    if 'weather_feature_names' in fitted and x.shape[1]!=len(fitted['weather_feature_names']):raise ValueError('Weather/crop feature schema differs')
    return predict(fitted,x,h,queries.leaf_index,strength),provenance


REQUIRED = [
    'global_target_index', 'field_id', 'endpoint_series', 'source',
    'site_id', 'leaf_index', 'date',
]


def validate(frame, *, observations):
    needed = REQUIRED + (['observed_percent'] if observations else [])
    if any(name not in frame for name in needed) or frame[needed].isna().any().any():
        raise ValueError('Complete field/series/site/source/date metadata required')
    try:
        leaf = frame.leaf_index.to_numpy(float)
    except (ValueError, TypeError) as error:
        raise ValueError('Recorded leaf codes must be numeric integers') from error
    if (not np.isfinite(leaf).all() or np.any(leaf != np.floor(leaf))
            or np.any((leaf < 0) | (leaf > 6))):
        raise ValueError('Recorded leaf codes must be finite integers within 0–6')
    if frame.global_target_index.duplicated().any():
        raise ValueError('Unique query/history target identities required')
    if observations:
        severity = frame.observed_percent.to_numpy(float)
        if not np.isfinite(severity).all() or np.any((severity < 0) | (severity > 100)):
            raise ValueError('Observed severity must be within 0–100 percent')


def checked_predict(fitted, weather_crop_features, queries, observed_history, *, strength=None):
    validate(queries, observations=False)
    validate(observed_history, observations=True)
    lookup = {
        key: part for key, part in observed_history.groupby(
            ['field_id', 'endpoint_series'], sort=False,
        )
    }
    for item in queries.itertuples(index=False):
        group = lookup.get((item.field_id, item.endpoint_series))
        if group is not None:
            for name in ['site_id','source','leaf_index']:
                if group[name].nunique() != 1 or group[name].iloc[0] != getattr(item, name):
                    raise ValueError('Query metadata conflicts with its recorded field series')
    x = np.asarray(weather_crop_features, float)
    names = fitted.get('weather_feature_names', [])
    if (x.ndim != 2 or len(x) != len(queries) or (names and x.shape[1] != len(names))
            or not np.isfinite(x).all()):
        raise ValueError('Complete schema-aligned weather/crop predictors required')
    leaf_columns = [f'leaf_{i}' for i in range(7)]
    if all(name in names for name in leaf_columns):
        supplied = x[:, [names.index(name) for name in leaf_columns]]
        if not np.array_equal(supplied, np.eye(7)[queries.leaf_index.to_numpy(int)]):
            raise ValueError('Leaf predictor schema conflicts with query leaf codes')
    cdf_columns = [f'crop_cdf_{stage}' for stage in [10,31,51,87]]
    if all(name in names for name in cdf_columns):
        cdf = x[:, [names.index(name) for name in cdf_columns]]
        if np.any((cdf < 0) | (cdf > 1)) or np.any(np.diff(cdf, axis=1) > 1e-10):
            raise ValueError('Ordered bounded crop CDFs required')
    return calibrated_predict(fitted, x, queries, observed_history, strength=strength)


def integer_identities(frame):
    if 'global_target_index' not in frame:
        raise ValueError('Integer target identities required')
    limits = np.iinfo(np.int64)
    identities = []
    for value in frame.global_target_index:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('Boolean target identities are invalid')
        if isinstance(value, Integral):
            identity = int(value)
        elif isinstance(value, Real) and math.isfinite(value) and value == math.trunc(value):
            identity = int(value)
        else:
            raise ValueError('Target identities must be finite numeric integers')
        if identity < limits.min or identity > limits.max:
            raise ValueError('Target identities must fit signed 64-bit integers')
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError('Unique integer target identities required')
    normalized = frame.copy()
    normalized['global_target_index'] = np.asarray(identities, dtype=np.int64)
    return normalized


def predict_queries(fitted, weather_crop_features, queries, observed_history, *, strength=None):
    return checked_predict(
        fitted, weather_crop_features, integer_identities(queries),
        integer_identities(observed_history), strength=strength,
    )


def basic_design(z, fields, frame, base_predictions=None, *, geography=False):
    fi=frame.field_index.to_numpy(int);day_raw=frame.day_index.to_numpy(float);leaf_raw=frame.leaf_index.to_numpy(float)
    if (not np.isfinite(day_raw).all() or not np.equal(day_raw,np.floor(day_raw)).all()
            or np.any(day_raw<1) or np.any(day_raw>z['features'].shape[1])):
        raise ValueError('Valid integer assessment days required')
    if (not np.isfinite(leaf_raw).all() or not np.equal(leaf_raw,np.floor(leaf_raw)).all()
            or np.any(leaf_raw<0) or np.any(leaf_raw>=7)):
        raise ValueError('Source leaf indices must lie in 0–6')
    day=day_raw.astype(int)-1;leaf=leaf_raw.astype(int)
    if np.any(fi<0) or np.any(fi>=len(z['features'])) or not z['valid'][fi,day].all():
        raise ValueError('Assessment outside valid field/day support')
    arrays=[z['features'][fi,day],np.eye(7)[leaf]]
    names=list(z['names'])+[f'leaf_{k}' for k in range(7)]
    if geography:
        if fields.field_index.duplicated().any():raise ValueError('Duplicate field registry')
        coordinates=fields.set_index('field_index').loc[fi,['latitude','longitude']].to_numpy(float)
        if (not np.isfinite(coordinates).all() or np.any(abs(coordinates[:,0])>90)
                or np.any(abs(coordinates[:,1])>180)):
            raise ValueError('Valid geographic coordinates required')
        arrays.append(coordinates/np.array([90.,180.]));names += ['latitude_scaled','longitude_scaled']
    if base_predictions is not None:
        if not np.array_equal(frame.global_target_index.to_numpy(),base_predictions.global_target_index.to_numpy()):
            raise ValueError('Base prediction target identity/order mismatch')
        values=base_predictions[['base_percent','base_occurrence']].to_numpy(float)
        if (not np.isfinite(values).all() or np.any((values[:,0]<0)|(values[:,0]>100))
                or np.any((values[:,1]<0)|(values[:,1]>1))):
            raise ValueError('Bounded base output required')
        arrays.append(values/np.array([100.,1.]));names += ['base_severity_fraction','base_occurrence']
    x=np.column_stack(arrays)
    if not np.isfinite(x).all():raise ValueError('Finite ML predictors required')
    return x,names


def design(z, frame):
    x, names = basic_design(z, None, frame)
    fi = frame.field_index.to_numpy(int)
    day = frame.day_index.to_numpy(int)-1
    daily = z['features']; lookup = {name:i for i,name in enumerate(z['names'])}
    cdf = [daily[:,:,lookup[f'crop_cdf_{stage}']] for stage in [10,31,51,87]]
    current = [value[fi,day] for value in cdf]
    extra = [1-current[0], current[0]-current[1], current[1]-current[2],
             current[2]-current[3], current[3]]
    labels = ['phase_before_10','phase_10_31','phase_31_51','phase_51_87','phase_after_87']
    for stage, value in zip([51,87], cdf[2:]):
        for lag in [7,21]:
            extra.append(value[fi,day]-value[fi,np.maximum(day-lag,0)])
            labels.append(f'crop_cdf_{stage}_change_{lag}d')
    for stage,value in zip([31,51,87], cdf[1:]):
        days = np.arange(value.shape[1])[None,:]
        # The most recent crossing is computed cumulatively, never from a
        # future crossing or a whole-season predicted event date.
        passed = np.maximum.accumulate(value>=.5,axis=1)
        first = np.minimum.accumulate(np.where(value>=.5,days,value.shape[1]),axis=1)
        extra.append(np.where(passed[fi,day],day-first[fi,day],0)/330.)
        labels.append(f'elapsed_since_predicted_{stage}_median_scaled')
    for stem in ['temperature','humidity','rain_log1p']:
        short = f'{stem}_mean_7d'; long = f'{stem}_mean_42d'
        if short in lookup and long in lookup:
            extra.append(daily[fi,day,lookup[short]]-daily[fi,day,lookup[long]])
            labels.append(f'{stem}_contrast_7d_42d')
    result = np.column_stack([x,*extra])
    if not np.isfinite(result).all():raise ValueError('Finite prefix predictors required')
    return result,names+labels


