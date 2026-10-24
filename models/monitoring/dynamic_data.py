"""Daily causal sequence arrays for native-code disease-state experiments."""
import numpy as np
import pandas as pd


def build_sequences(weather, fields, observations, queries):
    """Align queries; history and forcing on a query day are consumed afterwards."""
    if 'observed_percent' in queries:
        raise ValueError('Current query labels must be absent')
    if not len(queries):
        raise ValueError('Nonempty queries required')
    required=['global_target_index','field_id','site_id','source','leaf_index','date']
    obs=observations.copy(); q=queries.copy(); w=weather.copy(); f=fields.copy()
    for frame in [obs,q]:
        if frame[required].isna().any().any() or frame.global_target_index.duplicated().any():
            raise ValueError('Complete unique observation identities required')
        if not frame.leaf_index.isin(range(7)).all() or not frame.source.isin(['BASF','Corteva']).all():
            raise ValueError('Native source and leaf codes required')
        frame['date']=pd.to_datetime(frame.date,format='mixed',errors='raise').dt.normalize()
    if not np.isfinite(obs.observed_percent).all() or not obs.observed_percent.between(0,100).all():
        raise ValueError('Finite percentage observations required')
    if obs.duplicated(['field_id','leaf_index','date']).any() or f.field_id.duplicated().any():
        raise ValueError('Duplicate field or daily leaf observation')
    w['date']=pd.to_datetime(w.date,format='mixed',errors='raise').dt.normalize()
    if w.duplicated(['location_id','date']).any():
        raise ValueError('Duplicate weather day')
    f=f.set_index('field_id'); wg={k:g.set_index('date') for k,g in w.groupby('location_id')}
    field_ids=sorted(q.field_id.unique()); n=len(field_ids)
    lengths=[]; starts=[]; finals=[]
    for field in field_ids:
        start=pd.Timestamp(f.loc[field,'sowing_date']).normalize()
        final=q.loc[q.field_id.eq(field),'date'].max()
        if start >= q.loc[q.field_id.eq(field),'date'].min():
            raise ValueError('Queries require earlier post-sowing forcing')
        starts.append(start); finals.append(final); lengths.append((final-start).days+1)
    size=max(lengths)
    result=dict(field_ids=field_ids, weather=np.zeros((n,size,6),np.float32),
        valid_weather=np.zeros((n,size),bool), observed=np.zeros((n,size,7),np.float32),
        observed_mask=np.zeros((n,size,7),bool), static=np.zeros((n,3),np.float32),
        sources=np.zeros(n,np.int64), history_ids=[], starts=[str(x.date()) for x in starts],
        query_ids=q.global_target_index.to_numpy(np.int64), query_field=np.zeros(len(q),np.int64),
        query_day=np.zeros(len(q),np.int64), query_leaf=q.leaf_index.to_numpy(np.int64))
    for i,field in enumerate(field_ids):
        meta=f.loc[field]; here=q.field_id.eq(field); context=obs.loc[obs.field_id.eq(field)]
        if (not q.loc[here,'source'].eq(meta.source).all() or not q.loc[here,'site_id'].eq(meta.site_id).all()
                or not context.source.eq(meta.source).all() or not context.site_id.eq(meta.site_id).all()):
            raise ValueError('Conflicting field/site/source metadata')
        if meta.site_id not in wg:
            raise ValueError('Missing site weather')
        dates=pd.date_range(starts[i],finals[i]-pd.Timedelta(days=1))
        raw=wg[meta.site_id].reindex(dates)[['tmean_c','rh_mean_pct','rain_mm']].to_numpy(float)
        if not np.isfinite(raw).all() or np.any((raw[:,1]<0)|(raw[:,1]>100)) or np.any(raw[:,2]<0):
            raise ValueError('Incomplete or invalid earlier weather')
        day=np.asarray(dates.dayofyear)
        result['weather'][i,:len(dates)]=np.column_stack([(raw[:,0]-12)/12,(raw[:,1]-80)/20,
            np.log1p(raw[:,2])/2,np.sin(2*np.pi*day/365.25),np.cos(2*np.pi*day/365.25),
            np.maximum(raw[:,0],0)/18])
        result['valid_weather'][i,:len(dates)]=True
        context=context.loc[context.date.lt(finals[i])]
        if context.date.lt(starts[i]).any():
            raise ValueError('Observation before season start')
        di=(context.date-starts[i]).dt.days.to_numpy(int); li=context.leaf_index.to_numpy(int)
        result['observed'][i,di,li]=context.observed_percent.to_numpy()/100
        result['observed_mask'][i,di,li]=True
        result['history_ids'].append(context.global_target_index.astype(int).tolist())
        result['static'][i]=[(float(meta.latitude)-50)/10,float(meta.longitude)/10,float(meta.source=='Corteva')]
        result['sources'][i]=int(meta.source=='Corteva')
        result['query_field'][here.to_numpy()]=i
        result['query_day'][here.to_numpy()]=(q.loc[here,'date']-starts[i]).dt.days.to_numpy()
    return result


def subset_sequences(batch, field_ids):
    """Select complete field sequences without sharing any other field state."""
    keep=[batch['field_ids'].index(x) for x in field_ids]
    lookup={old:new for new,old in enumerate(keep)}
    mask=np.isin(batch['query_field'],keep)
    out={k:v for k,v in batch.items() if k not in ['field_ids','history_ids','starts']}
    for key in ['weather','valid_weather','observed','observed_mask','static','sources']:
        out[key]=batch[key][keep].copy()
    for key in ['query_ids','query_day','query_leaf']:
        out[key]=batch[key][mask].copy()
    out['query_field']=np.array([lookup[x] for x in batch['query_field'][mask]],np.int64)
    for key in ['field_ids','history_ids','starts']:
        out[key]=[batch[key][x] for x in keep]
    return out
