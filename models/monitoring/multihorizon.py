"""Additional prediction contexts from original observed within-field intervals."""
import numpy as np
import pandas as pd

from .history import build_history
from .weather_sequence_model import design
from .infection_event_features import EVENT_NAMES
from model.seasonal_septoria.wetness import duration_exposure


def context_schedule(observations,max_extra=3):
    obs=observations.reset_index(drop=True)
    dates=pd.to_datetime(obs.date,format='mixed')
    unique={k:np.sort(dates.loc[g.index].unique()) for k,g in obs.groupby('field_id')}
    original=[];extra=[]
    for i,row in obs.iterrows():
        past=unique[row.field_id][unique[row.field_id]<dates.iloc[i]]
        record=dict(original_position=i,target_date=str(dates.iloc[i].date()),field_id=row.field_id)
        original.append(dict(record,original=True,cutoff=str(pd.Timestamp(past[-1]).date()) if len(past) else None))
        alternatives=past[:-1]
        if len(alternatives)>max_extra:
            alternatives=alternatives[np.unique(np.linspace(0,len(alternatives)-1,max_extra).astype(int))]
        for cutoff in alternatives:
            extra.append(dict(record,original=False,cutoff=str(pd.Timestamp(cutoff).date())))
    return pd.DataFrame(original+extra)


def context_weights(schedule,original_weights):
    index=schedule.original_position.to_numpy(int);original=schedule.original.to_numpy(bool)
    counts=np.bincount(index[~original],minlength=len(original_weights))
    factor=np.where(original,np.where(counts[index]>0,.5,1.),.5/np.maximum(counts[index],1))
    return np.asarray(original_weights)[index]*factor


def build_contexts(observations,arrays,history,fields,weather,max_extra=3):
    obs=observations.reset_index(drop=True);schedule=context_schedule(obs,max_extra)
    original_x,original_anchor,original_available=design(arrays,history,obs.drop(columns='observed_percent'))
    extra=schedule.loc[~schedule.original].reset_index(drop=True)
    if not len(extra):
        return dict(schedule=schedule,x=original_x,anchor=original_anchor,available=original_available,
            frame=obs.copy(),provenance=pd.DataFrame())
    dated=obs.assign(_date=pd.to_datetime(obs.date,format='mixed'))
    groups=dict(tuple(dated.groupby('field_id')));aliases={};contexts=[];queries=[];origin={}
    nextid=int(min(obs.global_target_index.min(),0))-1
    for row in extra.itertuples(index=False):
        key=(row.field_id,row.cutoff)
        if key not in aliases:
            alias=row.field_id+'|context_through='+row.cutoff;aliases[key]=alias
            past=groups[row.field_id].loc[groups[row.field_id]._date.le(pd.Timestamp(row.cutoff))].copy()
            for record in past.drop(columns='_date').to_dict('records'):
                origin[nextid]=int(record['global_target_index']);record.update(field_id=alias,global_target_index=nextid)
                contexts.append(record);nextid-=1
        q=obs.iloc[row.original_position].drop(labels='observed_percent').to_dict()
        q.update(field_id=aliases[key],global_target_index=nextid);nextid-=1;queries.append(q)
    q=pd.DataFrame(queries);context=pd.DataFrame(contexts)
    h,provenance=build_history(context,q)
    mapped=extra.original_position.to_numpy(int)
    event=np.asarray(arrays['history_events'])[mapped].copy()
    # Only these two registered event summaries depend on the disease anchor.
    schema_station_columns=event.shape[1]-len(EVENT_NAMES)-3
    columns={n:schema_station_columns+EVENT_NAMES.index(n) for n in ['gap_wet_dose','gap_thermal_units']}
    wg={k:g.set_index('date') for k,g in weather.assign(date=pd.to_datetime(weather.date,format='mixed')).groupby('location_id')}
    field_index=fields.set_index('field_id');prefix={}
    for field in extra.field_id.unique():
        meta=field_index.loc[field];start=pd.Timestamp(meta.sowing_date)
        end=pd.to_datetime(obs.loc[obs.field_id.eq(field),'date']).max()-pd.Timedelta(days=1)
        a=wg[meta.site_id].reindex(pd.date_range(start,end))[['tmean_c','tmax_c','rh_mean_pct','rain_mm']].to_numpy(float)
        if not np.isfinite(a).all():raise ValueError('Complete causal interval weather required')
        wet=duration_exposure(a[:,0],a[:,1],a[:,2],a[:,3])['exposure']
        prefix[field]=(start,np.r_[0,np.cumsum(wet)],np.r_[0,np.cumsum(np.maximum(a[:,0],0)/18)])
    anchor_dates=pd.to_datetime(provenance.latest_same_date,format='mixed').fillna(pd.to_datetime(provenance.latest_any_date,format='mixed'))
    for i,row in enumerate(extra.itertuples(index=False)):
        start,wet,thermal=prefix[row.field_id]
        first=(anchor_dates.iloc[i]-start).days+1;last=(pd.Timestamp(row.target_date)-start).days
        event[i,columns['gap_wet_dose']]=wet[last]-wet[first]
        event[i,columns['gap_thermal_units']]=thermal[last]-thermal[first]
    xa,aa,ha=design(dict(process=np.asarray(arrays['process'])[mapped],history_events=event),h,q)
    provenance['original_target_index']=obs.global_target_index.to_numpy()[mapped]
    provenance['cutoff_date']=extra.cutoff
    provenance['prior_original_target_ids']=[[origin[int(k)] for k in ids] for ids in provenance.prior_target_ids]
    provenance['context_is_new_observation']=False
    appended=obs.iloc[mapped].copy().reset_index(drop=True)
    return dict(schedule=schedule,x=np.vstack([original_x,xa]),anchor=np.r_[original_anchor,aa],
        available=np.r_[original_available,ha],frame=pd.concat([obs,appended],ignore_index=True),provenance=provenance)
