"""Causal conditioning of joint development profiles on earlier BBCH intervals.

The likelihood is a fixed working observation model. It constrains milestone
rank, without interpreting numerical BBCH distances as physiological distances.
"""
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import expit

from .phenology_transfer import development_increments, STAGES

NAMES=([f'pass_BBCH{s}' for s in STAGES]+
       [f'mean_clock_ratio_BBCH{s}_capped4' for s in STAGES]+
       [f'sd_clock_ratio_BBCH{s}_capped4' for s in STAGES]+
       ['clock_mass_gdd','clock_mass_tp','clock_mass_tpv','mean_sowing_nonleap_day_365',
        'sd_sowing_days_30','profile_entropy_0to1','profile_ess_fraction'])


def coupled_profiles(records,weather,parameters):
    complete=[r for r in records if all(d is not None for d in r['true_dates'])]
    if not complete or any(r['END_DATE']>='2013-01-01' or not r['eligible'] or
                           r['observation_order_conflict'] for r in complete):
        raise ValueError('Ordered qualified historical complete profiles required')
    count=Counter(int(r['PEP_ID']) for r in complete)
    thresholds=[];ids=[];weights=[]
    for record in complete:
        daily=np.asarray(weather[record['row_start']:record['row_stop']],float)
        dates=pd.date_range(record['SOWING_DATE'],periods=len(daily))
        before=np.vstack([np.zeros((1,3)),np.cumsum(development_increments(
            daily,dates,record['lat'],parameters),axis=0)])
        positions=np.array([(pd.Timestamp(d)-dates[0]).days for d in record['true_dates']])
        thresholds.append(before[positions].T)
        ids.append(f'{record["PEP_ID"]}|{record["SOWING_DATE"]}')
        weights.append(1/count[int(record['PEP_ID'])])
    thresholds=np.asarray(thresholds)
    if not np.isfinite(thresholds).all() or np.any(np.diff(thresholds,axis=2)<-1e-10):
        raise ValueError('Finite ordered joint thresholds required')
    weights=np.asarray(weights);weights/=weights.sum()
    return dict(thresholds=thresholds,weight=weights,cycle_ids=ids,
                station_ids=[int(r['PEP_ID']) for r in complete],
                parameters=parameters,event_clock='Strictly before observed event date')


def rank_probability(cumulative,thresholds,scale=60.):
    cumulative=np.asarray(cumulative,float);thresholds=np.asarray(thresholds,float)
    if thresholds.shape!=(len(cumulative),4) or scale<=0 or not np.isfinite(thresholds).all():
        raise ValueError('Four finite ordered milestones and positive transition scale required')
    if not np.isfinite(cumulative).all() or np.any(np.diff(thresholds,axis=1)<-1e-10):
        raise ValueError('Finite clock and ordered thresholds required')
    passed=expit((cumulative[:,None]-thresholds)/scale)
    return np.column_stack([1-passed[:,0],passed[:,:-1]-passed[:,1:],passed[:,-1]])


def interval_update(weight,probability,lower,upper,contamination=.05):
    weight=np.asarray(weight,float);probability=np.asarray(probability,float)
    if not 0<=lower<=upper<=99 or not 0<=contamination<1:
        raise ValueError('Valid BBCH interval and contamination required')
    lo=int(np.sum(np.asarray(STAGES)<=lower));hi=int(np.sum(np.asarray(STAGES)<=upper))
    if lo==0 and hi==4:return weight.copy()
    likelihood=(1-contamination)*probability[:,lo:hi+1].sum(axis=1)+contamination*(hi-lo+1)/5
    posterior=weight*likelihood
    if not np.isfinite(posterior).all() or posterior.sum()<=0:raise ValueError('Invalid stage posterior')
    return posterior/posterior.sum()


def align_field(starts,clocks,query_dates,calendar,bank,scale=60.,contamination=.05):
    """Clocks[start][d] contains weather strictly before start + d days.

    Inputs belong to one field. Only calendar records before each query update
    the mixture; other fields never enter this function.
    """
    starts=pd.DatetimeIndex(starts);queries=pd.DatetimeIndex(query_dates)
    theta=np.asarray(bank['thresholds'],float).transpose(1,0,2)
    profiles=len(bank['weight']);nodes=len(starts);size=nodes*3*profiles
    thresholds=np.broadcast_to(theta[None],(nodes,3,profiles,4)).reshape(size,4)
    prior=np.broadcast_to(np.asarray(bank['weight'])[None,None,:]/(nodes*3),
                          (nodes,3,profiles)).reshape(-1).copy()
    np.testing.assert_allclose(prior.sum(),1,atol=1e-12)
    model_index=np.broadcast_to(np.arange(3)[None,:,None],(nodes,3,profiles)).reshape(-1)
    days=np.array([(pd.Timestamp(2001,s.month,s.day)-pd.Timestamp('2001-01-01')).days+1 for s in starts])
    sow=np.broadcast_to(days[:,None,None],(nodes,3,profiles)).reshape(-1)
    cal=calendar.copy();cal['date']=pd.to_datetime(cal.date).dt.normalize()
    cal=cal.loc[cal.date.ge(starts.min())].sort_values('date')
    if cal.date.duplicated().any():raise ValueError('Unique qualified same-field stage dates required')
    if len(cal) and (not np.isfinite(cal[['stage_from','stage_to']]).all().all() or
                    not (cal.stage_from.ge(0)&cal.stage_to.le(99)&cal.stage_from.le(cal.stage_to)).all()):
        raise ValueError('Qualified BBCH intervals required')
    events=list(cal.itertuples());cursor=0;posterior=prior.copy();latest=None
    output={key:np.zeros((len(queries),len(NAMES))) for key in ['filtered','prior']}
    output.update(filtered_probability=np.zeros((len(queries),4)),prior_probability=np.zeros((len(queries),4)),
                  stage_available=np.zeros(len(queries),bool),stage_count=np.zeros(len(queries),int),
                  stage_age_days=np.zeros(len(queries)),posterior_ess=np.zeros(len(queries)))

    def clock_at(date):
        values=[]
        for start in starts:
            day=(date-start).days
            if day<0:values.append(np.zeros(3))
            else:values.append(np.asarray(clocks[str(start.date())])[day])
        return np.broadcast_to(np.asarray(values)[:,:,None],(nodes,3,profiles)).reshape(-1)

    def summarize(weight,cumulative,rank):
        passed=np.array([np.sum(rank[:,j+1:],axis=1) for j in range(4)]).T
        ratio=np.clip(cumulative[:,None]/np.maximum(thresholds,1.),0,4)
        mean=weight@ratio;sd=np.sqrt(np.maximum(0,weight@(ratio-mean)**2))
        sow_mean=float(weight@sow);sow_sd=np.sqrt(max(0,float(weight@(sow-sow_mean)**2)))
        entropy=-np.sum(weight*np.log(np.maximum(weight,1e-300)))/np.log(size)
        ess=1/np.sum(weight**2)
        return np.r_[weight@passed,mean,sd,[weight[model_index==j].sum() for j in range(3)],
                     sow_mean/365,sow_sd/30,entropy,ess/size],ess

    for index in np.argsort(queries,kind='stable'):
        now=queries[index]
        while cursor<len(events) and events[cursor].date<now:
            event=events[cursor]
            rank=rank_probability(clock_at(event.date),thresholds,scale)
            posterior=interval_update(posterior,rank,event.stage_from,event.stage_to,contamination)
            latest=event.date;cursor+=1
        cumulative=clock_at(now);rank=rank_probability(cumulative,thresholds,scale)
        output['prior'][index],_=summarize(prior,cumulative,rank)
        output['filtered'][index],output['posterior_ess'][index]=summarize(posterior,cumulative,rank)
        output['prior_probability'][index]=output['prior'][index,:4]
        output['filtered_probability'][index]=output['filtered'][index,:4]
        output['stage_available'][index]=latest is not None
        output['stage_count'][index]=cursor
        output['stage_age_days'][index]=0 if latest is None else (now-latest).days
    return output
