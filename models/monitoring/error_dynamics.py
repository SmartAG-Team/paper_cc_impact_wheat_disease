"""Learn nonlinear dynamics of strictly earlier, honestly predicted field errors."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def error_states(frame,baseline):
    frame=frame.reset_index(drop=True)
    error=(frame.observed_percent.to_numpy(float)-np.asarray(baseline,float))/100
    dates=pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
    codes=frame.leaf_index.to_numpy(int)
    names=['error_history_available','error_count_10','error_days_10','latest_error_age_30',
        'latest_day_error_mean','latest_day_error_min','latest_day_error_max','all_error_mean','all_error_std',
        'same_error_available','same_latest_error_fraction','same_error_age_30','same_error_count_10','same_error_change_per30']
    names += [f'{kind}_error_ewma_{horizon}d_{summary}' for kind in ['field','same']
        for horizon in [7,21,42] for summary in ['mean','absolute']]
    names += [f'code{k}_error_{kind}' for k in range(7) for kind in ['available','latest','age30']]
    result=np.zeros((len(frame),len(names)))
    for _,index in frame.groupby('field_id',sort=False).groups.items():
        index=np.asarray(index);index=index[np.argsort(dates[index],kind='stable')]
        for q in index:
            past=index[dates[index]<dates[q]]
            if not len(past):continue
            last=past[dates[past]==dates[past].max()];same=past[codes[past]==codes[q]]
            result[q,:9]=[1,len(past)/10,len(np.unique(dates[past]))/10,(dates[q]-dates[last[0]])/30,
                error[last].mean(),error[last].min(),error[last].max(),error[past].mean(),error[past].std()]
            if len(same):
                rate=30*(error[same[-1]]-error[same[-2]])/(dates[same[-1]]-dates[same[-2]]) if len(same)>1 else 0.
                result[q,9:14]=[1,error[same[-1]],(dates[q]-dates[same[-1]])/30,len(same)/10,rate]
            at=14
            for ix in [past,same]:
                for horizon in [7,21,42]:
                    if len(ix):
                        weight=np.exp(-(dates[q]-dates[ix])/horizon)
                        result[q,at:at+2]=[np.average(error[ix],weights=weight),np.average(np.abs(error[ix]),weights=weight)]
                    at+=2
            for k in range(7):
                part=past[codes[past]==k]
                if len(part):result[q,at:at+3]=[1,error[part[-1]],(dates[q]-dates[part[-1]])/30]
                at+=3
    if not np.isfinite(result).all():raise ValueError('Finite causal error states required')
    return result,names


def fit_error_head(raw,states,frame,baseline,weights,max_iter=300):
    warm=np.asarray(states)[:,0]>0;y=frame.observed_percent.to_numpy(float)
    design=np.column_stack([raw,states,np.asarray(baseline)/100])
    fitted=dict(features=design.shape[1],model=None)
    if not warm.any():return fitted
    w=np.asarray(weights,float)[warm];w/=w.mean()
    model=HistGradientBoostingRegressor(max_iter=max_iter,max_depth=2,max_leaf_nodes=7,
        min_samples_leaf=12,learning_rate=.03,l2_regularization=10,early_stopping=False,random_state=20261022)
    model.fit(design[warm],(y[warm]-np.asarray(baseline)[warm])/100,sample_weight=w)
    fitted['model']=model;return fitted


def predict_error_head(fitted,raw,states,baseline,shrinkage=1.):
    baseline=np.asarray(baseline,float);result=baseline.copy();warm=np.asarray(states)[:,0]>0
    if fitted['model'] is not None and warm.any():
        design=np.column_stack([raw,states,baseline/100])
        result[warm]=np.clip(baseline[warm]+100*shrinkage*fitted['model'].predict(design[warm]),0,100)
    return result
