"""Anchor prediction to strictly earlier native-code field observations."""
import numpy as np
import pandas as pd
from analysis.final_study.metrics import hierarchical_weights
from .models import _category,_estimator


def field_history_anchor(history,queries):
    h=history.reset_index(drop=True)
    codes=queries.leaf_index.to_numpy(int)
    if len(h)!=len(codes) or not queries.leaf_index.isin(range(7)).all():
        raise ValueError('Aligned native leaf codes required')
    available=h[[f'other_code_{i}_available' for i in range(7)]].to_numpy(float)
    severity=h[[f'other_code_{i}_severity' for i in range(7)]].to_numpy(float)
    age=h[[f'other_code_{i}_age' for i in range(7)]].to_numpy(float)
    same=h.prior_assessment_available.to_numpy(float)
    if (not np.isfinite(np.column_stack([available,severity,age,same])).all()
        or not np.isin(available,[0,1]).all() or not np.isin(same,[0,1]).all()
        or np.any((severity<0)|(severity>1)) or np.any(age<0)):
        raise ValueError('Finite bounded history and nonnegative ages required')
    if available[np.arange(len(codes)),codes].any():
        raise ValueError('Other-leaf context cannot contain the query code')
    mask=available.astype(bool)
    recent=np.where(mask,age,np.inf).min(axis=1)
    latest=mask&np.isclose(age,recent[:,None],rtol=0,atol=1e-12)
    delta=np.arange(7)[None,:]-codes[:,None]
    distance=np.abs(delta)
    nearest=np.where(latest,distance,np.inf).min(axis=1)
    chosen=latest&(distance==nearest[:,None])
    count=chosen.sum(axis=1)
    anchor=np.divide((chosen*severity).sum(axis=1),count,out=np.zeros(len(h)),where=count>0)
    signed=np.divide((chosen*delta).sum(axis=1),count,out=np.zeros(len(h)),where=count>0)
    chosen_age=np.where(count>0,recent,0)
    chosen_distance=np.where(count>0,nearest,0)
    has_same=same==1
    anchor=np.where(has_same,h.latest_severity_fraction.to_numpy(float),anchor)
    chosen_age=np.where(has_same,h.days_since_latest_scaled.to_numpy(float),chosen_age)
    signed=np.where(has_same,0,signed)
    chosen_distance=np.where(has_same,0,chosen_distance)
    has_any=has_same|(count>0)
    features=np.column_stack([anchor,has_any,chosen_age,signed/6,chosen_distance/6])
    if not np.isfinite(features).all() or np.any((anchor<0)|(anchor>1)):
        raise ValueError('Finite bounded field-history anchor required')
    return anchor,features


def history_groups(history,structure):
    if structure=='field_residual':
        return {'all':np.ones(len(history),bool)}
    if structure!='field_split':
        raise ValueError('Registered field-history structure required')
    available=(history.prior_assessment_available.to_numpy()>0)|(history.other_leaf_available.to_numpy()>0)
    return {'none':~available,'history':available}


def _baseline(fitted,history,frame):
    anchor,features=field_history_anchor(history,frame)
    fallback=np.asarray([fitted['means'].get(key,fitted['pooled']) for key in _category(frame)])
    return np.where(features[:,1]>0,anchor,fallback),features


def fit_field_history(X,H,training,spec):
    x=np.asarray(X,float);h=H.reset_index(drop=True);frame=training.reset_index(drop=True)
    if len(x)!=len(frame) or len(h)!=len(frame) or not np.isfinite(x).all():
        raise ValueError('Complete aligned predictors required')
    y=frame.observed_percent.to_numpy(float)/100
    if not np.isfinite(y).all() or np.any((y<0)|(y>1)):
        raise ValueError('Bounded observed severity required')
    w=hierarchical_weights(frame);w=w/w.mean()
    pooled=float(np.average(y,weights=w));category=_category(frame)
    fitted=dict(spec=dict(spec),pooled=pooled,means={},models={},history_names=h.columns.tolist(),
        n_features=x.shape[1],training_ids=frame.global_target_index.astype(int).tolist())
    for key in np.unique(category):
        mask=category==key
        fitted['means'][key]=float((np.sum(w[mask]*y[mask])+10*pooled)/(w[mask].sum()+10))
    baseline,extra=_baseline(fitted,h,frame)
    design=np.column_stack([x,h.to_numpy(float),extra])
    for label,mask in history_groups(h,spec['structure']).items():
        if not mask.any():
            fitted['models'][label]=None;continue
        target=y if label=='none' else y-baseline
        head_weights=w[mask]/w[mask].mean()
        estimator=_estimator(spec)
        estimator.fit(design[mask],target[mask],sample_weight=head_weights)
        fitted['models'][label]=estimator
    return fitted


def predict_field_history(fitted,X,H,queries):
    x=np.asarray(X,float);h=H.reset_index(drop=True)
    if x.shape!=(len(queries),fitted['n_features']) or h.columns.tolist()!=fitted['history_names']:
        raise ValueError('Aligned query predictor schema required')
    baseline,extra=_baseline(fitted,h,queries)
    design=np.column_stack([x,h.to_numpy(float),extra]);prediction=baseline.copy()
    for label,mask in history_groups(h,fitted['spec']['structure']).items():
        estimator=fitted['models'][label]
        if mask.any() and estimator is not None:
            result=estimator.predict(design[mask])
            prediction[mask]=result if label=='none' else baseline[mask]+result
    if not np.isfinite(prediction).all():
        raise ValueError('Finite prediction required')
    return 100*np.clip(prediction,0,1)


def fit_ensemble(X,H,training,spec,members=20,fraction=.8,seed=20261022):
    clusters=np.asarray(sorted(training.coordinate_year.unique()))
    count=max(1,int(np.ceil(len(clusters)*fraction)))
    rng=np.random.default_rng(seed);fitted=[]
    for _ in range(members):
        selected=rng.choice(clusters,size=count,replace=False)
        positions=np.flatnonzero(training.coordinate_year.isin(selected))
        fitted.append(fit_field_history(np.asarray(X)[positions],H.iloc[positions],training.iloc[positions],spec))
    return dict(members=fitted,training_ids=training.global_target_index.astype(int).tolist(),
                seed=seed,fraction=fraction,clusters_per_member=count)


def predict_ensemble(fitted,X,H,queries):
    return np.mean([predict_field_history(member,X,H,queries) for member in fitted['members']],axis=0)
