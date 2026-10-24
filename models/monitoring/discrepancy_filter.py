"""Causal Gaussian discrepancy correction of an honestly cross-fitted baseline."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.special import expit,logit


def _prepare(frame,baseline,link):
    if link not in ['linear','logit']:raise ValueError('Registered discrepancy link required')
    base=np.asarray(baseline,float)/100;y=frame.observed_percent.to_numpy(float)/100
    if len(base)!=len(frame) or not np.isfinite(np.r_[base,y]).all() or np.any((base<0)|(base>1)):
        raise ValueError('Finite aligned percentage baselines and observations required')
    b=logit(np.clip(base,.005,.995)) if link=='logit' else base
    residual=(logit(np.clip(y,.005,.995)) if link=='logit' else y)-b
    days=pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
    codes=frame.leaf_index.to_numpy(int);groups=[]
    for _,idx in frame.reset_index(drop=True).groupby('field_id',sort=False).groups.items():
        ix=np.asarray(idx);d=days[ix]
        groups.append(dict(ix=ix,lag=np.abs(d[:,None]-d[None,:]).astype(float),
            same_code=codes[ix,None]==codes[None,ix],
            queries=[(np.flatnonzero(d==date),np.flatnonzero(d<date)) for date in np.unique(d)]))
    return dict(base=b,raw_base=base,residual=residual,groups=groups,link=link)


def _predict(parameters,prepared):
    shared,code,t_shared,t_code=np.exp(parameters);correction=np.zeros(len(prepared['base']))
    has_history=np.zeros(len(prepared['base']),bool)
    for group in prepared['groups']:
        distance=np.sqrt(3)*group['lag']/t_shared
        covariance=shared**2*(1+distance)*np.exp(-distance)
        covariance+=code**2*group['same_code']*np.exp(-group['lag']/t_code)
        for query,past in group['queries']:
            if not len(past):continue
            has_history[group['ix'][query]]=True
            covariance_past=covariance[np.ix_(past,past)]+np.eye(len(past))
            alpha=cho_solve(cho_factor(covariance_past,lower=True,check_finite=False),
                prepared['residual'][group['ix'][past]],check_finite=False)
            correction[group['ix'][query]]=covariance[np.ix_(query,past)]@alpha
    latent=prepared['base']+correction
    value=expit(latent) if prepared['link']=='logit' else np.clip(latent,0,1)
    value=np.where(has_history,value,prepared['raw_base'])
    return 100*value


def predict_discrepancy(parameters,frame,baseline,link):
    return _predict(parameters,_prepare(frame,baseline,link))


def fit_discrepancy(frame,baseline,weights,link):
    prepared=_prepare(frame,baseline,link);w=np.asarray(weights,float);w=w/w.sum()
    target=frame.observed_percent.to_numpy(float)/100;initial=np.log([1.,1.,30.,30.])
    trace=[]
    def objective(parameters):
        error=_predict(parameters,prepared)/100-target
        value=float(np.dot(w,error**2)+1e-5*np.sum((parameters-initial)**2))
        trace.append(value);return value
    bounds=np.log([[.01,100.],[.01,100.],[2.,180.],[2.,180.]])
    result=minimize(objective,initial,method='L-BFGS-B',bounds=bounds,
        options=dict(maxiter=60,ftol=1e-10))
    if not np.isfinite(result.fun):raise RuntimeError('Nonfinite discrepancy fit')
    return dict(parameters=result.x,link=link,converged=bool(result.success),
        message=str(result.message),iterations=result.nit,objective=float(result.fun),trace=trace,
        fitting='Weighted prequential MSE on inner-coordinate-held-out baseline predictions; unit nugget fixes covariance scale')
