"""Population and native-code field-trajectory covariance with causal conditioning."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit


def gaussian_update(mean,history_mean,cross_covariance,history_covariance,observations):
    if not len(observations):return np.asarray(mean,float).copy()
    if not all(np.isfinite(x).all() for x in [mean,history_mean,cross_covariance,history_covariance,observations]):
        raise ValueError('Finite Gaussian conditioning inputs required')
    return mean+cross_covariance@cho_solve(cho_factor(history_covariance,lower=True),observations-history_mean)


def _metadata(frame):
    if not frame.leaf_index.isin(range(7)).all() or frame[['field_id','date']].isna().any().any():
        raise ValueError('Valid field/date/native-code metadata required')
    return dict(field=frame.field_id.to_numpy(),leaf=frame.leaf_index.to_numpy(int),
        day=pd.to_datetime(frame.date,format='mixed').to_numpy(dtype='datetime64[D]').astype(float))


def _kernel(theta,x,a,z,b):
    amplitude,field_sd,noise,length,tau=np.exp(theta[:5]);rho=expit(theta[5])
    distance=cdist(x,z,'sqeuclidean')/x.shape[1]
    population=amplitude**2*np.exp(-distance/(2*length**2))
    age=abs(a['day'][:,None]-b['day'][None,:])
    samefield=a['field'][:,None]==b['field'][None,:]
    sameleaf=a['leaf'][:,None]==b['leaf'][None,:]
    field=field_sd**2*samefield*np.exp(-age/tau)*(rho+(1-rho)*sameleaf)
    return population+field


def _features(fitted,x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=len(fitted['mean']) or not np.isfinite(x).all():
        raise ValueError('Finite aligned GP features required')
    return (np.clip((x-fitted['mean'])/fitted['scale'],-6,6)@fitted['components'].T)/fitted['component_scale']


def fit_gp(x,training,baseline,weights,max_iter=40):
    x=np.asarray(x,float);baseline=np.asarray(baseline,float);w=np.asarray(weights,float)
    y=training.observed_percent.to_numpy(float)
    if x.shape[0]!=len(y) or baseline.shape!=y.shape or w.shape!=y.shape or np.any(w<=0):
        raise ValueError('Aligned positive-weight GP training inputs required')
    if not np.isfinite(np.r_[x.ravel(),baseline,w,y]).all() or np.any((y<0)|(y>100)):
        raise ValueError('Finite bounded GP targets required')
    w=w/w.mean();normalized=w/w.sum()
    mean=np.average(x,axis=0,weights=w);scale=np.maximum(.01,np.sqrt(np.average((x-mean)**2,axis=0,weights=w)))
    raw=np.clip((x-mean)/scale,-6,6)
    _,_,v=np.linalg.svd(raw*np.sqrt(normalized[:,None]),full_matrices=False)
    components=v[:min(16,len(y)-1,x.shape[1])]
    component_scale=np.maximum(.1,np.sqrt(np.average((raw@components.T)**2,axis=0,weights=w)))
    fitted=dict(mean=mean,scale=scale,components=components,component_scale=component_scale,
        train_metadata=_metadata(training),train_ids=training.global_target_index.astype(int).tolist())
    z=_features(fitted,x);a=fitted['train_metadata'];residual=y-baseline;n=len(y)
    distance=cdist(z,z,'sqeuclidean')/z.shape[1]
    age=abs(a['day'][:,None]-a['day'][None,:]);samefield=a['field'][:,None]==a['field'][None,:]
    sameleaf=(a['leaf'][:,None]==a['leaf'][None,:]).astype(float)
    trace=[]
    def objective(theta):
        amplitude,field_sd,noise,length,tau=np.exp(theta[:5]);rho=expit(theta[5])
        pop=amplitude**2*np.exp(-distance/(2*length**2))
        time=field_sd**2*samefield*np.exp(-age/tau)
        field=time*(rho+(1-rho)*sameleaf)
        diagonal=noise**2/w
        covariance=pop+field+np.diag(diagonal+1e-6)
        factor=cho_factor(covariance,lower=True,check_finite=False)
        alpha=cho_solve(factor,residual,check_finite=False)
        value=.5*residual@alpha+np.log(np.diag(factor[0])).sum()+.5*n*np.log(2*np.pi)
        inverse=cho_solve(factor,np.eye(n),check_finite=False)
        score=inverse-np.outer(alpha,alpha)
        derivatives=[2*pop,2*field,np.diag(2*diagonal),pop*distance/length**2,
            field*age/tau,time*(1-sameleaf)*rho*(1-rho)]
        gradient=np.array([.5*np.einsum('ij,ij->',score,d) for d in derivatives])
        trace.append(dict(objective=float(value),theta=theta.tolist()))
        return float(value),gradient
    theta=np.r_[np.log([15.,15.,12.,1.,30.]),0.]
    bounds=[(np.log(.1),np.log(60)),(np.log(.1),np.log(60)),(np.log(3),np.log(45)),
        (np.log(.1),np.log(10)),(np.log(5),np.log(365)),(-4.6,4.6)]
    result=minimize(objective,theta,jac=True,method='L-BFGS-B',bounds=bounds,
        options=dict(maxiter=max_iter,ftol=1e-6,gtol=1e-4,maxls=15))
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():raise RuntimeError('Nonfinite GP optimization')
    covariance=_kernel(result.x,z,a,z,a)+np.diag(np.exp(2*result.x[2])/w+1e-6)
    factor=cho_factor(covariance,lower=True)
    fitted.update(theta=result.x,z=z,factor=factor,alpha=cho_solve(factor,residual),
        trace=trace,converged=bool(result.success),convergence=str(result.message),iterations=int(result.nit))
    return fitted


def predict_gp(fitted,x,queries,baseline,context_x,context,context_baseline,assimilate=True):
    if 'observed_percent' in queries:raise ValueError('Query labels must be absent')
    if set(queries.field_id)&set(fitted['train_metadata']['field']):
        raise ValueError('This forecast contract requires held field identities')
    q=_metadata(queries);z=_features(fitted,x);train=fitted['train_metadata'];theta=fitted['theta']
    cross=_kernel(theta,z,q,fitted['z'],train)
    mean=np.asarray(baseline,float)+cross@fitted['alpha']
    if not assimilate:return np.clip(mean,0,100)
    context=context.reset_index(drop=True);dates=pd.to_datetime(context.date,format='mixed')
    cutoff=queries.assign(_date=pd.to_datetime(queries.date,format='mixed')).groupby('field_id')._date.max()
    mask=context.field_id.isin(cutoff.index)&dates.lt(context.field_id.map(cutoff))
    index=np.flatnonzero(mask.to_numpy())
    if not len(index):return np.clip(mean,0,100)
    h=context.iloc[index];hz=_features(fitted,np.asarray(context_x)[index]);hm=_metadata(h)
    hc=_kernel(theta,hz,hm,fitted['z'],train)
    hmean=np.asarray(context_baseline)[index]+hc@fitted['alpha']
    solved=cho_solve(fitted['factor'],hc.T)
    noise=np.exp(2*theta[2]);output=mean.copy()
    for field in queries.field_id.unique():
        qi=np.flatnonzero(queries.field_id.eq(field));hi=np.flatnonzero(h.field_id.eq(field))
        if not len(hi):continue
        qz=z[qi];qq={k:v[qi] for k,v in q.items()};hh={k:v[hi] for k,v in hm.items()}
        qh=_kernel(theta,qz,qq,hz[hi],hh)-cross[qi]@solved[:,hi]
        hh_cov=_kernel(theta,hz[hi],hh,hz[hi],hh)-hc[hi]@solved[:,hi]
        hh_cov=(hh_cov+hh_cov.T)/2+np.eye(len(hi))*(noise+1e-6)
        for i,row in enumerate(qi):
            earlier=hh['day']<q['day'][row]
            if earlier.any():
                output[row]=gaussian_update(mean[row:row+1],hmean[hi][earlier],
                    qh[i:i+1,earlier],hh_cov[np.ix_(earlier,earlier)],h.observed_percent.to_numpy()[hi][earlier])[0]
    if not np.isfinite(output).all():raise RuntimeError('Nonfinite GP forecast')
    return np.clip(output,0,100)
