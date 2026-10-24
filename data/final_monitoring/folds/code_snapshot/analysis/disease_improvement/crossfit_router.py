"""Coordinate-held-out expert learning and a regularized conditional mixture."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


def coordinate_folds(frame,folds=5,seed=20261022):
    coordinates=np.asarray([f'{a:.8f}|{b:.8f}' for a,b in zip(frame.latitude,frame.longitude)])
    groups=np.unique(coordinates)
    if len(groups)<2:raise ValueError('At least two training coordinates required')
    rng=np.random.default_rng(seed);rng.shuffle(groups)
    for held in np.array_split(groups,min(folds,len(groups))):
        test=np.flatnonzero(np.isin(coordinates,held))
        train=np.flatnonzero(~np.isin(coordinates,held))
        yield train,test


def fit_router(x,experts,y,weights,ridge=.001):
    x=np.asarray(x,float);p=np.asarray(experts,float)/100;y=np.asarray(y,float)/100
    w=np.asarray(weights,float)
    if x.ndim!=2 or p.ndim!=2 or len(x)!=len(y) or len(p)!=len(y) or w.shape!=y.shape:
        raise ValueError('Aligned inner-held-out expert predictions required')
    if not np.isfinite(np.r_[x.ravel(),p.ravel(),y,w]).all() or np.any(w<=0):
        raise ValueError('Finite positive-weight routing data required')
    w=w/w.sum();mean=np.average(x,axis=0,weights=w)
    scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=w)),.01)
    z=np.column_stack([np.ones(len(x)),np.clip((x-mean)/scale,-8,8)])
    shape=(z.shape[1],p.shape[1])
    def objective(flat):
        coefficients=flat.reshape(shape);prob=softmax(z@coefficients,axis=1)
        pred=(prob*p).sum(axis=1);error=pred-y
        loss=np.dot(w,error**2)+ridge*np.sum(coefficients**2)
        derivative=2*w[:,None]*error[:,None]*prob*(p-pred[:,None])
        gradient=z.T@derivative+2*ridge*coefficients
        return float(loss),gradient.ravel()
    result=minimize(objective,np.zeros(np.prod(shape)),jac=True,method='L-BFGS-B',
        options=dict(maxiter=300,ftol=1e-12,gtol=1e-8))
    if not np.isfinite(result.fun):raise RuntimeError('Nonfinite routing objective')
    return dict(mean=mean,scale=scale,coefficients=result.x.reshape(shape),ridge=ridge,
        converged=bool(result.success),message=str(result.message),iterations=result.nit,loss=float(result.fun))


def predict_router(fitted,x,experts):
    x=np.asarray(x,float);p=np.asarray(experts,float)
    z=np.column_stack([np.ones(len(x)),np.clip((x-fitted['mean'])/fitted['scale'],-8,8)])
    prob=softmax(z@fitted['coefficients'],axis=1)
    result=(prob*p).sum(axis=1)
    if not np.isfinite(result).all():raise ValueError('Finite routed predictions required')
    return result,prob
