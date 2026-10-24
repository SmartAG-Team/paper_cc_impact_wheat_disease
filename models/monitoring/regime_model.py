"""Jointly iterated full-range disease experts and predictor-dependent routing."""
import numpy as np
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingRegressor


def _head(config,gate=False):
    return HistGradientBoostingRegressor(max_iter=config['max_iter']//2 if gate else config['max_iter'],
        learning_rate=.05,max_depth=2 if gate else 3,max_leaf_nodes=7 if gate else 15,
        min_samples_leaf=config['min_samples_leaf'],l2_regularization=2.,early_stopping=False,random_state=20261022)


def fit_regime(x,y,baseline,weights,config=None):
    config=dict(dict(iterations=4,max_iter=160,min_samples_leaf=8),**(config or {}))
    x=np.asarray(x,float);y=np.asarray(y,float);baseline=np.asarray(baseline,float);w=np.asarray(weights,float)
    if not np.isfinite(x).all() or not np.isfinite(np.r_[y,baseline,w]).all() or np.any(w<=0):
        raise ValueError('Finite aligned positive-weight regime training inputs required')
    if x.shape[0]!=len(y) or y.shape!=baseline.shape or y.shape!=w.shape or np.any((y<0)|(y>100)):
        raise ValueError('Aligned bounded regime observations required')
    w=w/w.mean();labels=np.digitize(y,[10.,50.])
    responsibility=.85*np.eye(3)[labels]+.05
    fitted=dict(config=config,experts=[],gates=[],sigma=np.full(3,15.),trace=[])
    for iteration in range(config['iterations']):
        experts=[];gates=[];means=[];prob=[];sigmas=[]
        for k in range(3):
            head=_head(config);head.fit(x,y-baseline,sample_weight=w*responsibility[:,k])
            mu=np.clip(baseline+head.predict(x),0,100)
            variance=np.average((y-mu)**2,weights=w*responsibility[:,k])
            gate=_head(config,gate=True);gate.fit(x,responsibility[:,k],sample_weight=w)
            experts.append(head);gates.append(gate);means.append(mu)
            prob.append(np.maximum(gate.predict(x),.01));sigmas.append(max(8.,np.sqrt(variance)))
        means=np.column_stack(means);prob=np.column_stack(prob);prob/=prob.sum(axis=1,keepdims=True)
        sigma=np.asarray(sigmas)
        logdensity=-.5*((y[:,None]-means)/sigma)**2-np.log(sigma)
        responsibility=.97*softmax(np.log(prob)+logdensity,axis=1)+.01
        fitted.update(experts=experts,gates=gates,sigma=sigma)
        fitted['trace'].append(dict(iteration=iteration+1,
            training_rmse=float(np.sqrt(np.average((np.sum(prob*means,axis=1)-y)**2,weights=w))),
            sigma=sigma.tolist(),mean_responsibility=np.average(responsibility,axis=0,weights=w).tolist()))
    fitted['features']=x.shape[1]
    return fitted


def predict_regime(fitted,x,baseline,return_probabilities=False):
    x=np.asarray(x,float);baseline=np.asarray(baseline,float)
    if x.shape!=(len(baseline),fitted['features']) or not np.isfinite(x).all() or not np.isfinite(baseline).all():
        raise ValueError('Finite aligned regime predictors required')
    means=np.column_stack([np.clip(baseline+head.predict(x),0,100) for head in fitted['experts']])
    prob=np.column_stack([np.maximum(gate.predict(x),.01) for gate in fitted['gates']])
    prob/=prob.sum(axis=1,keepdims=True)
    prediction=np.sum(prob*means,axis=1)
    return (prediction,prob) if return_probabilities else prediction
