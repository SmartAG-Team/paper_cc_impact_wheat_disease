"""Joint field-level response experts with causal inference of a latent field class."""
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingRegressor


def _means(fitted,x,baseline):
    return np.clip(np.asarray(baseline,float)[:,None]/100+
        np.column_stack([head.predict(x) for head in fitted['experts']]),0,1)


def fit_latent_field(x,frame,baseline,weights,config=None):
    config=dict(dict(iterations=8,max_iter=160,min_samples_leaf=12),**(config or {}))
    x=np.asarray(x,float);y=frame.observed_percent.to_numpy(float)/100
    b=np.asarray(baseline,float)/100;w=np.asarray(weights,float);w=w/w.mean()
    fields,inverse=np.unique(frame.field_id.astype(str),return_inverse=True)
    field_weight=np.bincount(inverse,weights=w)
    score=np.bincount(inverse,weights=w*(y-b))/field_weight
    labels=np.digitize(score,np.quantile(score,[1/3,2/3]));responsibility=.9*np.eye(3)[labels]+1/30
    groups=[np.flatnonzero(inverse==k) for k in range(len(fields))]
    source=[str(frame.source.iloc[ix[0]]) for ix in groups]
    fitted=dict(config=config,experts=[],sigma=np.full(3,.15),prior=np.ones(3)/3,source_prior={},trace=[],
        features=x.shape[1],training_ids=frame.global_target_index.astype(int).tolist())
    for iteration in range(config['iterations']):
        experts=[]
        for k in range(3):
            head=HistGradientBoostingRegressor(max_iter=config['max_iter'],max_depth=2,max_leaf_nodes=7,
                min_samples_leaf=config['min_samples_leaf'],learning_rate=.05,l2_regularization=5.,
                early_stopping=False,random_state=20261022)
            head.fit(x,y-b,sample_weight=w*responsibility[inverse,k]);experts.append(head)
        fitted['experts']=experts;means=_means(fitted,x,baseline)
        fitted['sigma']=np.maximum(.08,np.sqrt(np.sum(w[:,None]*responsibility[inverse]*(y[:,None]-means)**2,axis=0)/
            np.sum(w[:,None]*responsibility[inverse],axis=0)))
        prior_mass=(field_weight[:,None]*responsibility).sum(axis=0)
        pseudo=field_weight.mean()
        fitted['prior']=(prior_mass+pseudo)/(prior_mass.sum()+3*pseudo)
        for value in np.unique(source):
            mask=np.asarray(source)==value;mass=(field_weight[mask,None]*responsibility[mask]).sum(axis=0)
            fitted['source_prior'][value]=(mass+pseudo)/(mass.sum()+3*pseudo)
        logdensity=-.5*((y[:,None]-means)/fitted['sigma'])**2-np.log(fitted['sigma'])
        updated=[]
        for key,ix in enumerate(groups):
            prior=fitted['source_prior'][source[key]]
            effective=min(4,frame.date.iloc[ix].nunique())
            likelihood=effective*np.average(logdensity[ix],axis=0,weights=w[ix])
            updated.append(.97*softmax(np.log(prior)+likelihood)+.01)
        responsibility=np.asarray(updated)
        fitted['trace'].append(dict(iteration=iteration+1,prior=fitted['prior'].tolist(),
            sigma_pp=(100*fitted['sigma']).tolist(),
            training_posterior_rmse=float(100*np.sqrt(np.average((np.sum(responsibility[inverse]*means,axis=1)-y)**2,weights=w)))))
    fitted.update(field_ids=fields.tolist(),field_responsibility=responsibility,training_weight=w)
    return fitted


def predict_latent_field(fitted,x,frame,baseline,forget_days=None):
    frame=frame.reset_index(drop=True);x=np.asarray(x,float);baseline=np.asarray(baseline,float)
    means=_means(fitted,x,baseline);y=frame.observed_percent.to_numpy(float)/100
    dates=pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
    probabilities=np.empty_like(means);prediction=baseline.copy()
    logdensity=-.5*((y[:,None]-means)/fitted['sigma'])**2-np.log(fitted['sigma'])
    for _,index in frame.groupby('field_id',sort=False).groups.items():
        ix=np.asarray(index);prior=fitted['source_prior'].get(str(frame.source.iloc[ix[0]]),fitted['prior'])
        for now in np.unique(dates[ix]):
            query=ix[dates[ix]==now];past=ix[dates[ix]<now]
            if not len(past):probabilities[query]=prior;continue
            known=dates[past];unique,counts=np.unique(known,return_counts=True)
            age=np.exp(-(now-unique)/forget_days) if forget_days is not None else np.ones(len(unique))
            # Each assessment day contributes unit mass split among its observed
            # native codes. Correlated histories supply at most four evidence units.
            date_weight=dict(zip(unique,age));date_count=dict(zip(unique,counts))
            weight=np.array([date_weight[d]/date_count[d] for d in known])
            likelihood=min(4,float(age.sum()))*np.average(logdensity[past],axis=0,weights=weight)
            posterior=.97*softmax(np.log(prior)+likelihood)+.01
            probabilities[query]=posterior
            prediction[query]=100*np.sum(posterior*means[query],axis=1)
    if not np.isfinite(prediction).all():raise RuntimeError('Nonfinite latent-field predictions')
    return prediction,probabilities
