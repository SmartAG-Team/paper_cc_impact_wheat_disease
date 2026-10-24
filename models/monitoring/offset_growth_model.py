"""Boosted conditional growth with a query-specific observed-severity offset."""
import numpy as np
from scipy.special import expit,logit
import lightgbm as lgb


def _offset(baseline,link):
    b=np.asarray(baseline,float)/100
    if not np.isfinite(b).all() or np.any((b<0)|(b>1)):
        raise ValueError('Finite bounded disease anchors required')
    if link not in ['logit','linear']:raise ValueError('Registered growth link required')
    return logit(np.clip(b,.005,.995)) if link=='logit' else b


def fit_growth(x,y,baseline,weights,link='logit',rounds=400,normalize_weights=True):
    x=np.asarray(x,float);y=np.asarray(y,float)/100;w=np.asarray(weights,float)
    if x.ndim!=2 or x.shape[0]!=len(y) or y.shape!=w.shape or y.shape!=np.shape(baseline):
        raise ValueError('Aligned growth-model training arrays required')
    if not np.isfinite(np.r_[x.ravel(),y,w]).all() or np.any(w<=0) or np.any((y<0)|(y>1)):
        raise ValueError('Finite bounded positive-weight growth training data required')
    offset=_offset(baseline,link)
    data=lgb.Dataset(x,label=y,weight=w/w.mean() if normalize_weights else w,init_score=offset)
    params=dict(objective='cross_entropy' if link=='logit' else 'regression',learning_rate=.03,
        num_leaves=15,max_depth=4,min_data_in_leaf=8,max_bin=31,lambda_l2=8.,
        feature_fraction=.9,seed=20261022,verbosity=-1,num_threads=1,deterministic=True,force_col_wise=True)
    model=lgb.train(params,data,num_boost_round=rounds)
    return dict(link=link,model_string=model.model_to_string(),params=params,rounds=rounds,features=x.shape[1])


def predict_growth(fitted,x,baseline):
    x=np.asarray(x,float);offset=_offset(baseline,fitted['link'])
    if x.shape!=(len(offset),fitted['features']) or not np.isfinite(x).all():
        raise ValueError('Finite aligned growth query predictors required')
    model=lgb.Booster(model_str=fitted['model_string'])
    latent=offset+model.predict(x,raw_score=True,num_threads=1)
    value=expit(latent) if fitted['link']=='logit' else latent
    return 100*np.clip(value,0,1)
