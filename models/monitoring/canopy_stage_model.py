"""A shared, stage-ordered crop-development classifier with original station mass."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def stage_cases(x,y,weights):
    x,y,weights=np.asarray(x,float),np.asarray(y,float),np.asarray(weights,float)
    if y.shape!=(len(x),3) or np.any((y!=0)&(y!=1)) or np.any(np.diff(y,axis=1)>0):
        raise ValueError('Three ordered binary stage-passage labels required')
    if x.ndim!=2 or not np.isfinite(x).all() or weights.shape!=(len(x),) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('Finite contexts and positive station/cycle/date weights required')
    return np.column_stack([np.repeat(x,3,axis=0),np.tile(np.arange(3),len(x))]),y.ravel(),np.repeat(weights/3,3)


def fit_canopy_stage(x,y,weights):
    design,target,mass=stage_cases(x,y,weights)
    if len(np.unique(target))!=2:
        raise ValueError('Both binary outcomes required in the auxiliary training population')
    model=HistGradientBoostingClassifier(max_iter=400,learning_rate=.05,max_depth=3,
        max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=5.,early_stopping=False,
        random_state=20261022,monotonic_cst=[0]*(design.shape[1]-1)+[-1])
    model.fit(design,target,sample_weight=mass/mass.mean())
    return dict(model=model,features=design.shape[1]-1,training_contexts=len(x),
        original_context_weight=np.asarray(weights,float),stage_case_count=3*len(x),
        encoded_case_weight=mass,encoded_weight_mean_normalization=float(mass.mean()),
        stage_index_values=[0,1,2],stage_bbch_values=[31,51,87],iterations=400)


def predict_canopy_stage(fitted,x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=fitted['features'] or not np.isfinite(x).all():
        raise ValueError('Matching finite crop-stage contexts required')
    design=np.column_stack([np.repeat(x,3,axis=0),np.tile(np.arange(3),len(x))])
    probability=fitted['model'].predict_proba(design)[:,1].reshape(-1,3)
    if np.any(np.diff(probability,axis=1)>1e-12):
        raise ValueError('Stage-passage probability ordering violated')
    return probability
