"""Hierarchy-weighted ICL through explicit, auditable context multiplicities."""
import numpy as np


def weighted_context(x,y,weights,size,seed):
    x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(weights,float)
    if x.ndim!=2 or y.ndim!=1 or y.shape!=w.shape or len(x)!=len(y) or not len(y):
        raise ValueError('Aligned original context rows and scientific weights required')
    if not np.isfinite(np.r_[x.ravel(),y,w]).all() or np.any(w<=0) or not isinstance(size,int) or size<1:
        raise ValueError('Finite positive weights and context size required')
    cumulative=np.cumsum(w/w.sum());cumulative[-1]=1.
    offset=np.random.default_rng(seed).random()
    positions=(offset+np.arange(size))/size
    indices=np.searchsorted(cumulative,positions,side='right')
    return x[indices],y[indices],indices
