"""Group-separated crop healthy-area/yield response estimation outside model/."""
import numpy as np


def _inputs(x,y,groups):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);groups=np.asarray(groups)
    if x.ndim!=1 or x.shape!=y.shape or x.shape!=groups.shape or not len(x):
        raise ValueError('Aligned one-dimensional predictor, response and group arrays are required.')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x<0) or not np.any(x>0):
        raise ValueError('Finite response and nonnegative, identifiable healthy-area losses are required.')
    if any(value is None or str(value).strip() in ('','nan') for value in groups):
        raise ValueError('Each record requires an explicit validation group.')
    return x,y,groups


def fit_group_balanced_slope(x,y,groups):
    """Nonnegative origin-constrained slope with equal total group weights.

    Each published genetic background receives equal weight, divided equally
    amongst its lines. This is a declared descriptive estimator, not an
    estimate of unavailable plot-level variance or covariance.
    """
    x,y,groups=_inputs(x,y,groups)
    unique,indices,counts=np.unique(groups,return_inverse=True,return_counts=True)
    weights=1/counts[indices]
    return max(0.,float(np.sum(weights*x*y)/np.sum(weights*x*x)))


def leave_group_out(x,y,groups):
    """Predict each group using a coefficient fitted only to other groups."""
    x,y,groups=_inputs(x,y,groups)
    unique=np.unique(groups)
    if len(unique)<2:raise ValueError('At least two groups are required for group-separated prediction.')
    predicted=np.full(len(x),np.nan);slopes=np.full(len(x),np.nan)
    training_groups=np.zeros(len(x),dtype=int)
    for group in unique:
        held=groups==group;training=~held
        slope=fit_group_balanced_slope(x[training],y[training],groups[training])
        slopes[held]=slope;predicted[held]=slope*x[held]
        training_groups[held]=len(np.unique(groups[training]))
    return dict(prediction=predicted,slope=slopes,training_groups=training_groups)
