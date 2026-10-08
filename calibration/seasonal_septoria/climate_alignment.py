"""Historical-only estimation of monthly climate alignment coefficients."""

import numpy as np
from scipy.special import logit
from model.seasonal_septoria.climate_alignment import _forcing, RH_EPSILON


def monthly_means(forcing, month):
    """Means in the correction coordinates, not humid hours or wetness."""
    arrays,months = _forcing(forcing,month)
    if len(np.unique(months))!=12:
        raise ValueError('All 12 baseline months are required.')
    rh = logit(np.clip(arrays['rh_mean_pct']/100,RH_EPSILON,1-RH_EPSILON))
    return {key:np.column_stack([value[:,months==m].mean(axis=1) for m in range(1,13)])
            for key,value in [('tmean_c',arrays['tmean_c']),
                              ('rh_logit',rh),('precipitation_mm',arrays['precipitation_mm'])]}


def alignment_from_monthly_means(reference, historical):
    """Coefficients from independently accumulated full-baseline monthly means."""
    keys = ('tmean_c','rh_logit','precipitation_mm')
    a,b = [{key:np.asarray(source[key],float) for key in keys} for source in (reference,historical)]
    shape = a['tmean_c'].shape
    if (len(shape)!=2 or shape[1]!=12 or not all(x.shape==shape and np.isfinite(x).all()
            for source in (a,b) for x in source.values())
            or np.any(a['precipitation_mm']<0) or np.any(b['precipitation_mm']<0)):
        raise ValueError('Complete finite 12-month climatologies required.')
    if np.any((b['precipitation_mm']==0)&(a['precipitation_mm']>0)):
        raise ValueError('Positive reference rain cannot be aligned from a zero model monthly mean.')
    ratio = np.divide(a['precipitation_mm'],b['precipitation_mm'],
                      out=np.ones(shape),where=b['precipitation_mm']!=0)
    return dict(temperature_offset=a['tmean_c']-b['tmean_c'],
                humidity_logit_offset=a['rh_logit']-b['rh_logit'],
                precipitation_ratio=ratio)


def fit_monthly_alignment(reference, historical, month):
    return alignment_from_monthly_means(monthly_means(reference,month),
                                        monthly_means(historical,month))


