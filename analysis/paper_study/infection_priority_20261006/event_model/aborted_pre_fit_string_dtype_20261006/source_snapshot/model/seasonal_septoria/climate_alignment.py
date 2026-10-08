"""Historical-only monthly mean alignment with explicit trend conventions."""

import numpy as np
from scipy.special import expit, logit

FIELDS = ('tmean_c','tmax_c','rh_mean_pct','precipitation_mm')
RH_EPSILON = .001


def _forcing(forcing, month):
    arrays = {key:np.asarray(forcing[key],float) for key in FIELDS}
    months = np.asarray(month,int)
    shape = arrays['tmean_c'].shape
    if (len(shape)!=2 or not all(x.shape==shape and np.isfinite(x).all() for x in arrays.values())
            or months.shape!=(shape[1],) or np.any((months<1)|(months>12))
            or np.any((arrays['rh_mean_pct']<0)|(arrays['rh_mean_pct']>100))
            or np.any(arrays['precipitation_mm']<0)
            or np.any(arrays['tmax_c']<arrays['tmean_c']-.001)):
        raise ValueError('Finite, consistent daily climate arrays and months1–12 required.')
    return arrays,months


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


def apply_monthly_alignment(forcing, month, coefficients):
    """Preserve additive temperature and multiplicative precipitation changes.

    A common temperature offset applies to mean and maximum, preserving their
    original difference. RH changes are preserved in bounded log-odds space.
    Rank order and zero-rain days remain unchanged within each month. This mean
    correction does not align wet-day frequency, tails or full distributions.
    """
    arrays,months = _forcing(forcing,month)
    shape = (len(arrays['tmean_c']),12)
    coefficients = {key:np.asarray(coefficients[key],float) for key in
                    ('temperature_offset','humidity_logit_offset','precipitation_ratio')}
    if (not all(x.shape==shape and np.isfinite(x).all() for x in coefficients.values())
            or np.any(coefficients['precipitation_ratio']<0)):
        raise ValueError('Finite per-cell 12-month correction coefficients required.')
    index = months-1
    offset = coefficients['temperature_offset'][:,index]
    rh = logit(np.clip(arrays['rh_mean_pct']/100,RH_EPSILON,1-RH_EPSILON))
    return dict(tmean_c=arrays['tmean_c']+offset,
                tmax_c=arrays['tmax_c']+offset,
                rh_mean_pct=100*expit(rh+coefficients['humidity_logit_offset'][:,index]),
                precipitation_mm=arrays['precipitation_mm']*coefficients['precipitation_ratio'][:,index])
