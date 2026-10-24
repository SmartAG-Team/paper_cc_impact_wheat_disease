"""Application of supplied monthly alignment coefficients with explicit trend conventions."""

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
