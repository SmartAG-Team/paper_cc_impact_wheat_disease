"""Fixed-population means and empirical annual canopy-damage statistics."""
import numpy as np


def _checked(values, weights, other=None):
    a=np.asarray(values,dtype=float)
    w=np.asarray(weights,dtype=float)
    if a.ndim!=2 or (other is not None and np.shape(other)!=a.shape):
        raise ValueError('Historical and future shape must agree: cells by years')
    if w.ndim!=1 or len(w)!=len(a) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Finite nonnegative weights must match cell identities')
    return a,w


def pairing_statistics(historical, future, weights, shift=0):
    a,w=_checked(historical,weights,future)
    b=np.asarray(future,dtype=float)[:,(np.arange(a.shape[1])+shift)%a.shape[1]]
    valid=np.isfinite(a)&np.isfinite(b)
    n=valid.sum(axis=1)
    den=float(w@n)
    total=float(w.sum()*a.shape[1])
    reference=float(w@np.where(valid,a,0).sum(axis=1)/den) if den else np.nan
    projected=float(w@np.where(valid,b,0).sum(axis=1)/den) if den else np.nan
    return {'reference':reference,'future':projected,'change':projected-reference,
            'valid_pairs':int(n[w>0].sum()),'valid_cells':int(((n>0)&(w>0)).sum()),
            'coverage_fraction':den/total if total else np.nan}


def separate_period_change(historical, future, weights):
    a,w=_checked(historical,weights,future)
    b=np.asarray(future,dtype=float)
    result={}
    for name,values in [('reference',a),('future',b)]:
        valid=np.isfinite(values)
        den=float(w@valid.sum(axis=1))
        result[name]=float(w@np.where(valid,values,0).sum(axis=1)/den) if den else np.nan
        result[name+'_coverage_fraction']=den/(w.sum()*a.shape[1]) if w.sum() else np.nan
    result['change']=result['future']-result['reference']
    return result


def annual_means(values,weights):
    a,w=_checked(values,weights)
    finite=np.isfinite(a)
    den=np.sum(w[:,None]*finite,axis=0)
    numerator=np.sum(w[:,None]*np.where(finite,a,0),axis=0)
    return {'mean':np.divide(numerator,den,out=np.full(a.shape[1],np.nan),where=den>0),
            'coverage_fraction':den/w.sum() if w.sum() else np.full(a.shape[1],np.nan),
            'valid_cells':np.sum(finite&(w[:,None]>0),axis=0)}


def distribution_statistics(historical,future):
    a,b=np.asarray(historical,dtype=float),np.asarray(future,dtype=float)
    if a.shape!=(30,) or b.shape!=(30,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Annual distribution requires 30 finite regional values in each period')
    result={}
    for name,x in [('historical',a),('future',b)]:
        result.update({name+'_years':len(x),name+'_mean':float(x.mean()),
            name+'_sd':float(x.std(ddof=1)),name+'_median':float(np.median(x)),
            name+'_q10':float(np.quantile(x,.1,method='linear')),
            name+'_q90':float(np.quantile(x,.9,method='linear')),
            name+'_top3_mean':float(np.sort(x)[-3:].mean())})
    result['mean_change']=float(b.mean()-a.mean())
    result['q90_change']=result['future_q90']-result['historical_q90']
    result['future_exceedance_years']=int((b>result['historical_q90']).sum())
    result['future_exceedance_fraction']=result['future_exceedance_years']/30
    return result


def production_components(total,rainfed,irrigated):
    # Provider total and management-specific rasters are separate published
    # quantities. Preserve their rounding residual instead of adjusting either.
    if not np.isfinite([total,rainfed,irrigated]).all() or min(total,rainfed,irrigated)<0:
        raise ValueError('Production components must be finite and nonnegative')
    return {'source_component_residual_tonnes':total-rainfed-irrigated,
            'irrigated_share_pct':100*irrigated/total if total else np.nan}
