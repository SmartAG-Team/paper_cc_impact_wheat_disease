"""Exact weighted error CDFs and paired site resampling of fixed forecasts."""
import numpy as np
import pandas as pd


def error_coverage(points,replicates=2000,seed=20261024):
    """Retain all empirical jumps, zero, and each integer threshold through100pp.

    Site multiplicities are shared across evaluation designs. Original relative
    weights within sampled sites are unchanged. Intervals describe fixed
    forecasts and do not include model fitting or adaptive selection uncertainty.
    """
    required={'evaluation','site_id','observed_percent','prediction_percent','scientific_weight'}
    if missing:=required-set(points):raise ValueError(f'Missing coverage inputs: {sorted(missing)}')
    if points.empty or points[list(required)].isna().any().any():raise ValueError('Coverage inputs must be nonempty and complete')
    values=points[['observed_percent','prediction_percent','scientific_weight']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,2]<=0).any():raise ValueError('Values must be finite and weights positive')
    if ((values[:,:2]<0)|(values[:,:2]>100)).any():raise ValueError('Disease percentages must lie within0–100')
    if replicates<2:raise ValueError('At least two resamples are required')
    sites=sorted(points.site_id.unique());positions={name:i for i,name in enumerate(sites)}
    multiplicities=np.random.default_rng(seed).multinomial(len(sites),np.full(len(sites),1/len(sites)),size=replicates)
    output=[]
    for design,part in points.groupby('evaluation',sort=False):
        errors=abs(part.prediction_percent-part.observed_percent).to_numpy(float)
        weights=part.scientific_weight.to_numpy(float)
        thresholds=np.unique(np.r_[np.arange(101,dtype=float),errors])
        site=part.site_id.map(positions).to_numpy(int)
        mass=np.zeros((len(sites),len(thresholds)),float)
        np.add.at(mass,(site,np.searchsorted(thresholds,errors)),weights)
        cumulative=mass.cumsum(axis=1)
        sampled_mass=multiplicities@cumulative
        denominator=sampled_mass[:,-1,None]
        sampled_share=np.divide(100*sampled_mass,denominator,out=np.full_like(sampled_mass,np.nan),where=denominator>0)
        lower,upper=np.nanquantile(sampled_share,[.025,.975],axis=0)
        output.append(pd.DataFrame(dict(evaluation=design,threshold_pp=thresholds,
            covered_weight_percent=np.clip(100*cumulative.sum(axis=0)/weights.sum(),0,100),
            lower95=np.clip(lower,0,100),upper95=np.clip(upper,0,100),
            assessments=len(part),sites=part.site_id.nunique(),weight_sum=weights.sum(),
            bootstrap_replicates=replicates,bootstrap_seed=seed)))
    return pd.concat(output,ignore_index=True)
