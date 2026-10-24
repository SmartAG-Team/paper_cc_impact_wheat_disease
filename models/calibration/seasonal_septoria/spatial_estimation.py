"""Design-based error for stratified area-proportional simulation samples."""

import numpy as np


def stratified_mcse(contributions,weights,strata):
    values=np.asarray(contributions,float);weights=np.asarray(weights,float);strata=np.asarray(strata)
    if values.shape[-1]!=len(weights) or len(strata)!=len(weights):
        raise ValueError('Spatial draw membership mismatch.')
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('Finite contributions and positive area weights required.')
    variance=np.zeros(values.shape[:-1])
    for stratum in np.unique(strata):
        selected=strata==stratum;n=int(selected.sum())
        if n<2 or not np.allclose(weights[selected],weights[selected][0],rtol=0,atol=1e-15):
            raise ValueError('Independent equally weighted replicate draws required within each stratum.')
        variance+=weights[selected].sum()**2*np.var(values[...,selected],axis=-1,ddof=1)/n
    return np.sqrt(variance)


def anchored_parameter_means(period_samples,full_point_mean,full_valid_fraction,weights,strata):
    """Estimate parameter perturbations around the exact full-grid point result.

    Samples have shape parameter × year × spatial draw; row zero is the frozen
    point parameter. Calendar/weather eligibility is parameter independent.
    The full valid fraction uses the same calendar-eligible area population.
    """
    samples=np.asarray(period_samples,float);weights=np.asarray(weights,float)
    if samples.ndim!=3 or not 0<full_valid_fraction<=1 or not np.isfinite(full_point_mean):
        raise ValueError('A complete period and its exact valid-area denominator are required.')
    valid=np.isfinite(samples[0])
    if not np.array_equal(np.isfinite(samples),np.broadcast_to(valid,samples.shape)):
        raise ValueError('Parameter draws change source/host eligibility.')
    perturbation=np.where(valid[None,:,:],samples-samples[0:1],0).mean(axis=1)/full_valid_fraction
    means=full_point_mean+perturbation@weights
    mcse=stratified_mcse(perturbation,weights,strata)
    return dict(means=means,mcse=mcse,contributions=perturbation)


def sampled_ratio_mean(values,valid_fraction,weights,strata):
    """Period mean and linearized sampling error with explicit missing coverage."""
    values=np.asarray(values,float);valid=np.asarray(valid_fraction,float);weights=np.asarray(weights,float)
    denominator=float(np.dot(weights,valid))
    if not 0<denominator<=1+1e-12:raise ValueError('No valid area-year exposure.')
    mean=(values@weights)/denominator
    influence=(values-mean[...,None]*valid)/denominator
    return dict(means=mean,mcse=stratified_mcse(influence,weights,strata),valid_fraction=denominator)
