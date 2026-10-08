"""Paired prediction-error bootstrap with complete repeated-location blocks."""
import numpy as np


def paired_location_bootstrap(frame,candidate,baseline,draws=5000,seed=20261004):
    errors={}
    for name in (candidate,baseline):
        sub=frame.loc[frame.model.eq(name)].copy()
        sub['sq']=(sub.predicted_percent-sub.observed_percent)**2
        # Original estimand: equal coordinate-year, then equal leaf sequence.
        units=sub.groupby(['location_id','coordinate_year','series_id']).sq.mean()
        errors[name]=units.groupby(['location_id','coordinate_year']).mean()
    shared=errors[candidate].index.intersection(errors[baseline].index)
    a,b=errors[candidate].loc[shared],errors[baseline].loc[shared]
    if not len(shared):raise ValueError('No shared coordinate-year observations.')
    locations=shared.get_level_values('location_id').unique()
    result={'candidate':candidate,'baseline':baseline,'n_locations':len(locations),
        'n_coordinate_years':len(shared),'rmse_improvement_pp':float(np.sqrt(b.mean())-np.sqrt(a.mean())),
        'ci_lower_pp':None,'ci_upper_pp':None,'interval_status':'insufficient_independent_locations'}
    if len(locations)<2:return result
    # Resample location sums AND denominators, preserving every year per location.
    a_sum=a.groupby('location_id').sum().reindex(locations).to_numpy()
    b_sum=b.groupby('location_id').sum().reindex(locations).to_numpy()
    count=a.groupby('location_id').size().reindex(locations).to_numpy()
    indices=np.random.default_rng(seed).integers(0,len(locations),(draws,len(locations)))
    denominator=count[indices].sum(axis=1)
    improvement=np.sqrt(b_sum[indices].sum(axis=1)/denominator)-np.sqrt(a_sum[indices].sum(axis=1)/denominator)
    result.update(ci_lower_pp=float(np.quantile(improvement,.025)),
        ci_upper_pp=float(np.quantile(improvement,.975)),interval_status='conditional_location_block_bootstrap')
    return result
