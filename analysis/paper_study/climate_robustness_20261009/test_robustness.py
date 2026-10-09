import numpy as np
import pytest
from .statistics import pairing_statistics, separate_period_change, annual_means, distribution_statistics


def test_complete_series_mean_is_invariant_to_arbitrary_pairing():
    historical=np.array([[1., 2., 8.], [0., 4., 3.]])
    future=np.array([[8., 1., 3.], [7., 3., 5.]])
    w=np.array([1., 4.])
    expected=np.average(future.mean(1)-historical.mean(1), weights=w)
    results=[pairing_statistics(historical,future,w,k) for k in range(3)]
    np.testing.assert_allclose([r['change'] for r in results], expected, atol=1e-14)
    assert all(r['coverage_fraction']==1 for r in results)


def test_missing_pairing_uses_common_seasons_and_fixed_full_denominator():
    a=np.array([[1.,np.nan,3.],[2.,4.,6.]])
    b=np.array([[2.,7.,np.nan],[5.,np.nan,9.]])
    w=np.array([10.,2.])
    for shift in range(3):
        pairs=[]
        for i in range(2):
            for y in range(3):
                future=b[i,(y+shift)%3]
                if np.isfinite(a[i,y]) and np.isfinite(future):
                    pairs.append((w[i],a[i,y],future))
        den=sum(x[0] for x in pairs)
        result=pairing_statistics(a,b,w,shift)
        assert result['change']==pytest.approx(sum(q*(v-u) for q,u,v in pairs)/den)
        assert result['coverage_fraction']==pytest.approx(den/(3*w.sum()))
        assert result['valid_pairs']==len(pairs)
        assert result['future']-result['reference']==pytest.approx(result['change'])
    assert pairing_statistics(a,b,w,0)['change']!=pairing_statistics(a,b,w,1)['change']


def test_separate_period_denominators_do_not_create_fictitious_pairs():
    a=np.array([[1.,np.nan,3.]])
    b=np.array([[np.nan,9.,np.nan]])
    r=separate_period_change(a,b,np.array([1.]))
    assert r['reference']==2
    assert r['future']==9
    assert r['change']==7
    assert np.isnan(pairing_statistics(a,b,np.array([1.]),0)['change'])


def test_annual_means_retain_missing_and_zero_damage_differently():
    r=annual_means(np.array([[0.,np.nan,2.],[0.,np.nan,np.nan]]),np.array([1.,3.]))
    np.testing.assert_allclose(r['mean'], [0.,np.nan,2.],equal_nan=True)
    np.testing.assert_allclose(r['coverage_fraction'],[1.,0.,.25])


def test_tail_statistics_are_empirical_years_not_pooled_climate_members():
    past=np.arange(30.)
    future=past+10.
    r=distribution_statistics(past,future)
    assert r['historical_q90']==pytest.approx(26.1)
    assert r['future_exceedance_fraction']==pytest.approx(13/30)
    assert r['historical_top3_mean']==28
    assert r['future_top3_mean']==38
    assert r['mean_change']==10
    assert r['future_years']==30


def test_incomplete_annual_distribution_rejected_for_fixed_support_analysis():
    with pytest.raises(ValueError,match='30 finite'):
        distribution_statistics(np.arange(29.),np.arange(30.))


def test_negative_or_incompatible_weights_fail():
    a=np.ones((2,3))
    with pytest.raises(ValueError,match='weights'):
        pairing_statistics(a,a,np.array([1.,-1.]),0)
    with pytest.raises(ValueError,match='shape'):
        pairing_statistics(a,np.ones((3,2)),np.ones(2),0)


def test_separately_published_production_components_keep_source_residual():
    from .statistics import production_components
    r=production_components(100.,80.1,20.)
    assert r['source_component_residual_tonnes']==pytest.approx(-.1)
    assert r['irrigated_share_pct']==20


def test_empty_and_zero_weight_domains_are_missing_not_damage_free():
    a=np.array([[1.,2.,3.]])
    r=pairing_statistics(a,a,np.array([0.]))
    assert np.isnan(r['change']) and np.isnan(r['coverage_fraction'])
    assert r['valid_pairs']==0
    r=pairing_statistics(np.full_like(a,np.nan),a,np.array([1.]))
    assert np.isnan(r['change']) and r['coverage_fraction']==0


def test_ensemble_requires_all_models_and_never_pools_years():
    import pandas as pd
    from .run import _ensemble
    frame=pd.DataFrame({'domain':['x']*3,'model':['a','b','c'],'value':[1.,2.,12.]})
    r=_ensemble(frame,['domain'],['value']).iloc[0]
    assert r.value==5 and r.value_gcm_min==1 and r.value_gcm_max==12
    assert np.isnan(_ensemble(frame.iloc[:2],['domain'],['value']).iloc[0].value)
    with pytest.raises(ValueError,match='Repeated climate'):
        _ensemble(pd.concat([frame,frame.iloc[:1]]),['domain'],['value'])
