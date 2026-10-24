"""Sampling units, exact boundaries and original weights for error coverage."""
import numpy as np
import pandas as pd
import pytest


def sample():
    return pd.DataFrame(dict(evaluation=['temporal']*4,site_id=['s1','s1','s2','s3'],
        observed_percent=[0.,0.,0.,0.],prediction_percent=[0.,10.,10.,100.],
        scientific_weight=[1.,2.,3.,4.]))


def test_coverage_includes_zero_and_tied_errors_with_original_weights():
    from paper.scripts.validation_error_coverage import error_coverage
    points=sample();before=points.copy(deep=True)
    curve=error_coverage(points,replicates=100,seed=123)
    q=curve.set_index('threshold_pp')
    assert q.loc[0,'covered_weight_percent']==pytest.approx(10.)
    assert q.loc[9,'covered_weight_percent']==pytest.approx(10.)
    assert q.loc[10,'covered_weight_percent']==pytest.approx(60.)
    assert q.loc[99,'covered_weight_percent']==pytest.approx(60.)
    assert q.loc[100,'covered_weight_percent']==pytest.approx(100.)
    assert np.all(np.diff(curve.covered_weight_percent)>=0)
    pd.testing.assert_frame_equal(points,before)


def test_bootstrap_resamples_sites_and_preserves_all_their_relative_weights():
    from paper.scripts.validation_error_coverage import error_coverage
    points=sample();other=points.assign(evaluation='spatial_50km',prediction_percent=[5.,20.,20.,90.])
    curve=error_coverage(pd.concat([points,other]),replicates=100,seed=123)
    multiplicities=np.random.default_rng(123).multinomial(3,[1/3]*3,size=100)
    for design,part in pd.concat([points,other]).groupby('evaluation'):
        for threshold in [0,10,20,100]:
            draws=[]
            for m in multiplicities:
                weights=part.scientific_weight.to_numpy()*part.site_id.map(dict(zip(['s1','s2','s3'],m))).to_numpy()
                draws.append(100*np.average(abs(part.prediction_percent-part.observed_percent)<=threshold,weights=weights))
            shown=curve[curve.evaluation.eq(design)&curve.threshold_pp.eq(threshold)].iloc[0]
            np.testing.assert_allclose([shown.lower95,shown.upper95],np.quantile(draws,[.025,.975]),rtol=0,atol=1e-12)


def test_noninteger_jumps_are_retained_and_invalid_weights_rejected():
    from paper.scripts.validation_error_coverage import error_coverage
    points=sample();points.loc[0,'prediction_percent']=.125
    curve=error_coverage(points,replicates=100,seed=123)
    assert .125 in set(curve.threshold_pp)
    points.loc[0,'scientific_weight']=0.
    with pytest.raises(ValueError,match='weight'):
        error_coverage(points,replicates=100,seed=123)
