import numpy as np
from model.seasonal_septoria.spatial_estimation import anchored_parameter_means,stratified_mcse


def test_exact_full_grid_point_anchor_and_constant_parameter_perturbation():
    weights=np.array([.3,.3,.2,.2]);strata=np.array(['a','a','b','b'])
    point=np.array([[1.,20.,50.,80.],[2.,21.,51.,81.]])
    samples=np.stack([point,point+7.])
    result=anchored_parameter_means(samples,42.,1.,weights,strata)
    np.testing.assert_array_equal(result['means'],[42.,49.])
    np.testing.assert_array_equal(result['mcse'],[0.,0.])


def test_mcse_matches_independent_two_stratum_variance_and_shared_draw_contrast():
    weights=np.array([.3,.3,.2,.2]);strata=np.array(['a','a','b','b'])
    values=np.array([[1.,3.,2.,6.]])
    # Two independent with-replacement draws: sample variances2 and8.
    expected=np.sqrt(.6**2*2/2+.4**2*8/2)
    np.testing.assert_allclose(stratified_mcse(values,weights,strata),[expected])
    np.testing.assert_array_equal(stratified_mcse(values-values,weights,strata),[0.])
