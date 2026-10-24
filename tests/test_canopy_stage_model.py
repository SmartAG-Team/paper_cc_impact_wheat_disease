import numpy as np
import pytest


def test_joint_stage_cases_conserve_original_context_mass_and_order():
    from models.monitoring.canopy_stage_model import stage_cases
    x=np.array([[1.,2.],[3.,4.]])
    y=np.array([[1,0,0],[1,1,0]])
    weights=np.array([.1,.7])
    expanded,target,mass=stage_cases(x,y,weights)
    np.testing.assert_array_equal(expanded[:3],[[1,2,0],[1,2,1],[1,2,2]])
    np.testing.assert_array_equal(target,y.ravel())
    np.testing.assert_allclose(mass.reshape(-1,3).sum(1),weights,atol=1e-16)
    with pytest.raises(ValueError,match='ordered binary'):
        stage_cases(x,np.array([[0,1,0],[1,1,0]]),weights)


def test_joint_stage_learner_learns_ordered_stage_probabilities():
    from models.monitoring.canopy_stage_model import fit_canopy_stage,predict_canopy_stage
    u=np.linspace(0,1,600)
    x=np.column_stack([u,np.cos(u*5)])
    y=(u[:,None]>=np.array([.25,.5,.75])).astype(int)
    fitted=fit_canopy_stage(x,y,np.ones(len(x)))
    pred=predict_canopy_stage(fitted,x)
    assert pred.shape==y.shape and np.isfinite(pred).all()
    assert np.all(np.diff(pred,axis=1)<=1e-12)
    assert np.mean((pred-y)**2)<.025
    assert np.all(pred[-1]-pred[0]>.8)
