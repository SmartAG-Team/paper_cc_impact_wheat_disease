import numpy as np
import pandas as pd
import pytest


def test_gaussian_update_matches_hand_calculation():
    from models.monitoring.field_gp_model import gaussian_update
    # A scalar covariance1 with noisy-history variance2 gives gain1/2.
    got=gaussian_update(np.array([0.]),np.array([0.]),np.array([[1.]]),np.array([[2.]]),np.array([2.]))
    np.testing.assert_allclose(got,[1.],atol=1e-12)


def test_gp_conditions_only_on_earlier_held_field_measurements():
    from models.monitoring.field_gp_model import fit_gp,predict_gp
    train=pd.DataFrame(dict(field_id=['a']*4+['b']*4,leaf_index=[0,1]*4,
        date=['2020-01-01','2020-01-03','2020-01-05','2020-01-07']*2,
        observed_percent=[10.,20.,30.,40.]*2,global_target_index=np.arange(8)))
    x=np.tile(np.arange(4),2)[:,None].astype(float)
    fit=fit_gp(x,train,np.full(8,20.),np.ones(8),max_iter=3)
    context=pd.DataFrame(dict(field_id=['c','c'],leaf_index=[0,0],date=['2020-01-02','2020-01-06'],
        observed_percent=[30.,60.],global_target_index=[20,21]))
    q=pd.DataFrame(dict(field_id=['c'],leaf_index=[0],date=['2020-01-04'],global_target_index=[30]))
    args=(np.array([[2.]]),q,np.array([20.]),np.array([[1.],[3.]]))
    p=predict_gp(fit,*args,context,np.array([20.,20.]))
    altered=context.copy();altered.loc[1,'observed_percent']=0
    np.testing.assert_array_equal(p,predict_gp(fit,*args,altered,np.array([20.,20.])))
    altered.loc[0,'observed_percent']=80
    assert predict_gp(fit,*args,altered,np.array([20.,20.]))[0]>p[0]
    with pytest.raises(ValueError,match='label'):
        predict_gp(fit,args[0],q.assign(observed_percent=99),*args[2:],context,np.array([20.,20.]))
