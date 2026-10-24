import numpy as np


def test_logit_growth_preserves_zero_change_and_restores_query_offset_once():
    from models.monitoring.offset_growth_model import fit_growth,predict_growth
    x=np.zeros((50,1));base=np.full(50,20.)
    fit=fit_growth(x,base,base,np.ones(50),link='logit',rounds=20)
    pred=predict_growth(fit,np.zeros((2,1)),np.array([10.,90.]))
    np.testing.assert_allclose(pred,[10.,90.],atol=1e-6)


def test_logit_growth_learns_above_the_old_blend_ceiling():
    from models.monitoring.offset_growth_model import fit_growth,predict_growth
    x=np.repeat([[-1.],[1.]],80,axis=0);y=np.repeat([10.,90.],80)
    fit=fit_growth(x,y,np.full(160,50.),np.ones(160),link='logit',rounds=150)
    pred=predict_growth(fit,np.array([[-1.],[1.]]),np.array([50.,50.]))
    assert pred[0]<25 and pred[1]>80


def test_task_weight_splitting_does_not_inflate_information_or_regularization():
    from models.monitoring.offset_growth_model import fit_growth,predict_growth
    x=np.repeat([[-1.],[1.]],80,axis=0);y=np.repeat([10.,90.],80);base=np.full(160,50.)
    original=fit_growth(x,y,base,np.ones(160),rounds=30,normalize_weights=False)
    duplicated=fit_growth(np.repeat(x,3,axis=0),np.repeat(y,3),np.repeat(base,3),
        np.full(480,1/3),rounds=30,normalize_weights=False)
    q=np.array([[-1.],[1.]])
    np.testing.assert_allclose(predict_growth(original,q,np.array([50.,50.])),
        predict_growth(duplicated,q,np.array([50.,50.])),atol=1e-5)
