import numpy as np
import pytest


def test_joint_regimes_learn_full_range_predictions_and_normalized_gates():
    from models.monitoring.regime_model import fit_regime,predict_regime
    x=np.repeat([[-1.],[1.]],60,axis=0)
    y=np.repeat([10.,90.],60)
    fitted=fit_regime(x,y,np.full(120,50.),np.ones(120),dict(iterations=2,max_iter=100,min_samples_leaf=5))
    pred,prob=predict_regime(fitted,np.array([[-1.],[1.]]),np.array([50.,50.]),return_probabilities=True)
    assert pred[0]<25 and pred[1]>80
    np.testing.assert_allclose(prob.sum(axis=1),1.,atol=1e-12)
    assert (prob>=0).all()
    with pytest.raises(ValueError):
        predict_regime(fitted,np.array([[np.nan]]),np.array([50.]))
