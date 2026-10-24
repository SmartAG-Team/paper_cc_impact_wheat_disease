import numpy as np
import pytest


def test_foundation_targets_restore_without_blend_ceiling():
    from models.monitoring.foundation_model import transform_target,restore_prediction
    y=np.array([0.,90.,100.]);base=np.array([20.,50.,70.])
    np.testing.assert_allclose(transform_target(y,base,'anchor'),[-20.,40.,30.])
    np.testing.assert_array_equal(restore_prediction(np.array([-20.,40.,30.]),base,'anchor'),y)
    np.testing.assert_array_equal(restore_prediction(np.array([-5.,120.,90.]),base,'direct'),[0.,100.,90.])
    with pytest.raises(ValueError):transform_target(y,base,'unknown')
    with pytest.raises(ValueError):restore_prediction(np.array([np.nan]),np.array([20.]),'direct')
