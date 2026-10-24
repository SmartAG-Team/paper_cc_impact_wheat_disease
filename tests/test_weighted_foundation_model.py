import numpy as np
import pytest


def test_systematic_context_preserves_scientific_mass_without_new_measurements():
    from models.monitoring.weighted_foundation_model import weighted_context
    x,y,indices=weighted_context(np.array([[10.],[20.]]),np.array([1.,9.]),np.array([1.,3.]),4,20261022)
    np.testing.assert_array_equal(indices,[0,1,1,1])
    np.testing.assert_array_equal(x,[[10.],[20.],[20.],[20.]])
    np.testing.assert_array_equal(y,[1.,9.,9.,9.])
    _,_,scaled=weighted_context(np.array([[10.],[20.]]),np.array([1.,9.]),np.array([100.,300.]),4,20261022)
    np.testing.assert_array_equal(indices,scaled)


def test_weighted_context_rejects_negative_or_nonfinite_scientific_weights():
    from models.monitoring.weighted_foundation_model import weighted_context
    for weights in [[1.,-1.],[1.,np.nan],[0.,1.]]:
        with pytest.raises(ValueError):weighted_context(np.zeros((2,1)),np.array([1.,2.]),weights,4,20261022)
