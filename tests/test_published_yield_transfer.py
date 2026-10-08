"""Arithmetic and leakage checks for the external published-yield benchmark."""
import numpy as np
import pytest
from calibration.yield_transfer import fit_group_balanced_slope, leave_group_out


def test_genetic_backgrounds_have_equal_total_weight():
    # Two related lines cannot outweigh one independent genetic background.
    assert fit_group_balanced_slope([1,1,1],[0,0,3],['a','a','b']) == 1.5


def test_origin_constrained_slope_reproduces_an_exact_response():
    assert fit_group_balanced_slope([1,2,3,4],[2,4,6,8],['a','a','b','c']) == 2.


def test_held_out_yields_do_not_affect_their_own_prediction():
    x=np.array([1,2,3,4,5.]);groups=np.array(['a','a','b','b','c'])
    y=2*x
    before=leave_group_out(x,y,groups)
    y[groups=='b']=[70,90]
    after=leave_group_out(x,y,groups)
    np.testing.assert_array_equal(before['prediction'][groups=='b'],after['prediction'][groups=='b'])
    np.testing.assert_array_equal(before['training_groups'][groups=='b'],[2,2])


def test_invalid_or_unidentifiable_inputs_are_rejected():
    with pytest.raises(ValueError):fit_group_balanced_slope([0,0],[1,2],['a','b'])
    with pytest.raises(ValueError):fit_group_balanced_slope([1,np.nan],[1,2],['a','b'])
    with pytest.raises(ValueError):leave_group_out([1,2],[1,2],['a','a'])
