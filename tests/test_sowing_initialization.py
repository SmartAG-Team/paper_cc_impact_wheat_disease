import numpy as np
import pandas as pd
import pytest


def test_estimated_sowing_changes_only_predictions_without_earlier_field_measurements():
    from models.monitoring.sowing_initialization import select_initialization
    h=pd.DataFrame(dict(prior_assessment_available=[0,1,0],other_leaf_available=[0,0,1]))
    q=pd.DataFrame(dict(leaf_index=[0,1,2]))
    calendar=np.array([10.,20.,30.]);estimated=np.array([40.,50.,60.])
    a,mask=select_initialization(calendar,estimated,h,q)
    b,_=select_initialization(calendar,estimated+10,h,q)
    np.testing.assert_array_equal(mask,[True,False,False])
    assert b[0]>a[0]
    np.testing.assert_array_equal(a[1:],b[1:])
    np.testing.assert_array_equal(a[1:],calendar[1:])
    with pytest.raises(ValueError,match='labels'):
        select_initialization(calendar,estimated,h,q.assign(observed_percent=[99.,99.,99.]))
