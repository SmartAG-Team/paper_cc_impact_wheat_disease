import numpy as np
import pandas as pd
from models.monitoring.history import SAME_NAMES,OTHER_NAMES


def empty_history(n):
    return pd.DataFrame(np.zeros((n,len(SAME_NAMES+OTHER_NAMES))),columns=SAME_NAMES+OTHER_NAMES)


def test_field_anchor_uses_same_leaf_then_latest_other_native_code_without_query_outcomes():
    from models.monitoring.field_history_models import field_history_anchor
    h=empty_history(4)
    q=pd.DataFrame(dict(leaf_index=[2,2,2,2],observed_percent=[100,0,60,20]))
    h.loc[0,['prior_assessment_available','latest_severity_fraction','days_since_latest_scaled']]=[1,.4,.2]
    for row in [0,1,2]:
        h.loc[row,['other_leaf_available','other_code_1_available','other_code_3_available']]=1
        h.loc[row,['other_code_1_severity','other_code_3_severity']]=[.9,.3]
        h.loc[row,['other_code_1_age','other_code_3_age']]=[.2,.1 if row==1 else .2]
    anchor,features=field_history_anchor(h,q)
    np.testing.assert_allclose(anchor,[.4,.3,.6,0])
    np.testing.assert_allclose(features[:,1],[1,1,1,0])
    np.testing.assert_allclose(features[:,2],[.2,.1,.2,0])
    assert features[1,3]==1/6 and features[1,4]==1/6
    assert features[2,3]==0 and features[2,4]==1/6
    changed=q.copy();changed.observed_percent=1
    np.testing.assert_array_equal(field_history_anchor(h,changed)[0],anchor)


def test_any_field_history_routes_other_leaf_cases_to_residual_head():
    from models.monitoring.field_history_models import history_groups
    h=empty_history(3)
    h.loc[1,'other_leaf_available']=1
    h.loc[2,'prior_assessment_available']=1
    groups=history_groups(h,'field_split')
    assert groups['none'].tolist()==[True,False,False]
    assert groups['history'].tolist()==[False,True,True]
