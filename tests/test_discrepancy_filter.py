import numpy as np
import pandas as pd


def test_discrepancy_updates_wait_until_after_all_same_day_leaf_queries():
    from models.monitoring.discrepancy_filter import predict_discrepancy
    frame=pd.DataFrame(dict(field_id=['a']*4+['b'],date=['2015-05-01','2015-05-02',
        '2015-05-02','2015-05-03','2015-05-03'],leaf_index=[0,0,1,0,0],
        observed_percent=[30.,40.,50.,60.,80.]))
    base=np.full(5,20.);params=np.log([2.,1.,30.,30.])
    first=predict_discrepancy(params,frame,base,'linear')
    # First visits have no residual context. One earlier +10pp innovation must
    # raise both second-day predictions, before either current label is seen.
    np.testing.assert_array_equal(first[[0,4]],[20.,20.])
    assert 20<first[2]<first[1]<30
    changed=frame.copy();changed.loc[1:3,'observed_percent']=[99.,1.,0.]
    second=predict_discrepancy(params,changed,base,'linear')
    np.testing.assert_array_equal(first[:3],second[:3])
    assert first[3]!=second[3]


def test_discrepancy_link_can_correct_predictions_above_frozen_blend_ceiling():
    from models.monitoring.discrepancy_filter import predict_discrepancy
    frame=pd.DataFrame(dict(field_id=['a','a'],date=['2015-05-01','2015-05-02'],
        leaf_index=[0,0],observed_percent=[95.,0.]))
    value=predict_discrepancy(np.log([10.,10.,30.,30.]),frame,np.array([40.,40.]),'logit')
    assert value[1]>90


def test_first_visit_endpoint_baselines_remain_exactly_zero_or_one_hundred():
    from models.monitoring.discrepancy_filter import predict_discrepancy
    frame=pd.DataFrame(dict(field_id=['a','b'],date=['2015-05-01']*2,
        leaf_index=[0,0],observed_percent=[50.,50.]))
    value=predict_discrepancy(np.log([1.,1.,30.,30.]),frame,np.array([0.,100.]),'logit')
    np.testing.assert_array_equal(value,[0.,100.])
