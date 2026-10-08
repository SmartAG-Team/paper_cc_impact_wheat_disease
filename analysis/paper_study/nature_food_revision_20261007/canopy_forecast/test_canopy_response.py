"""Guard empirical crop-response forecasting boundaries and physical bounds."""
import numpy as np,pandas as pd,pytest
from canopy_response import hierarchical_weights,extract_assessment_features,forecast_percent,fit_response,validate_site_fold

def fake_features():
 return pd.DataFrame(dict(global_target_index=np.arange(12),field_id=['t0']*4+['t1']*4+['v0']*4,site_id=['s0']*4+['s1']*4+['s2']*4,coordinate_year=['s0|2017']*4+['s1|2018']*4+['s2|2019']*4,leaf_index=[0,1,0,1]*3,available=np.ones(12,bool),crop_progress=np.tile([0,.1,.5,.8],3),leaf_thermal_age=np.tile([0,.1,.6,1.],3),cum_moisture=np.tile([0,.1,.3,.7],3)))

def fake_truth():
 return pd.DataFrame(dict(global_target_index=np.arange(12),partition=['calibration']*8+['reused_BASF2019']*4,observed_percent=np.tile([0,5,45,80],3)))

def test_equal_hierarchy_is_not_row_weighting():
 d=pd.DataFrame(dict(coordinate_year=['a']*6+['b'],field_id=['a1']*5+['a2','b1'],leaf_index=[0,0,0,0,1,0,0]));w=hierarchical_weights(d);d['w']=w
 assert d.groupby('coordinate_year').w.sum().to_dict()==pytest.approx({'a':.5,'b':.5})
 assert d.groupby(['coordinate_year','field_id']).w.sum().to_dict()==pytest.approx({('a','a1'):.25,('a','a2'):.25,('b','b1'):.5})
 assert d.groupby(['coordinate_year','field_id','leaf_index']).w.sum().loc[('a','a1',0)]==pytest.approx(.125)

def test_prediction_features_use_predicted_stage_and_1based_date():
 coords=pd.DataFrame(dict(global_target_index=[0],field_index=[0],day_index=[2],leaf_index=[0]));daily=dict(predicted_stage_code=np.array([[31,37,39]]),available=np.ones((1,3,7),bool),crop_progress=np.array([[0,.3,.7]]),leaf_thermal_age=np.broadcast_to(np.arange(3)[None,:,None],(1,3,7)),cum_moisture=np.zeros((1,3,7)))
 f=extract_assessment_features(coords,daily);assert f.predicted_stage_code.iloc[0]==37 and f.crop_progress.iloc[0]==.3 and f.leaf_thermal_age.iloc[0]==1
 with pytest.raises(ValueError,match='observed|outcome'):extract_assessment_features(coords.assign(stage_from=85),daily)
 with pytest.raises(ValueError,match='observed|outcome'):extract_assessment_features(coords.assign(value=99),daily)

def test_fixed_logistic_response_is_bounded_monotone_and_leaf_gated():
 f=pd.DataFrame(dict(leaf_index=[0]*4,available=[False,True,True,True],crop_progress=[0,.1,.4,1.],leaf_thermal_age=[0,.1,.4,1.],cum_moisture=[0,.1,.4,1.]));record=dict(variant='crop_moisture',leaf_intercepts=[-4.]*7,nonnegative_slopes=[1.,2.,.5]);p=forecast_percent(f,record)
 assert p[0]==0 and ((p>=0)&(p<=100)).all() and (np.diff(p)>=0).all()
 with pytest.raises(ValueError,match='nonnegative'):forecast_percent(f,{**record,'nonnegative_slopes':[-1.,2.,.5]})

def test_withheld_target_mutations_never_change_training_fit():
 f=fake_features();t=fake_truth();one=fit_response(f,t,np.arange(8),'crop_only',.001);changed=t.copy();changed.loc[changed.partition!='calibration','observed_percent']=[-100000,999999,0,100];two=fit_response(f,changed,np.arange(8),'crop_only',.001)
 np.testing.assert_array_equal(one['leaf_intercepts'],two['leaf_intercepts']);np.testing.assert_array_equal(one['nonnegative_slopes'],two['nonnegative_slopes'])
 with pytest.raises(ValueError,match='withheld|calibration'):fit_response(f,t,np.arange(12),'crop_only',.001)

def test_fold_guard_uses_whole_location_histories():
 f=fake_features();validate_site_fold(f,np.arange(4),np.arange(4,8))
 with pytest.raises(ValueError,match='location|site'):validate_site_fold(f,np.array([0,1]),np.array([2,3]))
