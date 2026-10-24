import numpy as np
import pandas as pd


def test_particle_curve_updates_withhold_current_day_and_keep_fields_separate():
    from models.monitoring.curve_particle_model import predict_curve_bank
    parameters=np.zeros((2,7,2));parameters[:,:,1]=10;parameters[1,:,0]=-2
    fitted=dict(banks={'BASF':dict(parameters=parameters,latitude=np.array([51.,51.]),
        longitude=np.array([0.,0.]),weight=np.array([1.,1.]),supported_codes=[0,1])},
        phase_shifts=[0.],slope_multipliers=[1.],sigma_pp=12.,clock_center=2.,pooled_fraction=.5)
    frame=pd.DataFrame(dict(field_id=['a']*4+['b'],source='BASF',latitude=51.,longitude=0.,
        date=['2015-05-01','2015-05-02','2015-05-02','2015-05-03','2015-05-03'],
        leaf_index=[0,0,1,0,0],observed_percent=[20.,80.,70.,0.,100.]))
    clock=np.array([1.8,2.,2.,2.2,2.2]);baseline=np.full(5,30.)
    first=predict_curve_bank(fitted,clock,frame,baseline)
    np.testing.assert_array_equal(first['same'][[0,4]],[30.,30.])
    assert first['same'][1]>20 and first['same'][2]==30
    assert first['field'][2]!=30
    changed=frame.copy();changed.loc[1:3,'observed_percent']=[0.,0.,100.]
    second=predict_curve_bank(fitted,clock,changed,baseline)
    np.testing.assert_array_equal(first['field'][:3],second['field'][:3])
    np.testing.assert_array_equal(first['free'],second['free'])
    assert first['same'][3]!=second['same'][3]


def test_regularized_curve_bank_learns_future_growth_from_historical_curves():
    from scipy.special import expit
    from models.monitoring.curve_particle_model import fit_curve_bank,predict_curve_bank
    clock=np.tile([1.6,1.8,2.],3)
    training=pd.DataFrame(dict(field_id=np.repeat(['a','b','c'],3),source='BASF',latitude=51.,longitude=0.,
        date=np.tile(['2014-05-01','2014-05-11','2014-05-21'],3),leaf_index=0,
        observed_percent=100*expit(8*(clock-2)),global_target_index=np.arange(9)))
    fitted=fit_curve_bank(clock,training,np.ones(9))
    query=pd.DataFrame(dict(field_id=['new']*2,source='BASF',latitude=51.,longitude=0.,
        date=['2015-05-21','2015-05-31'],leaf_index=0,observed_percent=[50.,0.]))
    value=predict_curve_bank(fitted,np.array([2.,2.2]),query,np.array([40.,40.]))
    assert value['same'][0]==40 and 65<value['same'][1]<100
