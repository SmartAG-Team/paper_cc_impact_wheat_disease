import numpy as np
import pandas as pd


def test_error_state_withholds_all_same_day_innovations_and_other_fields():
    from models.monitoring.error_dynamics import error_states
    frame=pd.DataFrame(dict(field_id=['a']*4+['b'],date=['2015-05-01','2015-05-02',
        '2015-05-02','2015-05-03','2015-05-03'],leaf_index=[0,0,1,0,0],
        observed_percent=[50.,70.,80.,90.,100.]))
    states,names=error_states(frame,np.full(5,20.))
    np.testing.assert_array_equal(states[[0,4]],np.zeros((2,len(names))))
    assert states[1,names.index('same_latest_error_fraction')]==.3
    altered=frame.copy();altered.loc[1:3,'observed_percent']=[0.,0.,0.]
    changed,_=error_states(altered,np.full(5,20.))
    np.testing.assert_array_equal(states[:3],changed[:3])
    assert not np.array_equal(states[3],changed[3])


def test_nonlinear_error_correction_uses_past_bias_and_preserves_first_visits():
    from models.monitoring.error_dynamics import error_states,fit_error_head,predict_error_head
    frame=pd.DataFrame(dict(field_id=np.repeat(np.arange(40),3),
        date=np.tile(['2014-05-01','2014-05-02','2014-05-03'],40),leaf_index=0,
        observed_percent=np.repeat(np.repeat([10.,90.],20),3)))
    base=np.full(120,40.);states,names=error_states(frame,base);raw=np.zeros((120,1))
    fit=fit_error_head(raw,states,frame,base,np.ones(120),max_iter=200)
    value=predict_error_head(fit,raw,states,base)
    np.testing.assert_array_equal(value[::3],np.full(40,40.))
    assert value[1]<20 and value[61]>80
