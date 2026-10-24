import numpy as np
import pandas as pd


def test_canopy_qa_rejects_cloud_backup_snow_shadow_and_fill():
    from models.monitoring.canopy_inputs import canopy_quality
    qc = np.array([0,2,32,8,64,0,0,0])
    extra = np.array([0,0,128,0,0,4,64,0])
    lai = np.array([20,20,20,20,20,20,20,255])
    np.testing.assert_array_equal(canopy_quality(qc,extra,lai,np.full(8,50),
        np.full(8,3),np.full(8,4)),[True,True,True,False,False,False,False,False])


def _series():
    return pd.DataFrame(dict(period_end=pd.to_datetime(['2015-05-01','2015-05-05','2015-05-09']),
        lai_mean=[2.,3.,9.],fpar_mean=[.4,.6,.9],lai_stdDev=.2,fpar_stdDev=.05,
        lai_uncertainty_mean=.3,fpar_uncertainty_mean=.04,valid_mean=.8,lai_count=200.,crop_fraction=.7))


def test_canopy_features_exclude_incomplete_or_not_yet_available_composites():
    from models.monitoring.canopy_inputs import canopy_prefix
    frame = _series(); q = pd.Timestamp('2015-05-14')
    first,names,supported = canopy_prefix(frame,q)
    assert supported and first[names.index('canopy_latest_lai')] == 3.
    altered = frame.copy(); altered.loc[2,'lai_mean'] = 999.
    second,_,_ = canopy_prefix(altered,q)
    np.testing.assert_array_equal(first,second)
    # A composite starting before a query can still contain future acquisitions.
    _,_,early = canopy_prefix(frame.iloc[1:],pd.Timestamp('2015-05-03'),lag_days=0)
    assert not early


def test_missing_canopy_is_flagged_and_never_treated_as_observed_zero():
    from models.monitoring.canopy_inputs import canopy_prefix
    frame = _series(); frame['valid_mean'] = 0.
    values,names,supported = canopy_prefix(frame,pd.Timestamp('2015-05-14'))
    assert not supported and values[names.index('canopy_available')] == 0
    assert np.isfinite(values).all()
