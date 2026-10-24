import numpy as np
import pandas as pd
import pytest
from test_dynamic_data import fixture


def test_microclimate_intervals_use_prior_weather_and_report_physical_proxy_bounds():
    from models.monitoring.microclimate_features import build_microclimate
    w,f,o=fixture();q=o.drop(columns='observed_percent')
    for key,value in dict(tmax_c=15.,tmin_c=5.,dewpoint_mean_c=5.,wind_mean_m_s=2.,
            shortwave_mj_m2=10.,rain_hours_gt_0_1mm=1.,rh_hours_ge_90pct=12.).items():w[key]=value
    provenance=pd.DataFrame(dict(global_target_index=q.global_target_index,latest_same_date=[None,None,'2020-01-02'],latest_any_date=[None,None,'2020-01-02']))
    a,names,ledger=build_microclimate(w,f,q,provenance)
    np.testing.assert_allclose(a[:,names.index('mc_wind_m_s_mean_7d')],[2.,2.,2.])
    assert a[2,names.index('mc_gap_days')]==2
    assert a[2,names.index('mc_gap_wet_lower_hours')]==24
    assert a[2,names.index('mc_gap_wet_upper_hours')]==26
    w.loc[1:,'shortwave_mj_m2']=99
    b,_,_=build_microclimate(w,f,q,provenance)
    np.testing.assert_array_equal(a[:2],b[:2])
    assert not np.array_equal(a[2],b[2])
    assert ledger.weather_end.tolist()==['2020-01-01','2020-01-01','2020-01-04']
    with pytest.raises(ValueError,match='label'):
        build_microclimate(w,f,o,provenance)
