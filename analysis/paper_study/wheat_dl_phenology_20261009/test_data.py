import numpy as np
import pandas as pd
import pytest
from analysis.paper_study.wheat_dl_phenology_20261009.prepare import pack_cycle, fit_normalizer


def daily():
    return pd.DataFrame(dict(DATE=pd.date_range('2020-10-01',periods=5),
        t_mean=[10.]*5,t_max=[15.]*5,t_min=[5.]*5,daylength=[11.]*5,
        CUMVER=[0.]*5,Cumulative_GDD=[10.,20.,30.,40.,50.],
        Cumulative_t_pp_v_GDD=[10.,20.,30.,40.,50.],t_pp_v_GDD=[10.]*5))


def test_prefix_stops_at_weather_gap_and_preserves_sowing_day_index():
    f=daily().drop(index=2)
    x,g,n=pack_cycle(f,pd.Timestamp('2020-10-01'),horizon=5)
    assert n==2 and g.tolist()==[10.,10.,0.,0.,0.]
    assert x[0,-1]==pytest.approx(1/366)
    assert (x[2:]==0).all()


def test_missing_start_is_not_backfilled():
    _,_,n=pack_cycle(daily().iloc[1:],pd.Timestamp('2020-10-01'),horizon=5)
    assert n==0


def test_duplicates_are_rejected_instead_of_arbitrarily_resolved():
    f=pd.concat([daily(),daily().iloc[:1]])
    with pytest.raises(ValueError,match='Duplicate'):pack_cycle(f,pd.Timestamp('2020-10-01'),5)


def test_normalization_uses_training_weather_and_not_padding_or_test_rows():
    x=np.array([[[1.,3.],[3.,5.],[999.,999.]],[[900.,900.],[900.,900.],[900.,900.]]])
    mean,std=fit_normalizer(x,np.array([2,3]),np.array([0]))
    np.testing.assert_allclose(mean,[2.,4.]);np.testing.assert_allclose(std,[1.,1.])
