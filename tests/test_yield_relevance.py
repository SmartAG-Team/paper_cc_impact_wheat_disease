import numpy as np
import pandas as pd
import pytest

from model.seasonal_septoria import yield_relevance as y


def test_earlier_damage_has_larger_yield_effect_despite_same_final_damage():
    dates=pd.date_range('2020-06-01',periods=10)
    area=np.ones((2,10,3))
    damage=np.zeros_like(area);damage[0]=.5;damage[1,-1]=.5
    result=y.upper_leaf_had_loss(dates,area,damage,[dates[0]]*2,[dates[-1]]*2)
    np.testing.assert_allclose(result['lost_had3'],[15,1.5])
    impact=y.transfer_had_yield_loss(result['lost_had3'],[.0141,.018,.0207])
    np.testing.assert_allclose(impact['absolute_loss_t_ha'][:,1],[.27,.027])
    assert impact['relative_loss_percent'] is None


def test_post_window_damage_has_no_grain_fill_effect():
    dates=pd.date_range('2020-06-01',periods=8)
    area=np.ones((1,8,4));loss=np.zeros_like(area);loss[:,6:]=1
    result=y.upper_leaf_had_loss(dates,area,loss,[dates[1]],[dates[5]])
    assert result['lost_had3'][0]==0
    assert result['reference_had3'][0]==15


def test_infection_overlap_uses_dates_and_excludes_late_or_absent_events():
    dates=pd.date_range('2020-06-01',periods=10)
    infection=np.array([[dates[0],dates[7],np.datetime64('NaT','D')]],dtype='datetime64[D]')
    result=y.top_three_infection_overlap(dates,infection,[dates[5]],[dates[9]])
    np.testing.assert_allclose(result['infected_grain_fill_days'],[[5,3,0]])
    np.testing.assert_allclose(result['infected_grain_fill_fraction'],[[1,.6,0]])
    assert result['days_with_any_top3_infected'][0]==5
    assert result['days_with_all_top3_infected'][0]==0


def test_incomplete_crop_window_is_missing_instead_of_zero_loss():
    dates=pd.date_range('2020-06-01',periods=5)
    area=np.ones((1,5,3));loss=area*.3
    result=y.upper_leaf_had_loss(dates,area,loss,[dates[0]],[dates[-1]+pd.Timedelta(days=1)])
    assert not result['complete_window'][0]
    assert np.isnan(result['lost_had3'][0])
    with pytest.raises(ValueError,match='order'):
        y.upper_leaf_had_loss(dates,area,loss,[dates[-1]],[dates[0]])


def test_yield_transfer_preserves_extrapolation_instead_of_hiding_it_by_clipping():
    result=y.transfer_had_yield_loss([200],[.0141,.018,.0207],reference_yield_t_ha=[2])
    np.testing.assert_allclose(result['relative_loss_percent'],[[141,180,207]])
    assert result['exceeds_reference_yield'].all()
    assert not result['coefficient_range_is_confidence_interval']
    with pytest.raises(ValueError):
        y.transfer_had_yield_loss([10],[.018],reference_yield_t_ha=[0])


def test_leaf_area_controls_contribution_without_assuming_equal_layer_weights():
    dates=pd.date_range('2020-06-01',periods=3)
    area=np.tile([2.,1.,.5],(1,3,1));loss=np.zeros_like(area);loss[:,:,0]=.5
    result=y.upper_leaf_had_loss(dates,area,loss,[dates[0]],[dates[-1]])
    assert result['lost_had3'][0]==3
    assert result['reference_had3'][0]==10.5
    np.testing.assert_allclose(result['lost_had_by_leaf'],[[3,0,0]])
