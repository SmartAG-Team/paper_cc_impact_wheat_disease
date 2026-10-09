import numpy as np
import pytest
from analysis.paper_study.wheat_dl_phenology_20261009.evaluate import point_metrics, paired_interval, choose_family


def test_failed_crossing_remains_in_denominator_and_cannot_improve_selection():
    y=np.array([[10,20],[10,20]])
    perfect=point_metrics(y,y.astype(float),366)
    missing=point_metrics(y,np.array([[10.,20.],[np.nan,20.]]),366)
    assert perfect['macro_penalized_mae']==0
    assert missing['macro_penalized_mae']==pytest.approx(91.5)
    assert missing['observed']==4 and missing['matched']==3


def test_paired_station_interval_retains_both_stages_and_shared_years():
    y=np.array([[10,30],[10,30],[10,30],[10,30]])
    baseline=y+4;candidate=y+2
    r=paired_interval(y,baseline,candidate,np.array([1,1,2,3]),draws=100)
    assert r['difference_days']==-2
    assert r['lower95']==-2 and r['upper95']==-2
    assert r['stations']==3


def test_validation_choice_ignores_any_supplied_test_scores():
    scores={'lstm':{'macro_penalized_mae':2.,'test_mae':900.},'tcn':{'macro_penalized_mae':3.,'test_mae':0.}}
    assert choose_family(scores)=='lstm'
    with pytest.raises(ValueError):choose_family({'lstm':{'macro_penalized_mae':float('nan')}})
