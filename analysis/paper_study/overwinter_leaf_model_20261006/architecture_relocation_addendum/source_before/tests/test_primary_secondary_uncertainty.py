"""Repeated station-years stay together and singleton uncertainty is undefined."""
import importlib
import numpy as np
import pandas as pd
import pytest


def module():
    try:return importlib.import_module('model.primary_secondary.uncertainty')
    except ModuleNotFoundError:pytest.fail('Location-block error uncertainty is not implemented')


def fixture(single=False):
    f=pd.DataFrame({'coordinate_year':['x|2017','x|2018','y|2019'],
        'location_id':['x','x','y'],'series_id':['a','b','c'],
        'observed_percent':[0.,0.,0.],'predicted_percent':[10.,10.,0.]})
    if single:f=f.iloc[:2].copy()
    f['model']='baseline'
    a=f.copy();a['model']='candidate';a['predicted_percent']=0.
    return pd.concat([f,a],ignore_index=True)


def test_repeated_coordinate_years_remain_one_block_without_changing_score_estimand():
    u=module()
    result=u.paired_location_bootstrap(fixture(),'candidate','baseline',draws=1000)
    assert result['n_locations']==2
    assert result['n_coordinate_years']==3
    assert result['rmse_improvement_pp']==pytest.approx(np.sqrt(200/3))
    assert result['ci_lower_pp']==0.
    assert result['ci_upper_pp']==10.


def test_one_location_cannot_produce_a_between_location_confidence_interval():
    u=module()
    result=u.paired_location_bootstrap(fixture(True),'candidate','baseline')
    assert result['n_locations']==1
    assert result['ci_lower_pp'] is None
    assert result['ci_upper_pp'] is None
    assert result['interval_status']=='insufficient_independent_locations'
