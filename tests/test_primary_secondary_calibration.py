"""Independent split and scoring behavior, rather than fitted parameter values."""
import importlib
import pandas as pd
import numpy as np
import pytest


def calibration():
    try:
        return importlib.import_module('calibration.primary_secondary.calibrate')
    except ModuleNotFoundError:
        pytest.fail('Grouped calibration is not implemented')


def test_forward_split_purges_test_coordinates_from_earlier_years():
    c=calibration()
    episodes=pd.DataFrame({'series_id':['a','b','c','d','e'], 'year':[2017,2017,2018,2019,2019],
        'location_id':['x','y','z','x','q'], 'Country':['A','B','A','A','B']})
    fold=next(x for x in c.make_folds(episodes) if x['name']=='forward_2019')
    assert fold['train']=={'b','c'}
    assert fold['test']=={'d','e'}
    assert 'a' not in fold['train']


def test_scoring_gives_equal_coordinate_year_weight_not_more_weight_to_dense_series():
    c=calibration()
    frame=pd.DataFrame({'coordinate_year':['x','x','x','y'], 'series_id':['a','a','b','c'],
        'observed_percent':[0,0,0,0], 'predicted_percent':[0,0,0,20]})
    out=c.score(frame)
    assert out['rmse_pp']==pytest.approx(np.sqrt(200))
    assert out['mae_pp']==pytest.approx(10)
    assert out['bias_pp']==pytest.approx(10)


def test_grouped_inner_folds_never_split_a_location():
    c=calibration()
    episodes=pd.DataFrame({'series_id':['a','b','c','d','e','f'],
        'location_id':['x','x','y','y','z','z']})
    splits=c.inner_folds(episodes,3)
    assert len(splits)==3
    for train,test in splits:
        a=set(episodes.loc[episodes.series_id.isin(train),'location_id'])
        b=set(episodes.loc[episodes.series_id.isin(test),'location_id'])
        assert not a.intersection(b)
        assert train|test==set(episodes.series_id)
