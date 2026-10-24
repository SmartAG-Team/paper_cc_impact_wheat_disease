"""Disease evaluation selects signs/brackets and preserves physical partitions."""
import importlib
import json
import numpy as np
import pandas as pd
import pytest
from models.calibration.seasonal_septoria.field_data import FieldData


def evaluation():
    try:return importlib.import_module('models.methods.seasonal_model.disease.run')
    except ModuleNotFoundError:pytest.fail('The isolated overwinter disease evaluation is missing')


def frozen_fit():
    from paper.analyses.source_support.paths import resolve
    path = resolve('analysis/paper_study/overwinter_leaf_model_20261006/disease/overwinter_source_model/frozen_selected_fit.json')
    return json.loads(path.read_text())['fitted']


def data_fixture():
    t=np.full((1,20),18.)
    target=pd.DataFrame(dict(field_id=['a','a'],field_index=[0,0],site_id=['x','x'],source=['BASF','BASF'],
        coordinate_year=['x|2018','x|2018'],leaf_index=[0,0],day_index=[8,15],
        endpoint_series=['a|F1','a|F1'],value=[np.nan,np.nan],season_year=[2018,2018]))
    meta=pd.DataFrame(dict(field_id=['a'],field_index=[0],forcing_days=[20],sowing_date=['2018-01-01']))
    d=FieldData(t,t*0+100,t*0+2,np.ones((1,20,8),bool),np.zeros((1,20,8)),meta,target,pd.DataFrame());d.maximum_temperature=t+2
    a=np.arange(1.,21.)[None,:]*50.
    thresholds={10:1.,31:10.,32:20.,33:30.,37:50.,39:100.,51:500.,65:700.,85:1500.}
    return d,a,thresholds


def test_prediction_requires_redacted_values_and_ignores_stage_metadata():
    r=evaluation();d,a,thresholds=data_fixture();fitted=frozen_fit()
    first,_=r.predict_record(d,a,thresholds,fitted)
    d.targets['stage_from']=99.
    second,_=r.predict_record(d,a,thresholds,fitted)
    np.testing.assert_array_equal(first.damage,second.damage)
    d.targets['value']=10.
    with pytest.raises(ValueError,match='redact'):r.predict_record(d,a,thresholds,fitted)


def test_no_sources_ablation_is_zero_and_juvenile_removal_obeys_stage31():
    r=evaluation();d,a,thresholds=data_fixture();fitted=frozen_fit()
    zero,_=r.predict_record(d,a,thresholds,fitted,ablation='no_sources')
    assert not zero.damage.any() and not zero.primary_flow.any() and not zero.secondary_flow.any()
    _,host=r.predict_record(d,a,thresholds,fitted,ablation='juvenile_removed_after31')
    assert not host.active[a>=thresholds[31],7].any()
    assert not host.area[a>=thresholds[31],7].any()

