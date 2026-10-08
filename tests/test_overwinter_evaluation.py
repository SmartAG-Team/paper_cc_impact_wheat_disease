"""Disease evaluation selects signs/brackets and preserves physical partitions."""
import importlib
import json
import numpy as np
import pandas as pd
import pytest
from calibration.seasonal_septoria.field_data import FieldData


def evaluation():
    try:return importlib.import_module('analysis.paper_study.overwinter_leaf_model_20261006.disease.run')
    except ModuleNotFoundError:pytest.fail('The isolated overwinter disease evaluation is missing')


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
    r=evaluation();d,a,thresholds=data_fixture();candidate=r.candidate_grid()[0]
    fitted=dict(**candidate,weather_preprocessing=dict(rain_rate_mm_hour=.46))
    first,_=r.predict_record(d,a,thresholds,fitted)
    d.targets['stage_from']=99.
    second,_=r.predict_record(d,a,thresholds,fitted)
    np.testing.assert_array_equal(first.damage,second.damage)
    d.targets['value']=10.
    with pytest.raises(ValueError,match='redact'):r.predict_record(d,a,thresholds,fitted)


def test_no_sources_ablation_is_zero_and_juvenile_removal_obeys_stage31():
    r=evaluation();d,a,thresholds=data_fixture();fitted=dict(**r.candidate_grid()[-1],weather_preprocessing=dict(rain_rate_mm_hour=.46))
    zero,_=r.predict_record(d,a,thresholds,fitted,ablation='no_sources')
    assert not zero.damage.any() and not zero.primary_flow.any() and not zero.secondary_flow.any()
    _,host=r.predict_record(d,a,thresholds,fitted,ablation='juvenile_removed_after31')
    assert not host.active[a>=thresholds[31],7].any()
    assert not host.area[a>=thresholds[31],7].any()


def test_balanced_detection_weights_keep_sensitivity_and_specificity_separate():
    r=evaluation();frame=pd.DataFrame(dict(coordinate_year=['x']*4,field_id=['a']*4,leaf_index=[0]*4,
        observed_positive=[False,False,True,True],predicted_positive=[False,True,True,False]))
    result=r.detection_metrics(frame)
    assert result['sensitivity']==.5 and result['specificity']==.5
    assert result['balanced_accuracy']==.5


def test_candidate_selection_uses_brackets_before_balanced_error():
    r=evaluation();records=[dict(candidate='later',candidate_order=0,primary_distance_days=1.,balanced_assessment_error=0.),
        dict(candidate='tie_bad',candidate_order=1,primary_distance_days=0.,balanced_assessment_error=.5),
        dict(candidate='tie_good',candidate_order=2,primary_distance_days=0.,balanced_assessment_error=.25)]
    assert r.select_candidate(records)['candidate']=='tie_good'


def test_primary_tie_diagnostic_is_json_serializable():
    r=evaluation();records=[dict(primary_distance_days=np.float64(0.)),dict(primary_distance_days=np.float64(0.)),
        dict(primary_distance_days=np.float64(1.))]
    assert json.loads(json.dumps(dict(ties=r.primary_tie_count(records,0.))))=={'ties':2}
