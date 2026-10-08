"""Forecast diagnostics must preserve field clustering and issue-time context."""
import importlib
import numpy as np
import pandas as pd
import pytest


def metrics():
    try:
        return importlib.import_module('model.primary_secondary.crop_protection_metrics')
    except ModuleNotFoundError:
        pytest.fail('Crop-protection forecast diagnostics are not implemented')


def sample():
    return pd.DataFrame({'coordinate_year':['x','x','x','y'],
        'series_id':['a','a','b','c'], 'observed_percent':[50,50,0,50],
        'predicted_percent':[30,30,30,0]})


def test_diagnostic_confusion_retains_equal_field_year_and_series_weights():
    result=metrics().threshold_diagnostics(sample(),25)
    assert result['weighted_true_positive']==pytest.approx(.25)
    assert result['weighted_false_positive']==pytest.approx(.25)
    assert result['weighted_false_negative']==pytest.approx(.5)
    assert result['sensitivity']==pytest.approx(1/3)
    assert result['false_alarm_fraction']==pytest.approx(.5)
    assert result['auc']==pytest.approx(1/6)


def test_auc_credits_ties_half_and_preserves_continuous_severity_ranking():
    f=pd.DataFrame({'coordinate_year':['x','x'], 'series_id':['a','b'],
        'observed_percent':[0,50], 'predicted_percent':[25,25]})
    assert metrics().threshold_diagnostics(f,25)['auc']==pytest.approx(.5)
    f['predicted_percent']=[10,11]
    assert metrics().threshold_diagnostics(f,25)['auc']==pytest.approx(1)


def test_absent_outcome_classes_have_undefined_rates():
    f=sample();f['observed_percent']=0;f['predicted_percent']=0
    result=metrics().threshold_diagnostics(f,25)
    assert result['auc'] is None
    assert result['sensitivity'] is None
    assert result['false_alarm_fraction'] is None


def test_repeating_a_complete_series_does_not_increase_its_field_weight():
    f=sample()
    dense=pd.concat([f,f.loc[f.series_id.eq('a')]],ignore_index=True)
    a=metrics().threshold_diagnostics(f,25)
    b=metrics().threshold_diagnostics(dense,25)
    for name in ('sensitivity','specificity','auc','weighted_prevalence'):
        assert a[name]==pytest.approx(b[name])


def test_forecast_context_uses_issue_time_stage_and_exact_horizon_boundaries():
    predictions=pd.DataFrame({'series_id':['a']*5, 'day':[7,8,14,28,29],
        'start':['2019-04-01']*5,
        'Date':['2019-04-08','2019-04-09','2019-04-15','2019-04-29','2019-04-30'],
        'conditioning':[False]*5,'GsFrom':[99]*5})
    episodes=pd.DataFrame({'series_id':['a'],'GsFrom':[32.], 'initial_percent':[2.]})
    x=metrics().attach_forecast_context(predictions,episodes)
    assert x.issue_stage.tolist()==[32.]*5
    assert x.stage_group.eq('GS31-33').all()
    assert x.horizon_group.tolist()==['01-07','08-14','08-14','15-28','29+']
    changed=predictions.copy();changed['GsFrom']=1
    y=metrics().attach_forecast_context(changed,episodes)
    assert x.stage_group.equals(y.stage_group)


@pytest.mark.parametrize('change',['day','conditioning','chronology'])
def test_conditioning_and_invalid_forecast_chronology_fail(change):
    p=pd.DataFrame({'series_id':['a'],'day':[7], 'start':['2019-04-01'],
        'Date':['2019-04-08'], 'conditioning':[False]})
    if change=='day':p['day']=0
    if change=='conditioning':p['conditioning']=True
    if change=='chronology':p['Date']='2019-04-09'
    e=pd.DataFrame({'series_id':['a'],'initial_percent':[2.]})
    with pytest.raises(ValueError):metrics().attach_forecast_context(p,e)


def test_percentage_diagnostic_rejects_missing_scores_and_out_of_range_cutoffs():
    f=sample();f.loc[0,'predicted_percent']=np.nan
    with pytest.raises(ValueError):metrics().threshold_diagnostics(f,25)
    with pytest.raises(ValueError):metrics().threshold_diagnostics(sample(),101)
