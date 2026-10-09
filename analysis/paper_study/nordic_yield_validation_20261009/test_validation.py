"""Checks for independent outcomes, missing stages and held-out-target leakage."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd


SOURCE=Path(__file__).with_name('run.py')


def analysis():
    assert SOURCE.exists(), 'Nordic yield validation implementation is absent'
    spec=importlib.util.spec_from_file_location('nordic_yield_analysis',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_source_cleaning_preserves_one_yield_per_trial_treatment():
    module=analysis();clean,endpoints,contrasts,quality=module.prepare()
    assert quality['source_records']==259
    assert quality['source_trial_treatment_outcomes']==190
    assert quality['postharvest_records']==8
    assert quality['duplicate_control_records']==3
    assert len(endpoints)==181
    assert not endpoints.duplicated(['registry_id','treatment_code']).any()
    assert len(contrasts)==62 and contrasts.registry_id.nunique()==5
    assert not ((contrasts.registry_id==72044)&(contrasts.treatment_code==13)).any()
    assert (contrasts.response_fraction<0).any()


def test_source_stage_zero_remains_unknown():
    module=analysis();clean,*_=module.prepare()
    part=clean[(clean.registry_id==72044)&(clean.date_disease=='2024-06-21')]
    assert len(part)==12
    assert part.verified_stage.isna().all()


def test_heldout_predictions_do_not_use_heldout_yields():
    module=analysis()
    frame=pd.DataFrame({'registry_id':[1,1,2,2,3,3],
                        'treatment_code':[2,3,2,3,2,3],
                        'response_fraction':[.01,.02,.1,.2,.3,.4],
                        'delta_severity':[.1,.2,.2,.3,.3,.4]})
    first=module.leave_trial_out(frame,'training_mean',[])
    changed=frame.copy();changed.loc[changed.registry_id==1,'response_fraction']=.9
    second=module.leave_trial_out(changed,'training_mean',[])
    np.testing.assert_allclose(first.loc[first.registry_id==1,'prediction'],.25,rtol=0,atol=1e-12)
    np.testing.assert_allclose(first.loc[first.registry_id==1,'prediction'],
                               second.loc[second.registry_id==1,'prediction'],rtol=0,atol=1e-12)
    assert all('1' not in x.split('|') for x in first.loc[first.registry_id==1,'training_trials'])


def test_positive_stage_model_has_zero_response_without_severity_reduction():
    module=analysis()
    frame=pd.DataFrame({'registry_id':[1,1,2,2],
                        'response_fraction':[.15,.10,.14,.09],
                        'delta_severity':[.1,.3,.1,.3],
                        'earlier_stage_basis':[.1,.3,.1,.3]})
    evaluation=pd.DataFrame({'delta_severity':[0.,.1,.2],
                             'earlier_stage_basis':[0.,.1,.2]})
    prediction,_=module.fit_predict(frame,evaluation,'positive_stage_linear',
                                    ['delta_severity','earlier_stage_basis'])
    assert abs(prediction[0])<1e-12
    assert np.all(np.diff(prediction)>=0)
