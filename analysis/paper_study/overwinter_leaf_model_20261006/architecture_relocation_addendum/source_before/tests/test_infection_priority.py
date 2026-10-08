from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from model.seasonal_septoria import infection_priority as p


def inputs():
    t=np.full((1,12),18.)
    data=SimpleNamespace(temperature=t,maximum_temperature=t+2,humidity=t*0+100,
        rain=t*0,metadata=pd.DataFrame([dict(field_index=0,field_id='field',sowing_date='2020-06-01',forcing_days=12)]),
        targets=pd.DataFrame({'value':[90.]}))
    config=dict(schema_version=1,model_id='test',event_model=dict(host_parameters=dict(flag_fraction=.3,leaf_interval=1.,expansion_units=1.),
        parameters=dict(dose_threshold=.5,dry_gap_days=3,delay_reference_days=2.),weather_preprocessing=dict(rain_rate_mm_hour=.46)),
        anthesis_threshold=5.,yield_slopes=[.0141,.018,.0207])
    return data,np.arange(1,13,dtype=float)[None,:],{10:1.,31:2.,51:3.,85:9.},config


def test_default_outputs_events_and_overlap_without_inventing_yield():
    data,a,t,c=inputs();result=p.predict_infection_priority(data,a,t,c)
    assert result['field_outputs'].iloc[0].complete_grain_fill_window
    assert result['yield_transfer'] is None
    assert not result['calibrated_occurrence_probability']
    assert result['infection_dates'].shape==(1,3)
    assert result['field_outputs'].iloc[0].anthesis_proxy_date==pd.Timestamp('2020-06-05')
    assert result['field_outputs'].iloc[0].soft_dough_date==pd.Timestamp('2020-06-09')


def test_caller_supplied_functional_area_supports_conditional_yield_transfer():
    data,a,t,c=inputs();area=np.ones((1,12,3));loss=area*.5
    result=p.predict_infection_priority(data,a,t,c,reference_leaf_area_index=area,functional_loss_fraction=loss)
    np.testing.assert_allclose(result['yield_transfer']['absolute_loss_t_ha'][:,1],[.135])
    assert result['yield_transfer']['relative_loss_percent'] is None
    with pytest.raises(ValueError):p.predict_infection_priority(data,a,t,c,reference_leaf_area_index=area)


def test_disease_magnitudes_do_not_change_any_infection_prediction():
    data,a,t,c=inputs();first=p.predict_infection_priority(data,a,t,c)
    data.targets['value']=np.nan
    second=p.predict_infection_priority(data,a,t,c)
    np.testing.assert_array_equal(first['infection_dates'],second['infection_dates'])
    pd.testing.assert_frame_equal(first['field_outputs'],second['field_outputs'])


def test_padded_weather_cannot_advance_field_event_clocks():
    data,a,t,c=inputs();data.metadata['forcing_days']=2
    result=p.predict_infection_priority(data,a,t,c,return_daily=True)
    raw=result['daily_events']
    assert np.all((raw.infection_day<=2)|(raw.infection_day==-1))
    assert np.all(raw.symptom_day==-1)
    assert not result['daily_valid_mask'][0,3:].any()
