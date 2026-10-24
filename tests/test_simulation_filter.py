import numpy as np
import torch


def _batch(fields=2,days=7):
    mask=np.zeros((fields,days,7),bool);mask[:,1,0]=True
    observed=np.zeros((fields,days,7),np.float32);observed[:,1,0]=np.linspace(.2,.8,fields)
    return dict(weather=np.zeros((fields,days,6),np.float32),valid_weather=np.ones((fields,days),bool),
        observed=observed,observed_mask=mask,static=np.zeros((fields,3),np.float32),
        sources=np.zeros(fields,int),
        query_field=np.repeat(np.arange(fields),2),query_day=np.tile([3,5],fields),
        query_leaf=np.zeros(2*fields,int),query_ids=np.arange(2*fields),field_ids=[str(i) for i in range(fields)])


def test_measurement_filter_withholds_day_q_values_weather_other_fields_and_padding():
    from models.monitoring.simulation_filter import MeasurementFilter
    torch.manual_seed(4);model=MeasurementFilter(hidden=12,dropout=.1);model.eval()
    batch=_batch()
    with torch.no_grad():first=model(batch).numpy()
    changed={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in batch.items()}
    changed['observed'][0,3:]=1;changed['observed_mask'][0,3:]=True;changed['weather'][0,3:]=2
    with torch.no_grad():second=model(changed).numpy()
    np.testing.assert_array_equal(first[[0,2,3]],second[[0,2,3]])
    # Changing the other field's entire history cannot affect either query here.
    changed={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in batch.items()}
    changed['observed'][1]=1;changed['observed_mask'][1]=True
    with torch.no_grad():other=model(changed).numpy()
    np.testing.assert_array_equal(first[:2],other[:2])
    padded={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in batch.items()}
    for key in ['weather','observed','observed_mask','valid_weather']:
        widths=[(0,0),(0,4)]+([(0,0)] if padded[key].ndim==3 else [])
        padded[key]=np.pad(padded[key],widths)
    with torch.no_grad():extended=model(padded).numpy()
    np.testing.assert_allclose(first,extended,atol=1e-5,rtol=0)


def test_measurement_filter_has_observer_gru_and_head_gradients_and_full_range_output():
    from models.monitoring.simulation_filter import MeasurementFilter
    torch.manual_seed(7);model=MeasurementFilter(hidden=12,dropout=0)
    result=model(_batch());((result/100-.9)**2).mean().backward()
    for parameter in [model.observation_encoder[0].weight,model.gru.weight_ih_l0,model.head[-1].weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    with torch.no_grad():model.head[-1].bias.fill_(6);value=model(_batch())
    assert torch.all(value>90) and torch.all(value<=100)


def test_filter_learns_from_prior_measurements_with_scientific_mass_and_replays():
    from models.monitoring.simulation_filter import fit_filter,predict_filter
    batch=_batch(fields=8,days=7)
    target=np.repeat(batch['observed'][:,1,0]*100,2)
    weights=np.tile([1.,3.],8)
    fitted=fit_filter(batch,target,weights,config=dict(hidden=12,dropout=0,epochs=150))
    prediction=predict_filter(fitted,batch)
    np.testing.assert_array_equal(fitted['training_weight'],weights)
    assert np.sqrt(np.average((prediction-target)**2,weights=weights))<8
    assert fitted['trace'][-1]['weighted_mse']<fitted['trace'][0]['weighted_mse']*.5
    np.testing.assert_array_equal(prediction,predict_filter(fitted,batch))


def test_pretraining_model_state_is_blind_to_real_disease_values():
    from models.monitoring.epidemic_world import world_inputs
    from models.monitoring.simulation_filter import pretrain_filter
    batch=_batch();batch['starts']=['2015-01-01']*2
    weights=np.array([1.,3.,2.,2.])
    config=dict(worlds=8,steps=3,batch_fields=4,hidden=12,dropout=0)
    first=pretrain_filter(world_inputs(batch,[2015,2015],weights),config)
    batch['observed'][:]=np.nan
    second=pretrain_filter(world_inputs(batch,[2015,2015],weights),config)
    assert first['trace']==second['trace']
    for key in first['state_dict']:
        np.testing.assert_array_equal(first['state_dict'][key],second['state_dict'][key])
