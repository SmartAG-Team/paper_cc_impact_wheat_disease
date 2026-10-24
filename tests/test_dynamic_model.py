import numpy as np
import torch
from test_dynamic_data import fixture


def batch():
    from models.monitoring.dynamic_data import build_sequences
    w,f,o=fixture()
    return build_sequences(w,f,o,o.drop(columns='observed_percent'))


def test_daily_states_conserve_mass_and_gradients_reach_transition_parameters():
    from models.monitoring.dynamic_model import DailyDiseaseModel
    torch.manual_seed(42); model=DailyDiseaseModel()
    out=model(batch())
    np.testing.assert_allclose(out['state'].detach().sum(-1),1.,atol=2e-6)
    assert torch.all(out['state']>=0)
    np.testing.assert_allclose(out['weights'].detach().sum(-1),1.,atol=1e-6)
    assert torch.all((out['mean']>=0)&(out['mean']<=100))
    out['mean'].sum().backward()
    assert model.rate_net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_same_day_observations_are_withheld_and_prior_changes_update_later_predictions():
    from models.monitoring.dynamic_model import DailyDiseaseModel
    torch.manual_seed(42); model=DailyDiseaseModel(); a=batch(); b=batch()
    b['observed'][0,1,:2]=.9
    pa=model(a)['mean'].detach().numpy(); pb=model(b)['mean'].detach().numpy()
    np.testing.assert_array_equal(pa[:2],pb[:2])
    assert pb[2]>pa[2]+1
    b['weather'][0,2:]=99
    pc=model(b)['mean'].detach().numpy()
    np.testing.assert_array_equal(pa[:2],pc[:2])


def test_direct_state_prediction_can_exceed_previous_blend_ceiling():
    from models.monitoring.dynamic_model import DailyDiseaseModel
    model=DailyDiseaseModel(); a=batch()
    a['observed'][0,1,:2]=.99
    with torch.no_grad():
        model.gain_logit.fill_(6)
        model.noise_logit.fill_(-8)
    out=model(a)['mean'].detach().numpy()
    assert out[2]>90  # A process prediction of 50 formerly capped the final value at75.


def test_near_saturated_state_recovers_tissue_mass_after_lower_observation():
    from models.monitoring.dynamic_model import DailyDiseaseModel
    torch.set_num_threads(1)
    days=700
    a=dict(weather=np.zeros((1,days,6),np.float32),valid_weather=np.ones((1,days),bool),
        observed=np.zeros((1,days,7),np.float32),observed_mask=np.zeros((1,days,7),bool),
        static=np.zeros((1,3),np.float32),sources=np.zeros(1,np.int64),
        query_field=np.array([0]),query_day=np.array([days-1]),query_leaf=np.array([0]))
    a['weather'][...,0]=.5;a['weather'][...,1]=1.;a['weather'][...,2]=1.;a['weather'][...,5]=1.
    a['observed_mask'][0,days-3]=True
    model=DailyDiseaseModel()
    with torch.no_grad():
        model.infection_logit.fill_(12);model.latent_logit.fill_(-8)
        out=model(a)
    np.testing.assert_allclose(out['state'].sum(-1),1.,atol=2e-6)
