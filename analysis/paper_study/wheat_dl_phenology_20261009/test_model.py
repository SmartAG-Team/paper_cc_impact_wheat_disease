import numpy as np
import pytest
import torch
from model.wheat_phenology_dl.network import WheatDevelopmentModel, observed_crps, quantile_days

@pytest.mark.parametrize('encoder',['lstm','tcn'])
def test_future_weather_does_not_change_past_predictions(encoder):
    torch.manual_seed(17)
    m=WheatDevelopmentModel([.08,.31,.56,1.],6,encoder=encoder,width=8)
    x=torch.randn(2,30,6);g=torch.ones(2,30)*12
    _,before=m(x,g,1000.)
    future=x.clone();future[:,15:]+=30
    _,after=m(future,g,1000.)
    torch.testing.assert_close(before[:,:,:15],after[:,:,:15])

@pytest.mark.parametrize('encoder',['lstm','tcn'])
def test_stage_distributions_are_ordered_and_cold_days_do_not_develop(encoder):
    m=WheatDevelopmentModel([.08,.31,.56,1.],6,encoder=encoder,width=8)
    x=torch.randn(2,40,6);g=torch.ones(2,40)*30;g[:,10:15]=0
    state,cdf=m(x,g,1000.)
    assert torch.all(torch.diff(state,dim=1)>=0)
    assert torch.all(torch.diff(cdf,dim=1)<=1e-7)
    assert torch.all(torch.diff(cdf,dim=2)>=-1e-7)
    torch.testing.assert_close(state[:,10:15],state[:,9:10].expand(-1,5))


def test_missing_observations_and_weather_tail_do_not_contribute_to_loss():
    cdf=torch.full((1,2,5),.4,requires_grad=True)
    loss=observed_crps(cdf,torch.tensor([[2,0]]),torch.tensor([1.,1.]),torch.tensor([3]))
    loss.backward()
    assert torch.equal(cdf.grad[:,1],torch.zeros((1,5)))
    assert torch.equal(cdf.grad[:,:,3:],torch.zeros((1,2,2)))
    assert float(loss.detach())==pytest.approx(.16+.36+.36)


def test_unreached_median_is_missing_not_a_horizon_date():
    cdf=np.array([[[.1,.3,.4],[.2,.5,.7]]])
    q=quantile_days(cdf,.5,np.array([3]))
    assert np.isnan(q[0,0]) and q[0,1]==2
    assert np.isnan(quantile_days(cdf,.5,np.array([1]))).all()


def test_invalid_or_maize_stage_layout_is_rejected():
    with pytest.raises(ValueError):WheatDevelopmentModel([.1,.4,.3,1.],6)
    with pytest.raises(ValueError):WheatDevelopmentModel([.1,.2,.3,.4,.5,.6,1.],6)
