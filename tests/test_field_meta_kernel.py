import numpy as np
import pandas as pd
import torch


def test_field_kernel_matches_all_past_ridge_update_and_withholds_entire_query_day():
    from models.monitoring.field_meta_kernel import prefix_layout,conditional_error
    frame=pd.DataFrame(dict(field_id=['a','a','a','a','b'],date=['2015-05-01','2015-05-02','2015-05-02','2015-05-03','2015-05-01']))
    phi=torch.tensor([[1.,0.],[1.,1.],[.5,1.],[1.,2.],[2.,2.]],dtype=torch.float64)
    noise=torch.ones(5,dtype=torch.float64)
    error=torch.tensor([.2,.3,-.1,.9,-4.],dtype=torch.float64)
    actual=conditional_error(phi,noise,error,prefix_layout(frame)).detach().numpy()
    expected=[]
    for i,row in frame.iterrows():
        past=np.flatnonzero(frame.field_id.eq(row.field_id)&pd.to_datetime(frame.date).lt(pd.Timestamp(row.date)))
        p=phi.numpy()[past]
        expected.append(phi.numpy()[i]@np.linalg.solve(np.eye(2)+p.T@p,p.T@error.numpy()[past]))
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    changed=error.clone();changed[1:3]=500;changed[3]=999;changed[4]=888
    after=conditional_error(phi,noise,changed,prefix_layout(frame)).detach().numpy()
    np.testing.assert_array_equal(actual[:3],after[:3])
    assert actual[0]==actual[4]==0
    permutation=np.array([4,2,0,3,1]);inverse=np.argsort(permutation)
    shuffled=conditional_error(phi[permutation],noise[permutation],error[permutation],prefix_layout(frame.iloc[permutation]))
    np.testing.assert_allclose(shuffled.detach().numpy()[inverse],actual,atol=1e-14)


def test_field_kernel_retains_early_support_beyond_32_records_and_has_finite_gradients():
    from models.monitoring.field_meta_kernel import prefix_layout,conditional_error,FieldMetaKernel
    frame=pd.DataFrame(dict(field_id='a',date=pd.date_range('2015-04-01',periods=41)))
    phi=torch.ones((41,2),dtype=torch.float64,requires_grad=True)
    noise=torch.ones(41,dtype=torch.float64,requires_grad=True)
    error=torch.zeros(41,dtype=torch.float64);error[0]=1
    delta=conditional_error(phi,noise,error,prefix_layout(frame))
    np.testing.assert_allclose(delta.detach().numpy()[-1],2/81,atol=1e-14)
    delta[-1].square().backward()
    assert torch.isfinite(phi.grad).all() and torch.isfinite(noise.grad).all()
    assert phi.grad.abs().sum()>0 and noise.grad.abs().sum()>0
    model=FieldMetaKernel(2).double()
    categories=torch.zeros((3,9),dtype=torch.float64);categories[:,0]=1
    categories[torch.arange(3),torch.tensor([2,3,8])]=1
    learned_phi,_=model.basis(torch.randn((3,2),dtype=torch.float64),categories)
    np.testing.assert_allclose(learned_phi.detach().square().sum(1).numpy(),.25**2,atol=1e-15)
    varied_layout=prefix_layout(pd.DataFrame(dict(field_id='x',date=pd.date_range('2015-01-01',periods=3))))
    delta=model(torch.tensor([[0.,1.],[1.,0.],[.4,-.5]],dtype=torch.float64),categories,
        torch.tensor([.2,-.1,.4],dtype=torch.float64),varied_layout)
    delta.square().sum().backward()
    assert model.context_basis[0].weight.grad.abs().sum()>0
    assert model.noise.weight.grad.abs().sum()>0
    assert torch.isfinite(model.context_basis[0].weight.grad).all()


def test_meta_kernel_learns_field_adaptation_and_preserves_first_visits():
    from models.monitoring.field_meta_kernel import fit_meta_kernel,predict_meta_kernel
    records=[]
    for field in range(12):
        for day in range(4):
            records.append(dict(field_id=str(field),date=f'2015-05-{day+1:02d}',leaf_index=0,source='BASF',
                global_target_index=len(records),observed_percent=70. if field%2 else 30.))
    frame=pd.DataFrame(records);x=np.zeros((len(frame),2));baseline=np.full(len(frame),50.)
    fitted=fit_meta_kernel(x,frame,baseline,np.ones(len(frame)),config=dict(members=1,epochs=200))
    assert fitted['fixed_prior_standard_deviation_fraction']==.25
    prediction=predict_meta_kernel(fitted,x,frame,baseline)
    first=np.arange(len(frame))%4==0
    np.testing.assert_array_equal(prediction[first],baseline[first])
    assert np.sqrt(np.mean((prediction[~first]-frame.observed_percent.to_numpy()[~first])**2))<8
    # The first-visit reference has a fixed, irreducible-to-this-filter loss term.
    fixed_cost=np.sum(((baseline[first]-frame.observed_percent.to_numpy()[first])/100)**2)/len(frame)
    trace=fitted['members'][0]['trace']
    assert trace[-1]['loss']-fixed_cost<(trace[0]['loss']-fixed_cost)*.5
    altered=frame.copy();altered.loc[3,'observed_percent']=100
    np.testing.assert_array_equal(predict_meta_kernel(fitted,x,altered,baseline)[:4],prediction[:4])
