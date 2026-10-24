"""Learn all-past field adaptation with a fixed-scale conditional Gaussian kernel."""
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn


def prefix_layout(frame):
    frame=frame.reset_index(drop=True)
    dates=pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]')
    if pd.isna(dates).any() or frame.field_id.isna().any():
        raise ValueError('Known field identity and assessment dates required')
    groups=[];order=[]
    for _,group in frame.groupby('field_id',sort=False):
        ix=group.index.to_numpy(int)
        ix=ix[np.argsort(dates[ix],kind='stable')]
        cut=np.searchsorted(dates[ix],dates[ix],side='left')
        groups.append(dict(index=ix,past_count=cut))
        order.extend(ix.tolist())
    return dict(groups=groups,inverse=np.argsort(order),rows=len(frame))


def conditional_error(phi,noise,residual,layout):
    """Exact Gaussian coefficient conditioning before each entire assessment day."""
    if phi.shape[0]!=layout['rows'] or noise.shape!=residual.shape or noise.shape!=(layout['rows'],):
        raise ValueError('Matching field contexts, observation noise and residuals required')
    if not torch.isfinite(phi).all() or not torch.isfinite(noise).all() or not torch.isfinite(residual).all() or torch.any(noise<=0):
        raise ValueError('Finite kernel quantities and positive observation noise required')
    dimension=phi.shape[1];identity=torch.eye(dimension,dtype=phi.dtype,device=phi.device)
    values=[]
    for group in layout['groups']:
        ix=torch.as_tensor(group['index'],device=phi.device)
        cut=torch.as_tensor(group['past_count'],device=phi.device)
        features=phi[ix];precision=noise[ix].reciprocal().square()
        outer=features[:,:,None]*features[:,None,:]*precision[:,None,None]
        information=features*(residual[ix]*precision)[:,None]
        # Separate field cumsums prevent numerical cancellation of other fields.
        moment=torch.cat([torch.zeros_like(outer[:1]),outer.cumsum(0)])[cut]
        rhs=torch.cat([torch.zeros_like(information[:1]),information.cumsum(0)])[cut]
        coefficient=torch.linalg.solve(identity[None]+moment,rhs[:,:,None]).squeeze(-1)
        values.append(torch.sum(features*coefficient,dim=1))
    return torch.cat(values)[torch.as_tensor(layout['inverse'],device=phi.device)]


class FieldMetaKernel(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(features+9,16),nn.Tanh())
        self.context_basis=nn.Sequential(nn.Linear(16,4),nn.Tanh())
        self.relative_amplitudes=nn.Parameter(torch.full((11,),float(np.log(np.expm1(1.)))))
        self.noise=nn.Linear(16,1)
        nn.init.zeros_(self.noise.weight)
        nn.init.constant_(self.noise.bias,float(np.log((.15-.02)/(.50-.15))))
        self.fixed_prior_sd=.25

    def basis(self,x,categories):
        hidden=self.body(torch.cat([x,categories],dim=1))
        raw=torch.cat([torch.ones_like(x[:,:1]),categories[:,2:],self.context_basis(hidden)],dim=1)
        amplitude=torch.cat([torch.ones_like(self.relative_amplitudes[:1]),
            torch.nn.functional.softplus(self.relative_amplitudes)])
        raw=raw*amplitude
        # The prior diagonal is fixed; MSE cannot identify an absolute noise/covariance scale.
        phi=self.fixed_prior_sd*raw/torch.linalg.vector_norm(raw,dim=1,keepdim=True)
        noise=.02+.48*torch.sigmoid(self.noise(hidden).squeeze(-1))
        return phi,noise

    def forward(self,x,categories,residual,layout):
        phi,noise=self.basis(x,categories)
        return conditional_error(phi,noise,residual,layout)


def _categories(frame):
    code=frame.leaf_index.to_numpy(int)
    if np.any((code<0)|(code>6)) or not frame.source.isin(['BASF','Corteva']).all():
        raise ValueError('Supported nominal source/native codes required')
    return np.column_stack([frame.source.eq('BASF'),frame.source.eq('Corteva'),np.eye(7)[code]]).astype(float)


def fit_meta_kernel(x,frame,baseline,weights,config=None):
    config=dict(dict(epochs=300,members=3,seed=20261022),**(config or {}))
    torch.set_num_threads(1)
    x=np.column_stack([np.asarray(x,float),np.asarray(baseline,float)/100])
    weights=np.asarray(weights,float)
    if not np.isfinite(x).all() or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('Finite contexts and positive scientific weights required')
    mass=weights/weights.sum();center=np.average(x,axis=0,weights=mass)
    scale=np.maximum(.01,np.sqrt(np.average((x-center)**2,axis=0,weights=mass)))
    xx=torch.tensor(np.clip((x-center)/scale,-8,8),dtype=torch.float64)
    categories=torch.tensor(_categories(frame),dtype=torch.float64)
    residual=torch.tensor((frame.observed_percent.to_numpy(float)-np.asarray(baseline))/100,dtype=torch.float64)
    weight=torch.tensor(mass,dtype=torch.float64);layout=prefix_layout(frame)
    fitted=dict(features=x.shape[1],center=center,scale=scale,config=config,members=[],
        training_ids=frame.global_target_index.astype(int).tolist(),training_baseline=np.asarray(baseline,float),
        training_weight=weights,fixed_prior_standard_deviation_fraction=.25,latent_coefficients=12,
        observation_noise_fraction_bounds=[.02,.50],input_nominal_category_count=9)
    for member in range(config['members']):
        torch.manual_seed(config['seed']+member)
        model=FieldMetaKernel(x.shape[1]).double()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        trace=[]
        for epoch in range(config['epochs']):
            optimizer.zero_grad()
            delta=model(xx,categories,residual,layout)
            loss=torch.sum(weight*(delta-residual).square())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite field-kernel prequential objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if epoch in [int(config['epochs']*.6),int(config['epochs']*.85)]:
                for group in optimizer.param_groups:group['lr']*=.3
            if epoch%50==0 or epoch+1==config['epochs']:
                trace.append(dict(epoch=epoch+1,loss=float(loss.detach()),training_rmse=float(100*loss.detach().sqrt())))
        fitted['members'].append(dict(state_dict=deepcopy(model.state_dict()),trace=trace))
    return fitted


def predict_meta_kernel(fitted,x,frame,baseline,shrinkage=1.):
    torch.set_num_threads(1)
    baseline=np.asarray(baseline,float)
    x=np.column_stack([np.asarray(x,float),baseline/100])
    xx=torch.tensor(np.clip((x-fitted['center'])/fitted['scale'],-8,8),dtype=torch.float64)
    categories=torch.tensor(_categories(frame),dtype=torch.float64)
    residual=torch.tensor((frame.observed_percent.to_numpy(float)-baseline)/100,dtype=torch.float64)
    layout=prefix_layout(frame);predictions=[]
    for member in fitted['members']:
        model=FieldMetaKernel(fitted['features']).double();model.load_state_dict(member['state_dict']);model.eval()
        with torch.no_grad():predictions.append(model(xx,categories,residual,layout).numpy())
    result=np.clip(baseline+100*shrinkage*np.mean(predictions,axis=0),0,100)
    for group in layout['groups']:
        first=group['index'][group['past_count']==0];result[first]=baseline[first]
    return result
