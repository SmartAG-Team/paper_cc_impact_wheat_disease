"""Compact differentiable disease dynamics with field-propensity filtering.

S/E1/E2/E3/V are unverified native-code tissue proxies. A shared Gaussian
field factor is integrated by quadrature, updated by censored-normal
observations, followed by a learned partial visible-state correction.
"""
from copy import deepcopy
import math
import time

import numpy as np
import torch
from torch import nn


def tensor_batch(batch):
    return {k:torch.tensor(batch[k]) for k in ['weather','valid_weather','observed',
        'observed_mask','static','sources','query_field','query_day','query_leaf']}


def censored_log_density(y, mu, sigma):
    z=(y-mu)/sigma
    interior=-.5*z.square()-torch.log(sigma)-.5*math.log(2*math.pi)
    return torch.where(y<=0,torch.special.log_ndtr(-mu/sigma),
        torch.where(y>=1,torch.special.log_ndtr((mu-1)/sigma),interior))


def censored_mean(mu,sigma):
    a=-mu/sigma; b=(1-mu)/sigma
    cdf=lambda z:.5*(1+torch.erf(z/math.sqrt(2)))
    pdf=lambda z:torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
    return mu*(cdf(b)-cdf(a))+sigma*(pdf(a)-pdf(b))+1-cdf(b)


class DailyDiseaseModel(nn.Module):
    def __init__(self, nodes=7, hidden=8):
        super().__init__()
        self.rate_net=nn.Sequential(nn.Linear(6,hidden),nn.Tanh(),nn.Linear(hidden,2))
        nn.init.zeros_(self.rate_net[-1].weight); nn.init.zeros_(self.rate_net[-1].bias)
        self.prior_net=nn.Linear(3,1,bias=False); nn.init.zeros_(self.prior_net.weight)
        self.leaf_effect=nn.Parameter(torch.zeros(2,7))
        self.infection_logit=nn.Parameter(torch.tensor(-2.2))
        self.latent_logit=nn.Parameter(torch.tensor(-1.1))
        self.initial_logit=nn.Parameter(torch.tensor(-6.))
        self.propensity_scale=nn.Parameter(torch.tensor(.4))
        self.splash_logit=nn.Parameter(torch.tensor(0.))
        self.noise_logit=nn.Parameter(torch.full((2,),-1.4))
        self.gain_logit=nn.Parameter(torch.full((2,),1.4))
        x,w=np.polynomial.hermite.hermgauss(nodes)
        self.register_buffer('nodes',torch.tensor(x*math.sqrt(2),dtype=torch.float32))
        self.register_buffer('prior_weights',torch.tensor(w/math.sqrt(math.pi),dtype=torch.float32))

    def forward(self, batch, assimilate=True):
        b=tensor_batch(batch) if not torch.is_tensor(batch['weather']) else batch
        weather=b['weather']; n,days,_=weather.shape; p=len(self.nodes)
        source=b['sources']; sigma=.025+.375*torch.sigmoid(self.noise_logit[source])
        u=self.prior_net(b['static'])+self.nodes[None,:]*(.1+1.9*torch.sigmoid(self.propensity_scale))
        leaf=.8*torch.tanh(self.leaf_effect[source])
        initial=.15*torch.sigmoid(self.initial_logit+u[:,:,None]+leaf[:,None,:])
        initial=initial.expand(n,p,7)
        state=torch.stack([1-initial,initial/3,initial/3,initial/3,torch.zeros_like(initial)],dim=-1)
        logw=self.prior_weights.log()[None,:].expand(n,p)
        rates=self.rate_net(weather)
        means=[]; components=[]; logweights=[]
        for day in range(days):
            mu=state[...,4]
            obsmean=censored_mean(mu,sigma[:,None,None])
            means.append((logw.exp()[:,:,None]*obsmean).sum(1))
            components.append(mu); logweights.append(logw)
            observed=b['observed'][:,day]; mask=b['observed_mask'][:,day]
            if assimilate and mask.any():
                ll=censored_log_density(observed[:,None,:],mu,sigma[:,None,None])
                newlog=logw+(ll*mask[:,None,:]).sum(-1)
                newlog=newlog-torch.logsumexp(newlog,dim=1,keepdim=True)
                # A 1% prior refresh avoids irreversible quadrature collapse.
                refreshed=.99*newlog.exp()+.01*self.prior_weights[None,:]
                logw=torch.where(mask.any(-1)[:,None],refreshed.log(),logw)
                gain=torch.sigmoid(self.gain_logit[source])[:,None,None]
                visible=mu+gain*(observed[:,None,:]-mu)
                visible=torch.where(mask[:,None,:],visible,mu)
                remainder=state[...,:4]
                # Normalize by actual tissue mass: 1-mu loses precision at saturation.
                total=remainder.sum(-1,keepdim=True)
                complement=(1-visible).clamp_min(0)[...,None]
                scaled=remainder/total.clamp_min(1e-8)*complement
                fallback=torch.cat([complement,torch.zeros_like(remainder[...,1:])],dim=-1)
                recovered=torch.where(total>1e-8,scaled,fallback)
                updated=torch.cat([recovered,visible[...,None]],dim=-1)
                state=torch.where(mask[:,None,:,None],updated,state)
            active=b['valid_weather'][:,day].float()[:,None,None]
            temp=weather[:,day,0]*12+12
            thermal=weather[:,day,5].clamp_min(0)
            humidity=torch.sigmoid(weather[:,day,1])
            rain=-torch.expm1(-torch.expm1(weather[:,day,2]*2)/2)
            suitability=torch.exp(-.5*((temp-18)/10).square())*(.05+.95*humidity)
            infectious=state[...,4].mean(-1,keepdim=True)
            logits=self.infection_logit+rates[:,day,0][:,None,None]+u[:,:,None]+leaf[:,None,:]
            hazard=.1*torch.sigmoid(logits)*suitability[:,None,None]
            hazard=hazard*(1+4*torch.sigmoid(self.splash_logit)*rain[:,None,None]*infectious)
            infected=state[...,0]*(-torch.expm1(-hazard))*active
            latent_days=5+75*torch.sigmoid(self.latent_logit+rates[:,day,1])
            q=(-torch.expm1(-3*thermal/latent_days))[:,None,None]*active
            transfer=state[...,1:4]*q[...,None]
            state=torch.stack([state[...,0]-infected,
                state[...,1]+infected-transfer[...,0],
                state[...,2]+transfer[...,0]-transfer[...,1],
                state[...,3]+transfer[...,1]-transfer[...,2],
                state[...,4]+transfer[...,2]],dim=-1)
        fi,di,li=[b[k] for k in ['query_field','query_day','query_leaf']]
        allmean=torch.stack(means,dim=1)
        allmu=torch.stack(components,dim=1)
        allweight=torch.stack(logweights,dim=1)
        return dict(mean=100*allmean[fi,di,li],
            component_mu=allmu[fi,di,:,li],log_weights=allweight[fi,di],sigma=sigma[fi],
            state=state,weights=logw.exp())


def fit_dynamic(batch, targets, weights, config=None):
    config=dict(dict(epochs=120,learning_rate=.02,weight_decay=.001,seed=20261022,
        nodes=7,hidden=8),**(config or {}))
    torch.set_num_threads(1); torch.manual_seed(config['seed'])
    model=DailyDiseaseModel(config['nodes'],config['hidden'])
    b=tensor_batch(batch)
    y=torch.as_tensor(np.asarray(targets)/100,dtype=torch.float32)
    w=torch.as_tensor(np.asarray(weights)/np.sum(weights),dtype=torch.float32)
    if y.shape!=w.shape or len(y)!=len(batch['query_ids']) or not torch.isfinite(y).all():
        raise ValueError('Aligned finite training targets and weights required')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    trace=[]; started=time.monotonic()
    for epoch in range(config['epochs']):
        optimizer.zero_grad(); result=model(b)
        ll=censored_log_density(y[:,None],result['component_mu'],result['sigma'][:,None])
        nll=-torch.logsumexp(result['log_weights']+ll,dim=1)
        mse=(result['mean']/100-y).square()
        loss=(w*(mse+.02*nll)).sum()
        loss=loss+.0001*model.leaf_effect.square().mean()+.0001*model.prior_net.weight.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite disease-state objective')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite disease-state gradient')
        optimizer.step()
        if epoch in [int(config['epochs']*.65),int(config['epochs']*.85)]:
            for group in optimizer.param_groups: group['lr']*=.4
        if epoch%10==0 or epoch+1==config['epochs']:
            trace.append(dict(epoch=epoch+1,loss=float(loss.detach()),
                training_rmse=float(torch.sqrt((w*mse).sum()).detach()*100),
                seconds=time.monotonic()-started))
    return dict(config=config,state_dict=deepcopy(model.state_dict()),trace=trace,
        training_query_ids=batch['query_ids'].astype(int).tolist(),training_field_ids=list(batch['field_ids']))


def predict_dynamic(fitted,batch,assimilate=True):
    torch.set_num_threads(1)
    model=DailyDiseaseModel(fitted['config']['nodes'],fitted['config']['hidden'])
    model.load_state_dict(fitted['state_dict']); model.eval()
    with torch.no_grad():
        value=model(batch,assimilate=assimilate)['mean'].numpy().astype(float)
    if not np.isfinite(value).all() or np.any((value<0)|(value>100)):
        raise RuntimeError('Invalid disease prediction')
    return value
