"""Small weather-memory network with native cross-leaf history and full-range output."""
from copy import deepcopy
import time

import numpy as np
import torch
from torch import nn

from .dynamic_model import tensor_batch
from .field_history_models import field_history_anchor
from analysis.final_study.metrics import hierarchical_weights


class WeatherSequenceModel(nn.Module):
    def __init__(self,features,hidden=8):
        super().__init__()
        self.weather_gru=nn.GRU(6,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(features+hidden,48),nn.Tanh(),nn.Dropout(.1),
            nn.Linear(48,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self,batch,x,baseline):
        b=tensor_batch(batch) if not torch.is_tensor(batch['weather']) else batch
        memories,_=self.weather_gru(b['weather'])
        before=torch.cat([torch.zeros_like(memories[:,:1]),memories[:,:-1]],dim=1)
        context=before[b['query_field'],b['query_day']]
        delta=self.head(torch.cat([x,context],dim=-1)).squeeze(-1)
        return 100*torch.clamp(baseline+.4*delta,0,1)


def design(arrays,history,queries):
    anchor,extra=field_history_anchor(history,queries)
    x=np.column_stack([arrays['process'],arrays['history_events'],history.to_numpy(float),extra])
    if 'observed_percent' in queries or not np.isfinite(x).all():
        raise ValueError('Finite label-free native predictor design required')
    return x,anchor,extra[:,1].astype(bool)


def _baseline(fitted,anchor,available,queries):
    categories=queries.source.astype(str)+'|'+queries.leaf_index.astype(str)
    fallback=np.array([fitted['means'].get(k,fitted['pooled']) for k in categories])
    return np.where(available,anchor,fallback)


def fit_sequence(batch,arrays,history,training,config=None):
    config=dict(dict(epochs=180,members=3,learning_rate=.006,weight_decay=.02,
        hidden=8,seed=20261022),**(config or {}))
    torch.set_num_threads(1)
    q=training.drop(columns='observed_percent')
    x,anchor,available=design(arrays,history,q)
    weights=hierarchical_weights(training); weights=weights/weights.sum()
    y=training.observed_percent.to_numpy(float)/100
    mean=np.average(x,axis=0,weights=weights);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights))
    scale=np.maximum(scale,.01);xx=torch.tensor(np.clip((x-mean)/scale,-10,10),dtype=torch.float32)
    fitted=dict(config=config,mean=mean,scale=scale,pooled=float(np.average(y,weights=weights)),means={},
        training_ids=training.global_target_index.astype(int).tolist(),training_fields=list(batch['field_ids']),members=[])
    categories=training.source.astype(str)+'|'+training.leaf_index.astype(str)
    for k in categories.unique():
        mask=categories.eq(k).to_numpy();mass=weights[mask].sum()
        fitted['means'][k]=float((np.sum(weights[mask]*y[mask])+.04*fitted['pooled'])/(mass+.04))
    baseline=torch.tensor(_baseline(fitted,anchor,available,q),dtype=torch.float32)
    target=torch.tensor(y,dtype=torch.float32);w=torch.tensor(weights,dtype=torch.float32)
    b=tensor_batch(batch);started=time.monotonic()
    for member in range(config['members']):
        torch.manual_seed(config['seed']+member)
        model=WeatherSequenceModel(x.shape[1],config['hidden'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        trace=[]
        for epoch in range(config['epochs']):
            model.train();optimizer.zero_grad();pred=model(b,xx,baseline)/100
            loss=(w*(pred-target).square()).sum()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite weather-sequence objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if epoch in [int(config['epochs']*.6),int(config['epochs']*.85)]:
                for group in optimizer.param_groups:group['lr']*=.3
            if epoch%30==0 or epoch+1==config['epochs']:
                trace.append(dict(epoch=epoch+1,training_rmse=float(loss.detach().sqrt()*100),seconds=time.monotonic()-started))
        fitted['members'].append(dict(state_dict=deepcopy(model.state_dict()),trace=trace))
    return fitted


def predict_sequence(fitted,batch,arrays,history,queries):
    torch.set_num_threads(1);x,anchor,available=design(arrays,history,queries)
    xx=torch.tensor(np.clip((x-fitted['mean'])/fitted['scale'],-10,10),dtype=torch.float32)
    baseline=torch.tensor(_baseline(fitted,anchor,available,queries),dtype=torch.float32)
    b=tensor_batch(batch);predictions=[]
    for member in fitted['members']:
        model=WeatherSequenceModel(x.shape[1],fitted['config']['hidden']);model.load_state_dict(member['state_dict']);model.eval()
        with torch.no_grad():predictions.append(model(b,xx,baseline).numpy())
    result=np.mean(predictions,axis=0).astype(float)
    if not np.isfinite(result).all() or np.any((result<0)|(result>100)):
        raise RuntimeError('Invalid weather-sequence predictions')
    return result
