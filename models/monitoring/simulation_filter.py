"""Recurrent measurement-state filtering with an explicit initialization prior."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn


def tensors(batch):
    if torch.is_tensor(batch['weather']):return batch
    result={key:torch.tensor(batch[key],dtype=torch.float32) for key in ['weather','observed','static']}
    result['observed_mask']=torch.tensor(batch['observed_mask'],dtype=torch.bool)
    for key in ['query_field','query_day','query_leaf']:result[key]=torch.tensor(batch[key],dtype=torch.long)
    return result


class MeasurementFilter(nn.Module):
    def __init__(self,hidden=48,dropout=.1):
        super().__init__()
        self.observation_encoder=nn.Sequential(nn.Linear(14,16),nn.Tanh())
        self.weather_encoder=nn.Sequential(nn.Linear(6,12),nn.Tanh())
        self.initial=nn.Sequential(nn.Linear(3,hidden),nn.Tanh())
        self.gru=nn.GRU(31,hidden,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Dropout(dropout),nn.Linear(hidden,7))

    def forward(self,batch):
        b=tensors(batch)
        observed=torch.where(b['observed_mask'],b['observed'],torch.zeros_like(b['observed']))
        memory=self.observation_encoder(torch.cat([observed,b['observed_mask'].float()],dim=2))
        weather=self.weather_encoder(b['weather'])
        static=b['static'][:,None,:].expand(-1,weather.shape[1],-1)
        initial=self.initial(b['static'])
        sequence,_=self.gru(torch.cat([weather,memory,static],dim=2),initial[None])
        # Hidden[day q] has consumed only weather/observations through day q-1.
        hidden=torch.cat([initial[:,None,:],sequence],dim=1)
        context=hidden[b['query_field'],b['query_day']]
        predictions=100*torch.sigmoid(self.head(context))
        return predictions[torch.arange(len(context)),b['query_leaf']]


def fit_filter(batch,targets,weights,initial_state=None,config=None):
    config=dict(dict(hidden=48,dropout=.1,epochs=150,seed=20261022,learning_rate=.002,
        weight_decay=.001,map_sum_coefficient=1e-4),**(config or {}))
    torch.set_num_threads(1);torch.manual_seed(config['seed'])
    model=MeasurementFilter(config['hidden'],config['dropout'])
    if initial_state is not None:model.load_state_dict(initial_state)
    anchor={name:parameter.detach().clone() for name,parameter in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    b=tensors(batch);y=torch.tensor(np.asarray(targets,float)/100,dtype=torch.float32)
    original_weight=np.asarray(weights,float)
    if y.shape!=(len(batch['query_ids']),) or original_weight.shape!=y.shape or not np.isfinite(original_weight).all() or np.any(original_weight<=0):
        raise ValueError('Aligned positive original scientific target mass required')
    mass=torch.tensor(original_weight/original_weight.sum(),dtype=torch.float32)
    trace=[]
    for epoch in range(config['epochs']):
        model.train();optimizer.zero_grad()
        prediction=model(b)/100
        mse=torch.sum(mass*(prediction-y).square())
        penalty=config['map_sum_coefficient']*sum((parameter-anchor[name]).square().sum()
            for name,parameter in model.named_parameters())
        loss=mse+penalty
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite measurement-filter objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if epoch in [int(config['epochs']*.6),int(config['epochs']*.85)]:
            for group in optimizer.param_groups:group['lr']*=.3
        if epoch%25==0 or epoch+1==config['epochs']:
            trace.append(dict(epoch=epoch+1,weighted_mse=float(mse.detach()),map_penalty=float(penalty.detach()),
                objective=float(loss.detach()),training_rmse=float(100*mse.detach().sqrt())))
    return dict(config=config,state_dict=deepcopy(model.state_dict()),anchor_state=anchor,trace=trace,
        training_weight=original_weight,training_query_ids=np.asarray(batch['query_ids'],int).tolist(),
        training_field_ids=list(batch['field_ids']),pretrained=initial_state is not None,
        scientific_mass_globally_normalized_full_batch=True,map_penalty_convention='coefficient times SUM squared deviation from initialization')


def predict_filter(fitted,batch):
    torch.set_num_threads(1)
    model=MeasurementFilter(fitted['config']['hidden'],fitted['config']['dropout'])
    model.load_state_dict(fitted['state_dict']);model.eval()
    with torch.no_grad():value=model(batch).numpy().astype(float)
    if not np.isfinite(value).all() or np.any((value<0)|(value>100)):
        raise RuntimeError('Invalid recurrent severity forecast')
    return value


def pretrain_filter(inputs,config=None):
    from .dynamic_data import subset_sequences
    from .epidemic_world import make_worlds
    config=dict(dict(hidden=48,dropout=.1,worlds=2048,steps=512,batch_fields=32,
        seed=20261022,learning_rate=.003,weight_decay=.001),**(config or {}))
    torch.set_num_threads(1);torch.manual_seed(config['seed'])
    worlds,target,weights,parameters,audit=make_worlds(inputs,config['worlds'],config['seed'])
    model=MeasurementFilter(config['hidden'],config['dropout'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    rng=np.random.default_rng(config['seed']+1);trace=[]
    for step in range(config['steps']):
        chosen=rng.choice(config['worlds'],min(config['batch_fields'],config['worlds']),replace=False)
        batch=subset_sequences(worlds,[worlds['field_ids'][int(i)] for i in chosen])
        index=batch['query_ids'];mass=torch.tensor(weights[index]/weights[index].sum(),dtype=torch.float32)
        y=torch.tensor(target[index]/100,dtype=torch.float32)
        model.train();optimizer.zero_grad()
        loss=torch.sum(mass*(model(batch)/100-y).square())
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite assumed-world pretraining objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step in [int(config['steps']*.6),int(config['steps']*.85)]:
            for group in optimizer.param_groups:group['lr']*=.3
        if step%50==0 or step+1==config['steps']:
            trace.append(dict(step=step+1,synthetic_weighted_mse=float(loss.detach()),
                synthetic_query_targets=len(index),synthetic_weight_mass=float(weights[index].sum())))
    return dict(config=config,state_dict=deepcopy(model.state_dict()),trace=trace,
        world_parameters=parameters,world_audit=audit,source_field_ids=list(inputs['source_field_ids']),
        real_disease_values_used=False,synthetic_worlds_are_not_measured_observations=True)
