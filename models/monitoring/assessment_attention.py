"""Learn query-dependent attention over strictly earlier field assessments."""
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn


def make_memory(frame, baseline, clock, max_memory=32):
    frame = frame.reset_index(drop=True)
    dates = pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
    baseline, clock = np.asarray(baseline, float), np.asarray(clock, float)
    y = frame.observed_percent.to_numpy(float) / 100
    code = frame.leaf_index.to_numpy(int)
    tokens = np.zeros((len(frame), max_memory, 16), dtype=np.float32)
    mask = np.zeros((len(frame), max_memory), dtype=bool)
    category = np.column_stack([frame.source.eq('BASF'), frame.source.eq('Corteva'), np.eye(7)[code]])
    for _, group in frame.groupby('field_id', sort=False):
        ix = group.index.to_numpy()
        order = ix[np.argsort(dates[ix], kind='stable')]
        for q in ix:
            past = order[dates[order] < dates[q]][-max_memory:]
            if not len(past):
                continue
            token = np.column_stack([category[past], y[past], baseline[past]/100,
                y[past]-baseline[past]/100, (dates[q]-dates[past])/30,
                clock[past]-2., clock[q]-clock[past], code[past] == code[q]])
            tokens[q, :len(past)] = token
            mask[q, :len(past)] = True
    return dict(tokens=tokens, mask=mask)


class AssessmentAttention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(features, 32), nn.LayerNorm(32), nn.Tanh())
        self.token = nn.Sequential(nn.Linear(16, 32), nn.Tanh())
        self.attention = nn.MultiheadAttention(32, 4, dropout=.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Dropout(.1), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, tokens, mask, baseline):
        query = self.query(x)
        null = torch.zeros((len(x), 1, 16), dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([null, tokens], dim=1)
        padding = torch.cat([torch.zeros((len(x), 1), dtype=torch.bool, device=mask.device), ~mask], dim=1)
        context = self.token(tokens)
        memory, _ = self.attention(query[:, None], context, context, key_padding_mask=padding, need_weights=False)
        delta = self.head(torch.cat([query, memory[:, 0]], dim=1)).squeeze(-1)
        return delta * mask.any(dim=1)


def fit_attention(x, frame, baseline, clock, weights, config=None):
    config = dict(dict(epochs=300, members=3, seed=20261022, max_memory=32), **(config or {}))
    torch.set_num_threads(1)
    x, baseline, weights = np.asarray(x, float), np.asarray(baseline, float), np.asarray(weights, float)
    x = np.column_stack([x, baseline/100])
    w = weights / weights.sum()
    center = np.average(x, axis=0, weights=w)
    scale = np.maximum(.01, np.sqrt(np.average((x-center)**2, axis=0, weights=w)))
    xx = torch.tensor(np.clip((x-center)/scale, -8, 8), dtype=torch.float32)
    memory = make_memory(frame, baseline, clock, config['max_memory'])
    tokens, mask = torch.tensor(memory['tokens']), torch.tensor(memory['mask'])
    base = torch.tensor(baseline/100, dtype=torch.float32)
    target = torch.tensor(frame.observed_percent.to_numpy(float)/100, dtype=torch.float32)
    mass = torch.tensor(w, dtype=torch.float32)
    fitted = dict(config=config, center=center, scale=scale, features=x.shape[1], members=[],
        training_ids=frame.global_target_index.astype(int).tolist(), training_weight=weights,
        training_baseline=baseline, training_memory=memory)
    for member in range(config['members']):
        torch.manual_seed(config['seed']+member)
        model = AssessmentAttention(x.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
        trace = []
        for epoch in range(config['epochs']):
            model.train()
            optimizer.zero_grad()
            delta = model(xx, tokens, mask, base)
            # Clip only at inference: clipped training means can strand an
            # overshooting neural forecast at a zero-gradient 0/100 endpoint.
            prediction = base+delta
            loss = (mass*(prediction-target).square()).sum()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite attention objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if epoch in [int(config['epochs']*.6), int(config['epochs']*.85)]:
                for group in optimizer.param_groups:
                    group['lr'] *= .3
            if epoch % 50 == 0 or epoch+1 == config['epochs']:
                trace.append(dict(epoch=epoch+1, training_rmse=float(100*loss.detach().sqrt())))
        fitted['members'].append(dict(state_dict=deepcopy(model.state_dict()), trace=trace))
    return fitted


def predict_attention(fitted, x, frame, baseline, clock, shrinkage=1.):
    torch.set_num_threads(1)
    baseline = np.asarray(baseline, float)
    x = np.column_stack([np.asarray(x, float), baseline/100])
    xx = torch.tensor(np.clip((x-fitted['center'])/fitted['scale'], -8, 8), dtype=torch.float32)
    memory = make_memory(frame, baseline, clock, fitted['config']['max_memory'])
    tokens, mask = torch.tensor(memory['tokens']), torch.tensor(memory['mask'])
    base = torch.tensor(baseline/100, dtype=torch.float32)
    members = []
    for member in fitted['members']:
        model = AssessmentAttention(fitted['features'])
        model.load_state_dict(member['state_dict'])
        model.eval()
        with torch.no_grad():
            members.append(model(xx, tokens, mask, base).numpy())
    value = np.clip(baseline+shrinkage*100*np.mean(members, axis=0), 0, 100)
    value[~memory['mask'].any(axis=1)] = baseline[~memory['mask'].any(axis=1)]
    return value
