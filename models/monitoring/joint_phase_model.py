"""Joint source-native epidemic curves with integrated field phase uncertainty."""
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit, log_ndtr, softmax


def _code(frame):
    return frame.leaf_index.to_numpy(int)+7*frame.source.eq('Corteva').to_numpy(int)


def _nodes():
    nodes, weights = np.polynomial.hermite.hermgauss(7)
    pressure = np.repeat(np.sqrt(2)*nodes, 3)
    growth = np.tile([.6, 1., 1/.6], 7)
    prior = np.repeat(weights/np.sqrt(np.pi), 3)*np.tile([.25, .5, .25], 7)
    return pressure, growth, prior


class JointPhase(nn.Module):
    def __init__(self, features, intercept):
        super().__init__()
        self.intercept = nn.Parameter(torch.tensor(intercept, dtype=torch.float32))
        self.log_slope = nn.Parameter(torch.full((14,), float(np.log(8.))))
        self.weather = nn.Parameter(torch.zeros((2, features)))
        self.log_pressure_sd = nn.Parameter(torch.zeros(2))
        pressure, growth, prior = _nodes()
        self.register_buffer('pressure', torch.tensor(pressure, dtype=torch.float32))
        self.register_buffer('growth', torch.tensor(growth, dtype=torch.float32))
        self.register_buffer('prior', torch.tensor(prior, dtype=torch.float32))

    def forward(self, x, clock, code):
        source = code//7
        intercept = self.intercept[code]+(x*self.weather[source]).sum(1)
        slope = self.log_slope.clamp(-3, 4).exp()[code]
        sd = self.log_pressure_sd.clamp(-2.3, 1.4).exp()[source]
        eta = (intercept[:, None]+slope[:, None]*(clock[:, None]-2.)*self.growth[None, :]
            +sd[:, None]*self.pressure[None, :])
        return 100*torch.sigmoid(eta)


def fit_phase(x, clock, frame, weights, epochs=400):
    torch.set_num_threads(1)
    frame = frame.reset_index(drop=True)
    x, clock, weights = np.asarray(x, float), np.asarray(clock, float), np.asarray(weights, float)
    w = weights/weights.sum()
    center = np.average(x, axis=0, weights=w)
    scale = np.maximum(.01, np.sqrt(np.average((x-center)**2, axis=0, weights=w)))
    code = _code(frame)
    y = frame.observed_percent.to_numpy(float)
    intercept = np.zeros(14)
    for k in np.unique(code):
        mask = code == k
        mean = np.clip(np.average(y[mask]/100, weights=w[mask]), .005, .995)
        intercept[k] = np.log(mean/(1-mean))-8*(np.average(clock[mask], weights=w[mask])-2.)
    model = JointPhase(x.shape[1], intercept)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    xx = torch.tensor(np.clip((x-center)/scale, -8, 8), dtype=torch.float32)
    cc = torch.tensor(clock, dtype=torch.float32); category = torch.tensor(code)
    target = torch.tensor(y, dtype=torch.float32)[:, None]
    fields, inverse = np.unique(frame.field_id.astype(str), return_inverse=True)
    field_mass = np.bincount(inverse, weights=w)
    row_weight = np.empty(len(frame))
    for k in range(len(fields)):
        mask = inverse == k
        row_weight[mask] = min(4, frame.date[mask].nunique())*w[mask]/field_mass[k]
    field_index = torch.tensor(inverse)
    rw = torch.tensor(row_weight, dtype=torch.float32)[:, None]
    fw = torch.tensor(field_mass, dtype=torch.float32)
    initial = torch.tensor(intercept, dtype=torch.float32)
    trace = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        mu = model(xx, cc, category)
        density = -.5*((target-mu)/15.)**2-np.log(15.)-.5*np.log(2*np.pi)
        density = torch.where(target == 0, torch.special.log_ndtr(-mu/15.), density)
        density = torch.where(target == 100, torch.special.log_ndtr((mu-100)/15.), density)
        likelihood = torch.zeros((len(fields), 21)).index_add(0, field_index, rw*density)
        loss = -(fw*torch.logsumexp(torch.log(model.prior)[None, :]+likelihood, dim=1)).sum()
        penalty = (.02*model.weather.square().mean()+.0002*(model.intercept-initial).square().mean()
            +.002*(model.log_slope-np.log(8.)).square().mean()+.002*model.log_pressure_sd.square().mean())
        objective = loss+penalty
        if not torch.isfinite(objective):
            raise RuntimeError('Nonfinite joint field likelihood')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if epoch % 50 == 0 or epoch+1 == epochs:
            trace.append(dict(epoch=epoch+1, weighted_marginal_negative_loglikelihood=float(loss.detach())))
    return dict(state_dict=deepcopy(model.state_dict()), features=x.shape[1], center=center, scale=scale,
        initial_intercept=intercept, trace=trace, epochs=epochs, noise_pp=15.,
        supported_codes=np.unique(code).tolist(), pooled_percent=float(np.average(y, weights=w)),
        training_ids=frame.global_target_index.astype(int).tolist(), training_weight=weights)


def predict_phase(fitted, x, clock, frame, baseline):
    torch.set_num_threads(1)
    frame = frame.reset_index(drop=True)
    baseline = np.asarray(baseline, float)
    code = _code(frame)
    known = np.isin(code, fitted['supported_codes']) & frame.source.isin(['BASF', 'Corteva']).to_numpy()
    model = JointPhase(fitted['features'], fitted['initial_intercept'])
    model.load_state_dict(fitted['state_dict']); model.eval()
    with torch.no_grad():
        xx = torch.tensor(np.clip((np.asarray(x)-fitted['center'])/fitted['scale'], -8, 8), dtype=torch.float32)
        curves = model(xx, torch.tensor(clock, dtype=torch.float32), torch.tensor(code)).numpy().astype(float)
    prior = model.prior.numpy().astype(float); prior /= prior.sum()
    result = dict(posterior=baseline.copy(), anchor=baseline.copy(),
        free=np.where(known, curves@prior, fitted['pooled_percent']), ess=np.zeros(len(frame)))
    dates = pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
    y = frame.observed_percent.to_numpy(float)
    sigma = fitted['noise_pp']
    density = -.5*((y[:, None]-curves)/sigma)**2-np.log(sigma)-.5*np.log(2*np.pi)
    density[y == 0] = log_ndtr(-curves[y == 0]/sigma)
    density[y == 100] = log_ndtr((curves[y == 100]-100)/sigma)
    for _, group in frame.groupby('field_id', sort=False):
        ix = group.index.to_numpy()
        for now in np.unique(dates[ix]):
            query = ix[(dates[ix] == now) & known[ix]]
            past = ix[(dates[ix] < now) & known[ix]]
            if not len(past):
                result['ess'][query] = 1/np.sum(prior**2)
                continue
            day, count = np.unique(dates[past], return_counts=True)
            count = dict(zip(day, count))
            weight = np.array([1./count[d] for d in dates[past]])
            log_likelihood = min(4, len(day))*np.average(density[past], axis=0, weights=weight)
            posterior = .99*softmax(np.log(prior)+log_likelihood)+.01*prior
            result['posterior'][query] = curves[query]@posterior
            result['anchor'][query] = result['posterior'][query]
            result['ess'][query] = 1/np.sum(posterior**2)
            for q in query:
                same = past[code[past] == code[q]]
                if len(same):
                    latest = same[dates[same] == dates[same].max()]
                    increment = (curves[q]-curves[latest].mean(axis=0))@posterior
                    result['anchor'][q] = np.clip(y[latest].mean()+increment, 0, 100)
    return result
