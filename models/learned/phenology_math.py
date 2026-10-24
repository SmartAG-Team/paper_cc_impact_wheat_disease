"""Original frozen phenology mathematical definitions, extracted verbatim."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn

STAGES = [10, 31, 51, 87]


HORIZON = 330


FEATURE_NAMES = ["tmean_c", "tmax_c", "tmin_c", "daylength_hours", "elapsed_days_over_330",
                 "day_of_year_sine", "day_of_year_cosine", "past_7d_mean_c", "past_30d_mean_c",
                 "past_mean_c", "past_frost_days_over_330", "past_hot_days_over_330",
                 "vernalization_over_40", "cumulative_gdd_over_1000", "cumulative_tp_over_1000",
                 "cumulative_tpv_over_1000", "sowing_day_sine", "sowing_day_cosine"]


def build_features(weather, dates, latitude, gates):
    """Only weather, known sowing and astronomical daylength enter predictors.

    The first daily increment includes sowing day to reproduce the wheat TPV
    convention. No observed stage, target duration, end date or future summary
    enters this function. Missing weather must be handled as a truncated prefix
    by its caller, never imputed from future weather.
    """
    weather = np.asarray(weather, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0 or len(weather) != len(dates) or weather.shape[1] != 4:
        raise ValueError("Expected nonempty daily mean/max/min/precipitation weather")
    if len(dates) > 1 and not np.all(np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1):
        raise ValueError("Weather dates must be consecutive")
    if not np.isfinite(weather).all() or not np.isfinite(latitude) or abs(latitude) > 90:
        raise ValueError("Weather and latitude must be finite")
    mean, maximum, minimum = weather[:, 0], weather[:, 1], weather[:, 2]
    if np.any(minimum > mean) or np.any(mean > maximum):
        raise ValueError("Daily temperature must satisfy minimum <= mean <= maximum")
    if np.any(weather[:, 3] < 0):
        raise ValueError("Precipitation must be nonnegative")
    day = dates.dayofyear.to_numpy()
    declination = .4093 * np.sin(2 * np.pi * (day - 81) / 365)
    daylength = 24 / np.pi * np.arccos(np.clip(-np.tan(np.radians(latitude)) * np.tan(declination), -1, 1))
    gdd = np.interp(mean, [0, 20, 30, 40], [0, 20, 20, 0], left=0, right=0)
    prior_gdd = np.cumsum(gdd) - gdd
    photo = (prior_gdd >= gates["photoperiod_onset_gdd"]) & (prior_gdd < gates["photoperiod_stop_gdd"])
    tp = gdd * np.where(photo, np.clip(1 - .09 * (16 - daylength), 0, 1), 1.)
    prior_tp = np.cumsum(tp) - tp
    response = np.interp(mean, [-4, 0, 10, 16], [0, 1, 1, 0])
    vernalization = np.zeros(len(weather)); factor = np.ones(len(weather)); cumulative = 0.
    for i in range(len(weather)):
        if prior_tp[i] >= gates["vernalization_onset_tpp"]:
            sensitive = prior_tp[i] < gates["vernalization_stop_tpp"]
            updated = cumulative + (response[i] if sensitive else 0.)
            if updated < 10 and maximum[i] > 30:
                updated = max(0., cumulative - .5 * (maximum[i] - 30))
            cumulative = updated
            if sensitive:
                factor[i] = .3 + .7 * min(cumulative / 40, 1.)
        vernalization[i] = cumulative
    increments = np.column_stack([gdd, tp, tp * factor])
    progress = np.cumsum(increments, axis=0)
    elapsed = np.arange(len(weather))
    series = pd.Series(mean)
    features = np.column_stack([
        mean, maximum, minimum, daylength, elapsed / HORIZON,
        np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25),
        series.rolling(7, min_periods=1).mean(), series.rolling(30, min_periods=1).mean(),
        np.cumsum(mean) / (elapsed + 1), np.cumsum(minimum < 0) / HORIZON,
        np.cumsum(maximum >= 30) / HORIZON, vernalization / 40,
        progress / 1000,
        np.full(len(weather), np.sin(2 * np.pi * day[0] / 365.25)),
        np.full(len(weather), np.cos(2 * np.pi * day[0] / 365.25)),
    ])
    return features.astype(np.float32), progress[:, 2].astype(np.float64), increments[:, 2].astype(np.float64)


def ordered_stage_cdf(intensity):
    """Exact one-forward-transition-per-day state probabilities, four stages.

    Adapted structurally from maize_clock/ordered_process.py. Float64 log-space
    calculations remain on CPU for portable, stable prefix sums.
    """
    intensity = intensity.to("cpu").double()
    if intensity.ndim != 3:
        raise ValueError("Expected batch x day x stage transition intensity")
    log_stay = -intensity
    log_event = torch.where(intensity > 0, torch.log((-torch.expm1(-intensity)).clamp_min(1e-30)),
                            torch.full_like(intensity, -1e30))
    previous = torch.cumsum(log_stay[:, :, 0], 1)
    states = [previous]
    for j in range(1, intensity.shape[2] + 1):
        initial = torch.zeros_like(previous[:, :1]) if j == 1 else torch.full_like(previous[:, :1], -1e30)
        incoming = torch.cat([initial, previous[:, :-1]], 1) + log_event[:, :, j - 1]
        if j < intensity.shape[2]:
            survival = torch.cumsum(log_stay[:, :, j], 1)
            previous = survival + torch.logcumsumexp(incoming - survival, 1)
        else:
            previous = torch.logcumsumexp(incoming, 1)
        states.append(previous)
    probability = torch.exp(torch.stack(states, 2))
    probability = probability / probability.sum(2, keepdim=True)
    cdf = torch.flip(torch.cumsum(torch.flip(probability[:, :, 1:], [2]), 2), [2])
    return cdf.transpose(1, 2).clamp(0, 1)


class WheatPhenology(nn.Module):
    def __init__(self, metadata, mode="stage_attention"):
        super().__init__()
        if mode not in {"ordered_process", "ordered_lstm", "stage_attention"}:
            raise ValueError("Unknown model family")
        self.mode = mode
        width = int(metadata.get("width", 32)); self.width = width
        self.register_buffer("feature_mean", torch.tensor(metadata["feature_mean"], dtype=torch.float32))
        self.register_buffer("feature_std", torch.tensor(metadata["feature_std"], dtype=torch.float32))
        thresholds = np.asarray(metadata["thresholds"], float)
        gaps = np.diff(np.r_[0., thresholds])
        if len(thresholds) != 4 or np.any(gaps <= 0):
            raise ValueError("Four increasing wheat thresholds required")
        self.raw_gaps = nn.Parameter(torch.tensor(np.log(np.expm1(gaps / 1000)), dtype=torch.float64), requires_grad=mode != "ordered_lstm")
        sigma = np.clip((np.asarray(metadata["sigma"]) - .025) / .475, .001, .999)
        self.raw_sigma = nn.Parameter(torch.tensor(np.log(sigma / (1 - sigma)), dtype=torch.float64), requires_grad=mode != "ordered_lstm")
        if mode != "ordered_process":
            self.encoder = nn.LSTM(len(FEATURE_NAMES), width, batch_first=True)
        if mode == "ordered_lstm":
            self.head = nn.Linear(width, 4)
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.copy_(torch.tensor(np.log(1000 / gaps), dtype=torch.float32))
        elif mode == "stage_attention":
            self.keys = nn.Linear(width, 32); self.values = nn.Linear(width, 32)
            self.query = nn.Linear(3, 32)
            self.stage_query = nn.Parameter(torch.randn(4, 2, 16) * .05)
            self.raw_decay = nn.Parameter(torch.zeros(4, 2))
            self.stage_embedding = nn.Embedding(4, 8)
            self.correction = nn.Sequential(nn.Linear(32 + width + 8, 32), nn.SiLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.correction[-1].weight); nn.init.zeros_(self.correction[-1].bias)

    def neural_device(self):
        return next(self.encoder.parameters()).device if self.mode != "ordered_process" else torch.device("cpu")

    def set_neural_device(self, device):
        if self.mode == "ordered_lstm":
            self.encoder.to(device); self.head.to(device)
        if self.mode == "stage_attention":
            for name in ["encoder", "keys", "values", "query", "stage_embedding", "correction"]:
                getattr(self, name).to(device)
            self.stage_query = nn.Parameter(self.stage_query.detach().to(device))
            self.raw_decay = nn.Parameter(self.raw_decay.detach().to(device))
        return self

    def thresholds(self):
        return torch.cumsum(torch.nn.functional.softplus(self.raw_gaps) + 1e-8, 0) * 1000

    def sigma(self):
        return .025 + .475 * torch.sigmoid(self.raw_sigma)

    def process_intensity(self, progress):
        progress = progress.to("cpu").double()
        z = (torch.log(progress.clamp_min(1e-12))[:, :, None] - torch.log(self.thresholds())[None, None, :]) / self.sigma()[None, None, :]
        survival = torch.special.log_ndtr(-z)
        survival = torch.where(progress[:, :, None] > 0, survival, torch.zeros_like(survival))
        previous = torch.cat([torch.zeros_like(survival[:, :1]), survival[:, :-1]], 1)
        return (previous - survival).clamp(0, 20)

    def transition_intensities(self, features, progress, increments):
        if self.mode == "ordered_process":
            return self.process_intensity(progress)
        x = ((features.to("cpu").float() - self.feature_mean) / self.feature_std).to(self.neural_device())
        hidden = self.encoder(x)[0]
        if self.mode == "ordered_lstm":
            rates = torch.exp(self.head(hidden).clamp(-12, 6)).to("cpu").double()
            return rates * increments.to("cpu").double()[:, :, None] / 1000
        context_input = torch.cat([x[:, 0, 3:4], x[:, 0, -2:]], 1)
        queries = self.query(context_input).reshape(len(x), 1, 2, 16) + self.stage_query[None]
        keys = self.keys(hidden).reshape(len(x), x.shape[1], 2, 16)
        values = self.values(hidden).reshape(len(x), x.shape[1], 2, 16)
        score = torch.einsum("njhd,nthd->ntjh", queries, keys) / 4
        position = (progress / self.thresholds()[-1]).float().to(hidden.device)
        score = score + position[:, :, None, None] * torch.sigmoid(self.raw_decay)[None, None]
        weight = torch.exp(score.clamp(-20, 20)).permute(0, 2, 3, 1).contiguous()
        values = values.permute(0, 2, 3, 1).contiguous()
        weighted = (weight[:, :, :, None] * values[:, None]).cumsum(-1)
        context = weighted / weight.cumsum(-1)[:, :, :, None]
        context = context.permute(0, 4, 1, 2, 3).reshape(len(x), x.shape[1], 4, 32)
        layer = self.correction[0]
        embedding = self.stage_embedding(torch.arange(4, device=hidden.device))[None, None]
        value = torch.nn.functional.linear(context, layer.weight[:, :32], layer.bias)
        value = value + torch.nn.functional.linear(hidden, layer.weight[:, 32:32 + self.width])[:, :, None]
        value = value + torch.nn.functional.linear(embedding, layer.weight[:, 32 + self.width:])
        delta = torch.tanh(self.correction[-1](self.correction[1](value)).squeeze(-1))
        return self.process_intensity(progress) * torch.exp(delta.to("cpu").double())

    def forward(self, features, progress, increments):
        return ordered_stage_cdf(self.transition_intensities(features, progress, increments))


def quantiles(cdf, levels=(.05, .5, .95)):
    cdf = np.asarray(cdf)
    result = np.full((*cdf.shape[:2], len(levels)), np.nan)
    for i, level in enumerate(levels):
        reached = cdf[:, :, -1] >= level
        result[:, :, i] = np.where(reached, (cdf >= level).argmax(2), np.nan)
    return result


def predict_batches(model, features, progress, increments, indices, batch=256):
    model.eval(); predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            ix = indices[start:start + batch]
            cdf = model(torch.as_tensor(np.asarray(features[ix])), torch.as_tensor(np.asarray(progress[ix])),
                        torch.as_tensor(np.asarray(increments[ix])))
            predictions.append(cdf.numpy().astype(np.float32))
    return np.concatenate(predictions)


def daily_features(x, progress, increments, base_cdf):
    x = np.asarray(x, dtype=np.float32)
    progress, increments, base = map(np.asarray, (progress, increments, base_cdf))
    if x.ndim != 3 or base.shape != (len(x), 4, x.shape[1]) or progress.shape != x.shape[:2] or increments.shape != x.shape[:2]:
        raise ValueError('Incompatible daily predictor shapes')
    state = base.transpose(0, 2, 1).astype(np.float32)
    previous = np.concatenate([np.zeros_like(state[:, :1]), state[:, :-1]], axis=1)
    result = np.concatenate([x, state, state - previous, (progress / 1000)[..., None],
                             (increments / 20)[..., None]], axis=2).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Daily predictors must be finite')
    return result


def forward_project(proposal, increments):
    proposal = np.asarray(proposal)
    increments = np.asarray(increments)
    if proposal.ndim != 3 or proposal.shape[1] != 4 or increments.shape != (len(proposal), proposal.shape[2]):
        raise ValueError('Incompatible probability/increment shapes')
    if not np.isfinite(proposal).all() or not np.isfinite(increments).all() or np.any(increments < 0):
        raise ValueError('Finite proposals and nonnegative thermal increments are required')
    ordered = np.minimum.accumulate(np.clip(proposal, 0, 1), axis=1)
    output = np.zeros_like(ordered, dtype=np.float32)
    state = np.zeros(proposal.shape[:2], dtype=np.float32)
    for day in range(proposal.shape[2]):
        active = increments[:, day] > 0
        state[active] = np.maximum(state[active], ordered[active, :, day])
        output[:, :, day] = state
    return output


def correct(base, residual, increments, strength, mask, variant):
    if variant not in ('raw','advance_only','horizon_floor'):
        raise ValueError('Unknown correction variant')
    if strength==0: return np.asarray(base).copy()
    residual=np.asarray(residual)
    if variant=='advance_only': residual=np.maximum(residual,0)
    proposal=np.asarray(base)+strength*np.asarray(mask)[None,:,None]*residual
    # Day329 is a fixed sowing-relative calendar day, never the last supplied
    # prefix day. Thermal pauses still take precedence over this lower bound.
    if variant=='horizon_floor' and proposal.shape[2]>329:
        eligible=np.asarray(base)[:,:,329]>=.5
        proposal[:,:,329]=np.where(eligible,np.maximum(proposal[:,:,329],.5),proposal[:,:,329])
    return forward_project(proposal,increments)


def geo_features(features, cycles):
    coordinates=cycles[['lat','lon']].to_numpy(dtype=np.float32)/100
    if len(features)!=len(cycles) or not np.isfinite(coordinates).all():
        raise ValueError('Known finite latitude/longitude are required')
    return np.concatenate([features,np.broadcast_to(coordinates[:,None],(len(features),features.shape[1],2))],axis=2)


def couple(stage_residual,occupancy_residual):
    result=np.asarray(stage_residual).copy()
    if np.asarray(occupancy_residual).shape!=(len(result),result.shape[2]): raise ValueError('Occupancy dimensions differ')
    result[:,3]=result[:,2]-occupancy_residual
    return result


def occupancy_prediction(head,features,batch=256):
    output=np.empty(features.shape[:2],np.float32)
    for start in range(0,len(features),batch):
        xx=features[start:start+batch]
        output[start:start+len(xx)]=head.predict(xx.reshape(-1,xx.shape[-1])).reshape(xx.shape[:2])
    return output


def epoch_features(features,cycles):
    sowing=pd.to_datetime(cycles.SOWING_DATE,errors='raise')
    epoch=(sowing.dt.year.to_numpy()+(sowing.dt.dayofyear.to_numpy()-1)/365.2425-2000)/10
    day=np.arange(features.shape[1])/365.2425/10
    return np.concatenate([features,(epoch[:,None]+day[None,:])[...,None]],axis=2).astype(np.float32)


def epoch_basis(features,stage):
    f=np.asarray(features);p=f[...,18+stage];e=f[...,-1];elapsed=f[...,4];lat=f[...,-3];lon=f[...,-2]
    return np.stack([p,p*p,p*p*p,elapsed,p*elapsed,e,e*p,e*p*p,e*p*p*p,lat,lon,lat*p,lon*p],axis=-1)


def predict_heads(package,features,coupled=False,batch=256):
    output=np.empty((len(features),4,features.shape[1]),np.float32)
    for start in range(0,len(features),batch):
        xx=features[start:start+batch]
        for stage,head in enumerate(package['heads']):
            inputs=epoch_basis(xx,stage) if package['family']=='ridge_epoch' else xx
            output[start:start+len(xx),stage]=head.predict(inputs.reshape(-1,inputs.shape[-1])).reshape(xx.shape[:2])
    if coupled:output=g.couple(output,g.occupancy_prediction(package['occupancy_head'],features))
    return output


def age_features(features,base,progress):
    features,base,progress=map(np.asarray,(features,base,progress))
    if base.shape!=(len(features),4,features.shape[1]) or progress.shape!=features.shape[:2]:raise ValueError('Age feature dimensions conflict')
    result=np.zeros((len(features),features.shape[1],9),np.float32)
    for stage in range(3):
        first=np.full(len(features),-1,int);at_cross=np.zeros(len(features))
        for day in range(features.shape[1]):
            newly=(first<0)&(base[:,stage,day]>=.5)
            first[newly]=day;at_cross[newly]=progress[newly,day]
            seen=first>=0
            result[:,day,stage*3]=seen
            result[:,day,stage*3+1]=np.where(seen,(day-first)/100,0)
            result[:,day,stage*3+2]=np.where(seen,(progress[:,day]-at_cross)/1000,0)
    return np.concatenate([features,result],axis=2)


g = SimpleNamespace(couple=couple, occupancy_prediction=occupancy_prediction)
