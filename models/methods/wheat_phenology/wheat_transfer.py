"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_transfer_20261010')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
STAGES = [10, 31, 51, 87]
STAGE_NAMES = ['first_leaf_appearance', 'first_node', 'heading', 'yellow_ripeness']
HORIZON = 330
DEFAULT_GATES = dict(photoperiod_onset_gdd=142.01, photoperiod_stop_gdd=1426.78, vernalization_onset_tpp=142.01, vernalization_stop_tpp=582.8)
FEATURE_NAMES = ['tmean_c', 'tmax_c', 'tmin_c', 'daylength_hours', 'elapsed_days_over_330', 'day_of_year_sine', 'day_of_year_cosine', 'past_7d_mean_c', 'past_30d_mean_c', 'past_mean_c', 'past_frost_days_over_330', 'past_hot_days_over_330', 'vernalization_over_40', 'cumulative_gdd_over_1000', 'cumulative_tp_over_1000', 'cumulative_tpv_over_1000', 'sowing_day_sine', 'sowing_day_cosine']

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def validate_splits(records):
    train = records.loc[records.partition.eq('train')]
    val = records.loc[records.partition.eq('val')]
    test = records.loc[records.partition.eq('independent')]
    if set(train.PEP_ID) & set(val.PEP_ID):
        raise ValueError('Training and validation station sets overlap')
    if not train.harvest_year.le(2015).all() or not val.harvest_year.le(2015).all() or (not test.harvest_year.ge(2017).all()):
        raise ValueError('Historical fitting and later testing violate time separation')
    strict = test.loc[test.historical_station_group.eq('test')]
    if set(strict.PEP_ID) & (set(train.PEP_ID) | set(val.PEP_ID)):
        raise ValueError('Reserved test station appears in fitting or validation')

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
        raise ValueError('Expected nonempty daily mean/max/min/precipitation weather')
    if len(dates) > 1 and (not np.all(np.diff(dates.values).astype('timedelta64[D]').astype(int) == 1)):
        raise ValueError('Weather dates must be consecutive')
    if not np.isfinite(weather).all() or not np.isfinite(latitude) or abs(latitude) > 90:
        raise ValueError('Weather and latitude must be finite')
    mean, maximum, minimum = (weather[:, 0], weather[:, 1], weather[:, 2])
    if np.any(minimum > mean) or np.any(mean > maximum):
        raise ValueError('Daily temperature must satisfy minimum <= mean <= maximum')
    if np.any(weather[:, 3] < 0):
        raise ValueError('Precipitation must be nonnegative')
    day = dates.dayofyear.to_numpy()
    declination = 0.4093 * np.sin(2 * np.pi * (day - 81) / 365)
    daylength = 24 / np.pi * np.arccos(np.clip(-np.tan(np.radians(latitude)) * np.tan(declination), -1, 1))
    gdd = np.interp(mean, [0, 20, 30, 40], [0, 20, 20, 0], left=0, right=0)
    prior_gdd = np.cumsum(gdd) - gdd
    photo = (prior_gdd >= gates['photoperiod_onset_gdd']) & (prior_gdd < gates['photoperiod_stop_gdd'])
    tp = gdd * np.where(photo, np.clip(1 - 0.09 * (16 - daylength), 0, 1), 1.0)
    prior_tp = np.cumsum(tp) - tp
    response = np.interp(mean, [-4, 0, 10, 16], [0, 1, 1, 0])
    vernalization = np.zeros(len(weather))
    factor = np.ones(len(weather))
    cumulative = 0.0
    for i in range(len(weather)):
        if prior_tp[i] >= gates['vernalization_onset_tpp']:
            sensitive = prior_tp[i] < gates['vernalization_stop_tpp']
            updated = cumulative + (response[i] if sensitive else 0.0)
            if updated < 10 and maximum[i] > 30:
                updated = max(0.0, cumulative - 0.5 * (maximum[i] - 30))
            cumulative = updated
            if sensitive:
                factor[i] = 0.3 + 0.7 * min(cumulative / 40, 1.0)
        vernalization[i] = cumulative
    increments = np.column_stack([gdd, tp, tp * factor])
    progress = np.cumsum(increments, axis=0)
    elapsed = np.arange(len(weather))
    series = pd.Series(mean)
    features = np.column_stack([mean, maximum, minimum, daylength, elapsed / HORIZON, np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25), series.rolling(7, min_periods=1).mean(), series.rolling(30, min_periods=1).mean(), np.cumsum(mean) / (elapsed + 1), np.cumsum(minimum < 0) / HORIZON, np.cumsum(maximum >= 30) / HORIZON, vernalization / 40, progress / 1000, np.full(len(weather), np.sin(2 * np.pi * day[0] / 365.25)), np.full(len(weather), np.cos(2 * np.pi * day[0] / 365.25))])
    return (features.astype(np.float32), progress[:, 2].astype(np.float64), increments[:, 2].astype(np.float64))

def ordered_stage_cdf(intensity):
    """Exact one-forward-transition-per-day state probabilities, four stages.

    Adapted structurally from maize_clock/ordered_process.py. Float64 log-space
    calculations remain on CPU for portable, stable prefix sums.
    """
    intensity = intensity.to('cpu').double()
    if intensity.ndim != 3:
        raise ValueError('Expected batch x day x stage transition intensity')
    log_stay = -intensity
    log_event = torch.where(intensity > 0, torch.log((-torch.expm1(-intensity)).clamp_min(1e-30)), torch.full_like(intensity, -1e+30))
    previous = torch.cumsum(log_stay[:, :, 0], 1)
    states = [previous]
    for j in range(1, intensity.shape[2] + 1):
        initial = torch.zeros_like(previous[:, :1]) if j == 1 else torch.full_like(previous[:, :1], -1e+30)
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

    def __init__(self, metadata, mode='stage_attention'):
        super().__init__()
        if mode not in {'ordered_process', 'ordered_lstm', 'stage_attention'}:
            raise ValueError('Unknown model family')
        self.mode = mode
        width = int(metadata.get('width', 32))
        self.width = width
        self.register_buffer('feature_mean', torch.tensor(metadata['feature_mean'], dtype=torch.float32))
        self.register_buffer('feature_std', torch.tensor(metadata['feature_std'], dtype=torch.float32))
        thresholds = np.asarray(metadata['thresholds'], float)
        gaps = np.diff(np.r_[0.0, thresholds])
        if len(thresholds) != 4 or np.any(gaps <= 0):
            raise ValueError('Four increasing wheat thresholds required')
        self.raw_gaps = nn.Parameter(torch.tensor(np.log(np.expm1(gaps / 1000)), dtype=torch.float64), requires_grad=mode != 'ordered_lstm')
        sigma = np.clip((np.asarray(metadata['sigma']) - 0.025) / 0.475, 0.001, 0.999)
        self.raw_sigma = nn.Parameter(torch.tensor(np.log(sigma / (1 - sigma)), dtype=torch.float64), requires_grad=mode != 'ordered_lstm')
        if mode != 'ordered_process':
            self.encoder = nn.LSTM(len(FEATURE_NAMES), width, batch_first=True)
        if mode == 'ordered_lstm':
            self.head = nn.Linear(width, 4)
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.copy_(torch.tensor(np.log(1000 / gaps), dtype=torch.float32))
        elif mode == 'stage_attention':
            self.keys = nn.Linear(width, 32)
            self.values = nn.Linear(width, 32)
            self.query = nn.Linear(3, 32)
            self.stage_query = nn.Parameter(torch.randn(4, 2, 16) * 0.05)
            self.raw_decay = nn.Parameter(torch.zeros(4, 2))
            self.stage_embedding = nn.Embedding(4, 8)
            self.correction = nn.Sequential(nn.Linear(32 + width + 8, 32), nn.SiLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)

    def neural_device(self):
        return next(self.encoder.parameters()).device if self.mode != 'ordered_process' else torch.device('cpu')

    def set_neural_device(self, device):
        if self.mode == 'ordered_lstm':
            self.encoder.to(device)
            self.head.to(device)
        if self.mode == 'stage_attention':
            for name in ['encoder', 'keys', 'values', 'query', 'stage_embedding', 'correction']:
                getattr(self, name).to(device)
            self.stage_query = nn.Parameter(self.stage_query.detach().to(device))
            self.raw_decay = nn.Parameter(self.raw_decay.detach().to(device))
        return self

    def thresholds(self):
        return torch.cumsum(torch.nn.functional.softplus(self.raw_gaps) + 1e-08, 0) * 1000

    def sigma(self):
        return 0.025 + 0.475 * torch.sigmoid(self.raw_sigma)

    def process_intensity(self, progress):
        progress = progress.to('cpu').double()
        z = (torch.log(progress.clamp_min(1e-12))[:, :, None] - torch.log(self.thresholds())[None, None, :]) / self.sigma()[None, None, :]
        survival = torch.special.log_ndtr(-z)
        survival = torch.where(progress[:, :, None] > 0, survival, torch.zeros_like(survival))
        previous = torch.cat([torch.zeros_like(survival[:, :1]), survival[:, :-1]], 1)
        return (previous - survival).clamp(0, 20)

    def transition_intensities(self, features, progress, increments):
        if self.mode == 'ordered_process':
            return self.process_intensity(progress)
        x = ((features.to('cpu').float() - self.feature_mean) / self.feature_std).to(self.neural_device())
        hidden = self.encoder(x)[0]
        if self.mode == 'ordered_lstm':
            rates = torch.exp(self.head(hidden).clamp(-12, 6)).to('cpu').double()
            return rates * increments.to('cpu').double()[:, :, None] / 1000
        context_input = torch.cat([x[:, 0, 3:4], x[:, 0, -2:]], 1)
        queries = self.query(context_input).reshape(len(x), 1, 2, 16) + self.stage_query[None]
        keys = self.keys(hidden).reshape(len(x), x.shape[1], 2, 16)
        values = self.values(hidden).reshape(len(x), x.shape[1], 2, 16)
        score = torch.einsum('njhd,nthd->ntjh', queries, keys) / 4
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
        return self.process_intensity(progress) * torch.exp(delta.to('cpu').double())

    def forward(self, features, progress, increments):
        return ordered_stage_cdf(self.transition_intensities(features, progress, increments))

def quantiles(cdf, levels=(0.05, 0.5, 0.95)):
    cdf = np.asarray(cdf)
    result = np.full((*cdf.shape[:2], len(levels)), np.nan)
    for i, level in enumerate(levels):
        reached = cdf[:, :, -1] >= level
        result[:, :, i] = np.where(reached, (cdf >= level).argmax(2), np.nan)
    return result

def predict_batches(model, features, progress, increments, indices, batch=256):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            ix = indices[start:start + batch]
            cdf = model(torch.as_tensor(np.asarray(features[ix])), torch.as_tensor(np.asarray(progress[ix])), torch.as_tensor(np.asarray(increments[ix])))
            predictions.append(cdf.numpy().astype(np.float32))
    return np.concatenate(predictions)

def replay(checkpoint, weather_path, sowing, latitude, output):
    """Export a whole supplied-weather prefix without observed terminal dates."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    package = torch.load(checkpoint, map_location='cpu', weights_only=False)
    frame = pd.read_csv(weather_path)
    frame = frame.rename(columns={'DATE': 'date', 't_mean': 'tmean_c', 't_max': 'tmax_c', 't_min': 'tmin_c', 'p_sum': 'precip_mm'})
    frame['date'] = pd.to_datetime(frame.date)
    sowing = pd.Timestamp(sowing)
    frame = frame.loc[frame.date.between(sowing, sowing + pd.Timedelta(days=HORIZON - 1))].sort_values('date')
    if frame.empty or frame.date.iloc[0] != sowing:
        raise ValueError('Weather must begin on supplied sowing day')
    if 'precip_mm' not in frame:
        frame['precip_mm'] = 0.0
    weather = frame[['tmean_c', 'tmax_c', 'tmin_c', 'precip_mm']].to_numpy()
    x, progress, inc = build_features(weather, frame.date, latitude, package['metadata']['gates'])
    model = WheatPhenology(package['metadata'], package['mode'])
    model.load_state_dict(package['state_dict'])
    cdf = predict_batches(model, x[None], progress[None], inc[None], np.array([0]))
    q = quantiles(cdf)[0]
    output.mkdir(parents=True)
    stages = []
    for j, stage in enumerate(STAGES):
        row = dict(stage=stage, endpoint=STAGE_NAMES[j], tail_probability=float(1 - cdf[0, j, -1]))
        for k, name in enumerate(['q05', 'q50', 'q95']):
            row[name + '_date'] = (sowing + pd.Timedelta(days=int(q[j, k]))).date().isoformat() if np.isfinite(q[j, k]) else None
        stages.append(row)
    pd.DataFrame(stages).to_csv(output / 'stage_quantiles.csv', index=False)
    daily = pd.DataFrame({'date': frame.date.dt.strftime('%Y-%m-%d')})
    for j, stage in enumerate(STAGES):
        daily[f'p_reached_{stage}'] = cdf[0, j]
    occupancy = np.concatenate([1 - cdf[:, :1], cdf[:, :-1] - cdf[:, 1:], cdf[:, -1:]], axis=1)[0]
    for j, name in enumerate(['before_10', '10_to_31', '31_to_51', '51_to_87', 'after_87']):
        daily['p_' + name] = occupancy[j]
    daily.to_csv(output / 'daily_host_calendar.csv', index=False)
    provenance = dict(status='complete', checkpoint_sha256=digest(checkpoint), weather_sha256=digest(weather_path), sowing=str(sowing.date()), latitude=latitude, days=len(frame), horizon_days=HORIZON, observed_stage_inputs=False, observed_end_date_inputs=False, maize_weights_used=False, interpretation='Retrospective or scenario supplied-weather phenology; not issue-time weather forecasting', stages=STAGES, unavailable_observed_stages=[37, 39, 65, 85], host_interface='Probabilistic occupancy of observed-stage intervals; leaf emergence/anthesis need a separate qualified model')
    (output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance
