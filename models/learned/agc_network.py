"""Selected canonical AGC architecture; original mathematical nodes."""
from contextlib import nullcontext
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
RESIDUAL_OBJECTIVE = 'process_residual_truncated_logistic_event_nll'


def _float32_scope(device):
    # Older Torch versions reject even a disabled MPS autocast context.
    return torch.autocast(device_type=device.type, enabled=False) if device.type in ('cpu', 'cuda') else nullcontext()


def pure_birth_distribution(initial_probability, rates):
    """Propagate row probabilities through daily five-state pure-birth generators.

    Rates have shape [B,H,4]. All dynamics are float32, even under autocast.
    Flux is computed directly across each threshold; no CDF subtraction occurs.
    """
    with _float32_scope(rates.device):
        rates = rates.float()
        p = initial_probability.float()
        if rates.ndim != 3 or rates.shape[-1] != 4 or p.shape != (rates.shape[0], 5):
            raise ValueError('Expected rates[B,H,4] and initial_probability[B,5]')
        if not torch.isfinite(rates).all() or (rates < 0).any():
            raise ValueError('Transition rates must be finite and nonnegative')
        if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(p.sum(-1), torch.ones_like(p[:, 0]), atol=1e-5):
            raise ValueError('Initial probabilities must be finite, nonnegative and sum to one')
        diagonal = torch.cat((-rates, torch.zeros_like(rates[..., :1])), -1)
        generator = torch.diag_embed(diagonal) + torch.diag_embed(rates, offset=1)
        transition = torch.linalg.matrix_exp(generator)
        # Remove roundoff-only negatives and row-sum drift before repeated products.
        transition = transition.clamp_min(0)
        transition = transition / transition.sum(-1, keepdim=True)
        probabilities, masses = [p], []
        for day in range(rates.shape[1]):
            daily = transition[:, day]
            masses.append(torch.stack([
                (p[:, :k] * daily[:, :k, k:].sum(-1)).sum(-1)
                for k in range(1, 5)
            ], -1))
            p = torch.bmm(p[:, None], daily).squeeze(1)
            probabilities.append(p)
        probability = torch.stack(probabilities, 1)
        cdf = probability[..., 1:].flip(-1).cumsum(-1).flip(-1)
        return {
            'stage_prob': probability,
            'stage_cdf': cdf,
            'stage_mean': cdf[:, 1:].sum(-1, keepdim=True),
            'event_mass': torch.stack(masses, 1),
        }


def gated_delta_reference(query, key, value, alpha, beta, valid):
    """Gated delta rule, Yang et al., ICLR 2025, equation (10).

    S_t = alpha*S_(t-1) + beta*(v-alpha*S_(t-1)@k) outer k.
    q/k/v are [B,L,heads,head_dim]; alpha/beta are [B,L,heads].
    Keys are normalized by the caller. Invalid history leaves memory unchanged.
    https://proceedings.iclr.cc/paper_files/paper/2025/file/
    4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf
    """
    batch, length, heads, _ = query.shape
    state = value.new_zeros(batch, heads, value.shape[-1], key.shape[-1])
    result = []
    for t in range(length):
        decayed = alpha[:, t, :, None, None] * state
        residual = value[:, t] - (decayed @ key[:, t, :, :, None]).squeeze(-1)
        candidate = decayed + beta[:, t, :, None, None] * residual[..., None] * key[:, t, :, None, :]
        state = torch.where(valid[:, t, None, None, None], candidate, state)
        output = (state @ query[:, t, :, :, None]).squeeze(-1)
        result.append(torch.where(valid[:, t, None, None], output, torch.zeros_like(output)))
    return torch.stack(result, 1)


class GatedDeltaBlock(nn.Module):
    """Pure PyTorch recurrent reference; no optimized external kernel is implied."""
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.heads = config.n_heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.gates = nn.Linear(width, config.n_heads * 2)
        self.output_gate = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.memory_norm = nn.LayerNorm(width // config.n_heads)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(width * 4, width))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, valid):
        z = self.norm(x)
        q, k, v = self.qkv(z).reshape(*z.shape[:2], 3, self.heads, -1).unbind(2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        alpha, beta = self.gates(z).reshape(*z.shape[:2], 2, self.heads).sigmoid().unbind(2)
        memory = gated_delta_reference(q, k, v, alpha, beta, valid)
        memory = self.memory_norm(memory).flatten(-2) * F.silu(self.output_gate(z))
        x = x + self.dropout(self.out(memory))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x.masked_fill(~valid[..., None], 0)


@dataclass(frozen=True)
class SeasonalConfig:
    """Seasonal-only input dimensions; no pre-sowing predictor path exists."""
    static_dim: int = 2
    future_dim: int = 12
    d_model: int = 128
    n_heads: int = 4
    layers: int = 2
    dropout: float = .1
    encoder: str = 'gated_delta'

    def __post_init__(self):
        if self.static_dim != 2 or self.future_dim != 12:
            raise ValueError('Seasonal inputs require static_dim=2 and future_dim=12')
        if self.n_heads < 1 or self.d_model < 2 or self.d_model % self.n_heads or self.layers < 1:
            raise ValueError('Positive layers and d_model divisible by positive n_heads are required')
        if self.encoder not in ('gated_delta', 'attention'):
            raise ValueError('Unknown seasonal encoder')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must lie in [0,1)')


def _future_mask(batch):
    valid = batch['future_valid']
    if valid.dtype != torch.bool or valid.shape != batch['known_future'].shape[:2]:
        raise ValueError('future_valid must be boolean [B,H]')
    if valid.shape[1] == 0 or not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError('Each valid season must be a nonempty prefix followed only by right padding')
    return valid


def _position(length, width, reference, offset=0):
    positions = (torch.arange(length, device=reference.device, dtype=reference.dtype) + offset)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=reference.device, dtype=reference.dtype)
                          * (-math.log(10000.) / width))
    return torch.stack((torch.sin(positions * frequency), torch.cos(positions * frequency)), -1).flatten(-2)[..., :width]


class SeasonalPhenformer(nn.Module):
    """Causal same-season encoder and daily, stage-specific prefix retrieval.

    Four queries per day combine that day's hidden state, separate coordinate
    context, and learned stage identity. The recurrent encoder is a reference
    implementation; daily stage retrieval remains quadratic in season length.
    """
    source_label = 'seasonal adaptation of AGC-PhenFormer v2'

    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.d_model
        self.seasonal_projection = nn.Linear(config.future_dim, width)
        self.context = nn.Sequential(nn.Linear(config.static_dim, width), nn.GELU(), nn.Linear(width, width))
        self.dropout = nn.Dropout(config.dropout)
        if config.encoder == 'gated_delta':
            self.encoder = nn.ModuleList([GatedDeltaBlock(config) for _ in range(config.layers)])
        else:
            layer = nn.TransformerEncoderLayer(width, config.n_heads, width * 4, config.dropout,
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.stage_embedding = nn.Parameter(torch.randn(1, 1, 4, width) * .02)
        self.cross_attention = nn.MultiheadAttention(width, config.n_heads, config.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.rate_network = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))

    def stage_features(self, batch):
        """Return causal per-day stage features without changing parameter names."""
        valid = _future_mask(batch)
        seasonal = batch['known_future'].masked_fill(~valid[..., None], 0)
        x = self.seasonal_projection(seasonal)
        x = self.dropout(x + _position(x.shape[1], x.shape[2], x))
        x = x.masked_fill(~valid[..., None], 0)
        length, width = x.shape[1:]
        if self.config.encoder == 'gated_delta':
            for block in self.encoder:
                x = block(x, valid)
        else:
            causal = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
            x = self.encoder(x, mask=causal, src_key_padding_mask=~valid)
        x = x.masked_fill(~valid[..., None], 0)
        context = self.context(batch['static'])
        query = self.stage_embedding + x[:, :, None, :] + context[:, None, None, :]
        query = query.flatten(1, 2)
        # Flattening preserves day-major, stage-minor order. Every query for day
        # d sees only keys k<=d. Right padding cannot create an all-masked row,
        # since every accepted season has at least its sowing day.
        query_days = torch.arange(length * 4, device=x.device) // 4
        keys = torch.arange(length, device=x.device)
        prefix_mask = keys[None, :] > query_days[:, None]
        retrieved, _ = self.cross_attention(query, x, x, attn_mask=prefix_mask,
                                             key_padding_mask=~valid, need_weights=False)
        stage = self.query_norm(query + retrieved).reshape(x.shape[0], length, 4, width)
        features = torch.cat((stage, x[:, :, None, :].expand(-1, -1, 4, -1)), -1)
        return features, valid

    def forward(self, batch):
        features, valid = self.stage_features(batch)
        correction = self.rate_network(features).squeeze(-1).float().masked_fill(~valid[..., None], 0)
        with _float32_scope(correction.device):
            prior = batch['rate_prior'].float()
            if prior.shape != (features.shape[0], 4) or not torch.isfinite(prior).all() or (prior <= 0).any():
                raise ValueError('Process rate_prior must be finite, positive and have shape [B,4]')
            log_rate = prior[:, None].log() + correction.clamp(-10, 10)
            rates = log_rate.clamp(max=math.log(100.)).exp().clamp(max=100.)
            clamped = ((correction.abs() > 10) | (log_rate > math.log(100.))) & valid[..., None]
            rates = rates.masked_fill(~valid[..., None], 0)
            p0 = rates.new_zeros((rates.shape[0], 5))
            p0[:, 0] = 1
            result = pure_birth_distribution(p0, rates)
        result.update(rates=rates, log_correction=correction, rate_clamp_count=clamped.sum().detach(),
                      objective='stage_balanced_composite_event_nll')
        return result


@dataclass(frozen=True)
class AblationConfig(SeasonalConfig):
    def __post_init__(self):
        if self.static_dim != 2 or self.future_dim not in (12, 14, 16):
            raise ValueError('Ablation feature dimensions must be static=2 and future=12,14,16')
        # Retain every architectural constraint without loosening the old config.
        SeasonalConfig(static_dim=2, future_dim=12, d_model=self.d_model, n_heads=self.n_heads,
                       layers=self.layers, dropout=self.dropout, encoder=self.encoder)


def point_stage(progress, thresholds):
    """Piecewise linear point score whose midpoints equal process thresholds."""
    knots = torch.cat([torch.zeros_like(thresholds[:, :1]), thresholds,
                       thresholds[:, -1:] + (thresholds[:, -1:] - thresholds[:, -2:-1])], 1)
    values = progress.new_tensor([0., .5, 1.5, 2.5, 3.5, 4.])
    interval = (progress[..., None] >= knots[:, None]).sum(-1).sub(1).clamp(0, 4)
    left = knots.gather(1, interval)
    right = knots.gather(1, interval+1)
    fraction = ((progress-left)/(right-left)).clamp(0, 1)
    score = values[interval] + fraction*(values[interval+1]-values[interval])
    # Prevent interpolation rounding a just-below-threshold value up to its
    # midpoint. Direct process crossing and midpoint crossing must agree.
    below_upper = torch.nextafter(values[interval+1], torch.full_like(score, -torch.inf))
    score = torch.where(progress < right, torch.minimum(score, below_upper), score)
    return score[..., None]


class ProcessResidualPhenformer(SeasonalPhenformer):
    source_label = 'bounded multiplicative residual on frozen daily T-P-V'

    def __init__(self, config):
        super().__init__(config)
        torch.nn.init.zeros_(self.rate_network[-1].weight)
        torch.nn.init.zeros_(self.rate_network[-1].bias)

    def forward(self, batch):
        features, valid = self.stage_features(batch)
        logits = self.rate_network(features).squeeze(-1).mean(-1)
        delta = (math.log(2.)*torch.tanh(logits)).masked_fill(~valid, 0)
        cumulative = batch['tpv_cumulative'].double().masked_fill(~valid, 0)
        increment = batch['tpv_increment'].double().masked_fill(~valid, 0)
        thresholds = batch['process_thresholds'].double()
        temperature = batch['residual_temperature'].double()
        if cumulative.shape != valid.shape or increment.shape != valid.shape:
            raise ValueError('Raw T-P-V arrays must have shape [B,H]')
        if not torch.isfinite(cumulative).all() or not torch.isfinite(increment).all() or (increment < 0).any() or (cumulative < 0).any():
            raise ValueError('Raw T-P-V development must be finite and nonnegative')
        previous = torch.cat([torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]], 1)
        # Pandas group cumsum and decimal CSV round trips differ by a few ULPs.
        # Bound error in a daily difference by 64 float64 eps at its cumulative
        # magnitude; this is ~1.4e-10 for progress=10,000, not an input rounding allowance.
        tolerance = 64*torch.finfo(torch.float64).eps*torch.maximum(cumulative.abs(), previous.abs()).clamp_min(1)
        if (((cumulative-previous-increment).abs() > tolerance) & valid).any():
            raise ValueError('Raw T-P-V cumulative differences must equal daily increments from sowing')
        if ((cumulative[:, 1:]-cumulative[:, :-1] < 0) & valid[:, 1:]).any():
            raise ValueError('Raw T-P-V cumulative development must be monotone')
        if thresholds.shape != (len(valid), 4) or not torch.isfinite(thresholds).all() or (thresholds[:, 0] <= 0).any() or (thresholds.diff(dim=1) <= 0).any():
            raise ValueError('Four finite positive ordered process thresholds required')
        if temperature.shape != (len(valid),) or not torch.isfinite(temperature).all() or (temperature <= 0).any():
            raise ValueError('Residual temperature must be finite and positive [B]')
        # The stored sowing-day cumulative already includes that day's increment.
        # Adding a residual preserves the original double values exactly at zero.
        corrected = cumulative + (increment*torch.expm1(delta.double())).cumsum(1)
        # Remove accepted float64 roundoff reversals with a causal projection.
        # Raw cumulative values are nonnegative/monotone, so zero correction
        # remains bit-identical to the stored process trajectory.
        corrected = corrected.clamp_min(0).cummax(dim=1).values
        lengths = valid.sum(1)
        last = corrected.gather(1, (lengths-1)[:, None])
        corrected = torch.where(valid, corrected, last)
        return dict(stage_mean=point_stage(corrected, thresholds), corrected_progress=corrected,
                    process_thresholds=thresholds, residual_temperature=temperature,
                    log_correction=delta, objective=RESIDUAL_OBJECTIVE,
                    stage_summary='monotone_point_stage')


