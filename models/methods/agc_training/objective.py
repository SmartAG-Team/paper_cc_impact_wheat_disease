"""Original canonical selected residual objective, with explicit native scope."""
import math
import torch
from torch.nn import functional as F
from analysis.final_models.agc_network import _future_mask
RESIDUAL_OBJECTIVE = 'process_residual_truncated_logistic_event_nll'


STATISTIC_KEYS = (
    'event_numerator', 'event_denominator',
    'auxiliary_numerator', 'auxiliary_denominator',
    'correction_numerator', 'correction_denominator',
    'regression_numerator', 'regression_denominator',
    'event_count', 'auxiliary_count', 'regression_count', 'floor_count',
    'auxiliary_floor_count', 'rate_clamp_count',
)


def _ratio(numerator, denominator):
    # Fractional cycle weights must not be rounded up to a denominator of one.
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)


def aggregate_loss_statistics(statistics, auxiliary_weight=.05, correction_weight=.001):
    """Reconstruct the objective after summing STATISTIC_KEYS over all batches.

    Statistics may be differentiable during training or summed detached tensors
    during validation. Average only globally supported stages, never batch means.
    Scalar regression uses its own sufficient statistics and objective.
    """
    supported = statistics['event_denominator'] > 0
    stage_loss = _ratio(statistics['event_numerator'], statistics['event_denominator'])
    event = (stage_loss * supported).sum() / supported.sum().clamp_min(1)
    auxiliary = _ratio(statistics['auxiliary_numerator'], statistics['auxiliary_denominator'])
    correction = _ratio(statistics['correction_numerator'], statistics['correction_denominator'])
    regression = _ratio(statistics['regression_numerator'], statistics['regression_denominator'])
    loss = event + auxiliary_weight * auxiliary + correction_weight * correction + regression
    return dict(loss=loss, event_loss=event.detach(), auxiliary_loss=auxiliary.detach(),
                correction_loss=correction.detach(), regression_loss=regression.detach(),
                supported_stages=supported.sum().detach())


def ablation_loss(outputs, batch):
    if outputs.get('objective') != RESIDUAL_OBJECTIVE:
        raise ValueError('Unsupported objective outside the frozen native process-residual experiment')
    valid = _future_mask(batch)
    progress = outputs['corrected_progress'].double()
    weight = batch['cycle_weight'].double()
    if weight.shape != (len(valid),) or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('cycle_weight must be finite nonnegative [B]')
    kind, lower, upper, limit = (batch[key].long() for key in ['event_kind','event_lower','event_upper','event_limit'])
    if any(value.shape != (len(valid), 4) for value in [kind, lower, upper, limit]):
        raise ValueError('Event metadata must have shape [B,4]')
    if ((kind < 0) | (kind > 2)).any():
        raise ValueError('Residual event kinds are missing=0, interval=1, right_censor=2')
    lengths = valid.sum(1)[:, None]
    interval, censor = kind == 1, kind == 2
    if (interval & ((lower < 0) | (upper <= lower) | (upper > limit) | (limit > lengths))).any():
        raise ValueError('Event interval must obey 0 <= lower < upper <= limit <= valid season')
    if (censor & ((limit < 0) | (limit > lengths))).any():
        raise ValueError('Censor limit must lie within the valid season')
    boundary = torch.cat([torch.zeros_like(progress[:, :1]), progress], 1)
    h = progress.shape[1]
    c0, c1 = boundary.gather(1, lower.clamp(0, h)), boundary.gather(1, upper.clamp(0, h))
    theta, tau = outputs['process_thresholds'], outputs['residual_temperature'][:, None]
    a, b = (c0-theta)/tau, (c1-theta)/tau
    difference = (c1-c0)/tau
    positive = difference > 0
    # Evaluate zero-mass branches at a benign difference to prevent 0*NaN
    # during backward; only exactly impossible intervals receive a finite floor.
    safe_difference = torch.where(positive, difference, torch.ones_like(difference))
    log_normalizer = F.logsigmoid(theta/tau)
    logmass = F.logsigmoid(b)+F.logsigmoid(-a)+torch.log(-torch.expm1(-safe_difference))-log_normalizer
    impossible = interval & ~positive
    logmass = torch.where(positive, logmass, torch.full_like(logmass, math.log(1e-12)))
    censor_progress = boundary.gather(1, limit.clamp(0, h))
    logsurvival = F.logsigmoid((theta-censor_progress)/tau)-log_normalizer
    log_likelihood = torch.where(interval, logmass, torch.where(censor, logsurvival, torch.zeros_like(logmass)))
    weights = weight[:, None]*(kind > 0)
    zero = progress.sum()*0
    stats = {key: zero for key in STATISTIC_KEYS}
    delta = outputs['log_correction'].double().masked_fill(~valid, 0)
    per_cycle = delta.square().sum(1)/valid.sum(1)
    stats.update(event_numerator=(-weights*log_likelihood).sum(0), event_denominator=weights.sum(0),
                 event_count=((kind > 0) & (weight[:, None] > 0)).sum(),
                 floor_count=(impossible & (weight[:, None] > 0)).sum(),
                 correction_numerator=(weight*per_cycle).sum(), correction_denominator=weight.sum())
    result = aggregate_loss_statistics(stats)
    result.update({key: value.detach() for key, value in stats.items()})
    return result

