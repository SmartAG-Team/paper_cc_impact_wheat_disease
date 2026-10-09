"""Wheat adaptation of maize_phen_dl's positive-development LSTM/TCN.

Fresh wheat weights, four native endpoints and wheat T-P-V forcing are required.
A shared scale preserves stage order for every marginal quantile. Prediction is
conditional on supplied daily weather; missing quantiles remain missing.
"""
import math
import numpy as np
import torch
from torch import nn

STAGES = (10, 31, 51, 85)


class CausalConv(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, 3, dilation=dilation)
        self.padding = 2 * dilation

    def forward(self, x):
        return x + torch.nn.functional.silu(self.conv(torch.nn.functional.pad(x, (self.padding, 0))))


class WheatDevelopmentModel(nn.Module):
    def __init__(self, thresholds, input_size, *, encoder='lstm', width=32):
        super().__init__()
        thresholds = np.asarray(thresholds, dtype=np.float32)
        if (thresholds.shape != (4,) or not np.isfinite(thresholds).all()
                or thresholds[0] <= 0 or np.any(np.diff(thresholds) <= 0)):
            raise ValueError('Four increasing native wheat thresholds required: GS10/31/51/85')
        self.register_buffer('thresholds', torch.tensor(thresholds))
        self.encoder_name = encoder
        if encoder == 'lstm':
            self.encoder = nn.LSTM(input_size, width, batch_first=True)
        elif encoder == 'tcn':
            self.encoder = nn.Sequential(nn.Conv1d(input_size, width, 1),
                                        *[CausalConv(width, d) for d in (1, 2, 4, 8)])
        else:
            raise ValueError('Expected lstm or tcn')
        self.head = nn.Linear(width, 1)
        with torch.no_grad():
            self.head.weight.mul_(.01)
            self.head.bias.zero_()
        self.raw_sigma = nn.Parameter(torch.tensor(math.log(.10 / .68)))

    def forward(self, x, thermal, scale):
        if x.ndim != 3 or thermal.shape != x.shape[:2] or scale <= 0:
            raise ValueError('Aligned batch/day inputs and positive thermal scale required')
        h = (self.encoder(x.transpose(1, 2)).transpose(1, 2)
             if self.encoder_name == 'tcn' else self.encoder(x)[0])
        multiplier = 2 * torch.sigmoid(self.head(h).squeeze(-1))
        state = torch.cumsum(thermal * multiplier / float(scale), dim=1)
        sigma = .02 + .78 * torch.sigmoid(self.raw_sigma)
        z = ((state.clamp_min(1e-12).log()[:, None, :]
              - self.thresholds.log()[None, :, None]) / sigma)
        cdf = torch.where(state[:, None, :] > 0,
                          .5 * (1 + torch.erf(z / math.sqrt(2))), torch.zeros_like(z))
        return state, cdf


def observed_crps(cdf, y, weights, lengths):
    days = torch.arange(1, cdf.shape[2] + 1, device=cdf.device)
    truth = (days[None, None, :] >= y[:, :, None]).to(cdf.dtype)
    supported = ((y > 0) & (y <= lengths[:, None]))[:, :, None]
    available = days[None, None, :] <= lengths[:, None, None]
    error = (cdf - truth).square() * supported * available
    return (error.sum(2) * weights[None, :]).sum() / len(y)


def quantile_days(cdf, probability, lengths):
    cdf = np.asarray(cdf)
    if cdf.ndim != 3 or not 0 < probability < 1:
        raise ValueError('Batch/stage/day CDF and interior probability required')
    available = np.arange(1, cdf.shape[2] + 1)[None, None, :] <= np.asarray(lengths)[:, None, None]
    crossed = (cdf >= probability) & available
    return np.where(crossed.any(2), crossed.argmax(2) + 1, np.nan)
