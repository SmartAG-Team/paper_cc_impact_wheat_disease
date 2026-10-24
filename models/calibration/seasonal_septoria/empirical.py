"""Calibration-only empirical comparators with causal daily weather features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from model.seasonal_septoria.regional import tpv_accumulation


def assessment_features(data, phenology):
    """Use weather through each assessment without reading disease or stage values."""
    accumulated = {}
    for meta in data.metadata.itertuples():
        field, length = int(meta.field_index), int(meta.forcing_days)
        dates = pd.date_range(meta.sowing_date, periods=length)
        t = data.temperature[field:field+1, :length]
        # T-P-V needs daily maximum temperature for hot-day devernalization.
        # The field input archive supplies it independently of observations.
        maximum = data.maximum_temperature[field:field+1, :length]
        accumulated[field] = tpv_accumulation(t, maximum, [meta.latitude],
            dates.dayofyear, np.ones_like(t, dtype=bool), phenology, True)[0]
    rows = []
    for target in data.targets.itertuples():
        field, end, leaf = int(target.field_index), int(target.day_index), int(target.leaf_index)
        if end < 1 or end > len(accumulated[field]):
            raise ValueError('Assessment outside forcing window.')
        day = pd.Timestamp(target.date).dayofyear
        active = data.host_active[field, :end, leaf]
        births = np.flatnonzero(active)
        row = {f'leaf_rank_{i+1}': float(leaf == i) for i in range(7)}
        row.update(elapsed_days=end, cumulative_tpv=accumulated[field][end-1],
            leaf_age_days=0 if not len(births) else end-int(births[0]),
            leaf_active=float(active[-1]), day_sin=np.sin(2*np.pi*day/365.2425),
            day_cos=np.cos(2*np.pi*day/365.2425))
        for window in [7, 21, 60]:
            start = max(0, end-window)
            t, h, p = [x[field, start:end] for x in
                       [data.temperature, data.humidity, data.rain]]
            moisture = 1-(1-1/(1+np.exp(-(h-85)/5)))*np.exp(-p/2)
            row.update({f'tmean_{window}d': float(t.mean()),
                f'rh_{window}d': float(h.mean()), f'rain_sum_{window}d': float(p.sum()),
                f'rain_days_{window}d': float(np.mean(p >= .2)),
                f'humid_days_{window}d': float(np.mean(h >= 85)),
                f'establishment_{window}d': float(np.mean(np.exp(-.5*((t-18)/8)**2)*moisture))})
        rows.append(row)
    return pd.DataFrame(rows, index=data.targets.index)


@dataclass
class WeightedRidge:
    penalty: float
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    estimator: Ridge | None = None

    def fit(self, x, y, weight):
        x, y, weight = np.asarray(x, float), np.asarray(y, float), np.asarray(weight, float)
        weight = weight*len(weight)/weight.sum()
        self.mean = np.average(x, axis=0, weights=weight)
        self.scale = np.sqrt(np.average((x-self.mean)**2, axis=0, weights=weight))
        self.scale[self.scale < 1e-12] = 1.
        self.estimator = Ridge(alpha=self.penalty, solver='svd').fit(
            (x-self.mean)/self.scale, y, sample_weight=weight)
        return self

    def predict(self, x):
        return np.clip(self.estimator.predict((np.asarray(x)-self.mean)/self.scale), 0, 100)


def fit_comparator(family, setting, x, y, weight):
    if family == 'weighted_ridge':
        return WeightedRidge(float(setting['penalty'])).fit(x, y, weight)
    if family == 'constrained_forest':
        return RandomForestRegressor(n_estimators=256, max_depth=setting['max_depth'],
            min_samples_leaf=setting['min_samples_leaf'], max_features=.8,
            bootstrap=False, random_state=20261005, n_jobs=1).fit(x, y, sample_weight=weight)
    raise ValueError('Unsupported comparator family.')
