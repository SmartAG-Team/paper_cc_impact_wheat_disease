"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_joint_forecast_20261010')
import numpy as np
import pandas as pd

def trailing_mean(values, width):
    """Mean of the current and up to width-1 preceding daily values."""
    values = np.asarray(values, dtype=float)
    cumulative = np.concatenate([np.zeros((len(values), 1)), np.cumsum(values, axis=1)], axis=1)
    ends = np.arange(1, values.shape[1] + 1)
    starts = np.maximum(0, ends - width)
    return (cumulative[:, ends] - cumulative[:, starts]) / (ends - starts)

def build(weather, parent_cdf, sowing_dates):
    """Return field × day × feature values, without outcome or identity inputs."""
    weather, parent_cdf = (np.asarray(weather, float), np.asarray(parent_cdf, float))
    if weather.ndim != 3 or weather.shape[2] != 3 or (not weather.shape[1]) or (parent_cdf.ndim != 3) or (parent_cdf.shape[:2] != (len(weather), 4)) or (not parent_cdf.shape[2]) or (len(sowing_dates) != len(weather)) or (not np.isfinite(weather).all()) or (not np.isfinite(parent_cdf).all()) or np.any((parent_cdf < 0) | (parent_cdf > 1)) or np.any((weather[:, :, 1] < 0) | (weather[:, :, 1] > 100)) or np.any(weather[:, :, 2] < 0):
        raise ValueError('Complete finite weather and four bounded crop CDFs required')
    arrays, names = ([], [])
    for width in [7, 21, 42]:
        for j, name in enumerate(['temperature', 'humidity', 'rain']):
            mean = trailing_mean(weather[:, :, j], width)
            arrays.append(np.log1p(mean) if name == 'rain' else mean)
            names.append(f'{name}_' + ('log1p_' if name == 'rain' else '') + f'mean_{width}d')
    n, days = weather.shape[:2]
    arrays.append(np.broadcast_to(np.arange(1, days + 1) / 330, (n, days)))
    names.append('elapsed_days_scaled')
    dates = [pd.date_range(start, periods=days) for start in sowing_dates]
    if any((d.isna().any() for d in dates)):
        raise ValueError('Valid sowing dates required')
    phase = np.asarray([2 * np.pi * (d.dayofyear.to_numpy() - 1) / 365.2425 for d in dates])
    arrays.extend([np.sin(phase), np.cos(phase)])
    names.extend(['calendar_sin', 'calendar_cos'])
    index = np.minimum(np.arange(days), parent_cdf.shape[2] - 1)
    for j, stage in enumerate([10, 31, 51, 87]):
        arrays.append(parent_cdf[:, j, index])
        names.append(f'crop_cdf_{stage}')
    return (np.stack(arrays, axis=2), names)
