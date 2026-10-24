"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study')
import numpy as np
import pandas as pd

def attach_maxima(data, weather):
    data.maximum_temperature = np.zeros_like(data.temperature)
    forcing = weather.copy()
    forcing['date'] = pd.to_datetime(forcing.date)
    for meta in data.metadata.itertuples():
        dates = pd.date_range(meta.first_forcing_date, meta.last_forcing_date)
        selected = forcing[forcing.location_id.eq(meta.site_id) & forcing.date.isin(dates)].sort_values('date')
        if not pd.DatetimeIndex(selected.date).equals(dates):
            raise ValueError('Maximum-temperature forcing coverage mismatch.')
        data.maximum_temperature[meta.field_index, :len(dates)] = selected.tmax_c

def endpoint(frame, final, upper):
    frame = frame[frame.leaf_index.lt(3)].copy() if upper else frame.copy()
    if final:
        frame = frame.sort_values('date').groupby(['field_id', 'endpoint_series']).tail(1).copy()
    frame['weight'] = 1 / frame.groupby('coordinate_year').field_id.transform('nunique') / frame.groupby('field_id').endpoint_series.transform('nunique')
    if not final:
        frame['weight'] /= frame.groupby(['field_id', 'endpoint_series']).value.transform('size')
    return frame

def paired_intervals(frame, cluster_ids, draws):
    error = frame.predicted_percent - frame.value
    comparator = frame.comparator_percent - frame.value
    scores = pd.DataFrame({'coordinate_year': frame.coordinate_year, 'w': frame.weight, 'mse': frame.weight * error ** 2, 'mae': frame.weight * abs(error), 'comparator_mse': frame.weight * comparator ** 2, 'comparator_mae': frame.weight * abs(comparator)}).groupby('coordinate_year').sum().reindex(cluster_ids)
    if scores.isna().any().any():
        raise ValueError('Bootstrap membership changed.')
    denominator = scores.w.to_numpy()[draws].sum(axis=1)
    model_rmse = np.sqrt(scores.mse.to_numpy()[draws].sum(axis=1) / denominator)
    other_rmse = np.sqrt(scores.comparator_mse.to_numpy()[draws].sum(axis=1) / denominator)
    mae_diff = (scores.mae - scores.comparator_mae).to_numpy()[draws].sum(axis=1) / denominator
    return {'rmse_difference': float(np.sqrt(scores.mse.sum() / scores.w.sum()) - np.sqrt(scores.comparator_mse.sum() / scores.w.sum())), 'rmse_difference_lower': float(np.quantile(model_rmse - other_rmse, 0.025)), 'rmse_difference_upper': float(np.quantile(model_rmse - other_rmse, 0.975)), 'mae_difference': float((scores.mae - scores.comparator_mae).sum() / scores.w.sum()), 'mae_difference_lower': float(np.quantile(mae_diff, 0.025)), 'mae_difference_upper': float(np.quantile(mae_diff, 0.975))}
