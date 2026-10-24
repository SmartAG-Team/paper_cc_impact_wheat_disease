"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/nature_food_fix_20261007/validation')
import numpy as np
import pandas as pd

def hierarchy_weights(frame):
    keys = ['coordinate_year', 'field_id', 'leaf_index']
    if frame.empty or frame[keys].isna().any().any():
        raise ValueError('Nonempty complete independent-group membership is required')
    fields = frame.groupby(keys[0]).field_id.transform('nunique').to_numpy(float)
    leaves = frame.groupby(keys[:2]).leaf_index.transform('nunique').to_numpy(float)
    records = frame.groupby(keys).leaf_index.transform('size').to_numpy(float)
    return 1.0 / (frame.coordinate_year.nunique() * fields * leaves * records)

def paired_bootstrap(frame, draws, *, kind):
    """Sample intact coordinate-year groups; never sample individual leaves.

    Groups are sorted lexical coordinate-year IDs. Every occurrence of a drawn
    group carries its entire field/leaf/date hierarchy and both paired models.
    Unavailable class denominators remain NaN in that bootstrap replicate.
    """
    weights = hierarchy_weights(frame)
    group_ids = sorted(frame.coordinate_year.unique())
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[1] != len(group_ids) or (not np.issubdtype(draws.dtype, np.integer)) or np.any(draws < 0) or np.any(draws >= len(group_ids)):
        raise ValueError('Valid integer cluster draws required')
    moment = pd.DataFrame({'coordinate_year': frame.coordinate_year, 'mass': weights})
    model = frame.model.to_numpy()
    benchmark = frame.benchmark.to_numpy()
    if kind == 'mean':
        moment['model_sum'] = weights * model
        moment['benchmark_sum'] = weights * benchmark
    elif kind == 'severity':
        observed = frame.observed.to_numpy(float)
        for label, prediction in [('model', model), ('benchmark', benchmark)]:
            error = prediction - observed
            moment[label + '_sse'] = weights * error ** 2
            moment[label + '_absolute'] = weights * abs(error)
            moment[label + '_error'] = weights * error
    elif kind == 'detection':
        observed = frame.observed.to_numpy(bool)
        moment['positive'] = weights * observed
        moment['negative'] = weights * ~observed
        for label, prediction in [('model', model.astype(bool)), ('benchmark', benchmark.astype(bool))]:
            moment[label + '_tp'] = weights * (observed & prediction)
            moment[label + '_tn'] = weights * (~observed & ~prediction)
    else:
        raise ValueError('Unknown paired bootstrap endpoint')
    columns = list(moment.drop(columns='coordinate_year').columns)
    groups = moment.groupby('coordinate_year')[columns].sum().reindex(group_ids)
    values = groups.to_numpy(float)
    sampled = values[draws].sum(axis=1)
    full = values.sum(axis=0, keepdims=True)

    def scores(array):
        v = {col: array[:, i] for i, col in enumerate(columns)}
        result = {}
        if kind == 'mean':
            for label in ('model', 'benchmark'):
                result[label] = v[label + '_sum'] / v['mass']
            result['difference'] = result['model'] - result['benchmark']
        elif kind == 'severity':
            for label in ('model', 'benchmark'):
                result[label + '_rmse'] = np.sqrt(v[label + '_sse'] / v['mass'])
                result[label + '_mae'] = v[label + '_absolute'] / v['mass']
                result[label + '_bias'] = v[label + '_error'] / v['mass']
            for metric in ('rmse', 'mae', 'bias'):
                result[metric + '_difference'] = result['model_' + metric] - result['benchmark_' + metric]
        else:
            for label in ('model', 'benchmark'):
                for metric, num, den in [('sensitivity', 'tp', 'positive'), ('specificity', 'tn', 'negative')]:
                    result[label + '_' + metric] = np.divide(v[label + '_' + num], v[den], out=np.full(len(array), np.nan), where=v[den] > 0)
                result[label + '_balanced_accuracy'] = 0.5 * (result[label + '_sensitivity'] + result[label + '_specificity'])
            for metric in ('sensitivity', 'specificity', 'balanced_accuracy'):
                result[metric + '_difference'] = result['model_' + metric] - result['benchmark_' + metric]
        return result
    return ({key: float(value[0]) for key, value in scores(full).items()}, scores(sampled))
