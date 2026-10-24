"""Learn convex component weights from whole-coordinate-year OOB predictions."""
import numpy as np
import pandas as pd

from analysis.final_study.metrics import hierarchical_weights
from .models import predict_candidate
from .field_history_models import predict_field_history, predict_ensemble
from .robust_models import predict_cluster_ensemble
from .history_interval_hybrid import history_prediction

GROUPS = ('first_field', 'same_leaf', 'other_leaf_only')


def history_groups(history):
    values = history[['prior_assessment_available', 'other_leaf_available']].to_numpy(float)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError('Finite binary history availability required')
    same, other = values.T.astype(bool)
    return np.where(same, 'same_leaf', np.where(other, 'other_leaf_only', 'first_field'))


def membership_masks(ensemble, training):
    """True only for rows whose entire coordinate-year is absent from a member."""
    ids = training.global_target_index.to_numpy(int)
    if not training.global_target_index.is_unique or training.coordinate_year.isna().any():
        raise ValueError('Unique training identity and complete coordinate-years required')
    if set(ensemble['training_ids']) != set(ids) or len(ensemble['training_ids']) != len(ids):
        raise ValueError('Ensemble pool differs from calibration training pool')
    masks = []
    for member in ensemble['members']:
        fitted_ids = member['training_ids']
        if len(set(fitted_ids)) != len(fitted_ids) or not set(fitted_ids) <= set(ids):
            raise ValueError('Member training identity is invalid')
        inside = training.global_target_index.isin(fitted_ids)
        selected_groups = training.loc[inside, 'coordinate_year']
        whole = training.coordinate_year.isin(selected_groups)
        if not np.array_equal(inside, whole):
            raise ValueError('Member sampling must include whole coordinate-year groups')
        masks.append(~whole.to_numpy())
    return np.asarray(masks, bool)


def oob_predictions(process, original_history, event_history, designs, history, training):
    """Use only aligned members that never fitted the queried coordinate-year."""
    masks = membership_masks(process, training)
    for ensemble in [original_history, event_history]:
        if not np.array_equal(masks, membership_masks(ensemble, training)):
            raise ValueError('All component members must share coordinate-year samples')
    sums = np.zeros((len(training), 2))
    counts = masks.sum(axis=0)
    for i, mask in enumerate(masks):
        if not mask.any():
            continue
        query = training.loc[mask].drop(columns='observed_percent')
        h = history.iloc[np.flatnonzero(mask)]
        p = predict_candidate(process['members'][i], designs['process'][mask], h, query)
        old = predict_field_history(original_history['members'][i], designs['station'][mask], h, query)
        new = predict_field_history(event_history['members'][i], designs['event_station'][mask], h, query)
        selected, _ = history_prediction(old, new, h, query)
        sums[mask] += np.column_stack([p, selected])
    means = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
    return means[:, 0], means[:, 1], counts


def fit_weights(process, history_prediction, history, training, oob_counts):
    """Three fixed groups; ten equivalent neutral groups regularize each weight."""
    p, h, counts = np.asarray(process, float), np.asarray(history_prediction, float), np.asarray(oob_counts)
    if (p.shape != (len(training),) or h.shape != p.shape or counts.shape != p.shape
            or len(history) != len(training) or not np.isfinite(counts).all()
            or np.any(counts < 0) or np.any(counts != np.floor(counts))):
        raise ValueError('Aligned OOB components and nonnegative integer counts required')
    y = training.observed_percent.to_numpy(float)
    eligible = counts >= 2
    if (not np.isfinite(y).all() or np.any((y < 0)|(y > 100))
            or not np.isfinite(np.c_[p[eligible], h[eligible]]).all()
            or np.any((p[eligible] < 0)|(p[eligible] > 100)|(h[eligible] < 0)|(h[eligible] > 100))):
        raise ValueError('Bounded labels and eligible OOB predictions required')
    group = history_groups(history)
    weights = hierarchical_weights(training)
    # Full training weights retain missing-OOB mass rather than renormalizing it away.
    weights = weights/weights.sum()*training.coordinate_year.nunique()
    fitted = dict(prior_equivalent_coordinate_years=10., minimum_oob_members=2,
                  calibration_training_ids=training.global_target_index.astype(int).tolist(), groups={})
    for name in GROUPS:
        mask = eligible & (group == name)
        w = weights[mask]
        delta = h[mask]-p[mask]
        denominator = float(np.sum(w*delta**2))
        numerator = float(np.sum(w*delta*(y[mask]-p[mask])))
        mass = float(w.sum())
        penalty = 10.*denominator/mass if mass > 0 else 0.
        value = (numerator+.5*penalty)/(denominator+penalty) if denominator > 0 else .5
        fitted['groups'][name] = dict(history_weight=float(np.clip(value, 0, 1)),
            calibration_rows=int(mask.sum()), equivalent_coordinate_year_mass=mass,
            squared_disagreement=denominator, cross_product=numerator, neutral_penalty=penalty)
    return fitted


def apply_weights(fitted, process, history_prediction, history, queries):
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld')
    p, h = np.asarray(process, float), np.asarray(history_prediction, float)
    if (p.shape != (len(queries),) or h.shape != p.shape or len(history) != len(queries)
            or not np.isfinite(np.c_[p, h]).all() or np.any((p < 0)|(p > 100)|(h < 0)|(h > 100))):
        raise ValueError('Aligned bounded query components required')
    group = history_groups(history)
    values = np.asarray([fitted['groups'][name]['history_weight'] for name in GROUPS], float)
    if not np.isfinite(values).all() or np.any((values < 0)|(values > 1)):
        raise ValueError('Convex fitted weights required')
    w = np.asarray([fitted['groups'][name]['history_weight'] for name in group])
    return (1-w)*p+w*h


def predict(fitted, process, station, event_station, history, queries):
    """Future native-domain inference; qualification and causal inputs are upstream."""
    components = fitted['components']
    base = components['base']
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld')
    dates = pd.to_datetime(queries.date, errors='coerce', format='mixed')
    if (dates.isna().any() or not dates.dt.year.eq(queries.season_year).all()
            or not queries.season_year.ge(base['minimum_assessment_year']).all()
            or not queries.source.isin(['BASF', 'Corteva']).all()
            or not queries.leaf_index.isin(range(7)).all()):
        raise ValueError('Queries outside the frozen future native-source domain')
    arrays = [np.asarray(a, float) for a in [process, station, event_station]]
    if (any(a.shape != (len(queries), width) or not np.isfinite(a).all()
            for a, width in zip(arrays, [37, 53, 68]))
            or history.columns.tolist() != base['history_names']
            or history.shape != (len(queries), len(base['history_names']))
            or not np.isfinite(history.to_numpy(float)).all()):
        raise ValueError('Complete aligned predictor schema required')
    p = predict_cluster_ensemble(base['process'], arrays[0], history, queries)
    old = predict_ensemble(base['history'], arrays[1], history, queries)
    new = predict_ensemble(components['events_history'], arrays[2], history, queries)
    selected, _ = history_prediction(old, new, history, queries)
    return apply_weights(fitted['calibration'], p, selected, history, queries)
