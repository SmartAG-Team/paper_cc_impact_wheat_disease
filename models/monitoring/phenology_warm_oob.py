"""Measured-anchor-only phase growth with auditable out-of-bag forecasts."""
import numpy as np

from .phenology_phase_growth import fit_phase_growth, predict_phase_growth


def _category(frame):
    return frame.source.astype(str)+'|'+frame.leaf_index.astype(str)


def fit_warm_growth(x, frame, anchor, weights, phase, available, members=20, iterations=200):
    # Cold outcomes never enter the global fitting target. Their measured
    # values may legitimately appear in a later query's supplied past history.
    available = np.asarray(available, bool)
    anchor = np.asarray(anchor, float)
    if available.shape != (len(frame),) or anchor.shape != (len(frame),):
        raise ValueError('Aligned measured-history availability and percentage anchors required')
    if not np.isfinite(anchor[available]).all() or np.any((anchor[available] < 0) | (anchor[available] > 100)):
        raise ValueError('Finite bounded measured anchors required')
    warm = frame.iloc[np.flatnonzero(available)].copy()
    fitted = dict(training_ids=frame.global_target_index.astype(int).tolist(),
        warm_training_ids=warm.global_target_index.astype(int).tolist(),
        growth=None, member_categories=[])
    if not len(warm):
        return fitted
    fitted['growth'] = fit_phase_growth(np.asarray(x)[available], warm, anchor[available],
        np.asarray(weights)[available], np.asarray(phase)[available], members=members, iterations=iterations)
    by_id = warm.set_index('global_target_index')
    for member in fitted['growth']['members']:
        training = by_id.loc[member['training_ids']]
        fitted['member_categories'].append(sorted(_category(training).unique().tolist()))
    return fitted


def predict_warm_growth(fitted, x, queries, anchor, phase, baseline, available, oob=False):
    if 'observed_percent' in queries:
        raise ValueError('Forecast queries cannot contain assessment outcomes')
    x, anchor, phase = np.asarray(x, float), np.asarray(anchor, float), np.asarray(phase, float)
    baseline, available = np.asarray(baseline, float), np.asarray(available, bool)
    if x.shape[0] != len(queries) or baseline.shape != (len(queries),) or not np.isfinite(baseline).all():
        raise ValueError('Aligned finite predictors and reference forecasts required')
    growth = fitted['growth']
    members = growth['members'] if growth is not None else []
    eligible = np.zeros((len(queries), len(members)), dtype=bool)
    totals = {key:np.zeros(len(queries)) for key in ['shared', 'stage']}
    for k, (member, supported) in enumerate(zip(members, fitted['member_categories'], strict=True)):
        keep = available & _category(queries).isin(supported).to_numpy()
        if oob:
            keep &= ~queries.coordinate_year.astype(str).isin(member['clusters']).to_numpy()
        eligible[:, k] = keep
        if not keep.any():
            continue
        one = dict(growth, members=[member])
        values = predict_phase_growth(one, x[keep], anchor[keep], phase[keep])
        for key in totals:
            totals[key][keep] += values[key]
    counts = eligible.sum(axis=1)
    result = {key:np.divide(total, counts, out=baseline.copy(), where=counts > 0)
        for key, total in totals.items()}
    for value in result.values():
        # Weighted sums of bounded heads can exceed 100 by one floating ULP.
        value[counts > 0] = np.clip(value[counts > 0], 0, 100)
    result.update(member_count=counts, eligible_members=eligible)
    return result
