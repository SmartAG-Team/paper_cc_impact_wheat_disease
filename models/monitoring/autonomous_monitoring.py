"""Honest coordinate-OOB calibration and optional same-field prior updates."""
import numpy as np

from .autonomous_weather import OUTPUTS
from .discrepancy_filter import predict_discrepancy


def coordinate_oob(fitted, baseline, augmented, phase, frame):
    baseline, augmented, phase = [np.asarray(x, float) for x in [baseline, augmented, phase]]
    if any(len(a) != len(frame) for a in [baseline, augmented, phase]):
        raise ValueError('Aligned OOB features required')
    totals = {name: np.zeros(len(frame)) for name in OUTPUTS}
    count = np.zeros(len(frame), int)
    cluster = frame.coordinate_year.astype(str).to_numpy()
    gate = np.maximum(phase, fitted['growth']['phase_weight_floor']); gate /= gate.sum(1, keepdims=True)
    for member, reference in zip(fitted['growth']['members'], fitted['baseline']):
        use = ~np.isin(cluster, member['clusters'])
        if not use.any():
            continue
        shared = np.clip(member['shared'].predict(augmented[use]), 0, 100)
        experts = np.column_stack([np.clip(h.predict(augmented[use]), 0, 100) for h in member['stage']])
        staged = np.sum(gate[use]*experts, axis=1)
        values = [np.clip(reference.predict(baseline[use]), 0, 100), shared, staged]
        for name, value in zip(OUTPUTS, values):
            totals[name][use] += value
        count[use] += 1
    predictions = {name: np.divide(value, count, out=np.full(len(frame), np.nan), where=count > 0)
        for name, value in totals.items()}
    return dict(predictions=predictions, count=count)


def monitoring_forecast(parameters, field_queries, autonomous_prior, link):
    """Optional observation-conditioned posterior; never the climate free run.

    The inherited predictor masks all query-day/future outcomes and groups by
    exact field identity. Nominal code equality has no anatomical rank metric.
    """
    return predict_discrepancy(parameters, field_queries, autonomous_prior, link)
