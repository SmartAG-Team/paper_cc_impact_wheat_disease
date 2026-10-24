"""Observed-anchor growth with externally learned crop-development phase gates."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def phase_weights(probability):
    probability = np.asarray(probability, float)
    if probability.ndim != 2 or probability.shape[1] != 4 or not np.isfinite(probability).all():
        raise ValueError('Four ordered stage-passage probabilities required')
    if np.any((probability < 0) | (probability > 1)) or np.any(np.diff(probability, axis=1) > 1e-12):
        raise ValueError('Later stages cannot precede earlier stages')
    # Emergence is retained among the features; canopy gates split at 31/51/87.
    return np.column_stack([1-probability[:, 1], probability[:, 1]-probability[:, 2],
        probability[:, 2]-probability[:, 3], probability[:, 3]])


def fit_phase_growth(x, frame, anchor, weights, phase, members=20, iterations=200):
    x, anchor = np.asarray(x, float), np.asarray(anchor, float)
    weights, phase = np.asarray(weights, float), np.asarray(phase, float)
    if not np.isfinite(x).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Finite predictors and positive original scientific weights required')
    if phase.shape != (len(frame), 4) or (phase < -1e-12).any() or not np.allclose(phase.sum(1), 1):
        raise ValueError('Normalized nonnegative phase gates required')
    # A small fixed floor shares information with sparse phase experts.
    regularized_phase = np.maximum(phase, .01)
    regularized_phase /= regularized_phase.sum(1, keepdims=True)
    target = frame.observed_percent.to_numpy(float)-anchor
    cluster = frame.coordinate_year.astype(str).to_numpy()
    unique = np.unique(cluster)
    rng = np.random.default_rng(20261022)
    fitted = dict(features=x.shape[1], members=[], training_ids=frame.global_target_index.astype(int).tolist(),
        training_weight=weights, training_anchor=anchor, training_phase=phase,
        iterations=iterations, phase_weight_floor=.01)
    for member in range(members):
        keep = rng.choice(unique, max(1, int(np.ceil(.8*len(unique)))), replace=False)
        ix = np.flatnonzero(np.isin(cluster, keep))
        mass = weights[ix]/weights[ix].mean()

        def fit(gate):
            model = HistGradientBoostingRegressor(max_iter=iterations, learning_rate=.05, max_depth=3,
                max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=3., early_stopping=False,
                random_state=20261022+member)
            model.fit(x[ix], target[ix], sample_weight=mass*gate)
            return model

        shared = fit(np.ones(len(ix)))
        staged = [fit(regularized_phase[ix, k]) for k in range(4)]
        fitted['members'].append(dict(shared=shared, stage=staged, clusters=keep.tolist(),
            training_ids=frame.iloc[ix].global_target_index.astype(int).tolist()))
    return fitted


def predict_phase_growth(fitted, x, anchor, phase, baseline=None, available=None):
    x, anchor, phase = np.asarray(x, float), np.asarray(anchor, float), np.asarray(phase, float)
    gate = np.maximum(phase, fitted['phase_weight_floor']); gate /= gate.sum(1, keepdims=True)
    shared, staged = [], []
    for member in fitted['members']:
        shared.append(np.clip(anchor+member['shared'].predict(x), 0, 100))
        means = np.column_stack([np.clip(anchor+head.predict(x), 0, 100) for head in member['stage']])
        staged.append(np.sum(gate*means, axis=1))
    result = dict(shared=np.mean(shared, axis=0), stage=np.mean(staged, axis=0))
    if available is not None:
        mask = ~np.asarray(available, bool)
        for value in result.values():
            value[mask] = np.asarray(baseline)[mask]
    return result
