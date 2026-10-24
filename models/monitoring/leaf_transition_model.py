"""Learn bounded severity changes around observation-updated leaf-state proxies."""
import numpy as np

from analysis.final_study.metrics import hierarchical_weights
from .field_history_models import field_history_anchor
from .history_interval_hybrid import predict as predict_parent
from .leaf_state import STATE_NAMES
from .models import _estimator


def _design(X, history, states):
    x, z = np.asarray(X, float), np.asarray(states, float)
    if (x.ndim != 2 or len(x) != len(history) or z.shape != (len(x), len(STATE_NAMES))
            or not np.isfinite(x).all() or not np.isfinite(z).all()
            or not np.isfinite(history.to_numpy(float)).all()):
        raise ValueError('Aligned complete predictor schema required')
    if (np.any(z[:, :7] < 0) or not np.allclose(z[:, :7].sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError('Unit-mass leaf-state schema required')
    same = history.prior_assessment_available.to_numpy(float)
    other = history.other_leaf_available.to_numpy(float)
    if not np.isin(np.c_[same, other], [0, 1]).all():
        raise ValueError('Binary field history availability required')
    available = (same > 0) | (other > 0)
    if (not np.array_equal(z[:, STATE_NAMES.index('state_same_assimilated')], same)
            or not np.array_equal(z[:, STATE_NAMES.index('state_any_assimilated')], available.astype(float))):
        raise ValueError('State and field history availability disagree')
    return np.c_[x, history.to_numpy(float), z], z[:, 4:7].sum(axis=1), available


def fit_transition(X, history, states, training):
    design, baseline, available = _design(X, history, states)
    y = training.observed_percent.to_numpy(float)/100
    if len(y) != len(design) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError('Complete aligned bounded training severity required')
    # Preserve the full training hierarchy before restricting to observed-field transitions.
    weights = hierarchical_weights(training)
    estimator = None
    if available.any():
        estimator = _estimator({})
        head_weights = weights[available]/weights[available].mean()
        estimator.fit(design[available], (y-baseline)[available], sample_weight=head_weights)
    return dict(estimator=estimator, n_features=np.asarray(X).shape[1],
        state_names=STATE_NAMES.copy(), history_names=history.columns.tolist(),
        training_ids=training.global_target_index.astype(int).tolist(),
        transition_training_ids=training.loc[available, 'global_target_index'].astype(int).tolist())


def predict_transition(fitted, X, history, states, queries):
    if 'observed_percent' in queries:
        raise ValueError('Current query labels must be withheld')
    if (np.asarray(X).shape != (len(queries), fitted['n_features'])
            or len(history) != len(queries) or history.columns.tolist() != fitted['history_names']
            or fitted['state_names'] != STATE_NAMES):
        raise ValueError('Fitted transition predictor schema required')
    # Validates native code and compatible history without using the resulting anchor.
    field_history_anchor(history, queries)
    design, baseline, available = _design(X, history, states)
    value = baseline.copy()
    if fitted['estimator'] is not None and available.any():
        value[available] += fitted['estimator'].predict(design[available])
    return 100*np.clip(value, 0, 1)


def fit_ensemble(X, history, states, training, members=20, fraction=.8, seed=20261022):
    if members < 2 or not 0 < fraction < 1:
        raise ValueError('At least two members and a strict subsampling fraction required')
    clusters = np.asarray(sorted(training.coordinate_year.unique()))
    count = max(1, int(np.ceil(len(clusters)*fraction)))
    rng = np.random.default_rng(seed); fitted = []
    for _ in range(members):
        selected = rng.choice(clusters, size=count, replace=False)
        positions = np.flatnonzero(training.coordinate_year.isin(selected))
        fitted.append(fit_transition(np.asarray(X)[positions], history.iloc[positions],
            np.asarray(states)[positions], training.iloc[positions]))
    return dict(members=fitted, training_ids=training.global_target_index.astype(int).tolist(),
        seed=seed, fraction=fraction, clusters_per_member=count)


def predict_ensemble(fitted, X, history, states, queries):
    return np.mean([predict_transition(member, X, history, states, queries)
        for member in fitted['members']], axis=0)


def route_transition(parent, process, transition, history):
    old, p, new = [np.asarray(a, float) for a in [parent, process, transition]]
    if (old.shape != (len(history),) or old.shape != p.shape or old.shape != new.shape
            or not np.isfinite(np.c_[old, p, new]).all()
            or np.any((np.c_[old, p, new] < 0) | (np.c_[old, p, new] > 100))):
        raise ValueError('Aligned bounded parent/process/transition predictions required')
    flags = history[['prior_assessment_available', 'other_leaf_available']].to_numpy(float)
    if not np.isin(flags, [0, 1]).all():
        raise ValueError('Binary earlier field history flags required')
    mask = flags.any(axis=1)
    return np.where(mask, .5*(p+new), old), mask


def predict(fitted, process, station, event_station, history, states, queries):
    if ('observed_percent' in queries or queries.season_year.lt(fitted['minimum_assessment_year']).any()
            or not queries.leaf_index.isin(range(7)).all()):
        raise ValueError('Label-free future native-leaf queries required')
    parent = predict_parent(fitted['parent'], process, station, event_station, history, queries)
    from .robust_models import predict_cluster_ensemble
    p = predict_cluster_ensemble(fitted['parent']['base']['process'], process, history, queries)
    transition = predict_ensemble(fitted['transition'], event_station, history, states, queries)
    return route_transition(parent, p, transition, history)[0]
