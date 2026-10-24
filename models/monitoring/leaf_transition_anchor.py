"""State-feature ablation retaining the established measured-severity anchor."""
import numpy as np

from .field_history_models import fit_ensemble as fit_field, predict_ensemble as predict_field
from .history_interval_hybrid import predict as predict_parent
from .leaf_transition_model import _design, route_transition
from .robust_models import predict_cluster_ensemble


def fit_ensemble(X, history, states, training, members=20, fraction=.8, seed=20261022):
    _design(X, history, states)
    return fit_field(np.c_[X, states], history, training, dict(structure='field_split'),
        members=members, fraction=fraction, seed=seed)


def predict_ensemble(fitted, X, history, states, queries):
    if 'observed_percent' in queries:
        raise ValueError('Current query labels must be withheld')
    _design(X, history, states)
    return predict_field(fitted, np.c_[X, states], history, queries)


def predict(fitted, process, station, event_station, history, states, queries):
    if ('observed_percent' in queries or queries.season_year.lt(fitted['minimum_assessment_year']).any()
            or not queries.leaf_index.isin(range(7)).all()):
        raise ValueError('Label-free future native-leaf queries required')
    old = predict_parent(fitted['parent'], process, station, event_station, history, queries)
    process_value = predict_cluster_ensemble(fitted['parent']['base']['process'], process, history, queries)
    transition = predict_ensemble(fitted['transition'], event_station, history, states, queries)
    return route_transition(old, process_value, transition, history)[0]
