"""Leaf-conditional empirical sowing priors with original scientific mass."""
import numpy as np


def conditional_sowing_prior(forest, x_training, observed_days, weights, queries, quantiles=None):
    """Integrate observed-date spread within the forest's training partitions.

    Every tree gives its matched leaf equal ensemble mass; observations inside
    that leaf share mass in proportion to their original scientific weights.
    All admitted observations assigned to a leaf are included, regardless of
    tree bootstrap multiplicity. This is a defined empirical prior, not a
    calibrated prediction interval or the distribution of tree mean outputs.
    """
    x_training, queries = np.asarray(x_training, float), np.asarray(queries, float)
    observed_days, weights = np.asarray(observed_days, float), np.asarray(weights, float)
    quantiles = (np.arange(9)+.5)/9 if quantiles is None else np.asarray(quantiles, float)
    if (len(x_training) != len(observed_days) or weights.shape != observed_days.shape or
        not np.isfinite(np.r_[x_training.ravel(), queries.ravel(), observed_days, weights]).all() or
        (weights <= 0).any() or np.any((quantiles < 0) | (quantiles > 1))):
        raise ValueError('Finite aligned training data and positive scientific weights required')
    training_leaf = forest.apply(x_training)
    query_leaf = forest.apply(queries)
    if training_leaf.shape[1] != query_leaf.shape[1] or training_leaf.shape[1] == 0:
        raise ValueError('One frozen forest partition is required')
    conditional_weight = np.zeros((len(queries), len(observed_days)))
    for tree in range(training_leaf.shape[1]):
        order = np.argsort(training_leaf[:, tree], kind='stable')
        values = training_leaf[order, tree]
        for leaf in np.unique(query_leaf[:, tree]):
            qi = np.flatnonzero(query_leaf[:, tree] == leaf)
            lo = np.searchsorted(values, leaf, side='left')
            hi = np.searchsorted(values, leaf, side='right')
            ti = order[lo:hi]
            if not len(ti):
                raise ValueError('Every query leaf must have admitted training support')
            mass = weights[ti]/weights[ti].sum()/training_leaf.shape[1]
            conditional_weight[np.ix_(qi, ti)] += mass[None, :]
    conditional_weight /= conditional_weight.sum(axis=1, keepdims=True)
    order = np.argsort(observed_days, kind='stable')
    days = []
    for weight in conditional_weight:
        support = order[weight[order] > 0]
        cumulative = np.cumsum(weight[support]); cumulative[-1] = 1.
        positions = np.clip(np.searchsorted(cumulative, quantiles, side='left'), 0, len(support)-1)
        days.append(observed_days[support[positions]])
    days = np.asarray(days)
    mean = conditional_weight@observed_days
    variance = (conditional_weight*(observed_days[None, :]-mean[:, None])**2).sum(axis=1)
    return dict(days=days, quantiles=np.asarray(quantiles), mean=mean,
        standard_deviation=np.sqrt(np.maximum(variance, 0)), conditional_weight=conditional_weight,
        effective_records=1/np.sum(conditional_weight**2, axis=1))
