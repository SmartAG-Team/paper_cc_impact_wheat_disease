"""First-visit conditional severity distributions with mean-preserving soft ordinals."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

THRESHOLDS = np.arange(5., 101., 5.)


def _category(frame):
    return frame.source.astype(str)+'|'+frame.leaf_index.astype(str)


def ordinal_cases(x, y, weights):
    x, y, weights = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    if x.ndim != 2 or y.shape != (len(x),) or weights.shape != y.shape:
        raise ValueError('Aligned predictors, percentage targets and scientific weights required')
    if not np.isfinite(np.r_[x.ravel(),y,weights]).all() or np.any((y < 0) | (y > 100)) or (weights <= 0).any():
        raise ValueError('Finite percentage targets and positive original weights required')
    q = np.clip((y[:,None]-(THRESHOLDS-5))/5, 0, 1)
    n = len(THRESHOLDS)
    return dict(x=np.column_stack([np.repeat(x, 2*n, axis=0),
            np.tile(np.repeat(THRESHOLDS/100, 2), len(x))]),
        labels=np.tile([0,1], len(x)*n),
        weights=(weights[:,None,None]*np.stack([1-q,q], axis=-1)/n).ravel(),
        target_index=np.repeat(np.arange(len(x)), 2*n))


def _features(x, frame, categories):
    return np.column_stack([np.asarray(x, float),
        (_category(frame).to_numpy()[:,None] == np.asarray(categories)[None,:]).astype(float)])


def fit_cold_ordinal(x, frame, weights, available, members=20, iterations=200):
    cold = ~np.asarray(available, bool)
    frame = frame.reset_index(drop=True)
    if cold.shape != (len(frame),):
        raise ValueError('Aligned measured-history availability required')
    categories = sorted(_category(frame.loc[cold]).unique().tolist())
    fitted = dict(features=np.asarray(x).shape[1], categories=categories, members=[],
        training_ids=frame.global_target_index.astype(int).tolist(),
        cold_training_ids=frame.loc[cold,'global_target_index'].astype(int).tolist(),
        thresholds=THRESHOLDS, iterations=iterations)
    if not cold.any():
        return fitted
    xx = _features(x, frame, categories)
    clusters = frame.coordinate_year.astype(str).to_numpy()
    unique = np.unique(clusters[cold]); rng = np.random.default_rng(20261022)
    weights = np.asarray(weights, float)
    for member in range(members):
        keep = rng.choice(unique, max(1, int(np.ceil(.8*len(unique)))), replace=False)
        ix = np.flatnonzero(cold & np.isin(clusters, keep))
        mass = weights[ix]/weights[ix].mean(); y = frame.observed_percent.to_numpy(float)[ix]
        common = dict(max_iter=iterations, learning_rate=.05, max_depth=3, max_leaf_nodes=15,
            l2_regularization=3., early_stopping=False, random_state=20261022+member)
        constant = float(y[0]) if np.all(y == y[0]) else None
        mean, classifier = None, None
        if constant is None:
            mean = HistGradientBoostingRegressor(min_samples_leaf=10, **common)
            mean.fit(xx[ix], y, sample_weight=mass)
            cases = ordinal_cases(xx[ix], y, mass)
            classifier = HistGradientBoostingClassifier(min_samples_leaf=80,
                monotonic_cst=[0]*xx.shape[1]+[-1], **common)
            classifier.fit(cases['x'], cases['labels'], sample_weight=cases['weights'])
        fitted['members'].append(dict(mean=mean, ordinal=classifier, constant=constant,
            categories=sorted(_category(frame.iloc[ix]).unique().tolist()),
            clusters=keep.tolist(), training_ids=frame.global_target_index.iloc[ix].astype(int).tolist(),
            scientific_weight=weights[ix], normalized_weight=mass))
    return fitted


def predict_cold_ordinal(fitted, x, queries, baseline, available):
    if 'observed_percent' in queries:
        raise ValueError('Query outcomes must be absent')
    baseline, cold = np.asarray(baseline, float), ~np.asarray(available, bool)
    xx = _features(x, queries, fitted['categories'])
    totals = np.zeros((len(queries), len(THRESHOLDS)))
    mean = np.zeros(len(queries)); counts = np.zeros(len(queries), int)
    for member in fitted['members']:
        use = cold & _category(queries).isin(member['categories']).to_numpy()
        if not use.any():
            continue
        constant = member['constant']
        if constant is not None:
            p = np.tile(np.clip((constant-(THRESHOLDS-5))/5,0,1), (use.sum(),1))
            value = np.full(use.sum(), constant)
        else:
            expanded = np.column_stack([np.repeat(xx[use],len(THRESHOLDS),axis=0),
                np.tile(THRESHOLDS/100,use.sum())])
            p = member['ordinal'].predict_proba(expanded)[:,1].reshape(use.sum(),len(THRESHOLDS))
            value = np.clip(member['mean'].predict(xx[use]),0,100)
        if np.any(np.diff(p,axis=1) > 1e-12):
            raise RuntimeError('Conditional exceedance probabilities must decrease with threshold')
        totals[use] += p; mean[use] += value; counts[use] += 1
    probabilities = np.divide(totals,counts[:,None],out=np.full_like(totals,np.nan),where=counts[:,None]>0)
    ordinal = baseline.copy(); use = counts > 0
    ordinal[use] = np.clip(5*probabilities[use].sum(axis=1),0,100)
    return dict(ordinal=ordinal, mean=np.divide(mean,counts,out=baseline.copy(),where=use),
        threshold_probability=probabilities, member_count=counts)
