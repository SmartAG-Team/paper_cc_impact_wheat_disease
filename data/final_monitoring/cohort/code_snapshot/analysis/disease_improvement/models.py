"""Weighted residual, separate cold/warm and process-correction candidates."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from analysis.final_study.metrics import hierarchical_weights


def _category(frame):
    return (frame.source.astype(str)+':'+frame.leaf_index.astype(str)).to_numpy()


def _baseline(fitted, history, frame, X):
    if fitted['spec'].get('structure') == 'direct':
        return np.zeros(len(frame), dtype=float)
    if fitted['spec'].get('structure') == 'process':
        baseline = np.asarray(X)[:, fitted['spec']['process_baseline_column']]
        if not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 1)):
            raise ValueError('Bounded process-state baseline required')
        return baseline
    prior = np.array([fitted['means'].get(key, fitted['pooled']) for key in _category(frame)])
    return np.where(history.prior_assessment_available.to_numpy() > 0,
                    history.latest_severity_fraction.to_numpy(), prior)


def _estimator(spec):
    if spec.get('estimator', 'hgb') == 'rf':
        return RandomForestRegressor(n_estimators=160, max_depth=8,
            min_samples_leaf=spec.get('min_samples_leaf', 5), max_features=.8,
            n_jobs=1, random_state=20261021)
    return HistGradientBoostingRegressor(max_iter=spec.get('max_iter', 200),
        max_leaf_nodes=spec.get('max_leaf_nodes', 15), max_depth=spec.get('max_depth', 3),
        min_samples_leaf=spec.get('min_samples_leaf', 10),
        learning_rate=spec.get('learning_rate', .05), l2_regularization=spec.get('l2_regularization', 1.), early_stopping=False,
        random_state=20261021)


def fit_candidate(X, H, training, spec):
    if spec.get('source_specific', False):
        inner = dict(spec, source_specific=False)
        result = dict(source_models={}, fallback=fit_candidate(X, H, training, inner),
                      training_ids=training.global_target_index.astype(int).tolist())
        for source in sorted(training.source.unique()):
            mask = training.source.eq(source).to_numpy()
            result['source_models'][source] = fit_candidate(np.asarray(X)[mask], H.iloc[mask],
                                                            training.loc[mask], inner)
        return result
    x = np.asarray(X, float)
    h = H.reset_index(drop=True)
    frame = training.reset_index(drop=True)
    if len(x) != len(frame) or len(h) != len(frame) or not np.isfinite(x).all():
        raise ValueError('Complete aligned training predictors required')
    y = frame.observed_percent.to_numpy(float)/100
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError('Valid severity required')
    w = hierarchical_weights(frame); w = w/w.mean()
    pooled = float(np.average(y, weights=w))
    result = dict(spec=spec.copy(), pooled=pooled, means={}, models={},
        n_features=x.shape[1], history_names=h.columns.tolist(),
        training_ids=frame.global_target_index.astype(int).tolist())
    category = _category(frame)
    # Ten equivalent average-weight records shrink sparse source-leaf offsets.
    for key in np.unique(category):
        mask = category == key
        result['means'][key] = float((np.sum(w[mask]*y[mask])+10*pooled)/(w[mask].sum()+10))
    base = _baseline(result, h, frame, x)
    design = np.column_stack([x, h.to_numpy(float)])
    cold = h.prior_assessment_available.to_numpy() == 0
    split = spec.get('structure', 'residual') == 'split'
    groups = [('cold', cold), ('warm', ~cold)] if split else [('all', np.ones(len(y), bool))]
    for label, mask in groups:
        if not mask.any():
            result['models'][label] = None
            continue
        target = y if label == 'cold' else y-base
        # Preserve the full training estimand; scalar normalization changes no ratios.
        head_w = w[mask].copy(); head_w /= head_w.mean()
        estimator = _estimator(spec)
        estimator.fit(design[mask], target[mask], sample_weight=head_w)
        result['models'][label] = estimator
    return result


def predict_candidate(fitted, X, H, queries):
    if 'source_models' in fitted:
        output = np.empty(len(queries))
        for source in queries.source.unique():
            mask = queries.source.eq(source).to_numpy()
            local = fitted['source_models'].get(source, fitted['fallback'])
            output[mask] = predict_candidate(local, np.asarray(X)[mask], H.iloc[mask], queries.loc[mask])
        return output
    x = np.asarray(X, float); h = H.reset_index(drop=True)
    if (x.shape != (len(queries), fitted['n_features']) or len(h) != len(queries)
            or h.columns.tolist() != fitted['history_names'] or not np.isfinite(x).all()):
        raise ValueError('Complete schema-aligned query predictors required')
    base = _baseline(fitted, h, queries, x)
    design = np.column_stack([x, h.to_numpy(float)])
    output = base.copy()
    cold = h.prior_assessment_available.to_numpy() == 0
    split = fitted['spec'].get('structure', 'residual') == 'split'
    groups = [('cold', cold), ('warm', ~cold)] if split else [('all', np.ones(len(x), bool))]
    for label, mask in groups:
        estimator = fitted['models'][label]
        if mask.any() and estimator is not None:
            correction = estimator.predict(design[mask])
            output[mask] = correction if label == 'cold' else base[mask]+correction
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite severity prediction')
    return 100*np.clip(output, 0, 1)
