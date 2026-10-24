"""Ordered categorical boosting with direct or observed-anchor growth targets."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, __version__


def _matrix(x, frame):
    result = pd.DataFrame(np.asarray(x, float), columns=[f'x{i}' for i in range(np.shape(x)[1])])
    result['source'] = frame.source.astype(str).to_numpy()
    result['native_code'] = frame.leaf_index.astype(str).to_numpy()
    result['source_code'] = result.source+'|'+result.native_code
    result['country'] = frame.country.fillna('unknown').astype(str).to_numpy() if 'country' in frame else 'unknown'
    return result


def fit_ordered(x, frame, baseline, weights, formulation='direct', iterations=500, seed=20261022):
    if formulation not in ['direct', 'growth']:
        raise ValueError('Registered direct or anchored growth formulation required')
    baseline, weights = np.asarray(baseline, float), np.asarray(weights, float)
    target = frame.observed_percent.to_numpy(float)
    if not np.isfinite(np.asarray(x, float)).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Finite predictors and positive scientific weights required')
    if formulation == 'growth':
        target = target-baseline
    matrix = _matrix(x, frame)
    order = np.argsort(pd.to_datetime(frame.date).to_numpy(), kind='stable')
    config = dict(iterations=iterations, depth=4, learning_rate=.04, l2_leaf_reg=10.,
        border_count=32, loss_function='RMSE', boosting_type='Ordered', has_time=True,
        bootstrap_type='Bayesian', bagging_temperature=1., random_strength=1.,
        thread_count=1, random_seed=seed, verbose=False, allow_writing_files=False,
        allow_const_label=True, one_hot_max_size=2)
    fitted = dict(config=config, formulation=formulation, features=np.shape(x)[1],
        catboost_version=__version__, training_ids=frame.global_target_index.astype(int).tolist(),
        training_weight=weights, training_order=order, baseline=baseline,
        categorical_columns=['source', 'native_code', 'source_code', 'country'])
    if (matrix.nunique() <= 1).all():
        fitted['constant'] = float(np.average(target, weights=weights))
        fitted['model'] = None
    else:
        model = CatBoostRegressor(**config)
        model.fit(matrix.iloc[order], target[order], sample_weight=(weights/weights.mean())[order],
            cat_features=fitted['categorical_columns'])
        fitted['model'] = model
    return fitted


def predict_ordered(fitted, x, frame, baseline):
    if np.shape(x)[1] != fitted['features']:
        raise ValueError('Original ordered-model feature schema required')
    value = (np.full(len(frame), fitted['constant']) if fitted['model'] is None
        else fitted['model'].predict(_matrix(x, frame), thread_count=1))
    if fitted['formulation'] == 'growth':
        value = np.asarray(baseline, float)+value
    return np.clip(value, 0, 100)
