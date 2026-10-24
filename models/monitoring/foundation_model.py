"""Explicit full-range target formulations for the local tabular benchmark."""
import numpy as np


def _aligned(value,baseline,kind):
    a=np.asarray(value,float);b=np.asarray(baseline,float)
    if kind not in ['anchor','direct']:
        raise ValueError('Registered anchor or direct target required')
    if a.ndim!=1 or a.shape!=b.shape or not np.isfinite(np.r_[a,b]).all() or np.any((b<0)|(b>100)):
        raise ValueError('Aligned finite targets and bounded baselines required')
    return a,b


def transform_target(observed,baseline,kind):
    y,b=_aligned(observed,baseline,kind)
    if np.any((y<0)|(y>100)):raise ValueError('Bounded severity targets required')
    return y-b if kind=='anchor' else y.copy()


def restore_prediction(prediction,baseline,kind):
    p,b=_aligned(prediction,baseline,kind)
    return np.clip(p+b if kind=='anchor' else p,0,100)


def contextual_regressor(bundle):
    """Reconstruct the pinned, fitted ICL context; no neural weights are updated."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    version=ModelVersion(bundle['version'])
    model=TabPFNRegressor.create_default_for_version(version,model_path=bundle['checkpoint'],
        n_estimators=4,device=bundle['device'],random_state=20261022,n_preprocessing_jobs=1,
        fit_mode='fit_preprocessors',tuning_config=dict(calibrate_temperature=False),show_progress_bar=False)
    model.fit(bundle['context_x'],bundle['context_y'])
    return model
