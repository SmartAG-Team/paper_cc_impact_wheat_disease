"""Use all training assessments to estimate severity without disease history."""
import numpy as np
import pandas as pd

from .field_history_models import fit_ensemble,predict_ensemble
from .history import SAME_NAMES,OTHER_NAMES


def _empty_history(length):
    return pd.DataFrame(np.zeros((length,len(SAME_NAMES)+len(OTHER_NAMES))),columns=SAME_NAMES+OTHER_NAMES)


def fit_blind(features,training,*,members=20):
    """Original field-split estimator with history absent on every training row."""
    matrix=np.asarray(features,float)
    if matrix.shape!=(len(training),53) or not np.isfinite(matrix).all():
        raise ValueError('Complete 53-column weather, stage and native-code design required')
    return fit_ensemble(matrix,_empty_history(len(training)),training,dict(structure='field_split'),
                        members=members,fraction=.8,seed=20261022)


def predict_blind(fitted,features,queries):
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld')
    matrix=np.asarray(features,float)
    if matrix.shape!=(len(queries),53) or not np.isfinite(matrix).all():
        raise ValueError('Complete 53-column weather, stage and native-code design required')
    return predict_ensemble(fitted,matrix,_empty_history(len(queries)),queries)
