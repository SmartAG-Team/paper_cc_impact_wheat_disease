"""Subsample whole coordinate-year clusters to reduce fitting variance."""
import numpy as np
from .models import fit_candidate,predict_candidate


def fit_cluster_ensemble(X,H,training,spec,*,members=20,fraction=.8,seed=20261022):
    if members<2 or not 0<fraction<1:
        raise ValueError('At least two members and a subsampling fraction inside(0,1) required')
    clusters=np.asarray(sorted(training.coordinate_year.unique()))
    if not len(clusters):raise ValueError('Nonempty training clusters required')
    rng=np.random.default_rng(seed);fitted=[]
    count=max(1,int(np.ceil(len(clusters)*fraction)))
    for _ in range(members):
        selected=rng.choice(clusters,size=count,replace=False)
        positions=np.flatnonzero(training.coordinate_year.isin(selected))
        fitted.append(fit_candidate(np.asarray(X)[positions],H.iloc[positions],training.iloc[positions],spec))
    return dict(members=fitted,seed=seed,fraction=fraction,clusters_per_member=count,
        training_ids=training.global_target_index.astype(int).tolist(),
        sampling='whole coordinate-year groups without replacement; all rows of each selected group')


def predict_cluster_ensemble(fitted,X,H,queries):
    return np.mean([predict_candidate(member,X,H,queries) for member in fitted['members']],axis=0)
