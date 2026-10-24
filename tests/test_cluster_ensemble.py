import numpy as np
import pandas as pd


def test_ensemble_samples_complete_coordinate_years_and_is_reproducible():
    from models.monitoring.robust_models import fit_cluster_ensemble,predict_cluster_ensemble
    from models.monitoring.history import SAME_NAMES
    rows=[]
    for cluster in range(5):
        for i in range(4):
            rows.append(dict(global_target_index=4*cluster+i,coordinate_year=f's{cluster}|2018',field_id=f'f{cluster}',
                leaf_index=0,source='BASF',observed_percent=float(cluster*20),season_year=2018))
    frame=pd.DataFrame(rows);x=np.linspace(0,1,20)[:,None]
    history=pd.DataFrame(np.zeros((20,8)),columns=SAME_NAMES)
    spec=dict(structure='direct',max_iter=2,min_samples_leaf=2)
    a=fit_cluster_ensemble(x,history,frame,spec,members=4,fraction=.8)
    b=fit_cluster_ensemble(x,history,frame,spec,members=4,fraction=.8)
    for first,second in zip(a['members'],b['members']):
        assert first['training_ids']==second['training_ids']
        selected=frame.loc[frame.global_target_index.isin(first['training_ids'])]
        assert len(selected)==16
        assert selected.groupby('coordinate_year').size().eq(4).all()
    query=frame.drop(columns='observed_percent')
    p=predict_cluster_ensemble(a,x,history,query)
    np.testing.assert_array_equal(p,predict_cluster_ensemble(b,x,history,query))
    assert np.isfinite(p).all() and ((p>=0)&(p<=100)).all()


def test_cluster_ensemble_preserves_unequal_field_hierarchy():
    from models.monitoring.robust_models import fit_cluster_ensemble
    from models.monitoring.history import SAME_NAMES
    rows=[]
    for cluster,field,value,count in [('x','a',0.,10),('x','b',100.,1),('y','c',20.,8),('z','d',80.,1)]:
        for _ in range(count):
            rows.append(dict(global_target_index=len(rows),coordinate_year=cluster,field_id=field,
                leaf_index=0,source='BASF',observed_percent=value,season_year=2018))
    frame=pd.DataFrame(rows);x=np.zeros((len(frame),1))
    history=pd.DataFrame(np.zeros((len(frame),8)),columns=SAME_NAMES)
    fitted=fit_cluster_ensemble(x,history,frame,dict(structure='direct',max_iter=1,min_samples_leaf=100),
        members=4,fraction=.6)
    cluster_means={'x':50.,'y':20.,'z':80.}
    for member in fitted['members']:
        selected=frame.loc[frame.global_target_index.isin(member['training_ids'])]
        expected=np.mean([cluster_means[c] for c in selected.coordinate_year.unique()])/100
        assert np.isclose(member['pooled'],expected)
