import numpy as np
import pandas as pd


def test_inner_experts_never_train_on_any_year_of_the_held_coordinate():
    from models.monitoring.crossfit_router import coordinate_folds
    frame=pd.DataFrame(dict(latitude=np.repeat(np.arange(10),2),longitude=np.zeros(20),
        season_year=np.tile([2014,2015],10)))
    seen=np.zeros(20,int)
    for train,test in coordinate_folds(frame):
        assert not set(frame.iloc[train].latitude)&set(frame.iloc[test].latitude)
        assert len(train) and len(test)
        seen[test]+=1
    np.testing.assert_array_equal(seen,np.ones(20))


def test_router_learns_context_dependent_expert_skill_without_query_labels():
    from models.monitoring.crossfit_router import fit_router,predict_router
    x=np.repeat([[-1.],[1.]],100,axis=0)
    experts=np.tile([10.,90.],(200,1))
    y=np.repeat([10.,90.],100)
    fitted=fit_router(x,experts,y,np.ones(200))
    value,prob=predict_router(fitted,np.array([[-1.],[1.]]),np.array([[10.,90.],[10.,90.]]))
    assert value[0]<15 and value[1]>85
    np.testing.assert_allclose(prob.sum(axis=1),1.)
    assert np.all(prob>0)
