import pandas as pd


def test_goal_requires_both_designs_and_severe_improvement():
    from models.monitoring.innovation_common import goal_decision
    rows=[]
    for design,baseline in [('temporal',20.),('spatial_50km',30.)]:
        rows.extend([dict(evaluation=design,model='reference',rmse=baseline),
            dict(evaluation=design,model='new',rmse=.89*baseline)])
    severe=pd.DataFrame([dict(evaluation=d,model=m,rmse=r,bias=b)
        for d in ['temporal','spatial_50km'] for m,r,b in [('reference',40.,-30.),('new',35.,-20.)]])
    assert goal_decision(pd.DataFrame(rows),severe,'new','reference')['achieved']
    severe.loc[(severe.evaluation=='spatial_50km')&(severe.model=='new'),'bias']=-31
    assert not goal_decision(pd.DataFrame(rows),severe,'new','reference')['achieved']
    assert not goal_decision(pd.DataFrame(rows[:2]),severe,'new','reference')['achieved']
