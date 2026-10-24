import numpy as np
import pandas as pd


def test_stage_brier_normalizes_station_mass_with_read_only_pandas_arrays(monkeypatch):
    from models.monitoring.canopy_stage_transfer import _score
    queries=pd.DataFrame(dict(station_id=[1,1,2],station_cycle_date_weight=[.5,.5,1.]))
    original=queries.copy(deep=True)
    original_to_numpy=pd.Series.to_numpy
    def read_only_column(self,*args,**kwargs):
        value=original_to_numpy(self,*args,**kwargs).copy()
        value.flags.writeable=False
        return value
    monkeypatch.setattr(pd.Series,'to_numpy',read_only_column)
    y=np.zeros((3,3));pred=np.array([[1.,1.,1.],[0.,0.,0.],[0.,0.,0.]])
    score=_score(y,pred,queries,np.ones(3,bool))
    assert score['brier']==.25 and score['stations']==2
    subset=_score(y,pred,queries,np.array([True,False,True]))
    assert subset['brier']==.5 and subset['rows']==2
    pd.testing.assert_frame_equal(queries,original)
