"""Continental bookkeeping checks independent of biological equations."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper_study.nature_food_revision_20261007.continental_replay.driver import merge_registry,alignment_arrays,simulate_all_cells


def cells():
    return pd.DataFrame({'cell_id':['a','b','c'],'latitude':[50.,51.,52.],'longitude':[1.,2.,3.],
        'harvested_total_ha':[10.,20.,30.]})


def calendars():
    return pd.DataFrame({'cell_id':['c','a','b'],'planting_doy':[280.,270.,np.nan],
        'maturity_doy':[200.,190.,np.nan],'calendar_valid':[True,True,False]})


def test_calendar_join_preserves_every_grid_cell_and_registry_order():
    result=merge_registry(cells(),calendars())
    assert result.cell_id.tolist()==['a','b','c']
    assert result.planting_doy.iloc[0]==270 and result.planting_doy.iloc[2]==280
    assert result.calendar_valid.tolist()==[True,False,True]
    assert result.harvested_total_ha.sum()==60


def test_alignment_cannot_shift_cell_ids_or_accept_missing_months():
    frame=pd.DataFrame([{'cell_id':cell,'month':month,'temperature_offset':100*i+month,
        'humidity_logit_offset':0.,'precipitation_ratio':1.} for i,cell in enumerate(['c','a']) for month in range(1,13)])
    output=alignment_arrays(frame,['a','c'])
    assert output['temperature_offset'][0,0]==101
    assert output['temperature_offset'][1,11]==12
    with pytest.raises(ValueError):alignment_arrays(frame.iloc[:-1],['a','c'])
    with pytest.raises(ValueError):alignment_arrays(pd.concat([frame,frame.iloc[:1]]),['a','c'])


def test_batches_preserve_outcomes_and_missing_calendar_is_not_zero_disease():
    registry=merge_registry(cells(),calendars());dates=pd.date_range('2001-01-01',periods=2)
    weather={key:np.array([[1.,2.],[100.,100.],[3.,4.]]) for key in ['tmean_c','tmax_c','rh_mean_pct','precipitation_mm']}
    calls=[]
    def fake(dates,weather,sowing,latitude,*args):
        calls.extend(latitude.tolist())
        return pd.DataFrame({'status':['complete']*len(latitude),'valid_complete_season':[True]*len(latitude),
            'crop_response':weather['tmean_c'].sum(axis=1)}),{'mass_error':0.}
    one,_=simulate_all_cells(dates,weather,registry,{}, {},{},harvest_year=2001,chunk_size=1,simulator=fake)
    two,_=simulate_all_cells(dates,weather,registry,{}, {},{},harvest_year=2001,chunk_size=2,simulator=fake)
    pd.testing.assert_frame_equal(one,two)
    assert set(calls)=={50.,52.}
    assert one.cell_id.tolist()==['a','b','c']
    assert one.loc[1,'status']=='missing_calendar' and not one.loc[1,'valid_complete_season']
    assert np.isnan(one.loc[1,'crop_response'])
    assert one.crop_response.iloc[[0,2]].tolist()==[3.,7.]
