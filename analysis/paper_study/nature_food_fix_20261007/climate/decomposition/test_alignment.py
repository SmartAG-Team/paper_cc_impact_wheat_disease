import numpy as np
import pandas as pd
from analysis.paper_study.nature_food_fix_20261007.climate.decomposition.run import align_crop_day,FIELDS


def test_elapsed_day_alignment_retains_real_leap_day_without_repeating_weather():
    source={'dates':pd.date_range('2020-02-27','2020-03-03'),'sowing':np.array(['2020-02-27'],dtype='datetime64[D]'),
        'weather':{k:np.arange(6,dtype=float)[None,:] for k in FIELDS}}
    host={'dates':pd.date_range('2100-02-27','2100-03-03'),'sowing':np.array(['2100-02-27'],dtype='datetime64[D]')}
    aligned,details=align_crop_day(source,host,np.ones((1,5),bool))
    np.testing.assert_array_equal(aligned[FIELDS[0]],np.arange(5)[None,:])
    assert details['source_forcing_days']==6 and details['target_forcing_days']==5


def test_active_forcing_cannot_be_padded_or_fabricated():
    source={'dates':pd.date_range('2020-01-01',periods=2),'sowing':np.array(['2020-01-01'],dtype='datetime64[D]'),
        'weather':{k:np.zeros((1,2)) for k in FIELDS}}
    host={'dates':pd.date_range('2100-01-01',periods=3),'sowing':np.array(['2100-01-01'],dtype='datetime64[D]')}
    try:align_crop_day(source,host,np.ones((1,3),bool))
    except ValueError:return
    raise AssertionError('Unavailable active forcing must fail.')
