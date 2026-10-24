import numpy as np
import pandas as pd


def test_context_expansion_uses_only_original_earlier_dates_and_preserves_target_mass():
    from models.monitoring.multihorizon import context_schedule,context_weights
    o=pd.DataFrame(dict(global_target_index=[1,2,3,4],field_id=['a']*4,
        date=['2020-01-02','2020-01-05','2020-01-08','2020-01-10']))
    schedule=context_schedule(o,max_extra=3)
    extra=schedule.loc[~schedule.original]
    assert len(extra)==3
    assert (pd.to_datetime(extra.cutoff)<pd.to_datetime(extra.target_date)).all()
    # Last target has original latest-context plus alternatives ending Jan2 and Jan5.
    last=extra.loc[extra.original_position.eq(3)]
    assert last.cutoff.tolist()==['2020-01-02','2020-01-05']
    weight=context_weights(schedule,np.array([2.,3.,5.,7.]))
    np.testing.assert_allclose(np.bincount(schedule.original_position,weights=weight),[2.,3.,5.,7.])


def test_expanded_history_and_thermal_intervals_exclude_target_day_information():
    from models.monitoring.multihorizon import build_contexts
    from models.monitoring.history import build_history
    obs=pd.DataFrame(dict(global_target_index=[1,2,3,4],field_id=['a']*4,site_id=['s']*4,
        source=['BASF']*4,leaf_index=[0]*4,endpoint_series=['code0']*4,
        date=['2020-01-02','2020-01-05','2020-01-08','2020-01-10'],observed_percent=[20.,40.,60.,80.]))
    fields=pd.DataFrame([dict(field_id='a',site_id='s',sowing_date='2020-01-01')])
    weather=pd.DataFrame(dict(location_id=['s']*10,date=pd.date_range('2020-01-01',periods=10),
        tmean_c=[18.]*10,tmax_c=[20.]*10,rh_mean_pct=[90.]*10,rain_mm=[1.]*10))
    arrays=dict(process=np.zeros((4,37)),history_events=np.zeros((4,68)))
    h,_=build_history(obs,obs.drop(columns='observed_percent'))
    original=build_contexts(obs,arrays,h,fields,weather)
    target=original['schedule'].original_position.eq(2).to_numpy()
    extra=target&~original['schedule'].original.to_numpy()
    np.testing.assert_allclose(original['anchor'][extra],[.2])
    np.testing.assert_allclose(original['x'][extra,37+61],[5.])
    altered=obs.copy();altered.loc[2,'observed_percent']=99
    changed=weather.copy();changed.loc[changed.date.ge('2020-01-08'),'rain_mm']=99
    hh,_=build_history(altered,altered.drop(columns='observed_percent'))
    after=build_contexts(altered,arrays,hh,fields,changed)
    np.testing.assert_array_equal(original['x'][target],after['x'][target])
