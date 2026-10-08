"""Leaf ranks remain distinct from BBCH node stages and true leaf counts."""
import importlib
import numpy as np
import pandas as pd
import pytest


def module():
    try:
        return importlib.import_module('model.seasonal_septoria.leaf_phenology')
    except ModuleNotFoundError:
        pytest.fail('The independent leaf-phenology extension is not implemented')


def thresholds():
    return {10:100.,31:300.,32:350.,33:400.,37:600.,39:700.,51:900.,65:1100.,85:1500.}


def test_flag_visibility_and_full_unfolding_are_separate_from_active_weather_day():
    m=module();a=np.array([[500.,600.,650.,700.,750.]])
    result=m.leaf_host(a,np.full_like(a,18.),thresholds(),rank_spacing_units=120.,forcing_mask=np.ones_like(a,bool))
    assert result.visible[0,:,0].tolist()==[False,True,True,True,True]
    assert result.fully_unfolded[0,:,0].tolist()==[False,False,False,True,True]
    assert result.active[0,:,0].tolist()==[False,False,True,True,True]
    assert result.area[0,:,0].tolist()==[0.,0.,.5,1.,1.]
    assert result.stage_day_index[37].tolist()==[1]
    assert result.stage_day_index[39].tolist()==[3]
    assert result.visible_final_ranks_at_stage[37][0,0]
    assert not result.unfolded_final_ranks_at_stage[37][0,0]
    assert result.unfolded_final_ranks_at_stage[39][0,0]


def test_node_stage_is_not_assigned_the_same_number_of_leaves():
    m=module();a=np.array([[300.,350.,400.,450.]])
    result=m.leaf_host(a,np.full_like(a,18.),thresholds(),rank_spacing_units=120.,forcing_mask=np.ones_like(a,bool))
    # BBCH32 means a node stage; four modeled final ranks are visible here.
    index=result.stage_day_index[32][0]
    assert result.modeled_visible_final_leaf_count[0,index]==4
    assert result.active.shape==(1,4,8)
    assert result.visible_thresholds.tolist()==[600.,480.,360.,240.,120.,100.,100.]


def test_padded_development_and_temperature_do_not_create_leaves_or_stage_events():
    m=module();a=np.array([[200.,350.,1200.,1600.]])
    mask=np.array([[True,True,False,False]])
    result=m.leaf_host(a,np.array([[18.,18.,40.,40.]]),thresholds(),rank_spacing_units=120.,forcing_mask=mask)
    assert result.stage_day_index[37].tolist()==[-1]
    assert not result.active[:,2:].any()
    assert not result.area[:,2:].any()
    assert not result.renewal[:,2:].any()
    assert not result.visible[:,2:].any()


def test_future_accumulation_cannot_rewrite_previous_leaf_states():
    m=module();a=np.arange(100.,1100.,100.)[None,:];t=np.full_like(a,18.);mask=np.ones_like(a,bool)
    first=m.leaf_host(a,t,thresholds(),rank_spacing_units=120.,forcing_mask=mask)
    a[:,6:]+=1000.
    second=m.leaf_host(a,t,thresholds(),rank_spacing_units=120.,forcing_mask=mask)
    np.testing.assert_array_equal(first.active[:,:6],second.active[:,:6])
    np.testing.assert_array_equal(first.area[:,:6],second.area[:,:6])


def test_invalid_order_spacing_and_discontinuous_mask_are_rejected():
    m=module();a=np.array([[100.,200.,300.]])
    bad=thresholds();bad[39]=bad[37]
    with pytest.raises(ValueError):m.leaf_host(a,a,bad,rank_spacing_units=120.,forcing_mask=np.ones_like(a,bool))
    with pytest.raises(ValueError):m.leaf_host(a,a,thresholds(),rank_spacing_units=0.,forcing_mask=np.ones_like(a,bool))
    with pytest.raises(ValueError):m.leaf_host(a,a,thresholds(),rank_spacing_units=120.,forcing_mask=np.array([[True,False,True]]))


def runner():
    try:
        return importlib.import_module('analysis.paper_study.overwinter_leaf_model_20261006.phenology.run')
    except ModuleNotFoundError:
        pytest.fail('The independent ordered stage-threshold calibration is missing')


def test_ordered_fit_uses_stage_bounds_and_never_reused_or_disease_labels():
    r=runner()
    meta=pd.DataFrame([dict(field_id='a',field_index=0,sowing_date='2018-01-01',forcing_days=4,last_forcing_date='2018-01-04')])
    constraints=pd.DataFrame([dict(field_id='a',source='BASF',season_year=2018,event=stage,
        eligible_constraint=True,genuinely_bracketed=True,lower_exclusive=pd.Timestamp(lower),upper_inclusive=pd.Timestamp(upper))
        for stage,lower,upper in [(32,'2018-01-01','2018-01-02'),(33,'2018-01-02','2018-01-03'),
            (37,'2018-01-02','2018-01-03'),(39,'2018-01-03','2018-01-04')]])
    selected,curve=r.fit_ordered_thresholds(constraints,np.array([[110.,130.,160.,190.]]),meta,
        {31:100.,51:200.},grid=np.array([115.,125.,135.,145.,155.,165.,175.,185.]))
    assert selected['minimum_loss_days']==0.
    values=[selected['thresholds'][str(stage)] for stage in (32,33,37,39)]
    assert values==sorted(values) and len(set(values))==4
    assert 110<values[0]<=130 and 130<values[1]<values[2]<=160 and 160<values[3]<=190
    constraints['value']=42.
    with pytest.raises(ValueError):r.fit_ordered_thresholds(constraints,np.array([[110.,130.,160.,190.]]),meta,{31:100.,51:200.},grid=np.array([115.,125.,135.,145.,155.,165.,175.,185.]))


def test_incremental_leaf_object_matches_batch_and_retains_snapshot_states():
    m=module();a=np.arange(100.,1100.,100.)[None,:];t=np.full_like(a,18.);mask=np.ones_like(a,bool)
    batch=m.leaf_host(a,t,thresholds(),rank_spacing_units=120.,forcing_mask=mask)
    canopy=m.LeafCanopyModel(thresholds(),rank_spacing_units=120.)
    states=[canopy.step(a[0,day],t[0,day],mask[0,day]) for day in range(a.shape[1])]
    np.testing.assert_array_equal(np.stack([s.active for s in states])[None,:],batch.active)
    np.testing.assert_allclose(np.stack([s.area for s in states])[None,:],batch.area)
    np.testing.assert_allclose(np.stack([s.renewal for s in states])[None,:],batch.renewal)
    assert states[2].stage_progress[32]==0.
    assert states[3].stage_progress[32]==1.
    assert states[2].predicted_stage_code==31
    assert states[5].newly_crossed_stage[37]
    assert states[0].area.max()==0.  # stored snapshot is unchanged by later steps


def test_leaf_rates_are_pure_and_checkpoint_replay_preserves_config_identity():
    m=module();canopy=m.LeafCanopyModel(thresholds(),rank_spacing_units=120.)
    before=canopy.snapshot();rates=canopy.calc_rates(500.,18.,True)
    assert canopy.snapshot()==before
    state=canopy.integrate(rates)
    with pytest.raises(ValueError):state.area[0]=.5
    with pytest.raises(ValueError):canopy.integrate(rates)
    checkpoint=canopy.snapshot();expected=canopy.step(650.,18.,True)
    replay=m.LeafCanopyModel(thresholds(),rank_spacing_units=120.);replay.restore(checkpoint)
    actual=replay.step(650.,18.,True)
    np.testing.assert_array_equal(actual.active,expected.active)
    np.testing.assert_allclose(actual.renewal,expected.renewal)
    different=m.LeafCanopyModel(thresholds(),rank_spacing_units=80.)
    with pytest.raises(ValueError):different.restore(checkpoint)


def test_leaf_proposals_belong_to_one_instance_and_expire_after_restore():
    m=module();first=m.LeafCanopyModel(thresholds(),rank_spacing_units=120.)
    other=m.LeafCanopyModel(thresholds(),rank_spacing_units=120.)
    foreign=first.calc_rates(100.,18.,True)
    with pytest.raises(ValueError):other.integrate(foreign)
    before=first.snapshot();first.restore(before)
    with pytest.raises(ValueError):first.integrate(foreign)
    # A newly calculated proposal against the restored state still commits.
    state=first.integrate(first.calc_rates(100.,18.,True))
    assert state.predicted_stage_code==10
