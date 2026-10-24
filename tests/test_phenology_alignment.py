import numpy as np
import pandas as pd

from models.monitoring.phenology_alignment import (
    rank_probability, interval_update, align_field, coupled_profiles,
)


def toy():
    starts=pd.to_datetime(['2014-09-01','2014-09-21'])
    # Two joint profiles, with three clock definitions per profile.
    thresholds=np.array([[[40,100,200,400]]*3,[[80,200,400,800]]*3],float)
    bank=dict(thresholds=thresholds,weight=np.array([.5,.5]))
    clocks={str(s.date()):np.arange(151)[:,None]*np.ones((1,3))*5 for s in starts}
    queries=pd.to_datetime(['2014-10-20','2014-11-01','2014-12-01'])
    calendar=pd.DataFrame(dict(date=['2014-10-21','2014-11-15'],stage_from=[51,87],stage_to=[51,87]))
    return starts,bank,clocks,queries,calendar


def test_rank_distribution_and_interval_likelihood_are_normalized():
    p=rank_probability(np.array([100.,300.]),np.array([[10,50,200,400],[20,100,200,500]]),60.)
    np.testing.assert_allclose(p.sum(1),1)
    assert np.all(p>=0)
    prior=np.array([.5,.5])
    updated=interval_update(prior,p,51,86,.05)
    expected=prior*(.95*p[:,3]+.05/5)
    np.testing.assert_allclose(updated,expected/expected.sum())
    # A fully uninformative stage interval must preserve the prior exactly.
    np.testing.assert_array_equal(interval_update(prior,p,0,99,.05),prior)


def test_prior_stage_alignment_changes_phase_without_query_day_or_future_labels():
    starts,bank,clocks,queries,calendar=toy()
    result=align_field(starts,clocks,queries,calendar,bank)
    np.testing.assert_array_equal(result['filtered'][0],result['prior'][0])
    assert result['stage_available'].tolist()==[False,True,True]
    assert np.max(abs(result['filtered'][1]-result['prior'][1]))>.01
    changed=calendar.copy();changed.loc[1,['stage_from','stage_to']]=10
    again=align_field(starts,clocks,queries,changed,bank)
    np.testing.assert_array_equal(result['filtered'][:2],again['filtered'][:2])
    same=pd.concat([calendar,pd.DataFrame(dict(date=['2014-11-01'],stage_from=[99],stage_to=[99]))],ignore_index=True)
    again=align_field(starts,clocks,queries[:2],same,bank)
    np.testing.assert_array_equal(result['filtered'][:2],again['filtered'])
    assert result['posterior_ess'].min()>0
    assert np.all(np.diff(result['filtered_probability'],axis=1)<=1e-12)


def test_alignment_is_prefix_stable_and_order_invariant():
    starts,bank,clocks,queries,calendar=toy()
    full=align_field(starts,clocks,queries,calendar,bank)
    prefix=align_field(starts,clocks,queries[:2],calendar,bank)
    np.testing.assert_array_equal(full['filtered'][:2],prefix['filtered'])
    reordered=align_field(starts,clocks,queries[[2,0,1]],calendar.iloc[::-1],bank)
    np.testing.assert_array_equal(full['filtered'][[2,0,1]],reordered['filtered'])
    changed={k:v.copy() for k,v in clocks.items()}
    for start in starts:
        # Query-day and later weather first alter the following day's cumulative clock.
        q=(queries[1]-start).days
        changed[str(start.date())][q+1:]+=999
    again=align_field(starts,changed,queries[:2],calendar,bank)
    np.testing.assert_array_equal(full['filtered'][:2],again['filtered'])


def test_joint_profiles_use_before_event_weather_and_equal_station_mass():
    records=[]
    for i,station in enumerate([10,10,10,20]):
        records.append(dict(PEP_ID=station,SOWING_DATE=f'{2000+i}-09-01',END_DATE=f'{2000+i}-10-01',
            true_dates=[f'{2000+i}-09-{d:02d}' for d in [2,3,4,5]],eligible=True,
            observation_order_conflict=False,row_start=0,row_stop=10,lat=50))
    parameters=dict(photoperiod_onset_gdd=1000,photoperiod_stop_gdd=2000,
                    vernalization_onset_tpp=1000,vernalization_stop_tpp=2000)
    bank=coupled_profiles(records,np.tile([5,8,2,0],(10,1)),parameters)
    np.testing.assert_array_equal(bank['thresholds'][0],np.tile([5,10,15,20],(3,1)))
    np.testing.assert_allclose(bank['weight'],[1/6,1/6,1/6,.5])
