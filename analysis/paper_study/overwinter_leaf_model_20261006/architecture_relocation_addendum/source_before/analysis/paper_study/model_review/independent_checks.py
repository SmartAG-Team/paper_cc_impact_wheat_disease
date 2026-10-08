"""Read-only numerical, causal-input and partition checks of the seasonal predictor."""
from pathlib import Path
import os
import sys
import json
import hashlib
from dataclasses import replace
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
os.environ['NUMBA_CACHE_DIR']=str(HERE/'numba_cache')
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from scipy.linalg import expm
from model.seasonal_septoria.core import Parameters, simulate_season
from model.seasonal_septoria.field_data import FieldData, prepare_fields, _leaf_index
from model.seasonal_septoria.host import cohort_inputs
from model.seasonal_septoria.calibrate import fit, predict
from model.seasonal_septoria.endpoints import onset_distance


def hash_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    results=[]
    def check(name,condition,details=None):
        if not bool(condition):raise AssertionError(name)
        results.append(dict(check=name,status='passed',details=details))
    rng=np.random.default_rng(20261005)
    episodes,days,leaves=4,180,8
    t=rng.uniform(-8,42,(episodes,days));h=rng.uniform(0,100,t.shape);r=rng.exponential(3,t.shape)
    a=rng.random((episodes,days,leaves))>.15;renewal=rng.uniform(0,.5,a.shape)*a
    stress=simulate_season(t,h,r,a,renewal,Parameters(alpha=5,beta=100,latent_days=.05,nonsporulating_days=.1,infectious_days=.1))
    mass=float(abs(stress.state.sum(axis=-1)-1).max());minimum=float(stress.state.min())
    check('random_stress_mass_and_positivity',mass<1e-12 and minimum>-1e-13 and stress.state.max()<1+1e-12,dict(maximum_mass_error=mass,minimum_state=minimum))
    check('random_stress_observation_order',np.all(stress.pycnidia<=stress.damage+1e-13) and np.all(stress.infectious<=stress.pycnidia+1e-13))
    check('origin_compartments_reconcile',np.max(abs(stress.origin_total.sum(axis=-1)+stress.state[...,0]-1))<1e-12)
    for origin,flow in [(0,stress.primary_flow),(1,stress.secondary_flow)]:
        retained=stress.origin_total[:,:-1,:,origin]*(1-renewal)*a
        check(f'route_bookkeeping_with_daily_turnover_origin{origin}',np.max(abs(stress.origin_total[:,1:,:,origin]-retained-flow))<1e-12)
    constant_t=np.full((1,100),18.);constant_h=np.full_like(constant_t,95.);constant_r=np.full_like(constant_t,4.)
    active=np.ones((1,100,2),bool);zero=np.zeros_like(active,float)
    no_external=simulate_season(constant_t,constant_h,constant_r,active,zero,Parameters(alpha=0,beta=100))
    check('zero_disease_initialization_and_no_spontaneous_secondary',np.count_nonzero(no_external.state[...,1:])==0)
    external=simulate_season(constant_t,constant_h,constant_r,active,zero,Parameters(alpha=.02,beta=0))
    check('primary_only_has_zero_secondary_route',np.count_nonzero(external.secondary_flow)==0 and np.count_nonzero(external.origin_total[...,1])==0)
    dry=simulate_season(constant_t,constant_h,np.zeros_like(constant_r),active,zero,Parameters(alpha=.02,beta=100))
    check('no_rain_has_zero_splash_secondary',np.count_nonzero(dry.secondary_flow)==0)
    one=simulate_season(constant_t,constant_h,constant_r,active[:,:,:1],zero[:,:,:1],Parameters(alpha=.02,beta=2))
    second_inactive=active.copy();second_inactive[:,:,1]=False
    two=simulate_season(constant_t,constant_h,constant_r,second_inactive,zero,Parameters(alpha=.02,beta=2),leaf_ranks=[0,7])
    check('inactive_rank_cannot_import_or_export',np.array_equal(one.state[:,:,0],two.state[:,:,0]) and np.count_nonzero(two.state[:,:,1,1:])==0)
    future_t=constant_t.copy();future_t[:,50:]=40
    future_r=constant_r.copy();future_r[:,50:]=100
    future=simulate_season(future_t,constant_h,future_r,active,zero,Parameters(alpha=.02,beta=0))
    check('future_weather_cannot_change_prior_states',np.array_equal(external.state[:,:51],future.state[:,:51]))
    cold=simulate_season(np.zeros_like(constant_t),constant_h,constant_r,active,zero,Parameters(alpha=.02,beta=2))
    check('nonpositive_temperature_stops_latent_progress',np.count_nonzero(cold.damage)==0 and cold.origin_total[...,0].sum()>0)
    reset=active.copy();reset[:,40:60,0]=False
    reset_result=simulate_season(constant_t,constant_h,constant_r,reset,zero,Parameters(alpha=.02,beta=0))
    check('inactive_reset_removes_old_state',np.count_nonzero(reset_result.state[:,41:61,0,1:])==0)
    check('reactivated_cohort_starts_without_old_disease',np.count_nonzero(reset_result.state[:,60,0,1:])==0)
    # Exact continuous-time primary-only reference: matrix exponential of its linear generator.
    archive=ROOT/'analysis/paper_study/seasonal_calibration_v1'
    frozen=json.loads((archive/'frozen_main_fit.json').read_text())['parameters']
    p=replace(Parameters(**frozen),beta=0)
    establish=np.exp(-.5*((18-p.temperature_optimum)/p.temperature_width)**2)*(1-(1-1/(1+np.exp(-(95-p.humidity_midpoint)/p.humidity_scale)))*np.exp(-4/p.rain_scale))
    n=1+p.latent_stages+3;Q=np.zeros((n,n))
    rates=[p.alpha*establish]+[p.latent_stages/p.latent_days]*p.latent_stages+[1/p.nonsporulating_days,1/p.infectious_days]
    for i,rate in enumerate(rates):Q[i,i]-=rate;Q[i+1,i]+=rate
    initial=np.zeros(n);initial[0]=1
    reference=np.stack([expm(Q*d)@initial for d in range(101)])
    truth=reference[:,1+p.latent_stages:].sum(axis=-1)
    convergence=[]
    for dt in [.5,.25,.125,.0625]:
        trajectory=simulate_season(constant_t,constant_h,constant_r,active[:,:,:1],zero[:,:,:1],p,time_step=dt)
        difference=trajectory.damage[0,:,0]-truth
        convergence.append(dict(time_step_days=dt,maximum_damage_error_percentage_points=float(abs(difference).max()*100),
            terminal_damage_error_percentage_points=float(difference[-1]*100)))
    check('continuous_time_reference_convergence',all(convergence[i+1]['maximum_damage_error_percentage_points']<convergence[i]['maximum_damage_error_percentage_points'] for i in range(3)),convergence)
    (HERE/'time_step_convergence.csv').write_text(pd.DataFrame(convergence).to_csv(index=False))
    # Fit synthetic calibration fields only; a third field has withheld values and forcing.
    st=np.full((3,75),18.);sh=np.full_like(st,90.);sr=np.tile(np.array([.5,4.,2.])[:,None],(1,75))
    sa=np.ones((3,75,2),bool);sz=np.zeros(sa.shape)
    target=pd.DataFrame([dict(field_index=f,day_index=d,leaf_index=l,
        observation_operator='pycnidia' if l else 'damage_proxy',weight=.25,value=0.) for f in range(3) for d in [45,75] for l in [0,1]])
    synthetic=FieldData(st,sh,sr,sa,sz,pd.DataFrame(),target,pd.DataFrame())
    predicted,_=predict(synthetic,Parameters(alpha=.003,beta=1.5,latent_days=20.));synthetic.targets['value']=predicted
    selected=np.flatnonzero(synthetic.targets.field_index.lt(2))
    original_fit=fit(synthetic,selected,latent_days=20.)
    synthetic.targets.loc[synthetic.targets.field_index.eq(2),'value']=9999.
    synthetic.temperature[2]=9999.;synthetic.humidity[2]=-9999.;synthetic.rain[2]=-9999.
    changed_fit=fit(synthetic,selected,latent_days=20.)
    check('withheld_outcomes_and_forcing_do_not_affect_fit',original_fit['parameters']==changed_fit['parameters'] and original_fit['training_rmse']==changed_fit['training_rmse'])
    check('synthetic_known_parameter_recovery',abs(original_fit['parameters']['alpha']-.003)<1e-6 and abs(original_fit['parameters']['beta']-1.5)<.001,dict(parameters=original_fit['parameters'],training_rmse=original_fit['training_rmse']))
    try:fit(synthetic,selected[:-1]);partial_rejected=False
    except ValueError:partial_rejected=True
    check('partial_field_season_calibration_rejected',partial_rejected)
    # Archived membership and inner selection audit, without any external outcomes.
    targets=pd.read_csv(archive/'target_membership.csv');members=pd.read_csv(archive/'field_season_membership.csv')
    check('archived_complete_field_season_partition',targets.groupby('field_id').partition.nunique().max()==1 and members.field_id.nunique()==len(members))
    check('archived_chronology',targets.loc[targets.partition.eq('calibration'),'season_year'].max()<2019 and targets.loc[targets.partition.eq('validation'),'season_year'].eq(2019).all())
    check('all_archived_targets_untreated',targets.treatment.eq('Untreated').all())
    check('all_current_targets_preserve_intended_operator',targets.observation_operator.eq('damage_proxy').all() and targets.metric.eq('infection_percent_unspecified_basis').all())
    config=json.loads((archive/'configuration_before_fitting.json').read_text())
    check('membership_matches_prefit_hash',hash_file(archive/'target_membership.csv')==config['membership_sha256'])
    check('source_assessments_match_prefit_hash',hash_file(archive/'basf_source_assessments_snapshot.csv')==config['source_snapshot_sha256'])
    check('weather_matches_prefit_hash',hash_file(ROOT/'data/paper_study/field_weather/daily_weather.parquet')==config['weather_sha256'])
    training_ids=set(np.flatnonzero(targets.partition.eq('calibration')));selection=[]
    for duration in config['latent_days_candidates']:
        numerator=denominator=0.;seen=[]
        for fold in range(3):
            fit_record=json.loads((archive/f'inner_latent{duration:g}_fold{fold}.json').read_text())
            indices=fit_record['target_indices'];heldout=pd.read_parquet(archive/f'inner_latent{duration:g}_fold{fold}_predictions.parquet')
            check(f'inner_train_is_outer_training_{duration}_{fold}',set(indices)<=training_ids)
            selected_fields=set(targets.iloc[indices].field_id)
            check(f'inner_train_complete_fields_{duration}_{fold}',set(indices)==set(np.flatnonzero(targets.field_id.isin(selected_fields))))
            check(f'inner_spatial_site_separation_{duration}_{fold}',not set(targets.iloc[indices].site_id)&set(heldout.site_id))
            check(f'inner_heldout_is_outer_training_{duration}_{fold}',heldout.partition.eq('calibration').all())
            check(f'inner_declared_site_fold_{duration}_{fold}',heldout.site_id.map(config['site_fold']).eq(fold).all())
            expected_heldout=targets.loc[targets.partition.eq('calibration') & targets.site_id.map(config['site_fold']).eq(fold)]
            key=['field_id','endpoint_series','date']
            actual_keys=heldout[key].copy();expected_keys=expected_heldout[key].copy()
            actual_keys['date']=pd.to_datetime(actual_keys.date);expected_keys['date']=pd.to_datetime(expected_keys.date)
            check(f'inner_complete_heldout_membership_{duration}_{fold}',sorted(map(tuple,actual_keys.to_numpy()))==sorted(map(tuple,expected_keys.to_numpy())))
            error=heldout.predicted_percent-heldout.value;numerator+=float(np.sum(heldout.weight*error**2));denominator+=float(heldout.weight.sum());seen.extend(heldout.index.tolist())
        selection.append(dict(latent_days=duration,independent_pooled_rmse=float(np.sqrt(numerator/denominator))))
    table=pd.read_csv(archive/'training_only_latent_selection.csv').merge(pd.DataFrame(selection),on='latent_days',validate='one_to_one')
    table['absolute_difference']=abs(table.pooled_rmse-table.independent_pooled_rmse)
    check('all_archived_inner_candidate_scores_recompute',table.absolute_difference.max()<1e-10)
    check('frozen_latent_winner_matches_training_selection',float(table.loc[table.pooled_rmse.idxmin(),'latent_days'])==frozen['latent_days'])
    table.to_csv(HERE/'recomputed_inner_selection.csv',index=False)
    frozen_hashes=json.loads((archive/'frozen_parameters_before_validation.json').read_text())
    check('frozen_fit_hashes_preserved',all(hash_file(archive/name)==value for name,value in frozen_hashes.items()))
    # Equal-coordinate-year weights are a documented review finding, not equal-field-season weights.
    field_weights=targets.groupby(['partition','field_id','coordinate_year']).weight.sum().reset_index()
    field_weights.to_csv(HERE/'archived_field_weights.csv',index=False)
    check('coordinate_year_weights_sum_to_one',np.allclose(targets.groupby(['partition','coordinate_year']).weight.sum(),1))
    # Explicit synthetic demonstrations of eligibility and parser limits.
    defs={'dataset_id':['review']*2,'physical_unit':['a']*2,'site_id':['loc']*2,'season_year':[2019]*2,
        'country':['Germany']*2,'latitude':[50.]*2,'longitude':[10.]*2,'organ':['F1','F2'],
        'date':['2019-05-30','2019-05-31'],'value':[1.,2.],'metric':['lesion_count']*2,'unit':['count']*2,'endpoint_series':['a1','a2']}
    assessments=pd.DataFrame(defs);calendar=pd.DataFrame(dict(dataset_id=['review'],point_id=['loc'],crop_season=['winter_wheat'],water_system=['rainfed'],calendar_valid=[True],planting_doy=[1.],maturity_doy=[200.]))
    weather=pd.DataFrame(dict(location_id=['loc']*151,date=pd.date_range('2019-01-01',periods=151),tmean_c=[18.]*151,tmax_c=[23.]*151,rh_mean_pct=[90.]*151,precipitation_mm=[2.]*151))
    count_input=prepare_fields(assessments,calendar,weather)
    eligibility_issue=dict(count_metric_accepted_as_percent_proxy=len(count_input.targets)==2,parsed_F10_as_leaf_index=_leaf_index('F10'),parsed_LEAF14_as_leaf_index=_leaf_index('LEAF, 14'))
    # Observable dates and valid outcomes do not change host or weather inputs.
    assessments.metric='infection_percent_unspecified_basis';assessments.unit='percent'
    clean=prepare_fields(assessments,calendar,weather);assessments.value=[99.,0.];changed=prepare_fields(assessments,calendar,weather)
    check('disease_values_do_not_change_host_or_weather_inputs',all(np.array_equal(getattr(clean,k),getattr(changed,k)) for k in ['temperature','humidity','rain','host_active','host_renewal']))
    check('daily_onset_negative_bound_is_exclusive',onset_distance('2019-05-10','2019-05-10','2019-05-20')['delta_days']==-1)
    check('missing_positive_onset_prediction_remains_incompatible',not onset_distance(None,None,'2019-05-20')['compatible'])
    snapshots=json.loads((HERE/'review_source_hashes.json').read_text())
    changed_sources={name:dict(before=value,after=hash_file(ROOT/name)) for name,value in snapshots['source_hashes'].items() if hash_file(ROOT/name)!=value}
    changed_archives={name:dict(before=value,after=hash_file(ROOT/name)) for name,value in snapshots['prior_archive_hashes'].items() if hash_file(ROOT/name)!=value}
    receipt=dict(status='independent_checks_passed_with_reported_protocol_limits',completed_utc=datetime.now(timezone.utc).isoformat(),
        checks_passed=len(results),checks=results,time_step_convergence=convergence,eligibility_demonstrations=eligibility_issue,
        source_changes_since_review_snapshot=changed_sources,prior_archive_changes=changed_archives,
        external_Corteva_outcomes_read=False,external_Corteva_evaluation_performed=False,
        numerical_reference='Independent exact continuous-time matrix exponential for primary-only constant forcing',
        source_snapshot_timing='Retrospective review snapshot, not original execution evidence')
    (HERE/'independent_check_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],checks=len(results),source_changes=list(changed_sources),archive_changes=list(changed_archives),time_step_convergence=convergence,eligibility_demonstrations=eligibility_issue),indent=2))


if __name__=='__main__':run()
