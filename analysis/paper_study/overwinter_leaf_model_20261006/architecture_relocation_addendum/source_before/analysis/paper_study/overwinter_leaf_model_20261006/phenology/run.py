"""Stage-only ordered leaf-development extension of the frozen donor T-P-V."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

from analysis.paper_study.infection_priority_20261006.anthesis_clock import run as clock
from model.seasonal_septoria.leaf_phenology import leaf_host

ROOT=clock.ROOT
HERE=Path(__file__).resolve().parent
EVENTS=(32,33,37,39)
SPACINGS=(80.,120.,160.)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def stage_constraints(rows,source,strict=None):
    normalized,rejected,_=clock.build_stage_constraints(rows,source,strict)
    bounds=clock.intervals.make_constraints(normalized,events=[31,*EVENTS])
    bad=set(normalized.loc[~normalized.stage_order_valid,'field_id'])
    invalid=bounds.field_id.isin(bad)
    bounds.loc[invalid,['eligible_constraint','genuinely_bracketed']]=False
    bounds.loc[invalid,'reason']='temporally_inconsistent_stage_order'
    bounds['bracket_width_days']=(bounds.upper_inclusive-bounds.lower_exclusive).dt.days.where(bounds.genuinely_bracketed)
    return normalized,rejected,bounds


def _forward_backward(cost):
    k,n=cost.shape;forward=np.full_like(cost,np.inf);backward=np.full_like(cost,np.inf)
    forward[0]=cost[0];backward[-1]=cost[-1]
    for event in range(1,k):
        forward[event,1:]=cost[event,1:]+np.minimum.accumulate(forward[event-1])[:-1]
    for event in range(k-2,-1,-1):
        suffix=np.minimum.accumulate(backward[event+1,::-1])[::-1]
        backward[event,:-1]=cost[event,:-1]+suffix[1:]
    return forward,backward


def fit_ordered_thresholds(constraints,accumulation,metadata,fixed,*,grid=None):
    """Exact finite ordered-grid minimum of equal-field stage-interval distance."""
    if ('value' in constraints or not constraints.source.eq('BASF').all()
            or not constraints.season_year.isin([2017,2018]).all()):
        raise ValueError('Only disease-redacted original BASF 2017-2018 stage constraints may fit thresholds.')
    c31,c51=float(fixed[31]),float(fixed[51])
    grid=np.linspace(c31,c51,401)[1:-1] if grid is None else np.asarray(grid,float)
    if (not 0<c31<c51 or grid.ndim!=1 or len(grid)<len(EVENTS)
            or not np.isfinite(grid).all() or np.any(np.diff(grid)<=0)
            or np.any((grid<=c31)|(grid>=c51))):
        raise ValueError('A unique increasing development-unit grid between fixed C31 and C51 is required.')
    eligible=constraints.loc[constraints.eligible_constraint&constraints.event.isin(EVENTS)].copy()
    if eligible.empty or set(eligible.event)!=set(EVENTS):
        raise ValueError('Every fitted stage needs at least one valid calibration event bound.')
    weights=1/eligible.field_id.nunique()/eligible.groupby('field_id').event.transform('size')
    meta=metadata.set_index('field_id')
    cost=np.zeros((len(EVENTS),len(grid)))
    for index,row in eligible.iterrows():
        member=meta.loc[row.field_id];n=int(member.forcing_days)
        a=np.asarray(accumulation[int(member.field_index),:n],float)
        if not np.isfinite(a).all() or np.any(np.diff(a)<-1e-10):
            raise ValueError('Valid nondecreasing donor development is required.')
        position=np.searchsorted(a,grid,side='left');present=position<n
        predicted=np.datetime64(member.sowing_date,'D')+position.astype('timedelta64[D]')
        loss=np.zeros(len(grid))
        if pd.notna(row.lower_exclusive):
            lower=np.datetime64(row.lower_exclusive,'D')+np.timedelta64(1,'D')
            loss[present]=np.maximum((lower-predicted[present]).astype('timedelta64[D]').astype(float),0.)
        if pd.notna(row.upper_inclusive):
            upper=np.datetime64(row.upper_inclusive,'D')
            loss[present]+=np.maximum((predicted[present]-upper).astype('timedelta64[D]').astype(float),0.)
            loss[~present]=max(0.,float((np.datetime64(member.last_forcing_date,'D')+np.timedelta64(1,'D')-upper).astype('timedelta64[D]').astype(float)))
        cost[EVENTS.index(int(row.event))]+=float(weights.loc[index])*loss
    forward,backward=_forward_backward(cost)
    minimum=float(np.min(forward[-1]));profile=forward+backward-cost
    centers=[]
    for event in range(len(EVENTS)):
        minima=np.flatnonzero(np.isclose(profile[event],minimum,atol=1e-12,rtol=0))
        centers.append(float(np.median(grid[minima])))
    # Primary costs remain exact; secondary distance chooses a center of the
    # joint minimizing plateau without using either reused evaluation set.
    secondary=np.full_like(cost,np.inf);parent=np.full(cost.shape,-1,int)
    secondary[0]=((grid-centers[0])/(c51-c31))**2
    for event in range(1,len(EVENTS)):
        for point in range(event,len(grid)):
            best=float(np.min(forward[event-1,:point]))
            tied=np.flatnonzero(np.isclose(forward[event-1,:point],best,atol=1e-12,rtol=0))
            chosen=int(tied[np.argmin(secondary[event-1,tied])])
            parent[event,point]=chosen
            secondary[event,point]=secondary[event-1,chosen]+((grid[point]-centers[event])/(c51-c31))**2
    final=np.flatnonzero(np.isclose(forward[-1],minimum,atol=1e-12,rtol=0))
    selected=[int(final[np.argmin(secondary[-1,final])])]
    for event in range(len(EVENTS)-1,0,-1):selected.append(int(parent[event,selected[-1]]))
    selected=selected[::-1]
    assert np.all(np.diff(selected)>0)
    assert abs(sum(cost[event,point] for event,point in enumerate(selected))-minimum)<1e-10
    diagnostics=[];rows=[]
    for event,stage in enumerate(EVENTS):
        part=eligible.loc[eligible.event.eq(stage)]
        bands=[]
        for tolerance in (0.,1.,2.):
            accepted=grid[profile[event]<=minimum+tolerance+1e-12]
            bands.append(dict(joint_loss_tolerance_days=tolerance,threshold_low=float(accepted.min()),
                threshold_high=float(accepted.max()),n_grid_values=len(accepted),confidence_interval=False))
        diagnostics.append(dict(event=stage,selected_threshold=float(grid[selected[event]]),
            constrained_fields=part.field_id.nunique(),genuine_two_sided_fields=int(part.genuinely_bracketed.sum()),
            left_bounds=int(part.censoring.eq('left').sum()) if 'censoring' in part else None,
            right_bounds=int(part.censoring.eq('right').sum()) if 'censoring' in part else None,
            joint_profile_loss_tolerance_sets=bands,poorly_identified=bool(int(part.genuinely_bracketed.sum())<10
                or bands[0]['threshold_high']-bands[0]['threshold_low']>.1*(c51-c31)
                or bands[1]['threshold_high']-bands[1]['threshold_low']>.2*(c51-c31))))
        rows.extend(dict(event=stage,threshold=float(value),loss_component_days=float(cost[event,point]),
            joint_profile_loss_days=float(profile[event,point]) if np.isfinite(profile[event,point]) else None)
            for point,value in enumerate(grid))
    record=dict(thresholds={str(stage):float(grid[point]) for stage,point in zip(EVENTS,selected)},
        minimum_loss_days=minimum,grid_points=len(grid),grid_low=float(grid[0]),grid_high=float(grid[-1]),
        exact_joint_ordering=True,diagnostics=diagnostics,q=1.,numeric_bbch_interpolation=False,
        calibration_rule='equal fields, equal constrained events within field; open lower and closed upper dates',
        tie_rule='minimum joint interval loss, then squared distance to joint-profile minimizing plateau centers',
        disease_labels_used=False,evaluation_stage_values_used_for_selection=False,
        leaf_rank_spacing_calibrated=False,true_final_leaf_number_estimated=False)
    return record,pd.DataFrame(rows)


def _temperature(metadata,shape,weather):
    frame=weather.copy();frame['date']=pd.to_datetime(frame.date)
    groups={site:part.set_index('date') for site,part in frame.groupby('location_id')}
    t=np.zeros(shape)
    for meta in metadata.itertuples():
        dates=pd.date_range(meta.first_forcing_date,meta.last_forcing_date)
        values=groups[meta.site_id].reindex(dates).tmean_c.to_numpy(float)
        if not np.isfinite(values).all() or len(values)!=meta.forcing_days:raise ValueError('Original forcing is incomplete.')
        t[meta.field_index,:meta.forcing_days]=values
    return t


def main():
    if (HERE/'configuration_before_fitting.json').exists():raise FileExistsError('Use a new immutable phenology archive.')
    HERE.mkdir(parents=True,exist_ok=True)
    old=clock.ARCHIVE
    paths=dict(basf=ROOT/'analysis/paper_study/seasonal_calibration_v1/basf_source_assessments_snapshot.csv',
        basf_registry=ROOT/'analysis/paper_study/seasonal_calibration_v1/field_season_membership.csv',
        corteva=ROOT/'data/paper_study/observations/corteva_external_assessments.csv',
        corteva_registry=ROOT/'analysis/paper_study/corteva_external_seasonal_v1/external_input_membership.csv',
        strict=ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv',
        weather=ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet',
        donor=ROOT/'process_model/parameters/calibration.json',anthesis=ROOT/'model/seasonal_septoria/infection_priority_model.json',
        metadata=old/'original_field_input_membership.csv',prior_events=old/'frozen_event_predictions.csv',
        calibration_acc=old/'calibration_development_inputs.npz',development_acc=old/'reused_development_2019_development_inputs.npz',
        external_acc=old/'reused_external_strict_development_inputs.npz')
    dependencies=[*paths.values(),Path(__file__),ROOT/'model/seasonal_septoria/leaf_phenology.py',
        ROOT/'tests/test_leaf_phenology.py',Path(clock.__file__),Path(clock.intervals.__file__)]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in dependencies}
    donor=json.loads(paths['donor'].read_text());fixed={int(x['BBCH']):float(x['Cumulative_t_pp_v_GDD']) for x in donor['thresholds'] if x['BBCH'] in [10,31,51,85]}
    assert fixed=={10:142.01,31:542.3,51:986.65,85:1760.39}
    fixed[65]=float(json.loads(paths['anthesis'].read_text())['anthesis_threshold'])
    contract=dict(registered_utc=datetime.now(timezone.utc).isoformat(),fixed_q=1.,fixed_donor_thresholds=fixed,
        fitted_events=list(EVENTS),threshold_grid=dict(points=399,lower_exclusive=fixed[31],upper_exclusive=fixed[51],spacing=(fixed[51]-fixed[31])/400),
        ordering='C31<C32<C33<C37<C39<C51<C65<C85',numeric_bbch_interpolation=False,
        calibrates_stage_events_not_leaf_number=True,rank_spacing_scenarios_units=list(SPACINGS),
        rank_spacing_is_effective_TPV_index_not_measured_thermal_phyllochron=True,rank_spacing_is_not_fitted=True,
        visibility='C37 for flag tip; rank-offset effective thresholds bounded below by C10',
        unfolding='C39 for flag unfolded; fixed C39-C37 within-leaf unfolding duration for all modeled ranks',
        modeled_count='F1 through F7 ranks only; true final leaf count unobserved',
        daily_disease_availability='prior-day visibility and current positive unfolding capacity',
        stage_status='end-of-day visible/unfolded final ranks at inclusive stage crossing',
        natural_senescence_estimated=False,juvenile_slot='retained normalized reservoir; 300 positive-degree-day renewal',
        stage_column_whitelist=clock.intervals.STAGE_COLUMNS,disease_values_loaded=False,
        training_fields=28,training_years=[2017,2018],reused_evaluation_fields=dict(BASF2019=45,strict_external=143),
        primary_loss='equal-field mean censored stage-event distance',genuine_interval_metrics_separate=True,
        uncertainty='joint-grid loss-tolerance profiles; not confidence intervals',
        validation_label='retrospective reused stage development evidence; no untouched test',input_and_source_sha256=hashes)
    frozen_json(HERE/'configuration_before_fitting.json',contract)
    for path in dependencies:
        if path.suffix in ('.py','.json'):
            target=HERE/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    original=pd.read_csv(paths['metadata'],dtype={'source_unit':str})
    basf_registry=pd.read_csv(paths['basf_registry'],dtype={'source_unit':str})
    external_registry=pd.read_csv(paths['corteva_registry'],dtype={'source_unit':str})
    strict=set(pd.read_csv(paths['strict'],usecols=['source_unit'],dtype={'source_unit':str}).source_unit)
    external_registry=external_registry.loc[external_registry.source_unit.isin(strict)]
    raw_basf=pd.read_csv(paths['basf'],usecols=lambda c:c in clock.intervals.STAGE_COLUMNS,dtype={'physical_unit':str})
    raw_external=pd.read_csv(paths['corteva'],usecols=lambda c:c in clock.intervals.STAGE_COLUMNS,dtype={'physical_unit':str})
    projected=clock.intervals.project_stage_rows(raw_basf,'BASF')
    external=clock.intervals.project_stage_rows(raw_external,'Corteva',strict)
    groups=[]
    for partition,rows,registry,count in [
        ('calibration',projected.loc[projected.season_year.isin([2017,2018])],basf_registry.loc[basf_registry.season_year.isin([2017,2018])],28),
        ('reused_development_2019',projected.loc[projected.season_year.eq(2019)],basf_registry.loc[basf_registry.season_year.eq(2019)],45),
        ('reused_external_strict',external,external_registry,143)]:
        meta,a=clock._clock_inputs(partition,original,registry);assert len(meta)==count
        selected,excluded=clock.intervals.restrict_membership(rows,meta)
        normalized,rejected,bounds=stage_constraints(selected,'Corteva' if partition=='reused_external_strict' else 'BASF',strict)
        for item in (normalized,rejected,bounds):assert 'value' not in item
        groups.append((partition,meta,a,normalized,rejected,bounds,excluded))
    calibration=groups[0]
    _,meta,a,_,_,bounds,_=calibration
    # Exact q1 donor reconciliation precedes the extension fit.
    q1=clock.predict_clock(a,meta,{31:fixed[31]})
    prior=pd.read_csv(paths['prior_events'],parse_dates=['predicted_date'])
    prior=prior.loc[prior.model.eq('copied_tpv_q1')&prior.partition.eq('calibration')&prior.event.eq(31)]
    paired=q1.merge(prior[['field_id','predicted_date']],on='field_id',validate='one_to_one',suffixes=('_new','_donor'))
    assert len(paired)==28
    assert ((paired.predicted_date_new==paired.predicted_date_donor)|(paired.predicted_date_new.isna()&paired.predicted_date_donor.isna())).all()
    frozen_json(HERE/'donor_stage31_q1_reconciliation.json',dict(status='passed',fields=28,retuned_q=False,donor_threshold=fixed[31],reconciled_before_new_threshold_fit=True))
    bounds.to_csv(HERE/'calibration_stage_constraints_before_fitting.csv',index=False)
    selection,curve=fit_ordered_thresholds(bounds,a,meta,fixed)
    calibrated={**fixed,**{int(k):v for k,v in selection['thresholds'].items()}}
    selection['all_stage_thresholds']={str(k):v for k,v in sorted(calibrated.items())}
    selection['source_sha256']=hashes
    frozen_json(HERE/'calibrated_stage_thresholds.json',selection)
    curve.to_csv(HERE/'ordered_threshold_joint_profiles.csv',index=False)
    frozen_hash=sha(HERE/'calibrated_stage_thresholds.json')
    print(json.dumps(dict(status='thresholds_frozen',thresholds=selection['thresholds'],loss_days=selection['minimum_loss_days'])),flush=True)
    weather=pd.read_parquet(paths['weather'])
    predictions,scores,coverage,status_rows,metadata_all,normalized_all,rejected_all,excluded_all=[],[],[],[],[],[],[],[]
    for partition,meta,a,normalized,rejected,bounds,excluded in groups:
        pred=clock.predict_clock(a,meta,calibrated);pred['partition']=partition;predictions.append(pred)
        scored=clock.score_clock(bounds,pred);scored['partition']=partition;scores.append(scored)
        metadata_all.append(meta);normalized_all.append(normalized);rejected_all.append(rejected);excluded_all.append(excluded)
        for stage in (31,*EVENTS):
            constraints=bounds.loc[bounds.event.eq(stage)]
            part=scored.loc[scored.event.eq(stage)]
            coverage.append(dict(partition=partition,event=stage,original_fields=len(meta),stage_date_fields=normalized.field_id.nunique(),
                constrained_fields=int(constraints.eligible_constraint.sum()),genuine_two_sided=int(constraints.genuinely_bracketed.sum()),
                left_censored=int((constraints.eligible_constraint&constraints.censoring.eq('left')).sum()),
                right_censored=int((constraints.eligible_constraint&constraints.censoring.eq('right')).sum()),
                unavailable_or_rejected_fields=len(meta)-int(constraints.eligible_constraint.sum())))
        temperature=_temperature(meta,a.shape,weather)
        mask=np.arange(a.shape[1])[None,:]<meta.forcing_days.to_numpy()[:,None]
        for spacing in SPACINGS:
            host=leaf_host(a,temperature,calibrated,rank_spacing_units=spacing,forcing_mask=mask)
            assert host.active.shape==(*a.shape,8) and not host.active[~mask].any()
            for stage,indices in host.stage_day_index.items():
                for member in meta.itertuples():
                    index=int(indices[member.field_index]);visible=host.visible_final_ranks_at_stage[stage][member.field_index]
                    unfolded=host.unfolded_final_ranks_at_stage[stage][member.field_index]
                    status_rows.append(dict(field_id=member.field_id,partition=partition,event=stage,effective_rank_spacing_units=spacing,
                        predicted_date=pd.NaT if index<0 else pd.Timestamp(member.sowing_date)+pd.Timedelta(days=index),
                        stage_reached=index>=0,modeled_visible_final_rank_count=np.nan if index<0 else int(visible.sum()),
                        modeled_unfolded_final_rank_count=np.nan if index<0 else int(unfolded.sum()),
                        top3_visible_count=np.nan if index<0 else int(visible[:3].sum()),top3_unfolded_count=np.nan if index<0 else int(unfolded[:3].sum()),
                        F1_visible=None if index<0 else bool(visible[0]),F1_unfolded=None if index<0 else bool(unfolded[0]),
                        observed_leaf_count_available=False,spacing_calibrated=False))
    scored=pd.concat(scores,ignore_index=True)
    metrics=[]
    for (partition,event),group in scored.groupby(['partition','event']):
        for scope,part in [('all_constraints',group),('genuine_two_sided',group.loc[group.genuinely_bracketed])]:
            if part.empty:continue
            metrics.append(dict(partition=partition,event=event,scope=scope,n_field_events=len(part),
                mean_distance_days=float(part.distance_days.mean()),mean_signed_distance_days=float(part.signed_distance_days.mean()),
                compatible_fraction=float(part.compatible.mean()),missing_predictions=int(part.predicted_date.isna().sum()),
                prediction_bound_distances=int(part.distance_is_lower_bound.sum()),
                median_bracket_width_days=None if part.bracket_width_days.dropna().empty else float(part.bracket_width_days.median()),
                evaluation_role='calibration' if partition=='calibration' else 'retrospective_reused_validation'))
    pd.DataFrame(metrics).to_csv(HERE/'stage_censoring_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(HERE/'stage_observation_coverage.csv',index=False)
    pd.DataFrame(status_rows).to_csv(HERE/'modeled_final_leaf_ranks_at_stages.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(HERE/'predicted_stage_dates.csv',index=False)
    scored.to_csv(HERE/'stage_censoring_scores.csv',index=False)
    pd.concat(metadata_all,ignore_index=True).to_csv(HERE/'original_input_membership.csv',index=False)
    pd.concat(normalized_all,ignore_index=True).to_csv(HERE/'normalized_stage_dates.csv',index=False)
    pd.concat(rejected_all,ignore_index=True).to_csv(HERE/'rejected_stage_rows.csv',index=False)
    pd.concat(excluded_all,ignore_index=True).to_csv(HERE/'excluded_stage_rows.csv',index=False)
    errors=[]
    for row in scored.itertuples():
        if pd.isna(row.predicted_date):expected=max(0,(pd.Timestamp(row.forcing_end)+pd.Timedelta(days=1)-row.upper_inclusive).days) if pd.notna(row.upper_inclusive) else 0
        elif pd.notna(row.lower_exclusive) and row.predicted_date<=row.lower_exclusive:expected=(row.lower_exclusive+pd.Timedelta(days=1)-row.predicted_date).days
        elif pd.notna(row.upper_inclusive) and row.predicted_date>row.upper_inclusive:expected=(row.predicted_date-row.upper_inclusive).days
        else:expected=0
        errors.append(abs(expected-row.distance_days))
    assert max(errors)==0 and sha(HERE/'calibrated_stage_thresholds.json')==frozen_hash
    cal_score=scored.loc[scored.partition.eq('calibration')&scored.event.isin(EVENTS)]
    independent_objective=float(cal_score.groupby('field_id').distance_days.mean().mean())
    assert abs(independent_objective-selection['minimum_loss_days'])<1e-10
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    frozen_json(HERE/'verification.json',dict(status='passed',stage_score_rows_checked=len(scored),maximum_distance_error_days=max(errors),
        selected_objective_recomputed=independent_objective,donor_q1_preserved=True,leaf_spacing_not_calibrated=True,
        actual_absolute_leaf_count_not_invented=True,selection_frozen_before_reused_evaluation=True))
    frozen_json(HERE/'receipt.json',dict(status='complete',model='stage-anchored final leaf rank scenarios',threshold_selection_sha256=frozen_hash,
        original_field_counts=dict(calibration=28,reused_BASF2019=45,reused_strict_external=143),
        disease_values_used=False,true_leaf_counts_observed=False,rank_spacing_calibrated=False,
        untouched_validation=False,prior_stage_outcome_exposure=True,source_sha256=hashes,
        output_sha256={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name not in ['run.py','receipt.json']}))


if __name__=='__main__':main()
