"""Original-membership retrospective selection of seasonal source/progression hypotheses."""
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

from analysis.paper_study.structural_evaluation.run import (
    ROOT,DEFAULT_PATHS,load_inputs,subset_fields,estimate_training_rain_rate,
    write_frozen_json,_save_frame,save_frozen_csv,
)
from analysis.paper_study.structural_evaluation.membership import spatial_folds
from model.seasonal_septoria.infection_events import (
    symptom_brackets,onset_distance,equal_hierarchy_weights,simulate_events,EventParameters,
)
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.wetness import duration_exposure
from model.seasonal_septoria.overwinter import OverwinterParameters,simulate_overwinter

HERE=Path(__file__).resolve().parent
PHENOLOGY=HERE.parent/'phenology/calibrated_stage_thresholds.json'
FRACTION=.001
SIGN_COLUMNS=['global_target_index','field_id','field_index','site_id','source','dataset_id','physical_unit',
    'season_year','coordinate_year','endpoint_series','leaf_index','date','day_index','value','metric']
ABLATIONS=['baseline','no_local_source','secondary_off','imported_off','initial_ready_0','initial_ready_1',
    'residue_lifetime_x0.5','residue_lifetime_x2','spacing80','spacing120','spacing160','contact_off',
    'juvenile_removed_after31','lower_ranks_removed_after39','persistent_juvenile','no_sources']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_grid():
    return [dict(name=f'a{alpha:g}_b{beta:g}_L{delay}_s{spacing}_i{imported:g}',
        parameters=asdict(OverwinterParameters(primary_scale=alpha,secondary_scale=beta,latent_reference_days=float(delay))),
        rank_spacing_units=float(spacing),constant_imported_pressure=float(imported),juvenile_policy='handover_31_39')
        for alpha in (.0001,.001,.01,.1) for beta in (.1,1.,10.) for delay in (20,30)
        for spacing in (80,120,160) for imported in (0.,.1)]


def benchmark_grid():
    return [dict(name=f'phenology_s{spacing}_L{delay}',rank_spacing_units=float(spacing),delay_reference_days=float(delay))
        for spacing in (80,120,160) for delay in (20,30)]


def detection_metrics(frame):
    observed=frame.observed_positive.to_numpy(bool);predicted=frame.predicted_positive.to_numpy(bool)
    weight=equal_hierarchy_weights(frame)
    tp=observed&predicted;tn=~observed&~predicted;fp=~observed&predicted;fn=observed&~predicted
    positive=float(weight@observed);negative=float(weight@~observed)
    sensitivity=None if not positive else float(weight@tp)/positive
    specificity=None if not negative else float(weight@tn)/negative
    balanced=None if sensitivity is None or specificity is None else .5*(sensitivity+specificity)
    return dict(n=len(frame),positive_count=int(observed.sum()),negative_count=int((~observed).sum()),
        true_positive=int(tp.sum()),false_positive=int(fp.sum()),true_negative=int(tn.sum()),false_negative=int(fn.sum()),
        sensitivity=sensitivity,specificity=specificity,balanced_accuracy=balanced,
        weighted_accuracy=float(weight@(observed==predicted)),specificity_identifiable=bool(negative))


def select_candidate(records):
    minimum=min(record['primary_distance_days'] for record in records)
    tied=[record for record in records if np.isclose(record['primary_distance_days'],minimum,atol=1e-12,rtol=0)]
    return min(tied,key=lambda record:(record['balanced_assessment_error'],record['candidate_order']))


def primary_tie_count(records,minimum):
    return int(np.count_nonzero([np.isclose(record['primary_distance_days'],minimum,atol=1e-12,rtol=0) for record in records]))


def _forcing(data,accumulation,thresholds,fitted):
    if data.targets.value.notna().any():raise ValueError('Forecast targets must be redacted.')
    lengths=data.metadata.set_index('field_index').forcing_days.reindex(range(len(data.temperature))).to_numpy(int)
    mask=np.arange(data.temperature.shape[1])[None,:]<lengths[:,None]
    host=leaf_host(accumulation,data.temperature,thresholds,rank_spacing_units=fitted['rank_spacing_units'],forcing_mask=mask,
        juvenile_policy=fitted.get('juvenile_policy','handover_31_39'))
    exposure=duration_exposure(data.temperature,data.maximum_temperature,data.humidity,data.rain,
        fitted['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    return host,np.where(mask,exposure,0.),mask


def predict_record(data,accumulation,thresholds,fitted,*,ablation='baseline',detection_fraction=FRACTION):
    """Signs and observed BBCH are absent from the prediction boundary."""
    setting=dict(fitted)
    setting['juvenile_policy']='persistent' if ablation=='persistent_juvenile' else 'handover_31_39'
    if ablation.startswith('spacing'):setting['rank_spacing_units']=float(ablation[7:])
    host,exposure,mask=_forcing(data,accumulation,thresholds,setting)
    p=OverwinterParameters(**setting['parameters']);source=1.;pressure=setting['constant_imported_pressure']
    if ablation in ('no_local_source','no_sources'):source=0.
    if ablation in ('imported_off','no_sources'):pressure=0.
    if ablation=='secondary_off':p=replace(p,secondary_scale=0.)
    if ablation=='initial_ready_0':p=replace(p,initial_ready_fraction=0.)
    if ablation=='initial_ready_1':p=replace(p,initial_ready_fraction=1.)
    if ablation=='residue_lifetime_x0.5':p=replace(p,residue_decay_reference_days=p.residue_decay_reference_days*.5)
    if ablation=='residue_lifetime_x2':p=replace(p,residue_decay_reference_days=p.residue_decay_reference_days*2.)
    if ablation=='contact_off':p=replace(p,contact_fraction=0.)
    if ablation in ('juvenile_removed_after31','lower_ranks_removed_after39'):
        removed=accumulation>=thresholds[31 if ablation=='juvenile_removed_after31' else 39]
        slots=slice(7,8) if ablation=='juvenile_removed_after31' else slice(3,8)
        for array in (host.active,host.area,host.renewal):
            view=array[:,:,slots];view[removed]=0
    trajectory=simulate_overwinter(data.temperature,exposure,data.rain,host.active,host.renewal,host.area,mask,p,
        initial_local_source=source,imported_pressure=np.full_like(data.temperature,pressure),detection_fraction=detection_fraction)
    if np.max(abs(trajectory.state.sum(axis=-1)-1.))>1e-10:raise AssertionError('Tissue mass accounting failed.')
    return trajectory,host


def predict_benchmark(data,accumulation,thresholds,fitted):
    host,_,_=_forcing(data,accumulation,thresholds,fitted)
    trajectory=simulate_events(data.temperature,np.zeros_like(data.temperature),host.active,
        EventParameters(.1,1,fitted['delay_reference_days']),phenology_only=True)
    return trajectory,host


def _assessment(truth,trajectory,model,partition,scenario='baseline',cutoff=FRACTION):
    part=truth.targets.copy();field,day,leaf=[part[key].to_numpy(int) for key in ('field_index','day_index','leaf_index')]
    if np.any(day>truth.metadata.set_index('field_index').forcing_days.reindex(field).to_numpy()):raise ValueError('Assessment exceeds forcing horizon.')
    part['observed_positive']=part.value.gt(0)
    part['predicted_positive']=(trajectory.damage[field,day,leaf]>=cutoff if hasattr(trajectory,'damage') else trajectory.symptomatic[field,day,leaf])
    part['predicted_symptom_day']=trajectory.symptom_day[field,leaf]
    part['predicted_infection_day']=trajectory.infection_day[field,leaf]
    part=part.drop(columns='value');part['model']=model;part['partition']=partition;part['scenario']=scenario;part['detection_fraction']=cutoff
    return part


def _brackets(truth,trajectory,model,partition,scenario='baseline',cutoff=FRACTION):
    part=symptom_brackets(truth.targets,upper_three=False)
    f,l=part.field_index.to_numpy(int),part.leaf_index.to_numpy(int)
    part['forcing_end_day']=part.field_index.map(truth.metadata.set_index('field_index').forcing_days).to_numpy(int)
    part['predicted_symptom_day']=trajectory.symptom_day[f,l];part['predicted_infection_day']=trajectory.infection_day[f,l]
    part['onset_distance_days']=onset_distance(part,part.predicted_symptom_day.to_numpy(),part.forcing_end_day.to_numpy())
    part['compatible']=part.onset_distance_days.eq(0)
    part['predicted_positive']=part.predicted_symptom_day.ge(0)&part.predicted_symptom_day.le(part.last_assessment_day)
    sow=part.field_index.map(truth.metadata.set_index('field_index').sowing_date)
    for label in ('symptom','infection'):
        days=part[f'predicted_{label}_day'];part[f'predicted_{label}_date']=pd.to_datetime(sow)+pd.to_timedelta(days-1,unit='D')
        part.loc[days.lt(0),f'predicted_{label}_date']=pd.NaT
    part['model']=model;part['partition']=partition;part['scenario']=scenario;part['detection_fraction']=cutoff
    return part


def _fit_record(path,training,weather,settings,stage):
    preprocessing=dict(weather_operator='duration_proxy',**estimate_training_rain_rate(training.metadata,weather))
    record=dict(status='fit_complete',stage=stage,weather_preprocessing=preprocessing,candidates=settings,
        physical_training_fields=training.metadata.field_id.tolist(),training_sites=sorted(training.targets.site_id.unique().tolist()),
        training_target_count=len(training.targets),training_positive_count=int(training.targets.value.gt(0).sum()),
        sign_only=True,disease_magnitude_fit=False,initial_local_source=1.,evaluation_outcomes_used=False)
    write_frozen_json(path,record)
    return preprocessing


def _score_candidate(assessments,brackets,name,order):
    primary=brackets.loc[brackets.leaf_index.lt(3)&brackets.censoring.eq('two_sided')]
    if primary.empty:raise ValueError('No genuine Top3 two-sided symptom brackets; percentage fallback is forbidden.')
    detection=detection_metrics(assessments.loc[assessments.leaf_index.lt(3)])
    if detection['balanced_accuracy'] is None:raise ValueError('Both assessment classes are needed for the fixed tie-break.')
    return dict(candidate=name,candidate_order=order,primary_distance_days=float(equal_hierarchy_weights(primary)@primary.onset_distance_days.to_numpy()),
        primary_brackets=len(primary),primary_fields=primary.field_id.nunique(),
        primary_compatibility=float(equal_hierarchy_weights(primary)@primary.compatible.to_numpy()),
        balanced_assessment_error=1-detection['balanced_accuracy'],assessment_detection=detection)


def select_grid(folder,data,accumulation,thresholds,weather,folds,settings,*,benchmark=False):
    # The stage thresholds come from the complete original calibration stage set;
    # they are not re-fit in these disease-inner folds. No evaluation stages enter.
    bracket_parts={s['name']:[] for s in settings};assessment_parts={s['name']:[] for s in settings}
    for fold in folds:
        training=subset_fields(data,fold.train);redacted=subset_fields(data,fold.test,redact_values=True);truth=subset_fields(data,fold.test)
        fields=np.sort(data.targets.iloc[fold.test].field_index.unique())
        preprocessing=_fit_record(folder/f'{fold.name}_training.json',training,weather,settings,'inner_training')
        assert set(training.targets.site_id).isdisjoint(redacted.targets.site_id)
        for candidate in settings:
            fitted=dict(**candidate,weather_preprocessing=preprocessing)
            trajectory,_=(predict_benchmark(redacted,accumulation[fields],thresholds,fitted) if benchmark else predict_record(redacted,accumulation[fields],thresholds,fitted))
            a=_assessment(truth,trajectory,'phenology_only' if benchmark else 'overwinter_source_model','inner',scenario=candidate['name'])
            b=_brackets(truth,trajectory,'phenology_only' if benchmark else 'overwinter_source_model','inner',scenario=candidate['name'])
            a['inner_fold']=fold.name;b['inner_fold']=fold.name
            assessment_parts[candidate['name']].append(a);bracket_parts[candidate['name']].append(b)
    records=[];all_a=[];all_b=[]
    for order,candidate in enumerate(settings):
        a=pd.concat(assessment_parts[candidate['name']],ignore_index=True);b=pd.concat(bracket_parts[candidate['name']],ignore_index=True)
        records.append(_score_candidate(a,b,candidate['name'],order));all_a.append(a);all_b.append(b)
    selected=select_candidate(records);minimum=selected['primary_distance_days']
    chosen=next(item for item in settings if item['name']==selected['candidate'])
    record=dict(selected_candidate=chosen,selected_score=selected,candidate_scores=records,
        exact_primary_minimum_ties=primary_tie_count(records,minimum),
        objective='pooled inner equal-coordinate-year/field/leaf Top3 two-sided symptom-bracket distance',
        tie_break='lowest weighted balanced assessment error then registered enumeration',
        stage_fit_outside_disease_inner_folds=True,stage_fit_original_calibration_only=True,
        parameter_identification_limited=True,disease_magnitude_fit=False,reused_validation_outcomes_used=False)
    write_frozen_json(folder/'selection.json',record)
    _save_frame(folder/'all_inner_assessment_predictions.parquet',pd.concat(all_a,ignore_index=True))
    _save_frame(folder/'all_inner_bracket_predictions.parquet',pd.concat(all_b,ignore_index=True))
    return record


def _flow_and_stages(truth,trajectory,host,model,partition,scenario):
    flows=[];stages=[]
    for meta in truth.metadata.itertuples():
        field=int(meta.field_index);n=int(meta.forcing_days);sow=pd.Timestamp(meta.sowing_date)
        stage_dates={stage:pd.NaT if index[field]<0 else sow+pd.Timedelta(days=int(index[field])) for stage,index in host.stage_day_index.items()}
        stages.append(dict(field_id=meta.field_id,partition=partition,model=model,scenario=scenario,
            **{f'BBCH{stage}_date':date for stage,date in stage_dates.items()},
            complete_grain_fill_window=pd.notna(stage_dates[65]) and pd.notna(stage_dates[85]),
            grain_fill_days=np.nan if pd.isna(stage_dates[65]) or pd.isna(stage_dates[85]) else (stage_dates[85]-stage_dates[65]).days+1))
        if not hasattr(trajectory,'local_flow'):continue
        for leaf in range(8):
            local=float(trajectory.local_flow[field,:n,leaf].sum());imported=float(trajectory.imported_flow[field,:n,leaf].sum())
            splash=float(trajectory.splash_flow[field,:n,leaf].sum());contact=float(trajectory.contact_flow[field,:n,leaf].sum())
            flows.append(dict(field_id=meta.field_id,partition=partition,scenario=scenario,leaf_index=leaf,
                local_flow_total=local,imported_flow_total=imported,splash_flow_total=splash,contact_flow_total=contact,
                primary_flow_total=local+imported,secondary_flow_total=splash+contact,
                end_damage_fraction=float(trajectory.damage[field,n,leaf]),end_local_source=float(trajectory.residue[field,n].sum()),
                flow_is_causal_source_attribution=False))
    return pd.DataFrame(flows),pd.DataFrame(stages)


def summarize(a,b):
    detection=[];onsets=[];occurrence=[]
    keys=['partition','model','scenario','detection_fraction']
    for identity,group in a.groupby(keys):
        base=dict(zip(keys,identity));bounds=b
        for key,val in base.items():bounds=bounds.loc[bounds[key].eq(val)]
        for scope in ('top3','all_ordinal_leaves'):
            part=group.loc[group.leaf_index.lt(3)] if scope=='top3' else group
            bp=bounds.loc[bounds.leaf_index.lt(3)] if scope=='top3' else bounds
            detection.append(dict(**base,leaf_scope=scope,**detection_metrics(part)))
            occurrence.append(dict(**base,leaf_scope=scope,level='leaf_window',**detection_metrics(bp)))
            f=bp.groupby(['coordinate_year','field_id'],as_index=False).agg(observed_positive=('observed_positive','any'),predicted_positive=('predicted_positive','any'));f['leaf_index']=0
            occurrence.append(dict(**base,leaf_scope=scope,level='field_window',**detection_metrics(f)))
            for censoring,censored in bp.groupby('censoring'):
                w=equal_hierarchy_weights(censored)
                onsets.append(dict(**base,leaf_scope=scope,censoring=censoring,n=len(censored),
                    distance_days=float(w@censored.onset_distance_days.to_numpy()),compatible_fraction=float(w@censored.compatible.to_numpy()),
                    missed_positive=int((censored.observed_positive&censored.predicted_symptom_day.lt(0)).sum()),
                    observed_persistence_violations=int(censored.persistence_violations.sum())))
    return pd.DataFrame(detection),pd.DataFrame(onsets),pd.DataFrame(occurrence)


def main():
    if (HERE/'configuration_before_fitting.json').exists():raise FileExistsError('Use a new immutable disease experiment.')
    HERE.mkdir(parents=True,exist_ok=True)
    settings=candidate_grid();benchmarks=benchmark_grid();assert len(settings)==144
    strict_path=ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv'
    dependencies=[Path(__file__),ROOT/'tests/test_overwinter_evaluation.py',PHENOLOGY,strict_path,*DEFAULT_PATHS.values(),
        ROOT/'model/seasonal_septoria/overwinter.py',ROOT/'model/seasonal_septoria/leaf_phenology.py',
        ROOT/'model/seasonal_septoria/infection_events.py',ROOT/'model/seasonal_septoria/wetness.py',
        ROOT/'analysis/paper_study/structural_evaluation/run.py',ROOT/'analysis/paper_study/structural_evaluation/membership.py',
        ROOT/'analysis/paper_study/run_structural_canopy.py',ROOT/'model/seasonal_septoria/field_data.py']
    hashes={str(path.relative_to(ROOT)):sha(path) for path in dependencies}
    configuration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),validation_label='retrospective_development_validation',
        prior_outcomes_exposed=True,untouched_test=False,direct_infection_dates_observed=False,
        candidates=settings,candidate_count=144,phenology_benchmark_candidates=benchmarks,inner_location_folds=3,seed=20261006,
        calibration='original BASF2017-2018:28 fields/254 all-leaf assessments',evaluation='BASF2019:45 fields; strict location-disjoint Corteva:143 fields',
        primary_detection_fraction=FRACTION,postfit_detection_sensitivities=[.0001,.01],disease_labels='zero vs positive only',
        primary_selection='pooled inner Top3 two-sided symptom bracket distance',tie_break='weighted balanced assessment error then fixed enumeration',
        no_percentage_fallback=True,minimum_required_two_sided_brackets=1,calibration_identification_limited=True,
        initial_local_source=1.,all_ecological_defaults_are_effective_unvalidated_hypotheses=asdict(OverwinterParameters()),
        rainfall_intensity='training-only meteorology median',stage_fit='frozen original-calibration stage-only fit, outside disease-inner folds',
        primary_juvenile_policy='handover_31_39; J=clip((C39-G)/(C39-C31),0,1)',
        same_fit_ablations=ABLATIONS,unmeasured_senescence_ablations=['juvenile_removed_after31','lower_ranks_removed_after39'],
        route_flows_are_causal_attribution=False,source_sha256=hashes)
    write_frozen_json(HERE/'configuration_before_fitting.json',configuration)
    for path in dependencies:
        if path.suffix in ('.py','.json'):
            target=HERE/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    data,weather,accumulation,_=load_inputs(DEFAULT_PATHS)
    thresholds={int(k):v for k,v in json.loads(PHENOLOGY.read_text())['all_stage_thresholds'].items()}
    strict=set(pd.read_csv(strict_path,usecols=['source_unit'],dtype={'source_unit':str}).source_unit)
    memberships=dict(calibration=np.flatnonzero(data.targets.source.eq('BASF')&data.targets.season_year.isin([2017,2018])),
        reused_BASF2019=np.flatnonzero(data.targets.source.eq('BASF')&data.targets.season_year.eq(2019)),
        reused_strict_Corteva=np.flatnonzero(data.targets.source.eq('Corteva')&data.targets.physical_unit.astype(str).isin(strict)))
    for partition,expected in [('calibration',28),('reused_BASF2019',45),('reused_strict_Corteva',143)]:
        assert data.targets.iloc[memberships[partition]].field_id.nunique()==expected
    assert len(memberships['calibration'])==254
    data.targets=data.targets[SIGN_COLUMNS].copy();data.targets['value']=data.targets.value.gt(0).astype(int)
    membership_parts=[]
    for partition,rows in memberships.items():
        part=data.targets.iloc[rows].copy();part['partition']=partition;membership_parts.append(part)
    _save_frame(HERE/'original_sign_only_target_membership.parquet',pd.concat(membership_parts,ignore_index=True))
    training=subset_fields(data,memberships['calibration']);train_fields=np.sort(data.targets.iloc[memberships['calibration']].field_index.unique())
    folds=spatial_folds(training.targets,n_splits=3,seed=20261006)
    inner=[]
    for fold in folds:
        for role,rows in [('train',fold.train),('validation',fold.test)]:
            part=training.targets.iloc[rows][['global_target_index','field_id','site_id','coordinate_year']].copy();part['fold']=fold.name;part['role']=role;inner.append(part)
    save_frozen_csv(HERE/'inner_location_membership_before_fitting.csv',pd.concat(inner,ignore_index=True))
    # Magnitude perturbations cannot alter signs, brackets or any forecast input.
    changed=training.targets.copy();changed['value']=changed.value.astype(float);positive=changed.value.gt(0)
    changed.loc[positive,'value']=np.linspace(.000001,100.,positive.sum())
    pd.testing.assert_frame_equal(symptom_brackets(training.targets),symptom_brackets(changed))
    if not symptom_brackets(training.targets).censoring.eq('two_sided').any():raise ValueError('No informative calibration brackets.')
    frozen={}
    for name,grid,benchmark in [('overwinter_source_model',settings,False),('phenology_only',benchmarks,True)]:
        folder=HERE/name;folder.mkdir()
        selection=select_grid(folder,training,accumulation[train_fields],thresholds,weather,folds,grid,benchmark=benchmark)
        preprocessing=_fit_record(folder/'full_calibration_training.json',training,weather,[selection['selected_candidate']],'post_inner_selection_refit')
        fitted=dict(**selection['selected_candidate'],weather_preprocessing=preprocessing)
        path=folder/'frozen_selected_fit.json'
        write_frozen_json(path,dict(status='fit_complete',fitted=fitted,selection_sha256=sha(folder/'selection.json'),
            calibration_fields=28,disease_magnitude_fit=False,validation_outcomes_used=False))
        frozen[name]=dict(path=str(path.relative_to(HERE)),sha256=sha(path))
        print(json.dumps(dict(model=name,selected=fitted['name'],score=selection['selected_score'])),flush=True)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    write_frozen_json(HERE/'fits_frozen_before_reused_validation.json',dict(models=frozen,validation_labels_scored=False,prior_outcome_exposure=True))
    a_parts=[];b_parts=[];flow_parts=[];stage_parts=[]
    for partition,rows in memberships.items():
        redacted=subset_fields(data,rows,redact_values=True);truth=subset_fields(data,rows);fields=np.sort(data.targets.iloc[rows].field_index.unique())
        for name,record in frozen.items():
            path=HERE/record['path'];assert sha(path)==record['sha256'];fitted=json.loads(path.read_text())['fitted']
            scenarios=['baseline'] if name=='phenology_only' else ABLATIONS
            for scenario in scenarios:
                trajectory,host=(predict_benchmark(redacted,accumulation[fields],thresholds,fitted) if name=='phenology_only' else predict_record(redacted,accumulation[fields],thresholds,fitted,ablation=scenario))
                a_parts.append(_assessment(truth,trajectory,name,partition,scenario));b_parts.append(_brackets(truth,trajectory,name,partition,scenario))
                flows,stages=_flow_and_stages(truth,trajectory,host,name,partition,scenario);flow_parts.append(flows);stage_parts.append(stages)
                if name=='overwinter_source_model' and scenario=='baseline':
                    daily=[]
                    for meta in truth.metadata.itertuples():
                        n=meta.forcing_days;i=meta.field_index
                        for leaf in range(3):
                            daily.append(pd.DataFrame(dict(field_id=meta.field_id,partition=partition,leaf_index=leaf,
                                date=pd.date_range(meta.sowing_date,periods=n),day_index=np.arange(1,n+1),damage_fraction=trajectory.damage[i,1:n+1,leaf],
                                affected_fraction=trajectory.state[i,1:n+1,leaf,1:].sum(axis=1),active=host.active[i,:n,leaf],area_capacity=host.area[i,:n,leaf])))
                    _save_frame(HERE/f'{partition}_baseline_daily_top3.parquet',pd.concat(daily,ignore_index=True))
                    for cutoff in (.0001,.01):
                        sensitivity,_=predict_record(redacted,accumulation[fields],thresholds,fitted,detection_fraction=cutoff)
                        a_parts.append(_assessment(truth,sensitivity,name,partition,'detection_sensitivity',cutoff))
                        b_parts.append(_brackets(truth,sensitivity,name,partition,'detection_sensitivity',cutoff))
    a=pd.concat(a_parts,ignore_index=True);b=pd.concat(b_parts,ignore_index=True)
    _save_frame(HERE/'all_assessment_sign_predictions.parquet',a);_save_frame(HERE/'all_first_onset_bracket_predictions.parquet',b)
    _save_frame(HERE/'leaf_pathway_flow_totals.parquet',pd.concat(flow_parts,ignore_index=True));_save_frame(HERE/'field_stage_dates.parquet',pd.concat(stage_parts,ignore_index=True))
    detection,onset,occurrence=summarize(a,b)
    for filename,frame in [('assessment_detection_metrics',detection),('onset_bracket_metrics',onset),('observed_window_occurrence_metrics',occurrence)]:
        save_frozen_csv(HERE/f'{filename}.csv',frame);_save_frame(HERE/f'{filename}.parquet',frame)
    # Independently recompute interval arithmetic, without model loss helpers.
    for row in b.itertuples():
        predicted=row.predicted_symptom_day
        if predicted<0:expected=max(30.,row.forcing_end_day+1-row.upper_day) if pd.notna(row.upper_day) else 0.
        elif row.censoring=='left':expected=max(0.,predicted-row.upper_day)
        elif row.censoring=='right':expected=max(0.,row.lower_day+1-predicted)
        else:expected=max(0.,row.lower_day+1-predicted,predicted-row.upper_day)
        assert expected==row.onset_distance_days
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    for record in frozen.values():assert sha(HERE/record['path'])==record['sha256']
    write_frozen_json(HERE/'independent_arithmetic_checks.json',dict(status='passed',bracket_rows_checked=len(b),
        positive_magnitude_invariant=True,physically_isolated_inner_training=True,redacted_prediction_targets=True,
        fits_frozen_before_reused_scoring=True,stage_fit_outside_inner_disease_folds_disclosed=True))
    write_frozen_json(HERE/'receipt.json',dict(status='complete',validation_label='retrospective_development_validation',
        untouched_test=False,direct_infection_dates_validated=False,calibration_fields=28,calibration_all_leaf_assessments=254,
        candidate_count=144,inner_location_folds=3,validation_fields=dict(BASF2019=45,strict_Corteva=143),
        ecological_parameter_identification_limited=True,route_flows_are_causal_attribution=False,
        source_sha256=hashes,output_sha256={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file()
            and 'source_snapshot' not in p.parts and '__pycache__' not in p.parts
            and not any(part.startswith('aborted_') for part in p.parts) and p.name not in ['run.py','receipt.json']}))


if __name__=='__main__':main()
