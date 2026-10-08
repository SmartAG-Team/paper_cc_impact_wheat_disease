"""Original-only selection diagnostic nesting stage and weather nuisance fits."""
from pathlib import Path
from datetime import datetime, timezone
import os,sys,json,hashlib,time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(HERE/'numba_cache'))
import numpy as np
import pandas as pd
from analysis.paper_study.nature_food_fix_20261007.validation.statistics import training_stage_membership
from analysis.paper_study.overwinter_leaf_model_20261006.disease import run as disease
from analysis.paper_study.overwinter_leaf_model_20261006.phenology import run as stage
from analysis.paper_study.infection_priority_20261006.anthesis_clock import run as flowering
from analysis.paper_study.structural_evaluation.run import subset_fields,estimate_training_rain_rate
from analysis.paper_study.run_structural_canopy import development_inputs
from calibration.seasonal_septoria.field_data import prepare_fields
from calibration.seasonal_septoria.infection_events import symptom_brackets

OUT=HERE/'nested_calibration'
ORIGINAL=ROOT/'analysis/paper_study/overwinter_leaf_model_20261006'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    start=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    sources=dict(basf=ROOT/'analysis/paper_study/seasonal_calibration_v1/basf_source_assessments_snapshot.csv',
        weather=ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet',
        calendars=ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv',
        donor=ROOT/'process_model/parameters/calibration.json',
        membership=ORIGINAL/'disease/original_sign_only_target_membership.parquet',
        folds=ORIGINAL/'disease/inner_location_membership_before_fitting.csv',
        stages=ORIGINAL/'phenology/calibration_stage_constraints_before_fitting.csv',
        flowering=ROOT/'analysis/paper_study/infection_priority_20261006/anthesis_clock/calibration_constraints_before_fitting.csv',
        stage_meta=ORIGINAL/'phenology/original_input_membership.csv',
        stage_acc=ROOT/'analysis/paper_study/phenology_assumptions_20261006/calibration_development_inputs.npz',
        original_disease_selection=ORIGINAL/'disease/overwinter_source_model/selection.json',
        original_benchmark_selection=ORIGINAL/'disease/phenology_only/selection.json',
        original_stage_fit=ORIGINAL/'phenology/calibrated_stage_thresholds.json')
    code=[Path(__file__),HERE/'statistics.py',Path(disease.__file__),Path(stage.__file__),Path(flowering.__file__),
        ROOT/'analysis/paper_study/structural_evaluation/run.py',
        ROOT/'calibration/seasonal_septoria/infection_events.py',
        ROOT/'model/seasonal_septoria/overwinter.py',ROOT/'model/seasonal_septoria/leaf_phenology.py',
        ROOT/'model/seasonal_septoria/infection_events.py',ROOT/'model/seasonal_septoria/wetness.py']
    hashes={str(path.relative_to(ROOT)):sha(path) for path in [*sources.values(),*code]}
    settings=disease.candidate_grid();benchmarks=disease.benchmark_grid()
    write_json(OUT/'contract_before_fitting.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes, scope='fully nested nuisance fitting inside original-only three-fold candidate selection',
        original_fields=28,calibration_years=[2017,2018],candidates=settings,benchmark_candidates=benchmarks,
        stage_refitting='C32,C33,C37,C39 joint ordered grid and independent C65 flowering grid on each training-fold field membership',
        meteorology_refitting='rainfall-duration scale on each training-fold unique location/date meteorology',
        donor_process='independently fitted German donor parameters/thresholds held fixed; not estimated from BASF/Corteva outcomes',
        selection='pooled equal-coordinate-year/field/leaf top-three genuine two-sided distance, balanced assessment error, original enumeration',
        no_new_untouched_validation=True,reused_validation_used_to_select=False,
        missing_fold_brackets='individual fold objective unavailable when no two-sided history; pooled objective uses only extant histories',
        identification='exact counts and joint minimizing plateaus, including zero two-sided stage evidence, are explicit; conventional plateau representatives are not identified physiological thresholds'))
    for path in code:
        target=OUT/'source_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    members=pd.read_parquet(sources['membership'])
    members=members.loc[members.partition.eq('calibration')].copy()
    raw=pd.read_csv(sources['basf'])
    raw=raw.loc[raw.season_year.isin([2017,2018])].copy()
    raw['source']='BASF'
    weather=pd.read_parquet(sources['weather'])
    calendars=pd.read_csv(sources['calendars'])
    assembled=prepare_fields(raw,calendars,weather)
    assembled.targets=assembled.targets.merge(members[['field_id','endpoint_series','date','global_target_index']].assign(
        date=lambda x:pd.to_datetime(x.date)),on=['field_id','endpoint_series','date'],validate='one_to_one')
    assert len(assembled.targets)==254
    donor=json.loads(sources['donor'].read_text())
    accumulation,_=development_inputs(assembled,weather,donor)
    assembled.targets=assembled.targets[disease.SIGN_COLUMNS].copy()
    assembled.targets['value']=assembled.targets.value.gt(0).astype(int)
    assert assembled.targets.field_id.nunique()==28
    members[['global_target_index','field_id','site_id','coordinate_year','endpoint_series','leaf_index','date','value']].to_csv(
        OUT/'original_calibration_target_membership.csv',index=False)
    folds=pd.read_csv(sources['folds'])
    folds.to_csv(OUT/'original_inner_membership.csv',index=False)
    constraints=pd.read_csv(sources['stages'],parse_dates=['lower_exclusive','upper_inclusive'])
    c65=pd.read_csv(sources['flowering'],parse_dates=['lower_exclusive','upper_inclusive'])
    meta=pd.read_csv(sources['stage_meta'])
    meta=meta.loc[meta.partition.eq('calibration')].copy()
    a=np.load(sources['stage_acc'])['accumulation']
    fixed={int(row['BBCH']):float(row['Cumulative_t_pp_v_GDD']) for row in donor['thresholds'] if row['BBCH'] in [10,31,51,85]}
    a_parts={s['name']:[] for s in settings+benchmarks}
    b_parts={s['name']:[] for s in settings+benchmarks}
    fold_counts=[];stage_counts=[]
    for fold,plan in folds.groupby('fold',sort=True):
        training_ids=set(plan.loc[plan.role.eq('train'),'global_target_index'])
        heldout_ids=set(plan.loc[plan.role.eq('validation'),'global_target_index'])
        train_rows=np.flatnonzero(assembled.targets.global_target_index.isin(training_ids))
        heldout_rows=np.flatnonzero(assembled.targets.global_target_index.isin(heldout_ids))
        training=subset_fields(assembled,train_rows)
        truth=subset_fields(assembled,heldout_rows)
        redacted=subset_fields(assembled,heldout_rows,redact_values=True)
        assert set(training.targets.site_id).isdisjoint(truth.targets.site_id)
        assert training_ids.isdisjoint(heldout_ids)
        assert training_ids|heldout_ids==set(members.global_target_index)
        training_fields=set(training.metadata.field_id)
        folder=OUT/fold;folder.mkdir(exist_ok=True)
        c=training_stage_membership(constraints,training_fields)
        flowering_c=training_stage_membership(c65,training_fields)
        assert set(c.field_id).isdisjoint(truth.targets.field_id)
        assert set(flowering_c.field_id).isdisjoint(truth.targets.field_id)
        c.to_csv(folder/'stage_training_membership_before_fitting.csv',index=False)
        flowering_c.to_csv(folder/'flowering_training_membership_before_fitting.csv',index=False)
        stage_metadata=meta.loc[meta.field_id.isin(training_fields)].copy()
        stage_metadata.to_csv(folder/'stage_weather_metadata_before_fitting.csv',index=False)
        fitted_stage,curve=stage.fit_ordered_thresholds(c,a,stage_metadata,fixed)
        fitted65,curve65,loss65=flowering.fit_threshold_grid(flowering_c,a,stage_metadata,fixed[51],fixed[85])
        thresholds={**fixed,65:fitted65['threshold_65'],**{int(k):v for k,v in fitted_stage['thresholds'].items()}}
        fitted_stage['all_stage_thresholds']={str(k):v for k,v in sorted(thresholds.items())}
        write_json(folder/'stage_nuisance_fit.json',fitted_stage)
        write_json(folder/'flowering_nuisance_fit.json',fitted65)
        curve.to_csv(folder/'stage_joint_loss_profiles.csv',index=False)
        curve65.to_csv(folder/'flowering_grid.csv',index=False)
        loss65.to_csv(folder/'flowering_field_losses.csv')
        preprocessing=dict(weather_operator='duration_proxy',**estimate_training_rain_rate(training.metadata,weather))
        write_json(folder/'weather_preprocessing.json',dict(**preprocessing,
            training_fields=sorted(training_fields),training_sites=sorted(training.metadata.site_id.unique())))
        for diagnostic in fitted_stage['diagnostics']:
            exact=diagnostic['joint_profile_loss_tolerance_sets'][0]
            stage_counts.append(dict(fold=fold,event=diagnostic['event'],
                selected_threshold=diagnostic['selected_threshold'],
                constrained_fields=diagnostic['constrained_fields'],genuine_two_sided_fields=diagnostic['genuine_two_sided_fields'],
                joint_exact_minimum_grid_values=exact['n_grid_values'],joint_exact_minimum_threshold_low=exact['threshold_low'],
                joint_exact_minimum_threshold_high=exact['threshold_high'],poorly_identified=diagnostic['poorly_identified'],
                threshold_status='no genuine two-sided records; representative of censored-loss plateau' if not diagnostic['genuine_two_sided_fields'] else 'weakly identified conventional joint-grid representative'))
        btrain=symptom_brackets(training.targets)
        btest=symptom_brackets(truth.targets)
        fold_counts.append(dict(fold=fold,training_fields=len(training_fields),heldout_fields=truth.metadata.field_id.nunique(),
            training_assessments=len(training.targets),heldout_assessments=len(truth.targets),
            training_top3_two_sided_histories=int(btrain.censoring.eq('two_sided').sum()),
            heldout_top3_two_sided_histories=int(btest.censoring.eq('two_sided').sum()),
            heldout_top3_two_sided_fields=btest.loc[btest.censoring.eq('two_sided'),'field_id'].nunique(),
            rain_rate_mm_hour=preprocessing['rain_rate_mm_hour'],stage_training_fields=stage_metadata.field_id.nunique(),
            flowering_two_sided_histories=fitted65['n_bracketed_fields']))
        fields=np.sort(assembled.targets.iloc[heldout_rows].field_index.unique())
        for label,grid,is_benchmark in [('overwinter_source_model',settings,False),('phenology_only',benchmarks,True)]:
            for index,candidate in enumerate(grid):
                fitted=dict(**candidate,weather_preprocessing=preprocessing)
                trajectory,_=(disease.predict_benchmark(redacted,accumulation[fields],thresholds,fitted) if is_benchmark else
                    disease.predict_record(redacted,accumulation[fields],thresholds,fitted))
                ass=disease._assessment(truth,trajectory,label,'original_calibration_inner',candidate['name'])
                brackets=disease._brackets(truth,trajectory,label,'original_calibration_inner',candidate['name'])
                ass['inner_fold']=fold;brackets['inner_fold']=fold
                a_parts[candidate['name']].append(ass);b_parts[candidate['name']].append(brackets)
                if index%24==0:
                    print(f'{fold}: {label} candidate {index+1}/{len(grid)}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
    pd.DataFrame(fold_counts).to_csv(OUT/'fold_information_counts.csv',index=False)
    pd.DataFrame(stage_counts).to_csv(OUT/'fold_stage_identification.csv',index=False)
    selections=[]
    for label,grid in [('overwinter_source_model',settings),('phenology_only',benchmarks)]:
        records=[];all_a=[];all_b=[]
        for order,candidate in enumerate(grid):
            assessments=pd.concat(a_parts[candidate['name']],ignore_index=True)
            brackets=pd.concat(b_parts[candidate['name']],ignore_index=True)
            assert len(assessments)==254
            assert assessments.global_target_index.nunique()==254
            records.append(disease._score_candidate(assessments,brackets,candidate['name'],order))
            all_a.append(assessments);all_b.append(brackets)
        selected=disease.select_candidate(records)
        original=json.loads(sources['original_disease_selection' if label=='overwinter_source_model' else 'original_benchmark_selection'].read_text())
        chosen=next(item for item in grid if item['name']==selected['candidate'])
        record=dict(model=label,selected_candidate=chosen,selected_score=selected,candidate_scores=records,
            exact_primary_minimum_ties=disease.primary_tie_count(records,selected['primary_distance_days']),
            all_stage_and_weather_nuisance_fits_inside_inner_training=True,
            original_candidate=original['selected_candidate']['name'],original_pooled_selection_distance_days=original['selected_score']['primary_distance_days'],
            original_nuisance_stage_fit_outside_disease_inner_folds=True,reused_validation_outcomes_used=False,
            untouched_validation=False,diagnostic_only=True)
        write_json(OUT/f'{label}_nested_selection.json',record)
        pd.DataFrame([{k:v for k,v in r.items() if k!='assessment_detection'} for r in records]).to_csv(OUT/f'{label}_candidate_scores.csv',index=False)
        pd.concat(all_a,ignore_index=True).to_csv(OUT/f'{label}_inner_assessment_predictions.csv',index=False)
        pd.concat(all_b,ignore_index=True).to_csv(OUT/f'{label}_inner_bracket_predictions.csv',index=False)
        selections.append(dict(model=label,nested_selected_candidate=chosen['name'],
            nested_pooled_distance_days=selected['primary_distance_days'],nested_primary_ties=record['exact_primary_minimum_ties'],
            nested_pooled_balanced_assessment_error=selected['balanced_assessment_error'],
            original_candidate=record['original_candidate'],original_pooled_distance_days=record['original_pooled_selection_distance_days'],
            top3_two_sided_histories=selected['primary_brackets'],top3_two_sided_fields=selected['primary_fields']))
    # The diagnostic final fits are archived separately; no evaluation outcomes
    # are scored and no original/publication fit is replaced.
    fullstage,fullcurve=stage.fit_ordered_thresholds(constraints,a,meta,fixed)
    full65,fullcurve65,_=flowering.fit_threshold_grid(c65,a,meta,fixed[51],fixed[85])
    fullweather=dict(weather_operator='duration_proxy',**estimate_training_rain_rate(assembled.metadata,weather))
    write_json(OUT/'full_original_stage_fit.json',fullstage)
    write_json(OUT/'full_original_flowering_fit.json',full65)
    write_json(OUT/'full_original_weather_fit.json',fullweather)
    for selection in selections:
        record=json.loads((OUT/f"{selection['model']}_nested_selection.json").read_text())
        write_json(OUT/f"{selection['model']}_diagnostic_frozen_full_original_fit.json",dict(
            status='fit_complete',fitted=dict(**record['selected_candidate'],weather_preprocessing=fullweather),
            stage_fit='full_original_stage_fit.json',flowering_fit='full_original_flowering_fit.json',
            untouched_validation=False,reused_validation_outcomes_scored=False,diagnostic_only=True))
    pd.DataFrame(selections).to_csv(OUT/'nested_selection_comparison.csv',index=False)
    assert all(sha(ROOT/path)==digest for path,digest in hashes.items())
    write_json(OUT/'receipt.json',dict(status='complete',elapsed_seconds=time.perf_counter()-start,
        source_hashes_unchanged=True,source_sha256=hashes,original_only=True,nuisance_fits_fully_nested=True,
        calibration_fields=28,calibration_assessments=254,top3_two_sided_histories=8,
        external_outcomes_loaded_by_selection=False,external_scores_computed=False,
        original_fits_changed=False,comparison=selections,fold_counts=fold_counts,
        stage_threshold_identification_weak=True))
    print(pd.DataFrame(selections).to_string(index=False),flush=True)
    print(pd.DataFrame(fold_counts).to_string(index=False),flush=True)
    print(pd.DataFrame(stage_counts).to_string(index=False),flush=True)


if __name__=='__main__':main()
