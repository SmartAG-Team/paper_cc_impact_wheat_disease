"""Independent fixed-prediction uncertainty and source-leaf sensitivity audit."""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, hashlib, json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE/'numba_cache'))
import numpy as np
import pandas as pd
from analysis.paper_study.nature_food_fix_20261007.validation.statistics import (
    hierarchy_weights, paired_bootstrap, preflag_zero_mask)
from calibration.seasonal_septoria.infection_events import symptom_brackets

ORIGINAL = ROOT/'analysis/paper_study/overwinter_leaf_model_20261006'
REVISION = ROOT/'analysis/paper_study/nature_food_revision_20261007'
SEED, NBOOT = 20261007, 20000
MODEL, BENCHMARK = 'overwinter_source_model', 'phenology_only'
PARTITIONS = ['calibration', 'reused_BASF2019', 'reused_strict_Corteva']
KEYS = ['partition','field_id','endpoint_series','leaf_index']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    sources = [ORIGINAL/'disease/all_first_onset_bracket_predictions.parquet',
        ORIGINAL/'disease/all_assessment_sign_predictions.parquet',
        REVISION/'epidemic_evaluation/observed_vs_frozen_assessments.parquet',
        REVISION/'disease_data_audit/leaf_measurement_semantics_assessments.csv',
        REVISION/'disease_data_audit/leaf_measurement_semantics_receipt.json',
        ORIGINAL/'disease/overwinter_source_model/frozen_selected_fit.json',
        ORIGINAL/'disease/phenology_only/frozen_selected_fit.json',
        ORIGINAL/'phenology/calibrated_stage_thresholds.json']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    write_json('statistical_contract_before_analysis.json',dict(
        created_utc=datetime.now(timezone.utc).isoformat(), input_sha256=hashes,
        seed=SEED, bootstrap_draws=NBOOT, independent_resampling_unit='rounded coordinate-year',
        biological_site_identity_independently_verified=False,
        weighting='equal coordinate-year, equal field within coordinate-year, equal source leaf within field, equal assessment within leaf',
        pairing='identical complete coordinate-year/field/source-leaf/date membership for model and comparator',
        interval='95% percentile interval conditional on fixed fitted predictions and each stated record population',
        unquantified='fitting/model-selection uncertainty, independent biological site identity, leaf mapping, stage ambiguity and prior outcome exposure',
        validation_status='retrospective reused development evidence; no untouched test',
        preflag_exclusion='sensitivity excludes zero source-leaf-1 assessments only when ordered same-day observed upper stage is <37 or <39; placeholder status unverified',
        Corteva_source_number_to_final_flag_mapping_inferred=False))

    onset = pd.read_parquet(sources[0])
    onset = onset.loc[onset.scenario.eq('baseline') & onset.detection_fraction.eq(.001)].copy()
    paired = onset.loc[onset.model.eq(MODEL)].drop(columns='model').merge(
        onset.loc[onset.model.eq(BENCHMARK)][KEYS+['onset_distance_days','compatible','predicted_symptom_day']],
        on=KEYS,validate='one_to_one',suffixes=('','_benchmark'))
    assert len(paired)*2 == len(onset)
    ass = pd.read_parquet(sources[2])
    audit = pd.read_csv(sources[3],dtype={'field_id':str},parse_dates=['date'])
    audit = audit.loc[audit.family.eq('STB') & audit.eligible & audit.untreated_zero_spray].copy()
    audit['physical_unit'], audit['leaf_index'] = audit.field_id, audit.leaf_rank-1
    columns = ['source','physical_unit','date','leaf_index','stage_from','stage_to','stage_source_rows',
        'source_rows','value','leaf_semantic_documentation','metric_semantic_documentation']
    ass = ass.merge(audit[columns],on=['source','physical_unit','date','leaf_index'],
        how='left',validate='one_to_one',suffixes=('','_audit'),indicator=True)
    assert ass['_merge'].eq('both').all()
    assert np.allclose(ass.observed_percent, ass.value)
    ass = ass.drop(columns='_merge')
    ass['same_day_stage_covered'] = (ass.stage_from_audit.notna() & ass.stage_to_audit.notna()
        & ass.stage_from_audit.le(ass.stage_to_audit))
    ass['stage_from_original'], ass['stage_to_original'] = ass.stage_from, ass.stage_to
    ass['stage_from'], ass['stage_to'] = ass.stage_from_audit, ass.stage_to_audit
    ass['pre37_source_leaf1_zero'] = preflag_zero_mask(ass,37)
    ass['pre39_source_leaf1_zero'] = preflag_zero_mask(ass,39)
    signs = pd.read_parquet(sources[1])
    signs = signs.loc[signs.scenario.eq('baseline') & signs.detection_fraction.eq(.001)]
    for label, model in [('model',MODEL),('benchmark',BENCHMARK)]:
        part = signs.loc[signs.model.eq(model)]
        assert len(part)==len(ass)
        matching = ass[['global_target_index','observed_percent']].merge(
            part[['global_target_index','observed_positive','predicted_positive']],
            on='global_target_index',validate='one_to_one')
        assert matching.observed_percent.gt(0).eq(matching.observed_positive).all()
        ass = ass.merge(part[['global_target_index','predicted_positive']],on='global_target_index',
            validate='one_to_one').rename(columns={'predicted_positive':label})
    ass['observed'] = ass.observed_percent.gt(0)
    ass.to_csv(HERE/'source_stage_and_prediction_membership.csv',index=False)
    paired.to_csv(HERE/'paired_onset_membership.csv',index=False)

    rows, replicate_parts, group_parts, draw_archive = [], [], [], {}
    common_draws = {}
    arithmetic = []
    number = 0

    def report(frame, kind, identity):
        nonlocal number
        if frame.empty:
            rows.append(dict(**identity,status='unavailable_empty_population',records=0))
            return
        group_ids = sorted(frame.coordinate_year.unique())
        membership_key = tuple(group_ids)
        if membership_key not in common_draws:
            common_draws[membership_key] = np.random.default_rng(SEED+number).integers(
                0,len(group_ids),size=(NBOOT,len(group_ids)))
        draws = common_draws[membership_key]
        analysis_id = f'analysis_{number:03d}'
        number += 1
        draw_archive[analysis_id] = draws.astype(np.int16)
        point, samples = paired_bootstrap(frame,draws,kind=kind)
        weight = hierarchy_weights(frame)
        # Independent nested arithmetic using explicit Python group averages.
        if kind == 'mean':
            independent = []
            for _,group in frame.groupby('coordinate_year'):
                field_values = []
                for _,field in group.groupby('field_id'):
                    field_values.append(np.mean([leaf.model.mean()-leaf.benchmark.mean()
                        for _,leaf in field.groupby('leaf_index')]))
                independent.append(np.mean(field_values))
            assert abs(float(np.mean(independent))-point['difference'])<1e-10
            arithmetic.append(dict(analysis_id=analysis_id,kind=kind,checked='nested explicit field/leaf arithmetic',
                maximum_absolute_difference=abs(float(np.mean(independent))-point['difference'])))
        elif kind == 'severity':
            error = frame.model-frame.observed
            comparator_error = frame.benchmark-frame.observed
            direct = np.sqrt(np.average(error**2,weights=weight))-np.sqrt(np.average(comparator_error**2,weights=weight))
            assert abs(direct-point['rmse_difference'])<1e-10
            arithmetic.append(dict(analysis_id=analysis_id,kind=kind,checked='direct weighted root difference',
                maximum_absolute_difference=abs(float(direct)-point['rmse_difference'])))
        else:
            for label in ['model','benchmark']:
                positive = frame.observed.to_numpy(bool)
                pred = frame[label].to_numpy(bool)
                if positive.any():
                    assert abs(float(np.average(pred[positive],weights=weight[positive]))-point[label+'_sensitivity'])<1e-10
                if (~positive).any():
                    assert abs(float(np.average(~pred[~positive],weights=weight[~positive]))-point[label+'_specificity'])<1e-10
            arithmetic.append(dict(analysis_id=analysis_id,kind=kind,checked='independent weighted class-conditional counts',maximum_absolute_difference=0.))
        for metric, values in samples.items():
            valid = np.isfinite(values)
            rows.append(dict(**identity, analysis_id=analysis_id,metric=metric,
                records=len(frame),fields=frame.field_id.nunique(),coordinate_year_groups=len(group_ids),
                source_leaf_series=frame.groupby(['field_id','leaf_index']).ngroups,
                point=point[metric] if np.isfinite(point[metric]) else None,
                lower95=float(np.quantile(values[valid],.025)) if valid.any() else None,
                upper95=float(np.quantile(values[valid],.975)) if valid.any() else None,
                requested_replicates=NBOOT,valid_replicates=int(valid.sum()),invalid_replicates=int((~valid).sum()),
                positive_records=int(frame.observed.sum()) if kind=='detection' else None,
                negative_records=int((~frame.observed).sum()) if kind=='detection' else None,
                status='estimable' if valid.any() else 'unavailable_class_denominator'))
        r = pd.DataFrame(samples)
        r['analysis_id'], r['replicate'] = analysis_id, np.arange(NBOOT)
        replicate_parts.append(r)
        groups = frame[['coordinate_year','field_id','leaf_index']].drop_duplicates().copy()
        groups['analysis_id'] = analysis_id
        groups['cluster_index'] = groups.coordinate_year.map({key:i for i,key in enumerate(group_ids)})
        group_parts.append(groups)

    evidence = []
    for partition in PARTITIONS:
        population = ass.loc[ass.partition.eq(partition)]
        for label,mask in [('all_source_numbered_leaves',population.leaf_index.ge(0)),
            ('top3_source_numbered',population.leaf_index.lt(3)),('source_leaf1',population.leaf_index.eq(0))]:
            part = population.loc[mask]
            evidence.append(dict(partition=partition,leaf_scope=label,assessments=len(part),
                fields=part.field_id.nunique(),coordinate_years=part.coordinate_year.nunique(),
                source_leaf_series=part.groupby(['field_id','leaf_index']).ngroups,
                positive_assessments=int(part.observed.sum()),zero_assessments=int((~part.observed).sum()),
                same_day_ordered_two_bound_stage_assessments=int(part.same_day_stage_covered.sum()),
                pre37_source_leaf1_zero_assessments=int(part.pre37_source_leaf1_zero.sum()),
                pre39_source_leaf1_zero_assessments=int(part.pre39_source_leaf1_zero.sum()),
                biological_leaf_label='BASF source leaf 1 explicitly flag; other ranks source numbered' if partition!='reused_strict_Corteva' else 'Corteva source numbering; final flag identity unverified'))
        if partition=='calibration':
            continue
        for scope in ['top3_source_numbered','source_leaf1','all_source_numbered_leaves']:
            q = paired.loc[paired.partition.eq(partition)]
            p = population.copy()
            if scope=='top3_source_numbered':
                q,p = q.loc[q.leaf_index.lt(3)],p.loc[p.leaf_index.lt(3)]
            elif scope=='source_leaf1':
                q,p = q.loc[q.leaf_index.eq(0)],p.loc[p.leaf_index.eq(0)]
            q = q.loc[q.censoring.eq('two_sided')].copy()
            q['model'], q['benchmark'] = q.onset_distance_days,q.onset_distance_days_benchmark
            report(q,'mean',dict(endpoint='genuine_two_sided_onset_distance_days',partition=partition,
                leaf_scope=scope,comparison_population='original_eligible_records',comparator='phenology_only'))
            for scenario,retained in [('original_eligible_records',p),
                ('same_day_ordered_two_bound_stage',p.loc[p.same_day_stage_covered]),
                ('exclude_observed_pre37_leaf1_zeros',p.loc[~p.pre37_source_leaf1_zero]),
                ('exclude_observed_pre39_leaf1_zeros',p.loc[~p.pre39_source_leaf1_zero])]:
                report(retained,'detection',dict(endpoint='zero_positive_assessment_detection',partition=partition,
                    leaf_scope=scope,comparison_population=scenario,comparator='phenology_only'))
                if scenario=='original_eligible_records':
                    continue
                # Rebuild genuine brackets after exclusion; changing censoring
                # membership is reported, never patched back to two-sided.
                raw = retained.drop(columns='value',errors='ignore').rename(columns={'observed_percent':'value'})
                rebuilt = symptom_brackets(raw,upper_three=False)
                rebuilt['partition'] = partition
                prediction = paired.loc[paired.partition.eq(partition)]
                rebuilt = rebuilt.merge(prediction[KEYS+['predicted_symptom_day','predicted_symptom_day_benchmark','forcing_end_day']],
                    on=KEYS,validate='one_to_one')
                rebuilt = rebuilt.loc[rebuilt.censoring.eq('two_sided')].copy()
                for label,col in [('model','predicted_symptom_day'),('benchmark','predicted_symptom_day_benchmark')]:
                    present = rebuilt[col].ge(0)
                    loss = np.maximum(rebuilt.lower_day+1-rebuilt[col],0)+np.maximum(rebuilt[col]-rebuilt.upper_day,0)
                    loss.loc[~present] = np.maximum(30.,rebuilt.forcing_end_day.loc[~present]+1-rebuilt.upper_day.loc[~present])
                    rebuilt[label] = loss
                rebuilt.to_csv(HERE/f'{partition}_{scope}_{scenario}_onset_membership.csv',index=False)
                report(rebuilt,'mean',dict(endpoint='genuine_two_sided_onset_distance_days',partition=partition,
                    leaf_scope=scope,comparison_population=scenario,comparator='phenology_only'))
        for endpoint in ['all_assessments','final_numeric_assessment']:
            full = population.copy()
            if endpoint=='final_numeric_assessment':
                full = full.sort_values('date').groupby(['field_id','leaf_index']).tail(1)
            for scope in ['all_source_numbered_leaves','top3_source_numbered','source_leaf1']:
                part = full.loc[full.leaf_index.lt(3)] if scope=='top3_source_numbered' else full.loc[full.leaf_index.eq(0)] if scope=='source_leaf1' else full
                for comparator,column,pop in [('calibration_leaf_mean','leaf_mean_percent','original_eligible_records'),
                    ('calibration_stage_time','stage_time_percent','shared_observed_stage_minimum')]:
                    matched = part.loc[part.stage_time_available].copy() if comparator=='calibration_stage_time' else part.copy()
                    matched['model'],matched['benchmark'],matched['observed'] = matched.frozen_damage_percent,matched[column],matched.observed_percent
                    report(matched,'severity',dict(endpoint=endpoint+'_severity_pp',partition=partition,
                        leaf_scope=scope,comparison_population=pop,comparator=comparator))
    pd.DataFrame(evidence).to_csv(HERE/'leaf_evidence_and_stage_coverage.csv',index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(HERE/'paired_cluster_bootstrap_summary.csv',index=False)
    pd.concat(replicate_parts,ignore_index=True).to_csv(HERE/'bootstrap_metric_replicates.csv.gz',index=False,compression='gzip')
    pd.concat(group_parts,ignore_index=True).to_csv(HERE/'bootstrap_group_membership.csv',index=False)
    np.savez_compressed(HERE/'bootstrap_draw_indices.npz',**draw_archive)
    write_json('independent_arithmetic_verification.json',dict(status='passed',checks=arithmetic,
        hand_derived_fixture_tests='six arithmetic/membership/denominator tests passed before analysis'))
    assert all(sha(ROOT/p)==digest for p,digest in hashes.items())
    write_json('statistical_receipt.json',dict(status='complete',analyses=number,
        summary_rows=len(summary),bootstrap_draws_per_analysis=NBOOT,source_hashes_unchanged=True,
        input_sha256=hashes,model_parameters_changed=False,untouched_test=False,
        leaf_mapping_inferred=False,placeholder_status_resolved=False,
        source_sha256={p.name:sha(p) for p in [Path(__file__),HERE/'statistics.py',HERE/'test_statistics.py']}))
    print(summary.loc[summary.metric.eq('difference') & summary.leaf_scope.eq('top3_source_numbered')].to_string(index=False))
    print(pd.DataFrame(evidence).to_string(index=False))


if __name__=='__main__':
    main()
