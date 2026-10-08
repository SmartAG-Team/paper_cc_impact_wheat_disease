"""Independent verification of source hashes, resampling records and nesting."""
from pathlib import Path
import sys,hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from analysis.paper_study.nature_food_fix_20261007.validation.exact_bootstrap import exact_mean_bootstrap


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    stats=json.loads((HERE/'statistical_receipt.json').read_text())
    nested=json.loads((HERE/'nested_calibration/receipt.json').read_text())
    for path,digest in {**stats['input_sha256'],**nested['source_sha256']}.items():
        assert sha(ROOT/path)==digest,path
    for name,digest in stats['source_sha256'].items():
        assert sha(HERE/name)==digest,name
    summary=pd.read_csv(HERE/'paired_cluster_bootstrap_summary.csv')
    groups=pd.read_csv(HERE/'bootstrap_group_membership.csv')
    draws=np.load(HERE/'bootstrap_draw_indices.npz')
    for identity,group in summary.groupby('analysis_id'):
        membership=groups.loc[groups.analysis_id.eq(identity)]
        matrix=draws[identity]
        assert matrix.shape==(20000,int(group.coordinate_year_groups.iloc[0]))
        assert membership.cluster_index.nunique()==matrix.shape[1]
        assert set(membership.cluster_index)==set(range(matrix.shape[1]))
    primary=summary.loc[summary.endpoint.eq('genuine_two_sided_onset_distance_days')
        &summary.leaf_scope.eq('top3_source_numbered') &summary.comparison_population.eq('original_eligible_records')
        &summary.metric.eq('difference')]
    original=pd.read_csv(HERE/'paired_onset_membership.csv')
    b=original.loc[original.partition.eq('reused_BASF2019') & original.leaf_index.lt(3)
        &original.censoring.eq('two_sided')].copy()
    group_means=[]
    for _,coordinate in b.groupby('coordinate_year'):
        fields=[]
        for _,field in coordinate.groupby('field_id'):
            fields.append(np.mean([leaf.onset_distance_days.mean()-leaf.onset_distance_days_benchmark.mean()
                for _,leaf in field.groupby('leaf_index')]))
        group_means.append(float(np.mean(fields)))
    assert group_means==[10.,-1.,2.,11.,-12.,1.5,1.5,3.5]
    exact,probability=exact_mean_bootstrap(group_means)
    cumulative=probability.cumsum()
    assert abs(probability.sum()-1)<1e-12
    limits=exact[np.searchsorted(cumulative,[.025,.975])]
    assert np.allclose(limits,[-2.75,6.3125])
    assert abs(exact@probability-2.0625)<1e-12
    pd.DataFrame(dict(mean_model_minus_benchmark_days=exact,probability=probability,
        cumulative_probability=cumulative)).to_csv(HERE/'exact_BASF_onset_bootstrap_distribution.csv',index=False)
    counts={}
    baseline_replicates=[]
    # Reading to EOF also verifies gzip integrity after serial artifact creation.
    for part in pd.read_csv(HERE/'bootstrap_metric_replicates.csv.gz',
            usecols=['analysis_id','replicate','difference'],chunksize=50000):
        for key,n in part.analysis_id.value_counts().items():counts[key]=counts.get(key,0)+int(n)
        q=part.loc[part.analysis_id.eq('analysis_000')]
        if len(q):baseline_replicates.append(q)
    assert len(counts)==stats['analyses']
    assert all(n==20000 for n in counts.values())
    mc=pd.concat(baseline_replicates).sort_values('replicate')
    assert len(mc)==20000
    independently_sampled=np.array(group_means)[draws['analysis_000']].mean(axis=1)
    np.testing.assert_allclose(mc.difference,independently_sampled,atol=1e-12,rtol=0)
    assert np.allclose(np.quantile(mc.difference,[.025,.975]),limits)
    # Identical record exclusions with no removed BASF rows must have exactly
    # identical paired bootstrap draws and interval endpoints.
    same=summary.loc[summary.partition.eq('reused_BASF2019') &summary.leaf_scope.eq('top3_source_numbered')
        &summary.endpoint.eq('genuine_two_sided_onset_distance_days') &summary.metric.eq('difference')]
    assert same.point.nunique()==same.lower95.nunique()==same.upper95.nunique()==1
    fold=pd.read_csv(HERE/'nested_calibration/original_inner_membership.csv')
    stage_counts=pd.read_csv(HERE/'nested_calibration/fold_stage_identification.csv')
    for key,plan in fold.groupby('fold'):
        training=set(plan.loc[plan.role.eq('train'),'field_id'])
        holdout=set(plan.loc[plan.role.eq('validation'),'field_id'])
        training_sites=set(plan.loc[plan.role.eq('train'),'site_id'])
        holdout_sites=set(plan.loc[plan.role.eq('validation'),'site_id'])
        assert training.isdisjoint(holdout) and training_sites.isdisjoint(holdout_sites)
        for name in ['stage_training_membership_before_fitting.csv','flowering_training_membership_before_fitting.csv']:
            nuisance=pd.read_csv(HERE/'nested_calibration'/key/name)
            assert set(nuisance.field_id)<=training
            assert 'value' not in nuisance
    original_stage=json.loads((ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/phenology/calibrated_stage_thresholds.json').read_text())
    full_stage=json.loads((HERE/'nested_calibration/full_original_stage_fit.json').read_text())
    full65=json.loads((HERE/'nested_calibration/full_original_flowering_fit.json').read_text())
    assert original_stage['thresholds']==full_stage['thresholds']
    assert original_stage['all_stage_thresholds']['65']==full65['threshold_65']
    for comparison in nested['comparison']:
        assert comparison['nested_selected_candidate']==comparison['original_candidate']
    result=dict(status='passed',input_and_source_hashes_unchanged=True,
        bootstrap_analyses=stats['analyses'],replicate_rows=sum(counts.values()),
        gzip_eof_and_integrity_verified=True,bootstrap_membership_indices_verified=True,
        independent_BASF_direct_draw_arithmetic_max_difference=float(np.max(abs(mc.difference-independently_sampled))),
        exact_BASF_onset_bootstrap=dict(coordinate_years=8,histories=13,
            ordered_draw_sequences=8**8,multinomial_count_configurations=6435,
            point_difference=2.0625,lower95=float(limits[0]),upper95=float(limits[1]),
            exact_probability_mass=float(probability.sum())),
        nested_stage_and_weather_membership_verified=True,
        full_original_stage_refit_matches_frozen_thresholds=True,
        full_original_flowering_refit_matches_frozen_threshold=True,
        nested_selected_settings_match_frozen_model_and_benchmark=True,
        zero_genuine_stage_bracket_records=stage_counts.loc[stage_counts.genuine_two_sided_fields.eq(0)].to_dict(orient='records'),
        untouched_test=False,
        bootstrap_conditions='fixed predictions, approximate rounded coordinate-year grouping, stated observed records; parameter and selection uncertainty excluded',
        executable_sha256={p.name:sha(p) for p in [Path(__file__),HERE/'exact_bootstrap.py']})
    (HERE/'final_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
