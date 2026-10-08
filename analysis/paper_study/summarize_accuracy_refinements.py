"""Independently verify and summarize versioned severity refinement results."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'analysis/paper_study'
SOURCES = dict(v2=BASE/'severity_refinement_v2_20261006',
               v3=BASE/'severity_hybrid_v3_20261006')
DEFAULT = BASE/'accuracy_review_20261006_verified'


def independent_scores(frame, final, upper):
    """Explicit nested means, independent of the calibration weight helper."""
    frame = frame.loc[frame.leaf_index.lt(3)].copy() if upper else frame.copy()
    if final:
        frame = frame.sort_values('date').groupby(['field_id', 'endpoint_series']).tail(1)
    error = frame.predicted_percent-frame.value
    scored = frame.assign(squared_error=error**2, absolute_error=abs(error),
                          signed_error=error, observed_squared=frame.value**2)
    columns = ['squared_error', 'absolute_error', 'signed_error', 'value', 'observed_squared']
    leaf = scored.groupby(['coordinate_year', 'field_id', 'endpoint_series'])[columns].mean()
    field = leaf.groupby(level=['coordinate_year', 'field_id']).mean()
    cluster = field.groupby(level='coordinate_year').mean()
    result = cluster.mean()
    return dict(rmse=float(np.sqrt(result.squared_error)), mae=float(result.absolute_error),
        bias=float(result.signed_error), weighted_r2=float(1-result.squared_error/
            (result.observed_squared-result.value**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError('Use a new verification destination.')
    dest.mkdir(parents=True)
    checks, difference, hashes = 0, 0., {}
    comparisons, evaluation = [], []
    for version, source in SOURCES.items():
        frame = pd.read_parquet(source/'frozen_evaluation_predictions.parquet')
        metrics = pd.read_csv(source/'severity_metrics.csv')
        for row in metrics.itertuples():
            sample = frame[frame.partition.eq(row.partition)&frame.model.eq(row.model)]
            calculated = independent_scores(sample, row.endpoint == 'final_numeric_assessment',
                                             row.leaf_scope == 'upper_three')
            for name, value in calculated.items():
                delta = abs(value-getattr(row, name))
                difference = max(difference, delta)
                checks += 1
                if delta > 1e-9:
                    raise ValueError(f'{version}: {name} failed independent verification.')
        if version == 'v2':
            snapshot = source/'source_snapshot/refinement.py'
            config = json.loads((source/'configuration_before_fitting.json').read_text())
            expected = config['input_and_source_sha256']['model/seasonal_septoria/refinement.py']
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
                raise ValueError('V2 source snapshot does not match its frozen calibration contract.')
            checks += 1
        # Standalone delivery labels identify reused evidence for both versions.
        metrics['partition'] = metrics.partition.replace({'validation':'reused_development_2019'})
        metrics['evidence_role'] = np.where(metrics.partition.eq('calibration'),
            'in_sample_fit', 'reused_development_evaluation')
        metrics['version'] = version
        evaluation.append(metrics)
        paired = pd.read_csv(source/'paired_improvement_intervals.csv')
        paired['partition'] = paired.partition.replace({'validation':'reused_development_2019'})
        paired['version'] = version
        paired['evidence_role'] = np.where(paired.partition.eq('calibration'),
            'in_sample_fit', 'reused_development_evaluation')
        comparisons.append(paired)
        for name in ['receipt.json', 'frozen_before_evaluation.json', 'severity_metrics.csv',
                     'frozen_evaluation_predictions.parquet', 'paired_improvement_intervals.csv']:
            path = source/name
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    complete = pd.concat(evaluation, ignore_index=True)
    complete.to_csv(dest/'verified_severity_metrics.csv', index=False)
    paired = pd.concat(comparisons, ignore_index=True)
    paired.to_csv(dest/'verified_paired_comparisons.csv', index=False)
    raw_onset = pd.read_csv(SOURCES['v2']/'onset_interval_predictions.csv')
    raw_onset['partition'] = raw_onset.partition.replace({'validation':'reused_development_2019'})
    onset_summary = []
    for labels, group in raw_onset.groupby(['partition', 'model', 'cutoff_percent', 'censoring']):
        rates = group.groupby(['coordinate_year', 'field_id']).compatible.mean().groupby('coordinate_year').mean()
        incompatible = group.loc[~group.compatible, 'delta_days'].dropna()
        onset_summary.append(dict(partition=labels[0], model=labels[1], cutoff_percent=labels[2],
            censoring=labels[3], leaf_series=len(group), coordinate_years=len(rates),
            coordinate_year_compatibility=float(rates.mean()),
            median_incompatible_signed_distance_days=None if incompatible.empty else float(incompatible.median()),
            evidence_role='in_sample_fit' if labels[0]=='calibration' else 'reused_development_evaluation'))
    onset = pd.DataFrame(onset_summary)
    onset.to_csv(dest/'verified_onset_metrics.csv', index=False)
    final = complete.loc[complete.endpoint.eq('final_numeric_assessment')
                         & complete.leaf_scope.eq('all_ordinal_leaves')]
    final = final.drop_duplicates(['partition', 'model']).copy()
    final.to_csv(dest/'final_severity_comparison.csv', index=False)
    # Reproduction of the original frozen predictions is an independent check.
    original = [
        ('BASF', BASE/'seasonal_calibration_v1/basf_frozen_predictions.parquet'),
        ('Corteva', BASE/'corteva_external_seasonal_v1/frozen_external_predictions.parquet')]
    new = pd.read_parquet(SOURCES['v2']/'frozen_evaluation_predictions.parquet')
    for source, path in original:
        old = pd.read_parquet(path).query("model == 'seasonal_seir'")
        baseline = new[new.source.eq(source)&new.model.eq('reference_v1')]
        keys = ['field_id', 'endpoint_series', 'date']
        matched = baseline[keys+['predicted_percent']].merge(old[keys+['predicted_percent']],
            on=keys, validate='one_to_one', how='outer', suffixes=('_new', '_old'))
        if matched.isna().any().any() or len(matched) != len(old):
            raise ValueError('Original frozen evaluation membership changed.')
        delta = float(abs(matched.predicted_percent_new-matched.predicted_percent_old).max())
        checks += 1; difference = max(difference, delta)
        if delta > 1e-9:
            raise ValueError('Reference prediction does not reproduce the original archive.')
    cv_v1 = json.loads((SOURCES['v2']/'receipt.json').read_text())['reference_cv_final_rmse']
    cv_v2 = json.loads((SOURCES['v2']/'receipt.json').read_text())['selected_cv_final_rmse']
    cv_v3 = json.loads((SOURCES['v3']/'receipt.json').read_text())['selection']['rmse']
    names = ['Original process', 'Final-assessment fit', 'Weather/process blend']
    models = ['reference_v1', 'selected_refinement_v2', 'selected_hybrid_v3']
    color = ['#24445C', '#B86B32', '#657D56']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={'width_ratios':[1, 1.8]})
    axes[0].bar(np.arange(3), [cv_v1, cv_v2, cv_v3], color=color, width=.65)
    axes[0].set_xticks(np.arange(3), ['Original', 'Final fit', 'Blend'])
    axes[0].set_ylabel('Final leaf-severity RMSE (percentage points)')
    axes[0].set_title('Calibration location-fold selection')
    axes[0].set_ylim(0, 42)
    for i, value in enumerate([cv_v1, cv_v2, cv_v3]):
        axes[0].text(i, value+.7, f'{value:.2f}', ha='center', fontsize=9)
    x = np.arange(2)
    partitions = ['reused_development_2019', 'reused_external_strict']
    for index, (model, name, shade) in enumerate(zip(models, names, color)):
        values = [float(final.loc[final.partition.eq(partition)&final.model.eq(model), 'rmse'].iloc[0])
                  for partition in partitions]
        positions = x+(index-1)*.24
        axes[1].bar(positions, values, width=.23, label=name, color=shade)
        for position, value in zip(positions, values):
            axes[1].text(position, value+.7, f'{value:.2f}', ha='center', fontsize=9)
    axes[1].set_xticks(x, ['BASF 2019\n45 field-seasons', 'Location-disjoint Corteva\n143 field-seasons'])
    axes[1].set_title('Reused development evaluation')
    axes[1].set_ylim(0, 42)
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.16), frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.text(.02, .02, 'Previously examined evaluation sources; these results are not an untouched test.', fontsize=9)
    fig.tight_layout(rect=[0, .08, 1, 1])
    fig.savefig(dest/'accuracy_comparison.png', dpi=180)
    fig.savefig(dest/'accuracy_comparison.pdf')
    plt.close(fig)
    def result(model, partition):
        return final[final.model.eq(model)&final.partition.eq(partition)].iloc[0]
    v1_ext, v2_ext, v3_ext = [result(model, 'reused_external_strict') for model in models]
    v1_year, v2_year, v3_year = [result(model, 'reused_development_2019') for model in models]
    interval = paired[paired.version.eq('v2')&paired.partition.eq('reused_external_strict')
        &paired.endpoint.eq('final_numeric_assessment')&paired.leaf_scope.eq('all_ordinal_leaves')].iloc[0]
    def onset_rate(model, cutoff):
        return 100*float(onset.loc[onset.model.eq(model)&onset.partition.eq('reused_external_strict')
            &onset.censoring.eq('interval')&onset.cutoff_percent.eq(cutoff), 'coordinate_year_compatibility'].iloc[0])
    report = f'''The original season-start process model remains the reference predictor. Neither calibration refinement established an improvement across both reused evaluation sources. Final leaf severity is the primary selection target; visible onset is a separate process diagnostic.

The final-assessment calibration fixes the secondary coefficient at zero, retains the 30-day reference latency and direct observation operator, and estimates the external-pressure coefficient from complete calibration field-seasons. Three location-grouped folds of the 2017–2018 BASF calibration sample selected this variant among twelve declared candidates. Calibration cross-validation RMSE decreased from {cv_v1:.2f} to {cv_v2:.2f} percentage points. The selected external-pressure coefficient is 0.085814 day⁻¹. Greater observation-mapping and latency flexibility did not improve calibration cross-validation sufficiently to select those variants.

In the location-disjoint Corteva sample, final leaf-severity RMSE decreased from {v1_ext.rmse:.2f} to {v2_ext.rmse:.2f} percentage points, a {100*(v1_ext.rmse-v2_ext.rmse)/v1_ext.rmse:.1f}% reduction. The paired RMSE difference was {interval.rmse_difference:.2f} percentage points, with a coordinate-year bootstrap 95% interval of {interval.rmse_difference_lower:.2f} to {interval.rmse_difference_upper:.2f}. MAE decreased from {v1_ext.mae:.2f} to {v2_ext.mae:.2f}, and signed bias improved from {v1_ext.bias:.2f} to {v2_ext.bias:.2f} percentage points. Weighted R² increased from {v1_ext.weighted_r2:.3f} to {v2_ext.weighted_r2:.3f}. Absolute severity error remains substantial. In the reused BASF 2019 sample, RMSE increased from {v1_year.rmse:.2f} to {v2_year.rmse:.2f}, and MAE increased from {v1_year.mae:.2f} to {v2_year.mae:.2f} percentage points.

The final-assessment calibration worsened onset timing. Among informative external 0.1% visibility intervals, equal-coordinate-year compatibility decreased from {onset_rate('reference_v1', .1):.1f}% to {onset_rate('selected_refinement_v2', .1):.1f}%. Corresponding compatibility at 1% and 5% decreased from {onset_rate('reference_v1', 1.):.1f}% and {onset_rate('reference_v1', 5.):.1f}% to {onset_rate('selected_refinement_v2', 1.):.1f}% and {onset_rate('selected_refinement_v2', 5.):.1f}%. These onset rates give equal weight to coordinate-years and then fields, with equal leaf-series weights within fields. The external-severity gain therefore supports an experimental endpoint-specific predictor, without establishing improved disease timing or general temporal transfer.

The regularized weather/process blend used causal weather and crop-development features with calibration-only fitting, standardization, penalty selection and blend selection. Its selected weights were 75% weighted ridge and 25% final-assessment process prediction, with ridge penalty 10 and final-assessment training loss. Calibration cross-validation RMSE decreased to {cv_v3:.2f} percentage points, whereas reused BASF 2019 and location-disjoint Corteva RMSEs were {v3_year.rmse:.2f} and {v3_ext.rmse:.2f}. The larger in-sample fit improvement did not transfer. The blend does not define an onset event; its archived onset diagnostics retain the original process model.

Model parameters and settings were frozen before newly calculated evaluation scores. The 2019 and Corteva sources had been examined in earlier development, including the preceding refinement cycle. These are reused development evaluations and cannot establish untouched test performance. Confidence intervals describe cluster resampling conditional on the selected models and these data; they exclude model-selection uncertainty. The original manuscript and regional climate projections retain their original fitted model. An unqualified replacement of that model is unsupported by these results.
'''
    (dest/'Accuracy_results.txt').write_text(report)
    (dest/'Reproduction_README.txt').write_text('''Runtime: project .venv/bin/python, from the project root.

Final-assessment refinement:
.venv/bin/python analysis/paper_study/refine_seasonal_accuracy.py --output analysis/paper_study/new_severity_refinement

Hybrid comparison (uses frozen v1 and v2 out-of-fold process predictions):
.venv/bin/python analysis/paper_study/refine_seasonal_hybrid.py --output analysis/paper_study/new_severity_hybrid

Independent arithmetic and reference reproduction:
.venv/bin/python analysis/paper_study/summarize_accuracy_refinements.py --output analysis/paper_study/new_accuracy_review

Regression suite:
.venv/bin/python -m pytest -q

calibration/seasonal_septoria/refinement.py provides fit_refinement and predict_refinement. Frozen experimental v2 parameters are in severity_refinement_v2_20261006/frozen_selected_fit.json. The v3 joblib bundle contains the weighted ridge estimator, ordered feature names, empirical weight, process fit and separate original onset fit. The source snapshots and recorded hashes identify the executed implementations; later live-source revisions may differ.

The original raw v2 archive retains its original partition labels. The verified delivery tables explicitly label the 2019 sample reused_development_2019. Numerical predictions are unchanged by that evidence-label amendment. V1 remains the default scientific reference; v2 is an experimental final-severity candidate and v3 is an unsuccessful transfer experiment.
''')
    receipt = dict(status='passed', independent_arithmetic_checks=checks,
        maximum_absolute_difference=difference, original_predictions_reproduced=True,
        evaluation_role_labels_verified=True, selected_for_default='reference_v1',
        untouched_test_evaluated=False, input_sha256=hashes)
    (dest/'independent_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
