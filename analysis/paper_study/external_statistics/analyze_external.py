#!/usr/bin/env python3
"""Frozen external prediction analysis; no model imports, fits or selection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/paper_study/external_statistics'
SOURCE=ROOT/'analysis/paper_study/corteva_external_seasonal_v1'
CAL=ROOT/'analysis/paper_study/seasonal_calibration_v1'
SEED=20261006
B=10000
MODELS=['seasonal_seir','calibration_only_leaf_rank_mean']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(predictions,geography,leaves,sensitivity,endpoint):
    q=predictions.copy()
    if geography=='strict_location_disjoint':q=q[q.strict_location_disjoint]
    if leaves=='upper_three':q=q[q.leaf_index.lt(3)]
    if sensitivity=='exclude_2015_58_oct30':q=q[~(q.source_unit.eq('2015-58')&q.date.eq(pd.Timestamp('2015-10-30')))]
    if endpoint=='final_numeric_assessment':q=q.sort_values('date').groupby(['model','field_id','endpoint_series']).tail(1)
    return q.copy()


def weights(q,weighting,endpoint):
    if weighting=='archived_selected_weights' and endpoint=='all_assessments':return q.weight.to_numpy()
    nfields=q.groupby('coordinate_year').field_id.transform('nunique')
    nleaf=q.groupby('field_id').endpoint_series.transform('nunique')
    ndates=q.groupby(['field_id','endpoint_series']).value.transform('size')
    result=1/nleaf/ndates
    if weighting!='literal_equal_field':result=result/nfields
    return result.to_numpy()


def main():
    HERE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    files=[p for p in SOURCE.iterdir() if p.is_file()]+[CAL/name for name in ['frozen_main_fit.json','frozen_primary_only_fit.json','frozen_leaf_rank_baseline.json','field_season_membership.csv']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    archive_receipt=json.loads((SOURCE/'receipt.json').read_text())
    for name,expected in archive_receipt['frozen_parameters'].items():assert sha(CAL/name)==expected
    predictions=pd.read_parquet(SOURCE/'frozen_external_predictions.parquet');predictions.date=pd.to_datetime(predictions.date)
    assert len(predictions)==5163 and len(predictions[predictions.model.eq('seasonal_seir')])==1721
    rng=np.random.default_rng(SEED);draws={};cluster_ids={};memberships=[]
    for geography in ['full_source_transfer','strict_location_disjoint']:
        q=selected(predictions,geography,'all_ordinal_leaves','all_dates','all_assessments')
        ids=sorted(q.coordinate_year.unique());assert len(ids)==(106 if geography=='full_source_transfer' else 104)
        cluster_ids[geography]=ids;draws[geography]=rng.integers(0,len(ids),size=(B,len(ids)))
        for index,coord in enumerate(ids):memberships.append(dict(geography=geography,cluster_index=index,coordinate_year=coord,
            source_field_ids=';'.join(sorted(q.loc[q.coordinate_year.eq(coord),'field_id'].unique()))))
    np.savez_compressed(OUT/'coordinate_year_draw_indices.npz',**draws)
    pd.DataFrame(memberships).to_csv(OUT/'coordinate_year_membership.csv',index=False)
    summaries=[];boot_rows=[];cluster_rows=[];archive_checks=[];point_rows=[]
    for geography in cluster_ids:
        ids=cluster_ids[geography];sample=draws[geography]
        for leaves in ['all_ordinal_leaves','upper_three']:
            for sensitivity in ['all_dates','exclude_2015_58_oct30']:
                for endpoint in ['all_assessments','final_numeric_assessment']:
                    q=selected(predictions,geography,leaves,sensitivity,endpoint)
                    for weighting in ['archived_selected_weights','literal_equal_coordinate_year','literal_equal_field']:
                        model_boot={};model_point={}
                        for model in MODELS:
                            g=q[q.model.eq(model)].copy();w=weights(g,weighting,endpoint);error=(g.predicted_percent-g.value).to_numpy()
                            g['analysis_weight']=w;g['weighted_mse']=w*error**2;g['weighted_mae']=w*abs(error);g['weighted_bias']=w*error
                            clusters=g.groupby('coordinate_year')[['weighted_mse','weighted_mae','weighted_bias','analysis_weight']].sum().reindex(ids,fill_value=0)
                            total=clusters.to_numpy();den=total[:,3].sum()
                            point=[np.sqrt(total[:,0].sum()/den),total[:,1].sum()/den,total[:,2].sum()/den]
                            sums=total[sample].sum(axis=1);boot=np.column_stack([np.sqrt(sums[:,0]/sums[:,3]),sums[:,1]/sums[:,3],sums[:,2]/sums[:,3]])
                            model_boot[model]=boot;model_point[model]=np.asarray(point)
                            common=dict(geography=geography,leaf_scope=leaves,sensitivity=sensitivity,endpoint=endpoint,weighting=weighting)
                            point_rows.append(dict(common,model=model,rmse_pp=point[0],mae_pp=point[1],bias_pp=point[2],assessment_cells=len(g),field_seasons=g.field_id.nunique(),leaf_series=g.endpoint_series.nunique(),coordinate_year_clusters=len(ids)))
                            clusters=clusters.reset_index().assign(model=model,**common);cluster_rows.append(clusters)
                            if sensitivity=='all_dates' and weighting=='archived_selected_weights':
                                original=pd.read_csv(SOURCE/'severity_metrics.csv');r=original[(original.model==model)&(original.geography==geography)&(original.leaf_scope==leaves)&(original.endpoint==endpoint)].iloc[0]
                                difference=np.max(abs(np.asarray(point)-r[['rmse','mae','bias']].to_numpy(float)));assert difference<1e-10
                                archive_checks.append(dict(common,model=model,max_metric_difference_pp=difference))
                        model_boot['paired_model_minus_baseline']=model_boot[MODELS[0]]-model_boot[MODELS[1]]
                        model_point['paired_model_minus_baseline']=model_point[MODELS[0]]-model_point[MODELS[1]]
                        for k,metric in enumerate(['rmse','mae','bias']):
                            for model,values in model_boot.items():
                                bounds=np.quantile(values[:,k],[.025,.975]);summaries.append(dict(common,metric=metric,comparison=model,
                                    estimate_pp=model_point[model][k],ci95_lower_pp=bounds[0],ci95_upper_pp=bounds[1],
                                    coordinate_year_clusters=len(ids),field_seasons=q.field_id.nunique(),leaf_series=q.endpoint_series.nunique(),resamples=B,seed=SEED))
                            boot_rows.append(pd.DataFrame(dict(common,bootstrap_id=np.arange(B),metric=metric,model_pp=model_boot[MODELS[0]][:,k],
                                baseline_pp=model_boot[MODELS[1]][:,k],paired_difference_pp=model_boot['paired_model_minus_baseline'][:,k])))
    pd.DataFrame(summaries).to_csv(HERE/'severity_bootstrap_intervals.csv',index=False)
    pd.DataFrame(point_rows).to_csv(HERE/'severity_point_estimates.csv',index=False)
    pd.DataFrame(archive_checks).to_csv(HERE/'archived_metric_reconstruction.csv',index=False)
    pd.concat(boot_rows,ignore_index=True).to_parquet(OUT/'severity_bootstrap_draws.parquet',index=False,compression='zstd')
    pd.concat(cluster_rows,ignore_index=True).to_csv(OUT/'severity_cluster_scores.csv',index=False)
    # The forecast dates stay frozen; sensitivity changes only the observed bounds.
    original_onset=pd.read_csv(SOURCE/'onset_interval_predictions.csv');observed=predictions[predictions.model.eq('seasonal_seir')]
    onset_rows=[]
    for geography in cluster_ids:
        for sensitivity in ['all_dates','exclude_2015_58_oct30']:
            obs=selected(observed,geography,'all_ordinal_leaves',sensitivity,'all_assessments')
            groups={key:g.sort_values('date') for key,g in obs.groupby(['field_id','endpoint_series'])}
            chosen=original_onset if geography=='full_source_transfer' else original_onset[original_onset.strict_location_disjoint]
            for r in chosen.itertuples():
                if (r.field_id,r.series) not in groups:continue
                g=groups[r.field_id,r.series];upper=g.loc[g.value.ge(r.cutoff_percent),'date'].min()
                negative=g.loc[g.value.lt(r.cutoff_percent),'date'];lower=negative[negative.lt(upper)].max() if pd.notna(upper) else negative.max()
                censor='right' if pd.isna(upper) else 'left' if pd.isna(lower) else 'interval';pred=pd.to_datetime(r.predicted_visible_date)
                if pd.isna(pred):compatible=pd.isna(upper);distance=0 if compatible else np.nan
                else:
                    distance=(pred-lower).days-1 if pd.notna(lower) and pred<=lower else (pred-upper).days if pd.notna(upper) and pred>upper else 0
                    compatible=distance==0
                if sensitivity=='all_dates':assert compatible==bool(r.compatible) and np.isclose(distance,r.delta_days,equal_nan=True)
                onset_rows.append(dict(geography=geography,sensitivity=sensitivity,model=r.model,field_id=r.field_id,series=r.series,
                    coordinate_year=g.iloc[0].coordinate_year,leaf_index=int(g.iloc[0].leaf_index),cutoff_percent=r.cutoff_percent,
                    censoring=censor,last_negative=None if pd.isna(lower) else str(lower.date()),first_positive=None if pd.isna(upper) else str(upper.date()),
                    predicted_visible_date=r.predicted_visible_date,compatible=compatible,bound_distance_days=distance,missing_prediction=pd.isna(pred),
                    infection_date_observed=False,exact_onset_error_identifiable=False))
    onset=pd.DataFrame(onset_rows);onset.to_csv(OUT/'onset_reconstructed_bounds.csv',index=False)
    onset_summary=[];onset_draws=[]
    for geography in cluster_ids:
        ids=cluster_ids[geography];sample=draws[geography]
        for leaves in ['all_ordinal_leaves','upper_three']:
            choice=onset[onset.geography.eq(geography)]
            if leaves=='upper_three':choice=choice[choice.leaf_index.lt(3)]
            for key,g in choice.groupby(['sensitivity','model','cutoff_percent','censoring']):
                rates=g.groupby(['coordinate_year','field_id']).compatible.mean().groupby('coordinate_year').mean()
                available=np.asarray([coord in rates.index for coord in ids],int);values=rates.reindex(ids,fill_value=0).to_numpy()
                den=available[sample].sum(axis=1);boot=np.divide(values[sample].sum(axis=1),den,out=np.full(B,np.nan),where=den>0)
                bounds=np.nanquantile(boot,[.025,.975]);sensitivity,model,cutoff,censoring=key
                violation=g.loc[~g.compatible,'bound_distance_days'].dropna();finite=g.bound_distance_days.dropna()
                common=dict(geography=geography,leaf_scope=leaves,sensitivity=sensitivity,model=model,cutoff_percent=cutoff,censoring=censoring)
                onset_summary.append(dict(common,source_leaf_series=len(g),field_seasons=g.field_id.nunique(),cohort_coordinate_year_clusters=len(ids),
                    stratum_coordinate_year_clusters=len(rates),compatible_count=int(g.compatible.sum()),unweighted_compatible_fraction=g.compatible.mean(),
                    equal_coordinate_year_compatible_fraction=rates.mean(),ci95_lower=bounds[0],ci95_upper=bounds[1],
                    resamples=B,finite_bootstrap_draws=int(np.isfinite(boot).sum()),seed=SEED,missing_predicted_events=int(g.missing_prediction.sum()),
                    finite_bound_distances=len(finite),mean_absolute_finite_bound_distance_days=finite.abs().mean(),
                    mean_absolute_incompatible_finite_distance_days=violation.abs().mean(),mean_signed_incompatible_finite_distance_days=violation.mean(),
                    median_signed_incompatible_finite_distance_days=violation.median(),exact_onset_error_identifiable=False))
                onset_draws.append(pd.DataFrame(dict(common,bootstrap_id=np.arange(B),compatible_fraction=boot,represented_stratum_clusters=den)))
    pd.DataFrame(onset_summary).to_csv(HERE/'onset_censoring_statistics.csv',index=False)
    pd.concat(onset_draws,ignore_index=True).to_parquet(OUT/'onset_bootstrap_draws.parquet',index=False,compression='zstd')
    # Classification is observed-window threshold detection, evaluated at the source field grain.
    occurrence=[]
    for geography in cluster_ids:
        for sensitivity in ['all_dates','exclude_2015_58_oct30']:
            q=selected(predictions,geography,'all_ordinal_leaves',sensitivity,'all_assessments')
            for model,g in q.groupby('model'):
                fields=g.groupby('field_id').agg(observed=('value','max'),forecast=('predicted_percent','max'))
                for cutoff in [.1,1.,5.]:
                    positive=fields.observed.ge(cutoff);warning=fields.forecast.ge(cutoff)
                    occurrence.append(dict(geography=geography,sensitivity=sensitivity,model=model,cutoff_percent=cutoff,field_seasons=len(fields),
                        positive_fields=int(positive.sum()),below_threshold_fields=int((~positive).sum()),warnings=int(warning.sum()),
                        sensitivity_observed_window=float(warning[positive].mean()) if positive.any() else np.nan,
                        specificity_below_threshold_window=float((~warning[~positive]).mean()) if (~positive).any() else np.nan,
                        full_season_absence_supported=False,proper_probability_calibration_identifiable=False,
                        always_positive_rule_has_same_classification=bool(warning.all())))
    pd.DataFrame(occurrence).to_csv(HERE/'observed_window_classification.csv',index=False)
    assert all(sha(ROOT/path)==h for path,h in hashes.items())
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='complete',source_sha256=hashes,source_archives_preserved=True,
        seed=SEED,resamples=B,full_coordinate_year_clusters=106,strict_coordinate_year_clusters=104,
        parameters_refitted=False,hyperparameters_reselected=False,model_imported=False,
        paired_predictions_held_fixed=True,parameter_uncertainty_included=False,
        weighting_definitions={'archived_selected_weights':'original target weights for all-assessment scores; final endpoints reweighted as the archive',
            'literal_equal_coordinate_year':'equal selected CY; equal fields within CY; equal leaf series within field; equal remaining assessments within series',
            'literal_equal_field':'equal selected fields; equal leaf series within field; equal remaining assessments within series; cluster bootstrap still uses CY'},
        upper_three_archived_all_assessment_weights_renormalized=False,
        sensitivity='only2015-58 original assessment2015-10-30 removed from scoring; frozen predictions and event dates unchanged; final endpoints and observed censor bounds reconstructed',
        source_flags_predate_external_scoring=True,true_infection_dates_observed=False,
        confidence_intervals='conditional paired CY percentile95%; onset strata use the same full-cohort draws, retaining undefined empty-stratum draws explicitly')
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    results=pd.DataFrame(summaries)
    print(results[(results.comparison=='paired_model_minus_baseline')&(results.sensitivity=='all_dates')&(results.weighting=='literal_equal_coordinate_year')&(results.endpoint=='final_numeric_assessment')].to_string(index=False),flush=True)


if __name__=='__main__':main()
