"""Freeze a training-selected season-start predictor before external evaluation."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from model.seasonal_septoria.field_data import prepare_fields
from model.seasonal_septoria.calibrate import fit,predict
from model.seasonal_septoria.core import Parameters
from model.seasonal_septoria.endpoints import chronological_partition,onset_distance,occurrence_scores

DEST = ROOT/'analysis/paper_study/seasonal_calibration_v1'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(frame):
    error = frame.predicted_percent-frame.value
    return dict(n=len(frame),field_seasons=int(frame.field_id.nunique()),
        coordinate_years=int(frame.coordinate_year.nunique()),
        rmse=float(np.sqrt(np.average(error**2,weights=frame.weight))),
        mae=float(np.average(abs(error),weights=frame.weight)),
        bias=float(np.average(error,weights=frame.weight)))


def main():
    if DEST.exists():
        raise FileExistsError('Calibration archive already exists; a new version is required.')
    DEST.mkdir(parents=True)
    full = pd.read_csv(ROOT/'data/paper_study/observations/assessments.csv')
    assessments = full.loc[full.dataset_id.eq('basf-wheat-diseases')].copy()
    assessments.to_csv(DEST/'basf_source_assessments_snapshot.csv',index=False)
    calendars = pd.read_csv(ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv')
    weather = pd.read_parquet(ROOT/'data/paper_study/field_weather/daily_weather.parquet')
    data = prepare_fields(assessments,calendars,weather)
    data.targets['partition'] = chronological_partition(data.targets,[2019])
    data.metadata['partition'] = chronological_partition(data.metadata,[2019])
    data.metadata.to_csv(DEST/'field_season_membership.csv',index=False)
    data.targets.to_csv(DEST/'target_membership.csv',index=False)
    data.excluded.to_csv(DEST/'excluded_source_assessments.csv',index=False)
    training = np.flatnonzero(data.targets.partition.eq('calibration'))
    validation = np.flatnonzero(data.targets.partition.eq('validation'))
    assert not (set(data.targets.iloc[training].field_id)&set(data.targets.iloc[validation].field_id))
    training_sites = sorted(data.targets.iloc[training].site_id.unique())
    site_fold = {site:i%3 for i,site in enumerate(training_sites)}
    configuration = dict(scope='season-start normalized leaf-fraction predictor',
        initialized_from_observed_disease=False,development_source='basf-wheat-diseases',
        calibration_harvest_years=[2017,2018],validation_harvest_years=[2019],
        prior_development_exposure_disclosed=True,prospective_external_source='corteva-2014-2018-external',
        crop_scenario='GGCMI winter rainfed calendar; frozen German winter TPV parameters',
        sowing_sensitivity_days=[-14,0,14],latent_days_candidates=[10.,20.,30.],
        selection='Three location-grouped calibration-only folds; minimum pooled weightedRMSE',
        site_fold=site_fold,primary_route='external weather-driven effective infection pressure',
        secondary_route='infectious-cohort rain-dependent pressure',
        inferred_primary_date_threshold_fraction=.0001,visible_diagnostic_cutoffs_percent=[.1,1.,5.],
        percent_observation_operator='Source infection percentage proxy; area denominator unspecified',
        secondary_origin_is_spore_type_measurement=False,
        baselines=['calibration_only_leaf_rank_mean','primary_only_seasonal'],
        source_snapshot_sha256=sha(DEST/'basf_source_assessments_snapshot.csv'),
        membership_sha256=sha(DEST/'target_membership.csv'),
        weather_sha256=sha(ROOT/'data/paper_study/field_weather/daily_weather.parquet'))
    write_json(DEST/'configuration_before_fitting.json',configuration)
    selection,predictions = [],[]
    for duration in configuration['latent_days_candidates']:
        total_sse,total_weight = 0.,0.
        for fold in range(3):
            heldout_sites = {site for site,v in site_fold.items() if v==fold}
            is_heldout = data.targets.site_id.isin(heldout_sites).to_numpy()
            fit_indices = training[~is_heldout[training]]
            heldout_indices = training[is_heldout[training]]
            if len(fit_indices)==0 or len(heldout_indices)==0:
                raise ValueError('Inner fold is empty.')
            result = fit(data,fit_indices,latent_days=duration)
            write_json(DEST/f'inner_latent{duration:g}_fold{fold}.json',result)
            forecast,_ = predict(data,Parameters(**result['parameters']))
            scored = data.targets.iloc[heldout_indices].copy()
            scored['predicted_percent'] = forecast[heldout_indices]
            scored['latent_days'],scored['inner_fold'] = duration,fold
            scored.to_parquet(DEST/f'inner_latent{duration:g}_fold{fold}_predictions.parquet',index=False)
            error = scored.predicted_percent-scored.value
            total_sse += float(np.sum(scored.weight*error**2))
            total_weight += float(scored.weight.sum())
            print(f'inner duration{duration:g} fold{fold}: {metrics(scored)}',flush=True)
        selection.append(dict(latent_days=duration,pooled_rmse=float(np.sqrt(total_sse/total_weight))))
    pd.DataFrame(selection).to_csv(DEST/'training_only_latent_selection.csv',index=False)
    chosen = min(selection,key=lambda row:row['pooled_rmse'])['latent_days']
    main_fit = fit(data,training,latent_days=chosen)
    primary_fit = fit(data,training,latent_days=chosen,beta_free=False)
    # Freeze all fitted parameters before calculating outer or external scores.
    write_json(DEST/'frozen_main_fit.json',main_fit)
    write_json(DEST/'frozen_primary_only_fit.json',primary_fit)
    rank_means = {}
    for rank,group in data.targets.iloc[training].groupby('leaf_index'):
        rank_means[int(rank)] = float(np.average(group.value,weights=group.weight))
    write_json(DEST/'frozen_leaf_rank_baseline.json',rank_means)
    frozen_hashes = {path.name:sha(path) for path in DEST.glob('frozen_*.json')}
    write_json(DEST/'frozen_parameters_before_validation.json',frozen_hashes)
    severity_rows,onset_rows,occurrence_rows,numerical = [],[],[],[]
    for name,fitted in [('seasonal_seir',main_fit),('primary_only_seasonal',primary_fit),
                        ('calibration_only_leaf_rank_mean',None)]:
        trajectory = None
        if fitted:
            forecast,trajectory = predict(data,Parameters(**fitted['parameters']))
            numerical.append(dict(model=name,mass_error=float(np.max(abs(trajectory.state.sum(axis=-1)-1))),
                minimum_state=float(trajectory.state.min()),maximum_damage=float(trajectory.damage.max())))
        else:
            global_mean = float(np.average(data.targets.iloc[training].value,weights=data.targets.iloc[training].weight))
            forecast = np.asarray([rank_means.get(int(rank),global_mean) for rank in data.targets.leaf_index])
        scored = data.targets.copy()
        scored['model'],scored['predicted_percent'] = name,forecast
        predictions.append(scored)
        for partition,group in scored.groupby('partition'):
            for subset,selected in [('all_ordinal_leaves',group),('upper_three',group[group.leaf_index.lt(3)])]:
                if selected.empty:
                    continue
                severity_rows.append(dict(model=name,partition=partition,endpoint='all_assessments',subset=subset,**metrics(selected)))
                final = selected.sort_values('date').groupby(['field_id','endpoint_series']).tail(1).copy()
                field_counts = final.groupby('coordinate_year').field_id.transform('nunique')
                leaf_counts = final.groupby('field_id').endpoint_series.transform('nunique')
                final['weight'] = 1/field_counts/leaf_counts
                severity_rows.append(dict(model=name,partition=partition,endpoint='final_numeric_assessment',subset=subset,**metrics(final)))
        if trajectory is None:
            continue
        for (field,series),group in scored.groupby(['field_index','endpoint_series']):
            row = group.iloc[0]
            meta = data.metadata.loc[data.metadata.field_index.eq(field)].iloc[0]
            organ = int(row.leaf_index)
            duration = int(meta.forcing_days)
            daily = trajectory.damage[field,1:duration+1,organ]*100
            for cutoff in configuration['visible_diagnostic_cutoffs_percent']:
                observed = group.sort_values('date')
                positives = observed[observed.value.ge(cutoff)]
                upper = None if positives.empty else positives.date.min()
                negatives = observed[observed.value.lt(cutoff)]
                lower = negatives.date.max() if upper is None else negatives.loc[negatives.date.lt(upper),'date'].max()
                lower = None if pd.isna(lower) else lower
                found = np.flatnonzero(daily>=cutoff)
                predicted = None if len(found)==0 else pd.Timestamp(meta.sowing_date)+pd.Timedelta(days=int(found[0]))
                distance = onset_distance(predicted,lower,upper)
                onset_rows.append(dict(model=name,field_id=row.field_id,series=series,partition=row.partition,
                    cutoff_percent=cutoff,last_negative=None if lower is None else str(lower.date()),
                    first_positive=None if upper is None else str(upper.date()),
                    predicted_visible_date=None if predicted is None else str(predicted.date()),
                    censoring='right' if upper is None else 'left' if lower is None else 'interval',**distance))
        for cutoff in configuration['visible_diagnostic_cutoffs_percent']:
            for partition,part in scored.groupby('partition'):
                trial = part.groupby('field_id').agg(observed=('value','max'),forecast=('predicted_percent','max'))
                occurrence_rows.append(dict(model=name,partition=partition,cutoff_percent=cutoff,
                    endpoint='any_ordinal_leaf_detected_at_assessment_dates',
                    **occurrence_scores(trial.observed.ge(cutoff),trial.forecast.ge(cutoff).astype(float))))
    pd.concat(predictions,ignore_index=True).to_parquet(DEST/'basf_frozen_predictions.parquet',index=False)
    pd.DataFrame(severity_rows).to_csv(DEST/'severity_metrics.csv',index=False)
    pd.DataFrame(onset_rows).to_csv(DEST/'visible_onset_interval_predictions.csv',index=False)
    pd.DataFrame(occurrence_rows).to_csv(DEST/'observed_window_detection_metrics.csv',index=False)
    write_json(DEST/'numerical_checks.json',numerical)
    write_json(DEST/'receipt.json',dict(status='complete',publication_ready=False,
        calibrated_field_seasons=int(data.metadata.partition.eq('calibration').sum()),
        validation_field_seasons=int(data.metadata.partition.eq('validation').sum()),
        calibrated_targets=len(training),validation_targets=len(validation),
        independent_external_source_evaluated=False,selected_latent_days=chosen,
        retained_frozen_parameter_hashes=frozen_hashes,prior_development_exposure=True,
        true_infection_dates_observed=False,negative_full_season_field_classes_available=False))
    print(pd.DataFrame(severity_rows).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
