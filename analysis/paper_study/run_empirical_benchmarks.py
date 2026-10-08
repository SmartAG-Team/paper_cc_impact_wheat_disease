"""Freeze calibration-only weather comparators and score European transfer."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.paper_study.calibrate_seasonal import sha, metrics
from calibration.seasonal_septoria.field_data import prepare_fields
from calibration.seasonal_septoria.endpoints import chronological_partition
from calibration.seasonal_septoria.empirical import assessment_features, fit_comparator

DEST = ROOT/'analysis/paper_study/empirical_benchmarks'
OUT = ROOT/'data/paper_study/empirical_benchmarks'
KEYS = ['field_id','endpoint_series','date']


def attach_maxima(data, weather):
    data.maximum_temperature = np.zeros_like(data.temperature)
    forcing = weather.copy(); forcing['date'] = pd.to_datetime(forcing.date)
    for meta in data.metadata.itertuples():
        dates = pd.date_range(meta.first_forcing_date,meta.last_forcing_date)
        selected = forcing[forcing.location_id.eq(meta.site_id)&forcing.date.isin(dates)].sort_values('date')
        if not pd.DatetimeIndex(selected.date).equals(dates):
            raise ValueError('Maximum-temperature forcing coverage mismatch.')
        data.maximum_temperature[meta.field_index,:len(dates)] = selected.tmax_c


def endpoint(frame,final,upper):
    frame = frame[frame.leaf_index.lt(3)].copy() if upper else frame.copy()
    if final:
        frame = frame.sort_values('date').groupby(['field_id','endpoint_series']).tail(1).copy()
    frame['weight'] = 1/frame.groupby('coordinate_year').field_id.transform('nunique')/frame.groupby('field_id').endpoint_series.transform('nunique')
    if not final:
        frame['weight'] /= frame.groupby(['field_id','endpoint_series']).value.transform('size')
    return frame


def paired_intervals(frame,cluster_ids,draws):
    error = frame.predicted_percent-frame.value
    comparator = frame.comparator_percent-frame.value
    scores = pd.DataFrame({'coordinate_year':frame.coordinate_year,'w':frame.weight,
        'mse':frame.weight*error**2,'mae':frame.weight*abs(error),
        'comparator_mse':frame.weight*comparator**2,
        'comparator_mae':frame.weight*abs(comparator)}).groupby('coordinate_year').sum().reindex(cluster_ids)
    if scores.isna().any().any(): raise ValueError('Bootstrap membership changed.')
    denominator = scores.w.to_numpy()[draws].sum(axis=1)
    model_rmse = np.sqrt(scores.mse.to_numpy()[draws].sum(axis=1)/denominator)
    other_rmse = np.sqrt(scores.comparator_mse.to_numpy()[draws].sum(axis=1)/denominator)
    mae_diff = (scores.mae-scores.comparator_mae).to_numpy()[draws].sum(axis=1)/denominator
    return {'rmse_difference':float(np.sqrt(scores.mse.sum()/scores.w.sum())-np.sqrt(scores.comparator_mse.sum()/scores.w.sum())),
        'rmse_difference_lower':float(np.quantile(model_rmse-other_rmse,.025)),
        'rmse_difference_upper':float(np.quantile(model_rmse-other_rmse,.975)),
        'mae_difference':float((scores.mae-scores.comparator_mae).sum()/scores.w.sum()),
        'mae_difference_lower':float(np.quantile(mae_diff,.025)),
        'mae_difference_upper':float(np.quantile(mae_diff,.975))}


def detection(frame,cutoff):
    observed = frame.value.ge(cutoff).to_numpy(); predicted = frame.predicted_percent.ge(cutoff).to_numpy()
    w = frame.weight.to_numpy(); positive,negative = w[observed].sum(),w[~observed].sum()
    issued = w[predicted].sum()
    return dict(assessment_records=len(frame),positive_records=int(observed.sum()),negative_records=int((~observed).sum()),
        sensitivity=float(w[observed&predicted].sum()/positive) if positive else None,
        specificity=float(w[~observed&~predicted].sum()/negative) if negative else None,
        precision=float(w[observed&predicted].sum()/issued) if issued else None,
        severity_ranking_auc=float(roc_auc_score(observed,frame.predicted_percent,sample_weight=w)) if positive and negative else None,
        score_is_occurrence_probability=False,labels_are_season_long_field_occurrence=False)


def main():
    if '--resume-scoring' in sys.argv:
        if (DEST/'receipt.json').exists():raise FileExistsError('Completed benchmark archive is immutable.')
        freeze=json.loads((DEST/'frozen_before_new_validation_scores.json').read_text())
        predictions=pd.read_parquet(OUT/'frozen_all_predictions.parquet')
        predictions.loc[predictions.source.eq('Corteva'),'partition']='external'
        original=OUT/'frozen_all_predictions_before_partition_completion.parquet'
        if not original.exists():(OUT/'frozen_all_predictions.parquet').rename(original)
        predictions.to_parquet(OUT/'frozen_all_predictions.parquet',index=False)
        return score_and_finish(predictions,freeze)
    if DEST.exists() or OUT.exists(): raise FileExistsError('Use a new benchmark version.')
    DEST.mkdir(parents=True);OUT.mkdir(parents=True)
    source_root = ROOT/'analysis/paper_study/seasonal_calibration_v1'
    protocol = json.loads((source_root/'configuration_before_fitting.json').read_text())
    settings = {'weighted_ridge':[{'penalty':x} for x in [.1,1.,10.,100.,1000.]],
        'constrained_forest':[{'max_depth':depth,'min_samples_leaf':leaf} for depth in [3,5] for leaf in [5,10,20]]}
    config = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        comparison_families_added_after_initial_external_scores_were_known=True,
        external_data_used_to_fit_or_select_parameters=False,
        calibration_years=[2017,2018],development_validation_years=[2019],
        site_fold=protocol['site_fold'],settings=settings,feature_windows_days=[7,21,60],
        prohibited_features=['observed_disease','observed_stage','observed_disease_initialization','post_assessment_weather','site_identity','country_identity'],
        feature_selection_using_validation=False,selection='minimum calibration-only three-site-fold pooled weighted RMSE',
        random_seed=20261005,forest_trees=256,forest_bootstrap=False,forest_max_features=.8,
        ridge_prediction_bounds=[0,100],weights='equal coordinate-year, field, source-leaf, assessment-date',
        sklearn_version=sklearn.__version__,source_code_sha256=sha(Path(__file__)),
        feature_code_sha256=sha(ROOT/'calibration/seasonal_septoria/empirical.py'))
    (DEST/'configuration_before_fitting.json').write_text(json.dumps(config,indent=2)+'\n')
    weather = pd.read_parquet(ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet')
    calendars = pd.read_csv(ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv')
    phenology = json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    source = pd.read_csv(source_root/'basf_source_assessments_snapshot.csv')
    basf = prepare_fields(source,calendars,weather);attach_maxima(basf,weather)
    basf.targets['partition'] = chronological_partition(basf.targets,[2019])
    features = assessment_features(basf,phenology)
    training = basf.targets.partition.eq('calibration').to_numpy()
    validation = ~training
    if set(basf.targets.loc[training,'field_id'])&set(basf.targets.loc[validation,'field_id']):
        raise ValueError('Field split leakage.')
    train = basf.targets.loc[training]
    models,cv_rows = {},[]
    for family,candidates in settings.items():
        for setting in candidates:
            sse,total_weight = 0.,0.
            for fold in range(3):
                heldout = basf.targets.site_id.map(protocol['site_fold']).eq(fold).to_numpy()
                fit_rows,test_rows = training&~heldout,training&heldout
                if set(basf.targets.loc[fit_rows,'site_id'])&set(basf.targets.loc[test_rows,'site_id']):
                    raise ValueError('Site leakage in inner selection.')
                fitted = fit_comparator(family,setting,features.loc[fit_rows],
                    basf.targets.loc[fit_rows,'value'],basf.targets.loc[fit_rows,'weight'])
                y = fitted.predict(features.loc[test_rows]); actual = basf.targets.loc[test_rows]
                sse += float(np.sum(actual.weight*(y-actual.value)**2));total_weight += float(actual.weight.sum())
            cv_rows.append(dict(family=family,setting=setting,pooled_rmse=float(np.sqrt(sse/total_weight))))
        selected = min([row for row in cv_rows if row['family']==family],key=lambda x:x['pooled_rmse'])
        fitted = fit_comparator(family,selected['setting'],features.loc[training],train.value,train.weight)
        models[family] = fitted
        bundle = dict(estimator=fitted,feature_names=features.columns.tolist(),setting=selected['setting'],
            calibration_field_ids=sorted(train.field_id.unique()),calibration_only=True,
            validation_or_external_targets_used=False)
        joblib.dump(bundle,DEST/f'frozen_{family}.joblib')
    (DEST/'calibration_only_cv.json').write_text(json.dumps(cv_rows,indent=2)+'\n')
    freeze = {p.name:sha(p) for p in DEST.glob('frozen_*.joblib')}
    (DEST/'frozen_before_new_validation_scores.json').write_text(json.dumps(freeze,indent=2)+'\n')
    # Newly calculated validation and external predictions follow freezing.
    external = pd.read_csv(ROOT/'data/paper_study/observations/corteva_external_assessments.csv')
    external = external[external.assessment_eligible.eq(True)].copy()
    external['endpoint_series'] = external.physical_unit.astype(str)+'|'+external.organ.astype(str)
    corteva = prepare_fields(external,calendars,weather);attach_maxima(corteva,weather)
    corteva.targets['partition'] = 'external'
    strict_units = set(pd.read_csv(ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv').source_unit.astype(str))
    predictions = []
    for name,data,x,archive in [('BASF',basf,features,source_root/'basf_frozen_predictions.parquet'),
            ('Corteva',corteva,assessment_features(corteva,phenology),ROOT/'analysis/paper_study/corteva_external_seasonal_v1/frozen_external_predictions.parquet')]:
        frozen = pd.read_parquet(archive)
        frozen['source'] = name
        if name=='Corteva':frozen['partition']='external'
        if len(frozen[frozen.model.eq('seasonal_seir')])!=len(data.targets):raise ValueError('Archive membership mismatch.')
        predictions.append(frozen)
        for family,fitted in models.items():
            scored = data.targets.copy();scored['predicted_percent'] = fitted.predict(x)
            scored['model'],scored['source'] = family,name
            predictions.append(scored)
        x.to_parquet(OUT/f'{name.lower()}_features.parquet',index=False)
    predictions = pd.concat(predictions,ignore_index=True)
    # Source identifiers can be integer-valued in BASF and strings in Corteva.
    # Their textual keys already define field identity throughout this analysis.
    predictions['physical_unit'] = predictions.physical_unit.astype(str)
    for column in predictions.select_dtypes(include=['object','str']).columns:
        nonmissing = predictions[column].dropna()
        if nonmissing.map(type).nunique()>1:
            predictions[column] = predictions[column].map(lambda value:
                None if pd.isna(value) else str(value)).astype('string')
    predictions['strict_location_disjoint'] = predictions.physical_unit.astype(str).isin(strict_units)&predictions.source.eq('Corteva')
    predictions.to_parquet(OUT/'frozen_all_predictions.parquet',index=False)
    score_and_finish(predictions,freeze)


def score_and_finish(predictions,freeze):
    models=['weighted_ridge','constrained_forest']
    for name,digest in freeze.items():
        if sha(DEST/name)!=digest:raise ValueError('Frozen comparator changed before scoring.')
    for _,part in predictions.groupby(['source','partition']):
        memberships={model:set(map(tuple,group[KEYS].to_numpy())) for model,group in part.groupby('model')}
        if set(memberships)!=set(models+['seasonal_seir','primary_only_seasonal','calibration_only_leaf_rank_mean']):
            raise ValueError('Comparator family is absent from a scored partition.')
        if not all(keys==memberships['seasonal_seir'] for keys in memberships.values()):
            raise ValueError('Scored comparator membership differs.')
    membership = pd.read_csv(ROOT/'data/paper_study/external_statistics/coordinate_year_membership.csv')
    external_draws = np.load(ROOT/'data/paper_study/external_statistics/coordinate_year_draw_indices.npz')
    score_rows,comparison_rows,detection_rows = [],[],[]
    for (source_name,partition),part in predictions.groupby(['source','partition']):
        scopes = [('full_source_transfer',part)]
        if source_name=='Corteva':scopes.append(('strict_location_disjoint',part[part.strict_location_disjoint]))
        for geography,scope in scopes:
            cluster_ids = sorted(scope.coordinate_year.unique())
            if source_name=='Corteva':
                ids = membership[membership.geography.eq(geography)].sort_values('cluster_index').coordinate_year.tolist()
                if ids!=cluster_ids:raise ValueError('Shared external bootstrap clusters differ.')
                draws = external_draws[geography]
            else:
                draws = np.random.default_rng(20261005).integers(0,len(cluster_ids),size=(10000,len(cluster_ids)))
            for final in [False,True]:
                for upper in [False,True]:
                    label = dict(source=source_name,partition=partition,geography=geography,
                        endpoint='final_leaf_assessment' if final else 'all_assessments',leaf_scope='upper_three' if upper else 'all_ordinal')
                    selected = {model:endpoint(group,final,upper) for model,group in scope.groupby('model')}
                    for model,frame in selected.items():
                        score_rows.append(dict(**label,model=model,**metrics(frame)))
                        if not final and not upper:
                            for cutoff in [.1,1.,5.]:detection_rows.append(dict(**label,model=model,cutoff_percent=cutoff,**detection(frame,cutoff)))
                    main = selected['seasonal_seir']
                    for family in models:
                        paired = main.merge(selected[family][KEYS+['predicted_percent']],on=KEYS,
                            validate='one_to_one',suffixes=('','_comparator'))
                        if len(paired)!=len(main):raise ValueError('Paired endpoint mismatch.')
                        paired = paired.rename(columns={'predicted_percent_comparator':'comparator_percent'})
                        comparison_rows.append(dict(**label,model='seasonal_seir',comparator=family,
                            **paired_intervals(paired,cluster_ids,draws)))
    pd.DataFrame(score_rows).to_csv(DEST/'severity_metrics.csv',index=False)
    pd.DataFrame(comparison_rows).to_csv(DEST/'paired_model_comparator_intervals.csv',index=False)
    pd.DataFrame(detection_rows).to_csv(DEST/'assessment_detection_metrics.csv',index=False)
    for name,digest in freeze.items():
        if sha(DEST/name)!=digest:raise ValueError('Frozen comparator changed during scoring.')
    point=predictions[predictions.model.eq('seasonal_seir')]
    receipt = dict(status='complete',frozen_comparator_sha256=freeze,
        calibration_targets=int(point.partition.eq('calibration').sum()),
        validation_targets=int(point.partition.eq('validation').sum()),external_targets=int(point.partition.eq('external').sum()),
        calibration_only_parameter_selection=True,comparison_family_timing_disclosed=True,
        causal_features_tests_required=True,external_draw_indices_sha256=sha(ROOT/'data/paper_study/external_statistics/coordinate_year_draw_indices.npz'),
        field_seir_parameters_refitted=False,source_code_sha256=sha(Path(__file__)))
    (DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    print(pd.DataFrame(score_rows).query("partition!='calibration' and endpoint=='final_leaf_assessment'").to_string(index=False),flush=True)


if __name__=='__main__':main()
