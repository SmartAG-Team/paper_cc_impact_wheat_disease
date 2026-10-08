"""Independent statistics and onset diagnostics for frozen nested predictions."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.paper_study.structural_evaluation.run import load_inputs, DEFAULT_PATHS, subset_fields
from calibration.seasonal_septoria.structural import predict_canopy
from calibration.seasonal_septoria.calibrate import predict as predict_seir
from model.seasonal_septoria.core import Parameters

BASE = ROOT/'analysis/paper_study'
HERE = Path(__file__).parent
DEST = HERE/'nested_summary'
KEYS = ['evaluation_kind', 'outer_fold', 'global_target_index']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected(frame, final=True, upper=False):
    frame = frame.loc[frame.leaf_index.lt(3)].copy() if upper else frame.copy()
    if final:
        frame = frame.sort_values('date').groupby(['field_id', 'endpoint_series']).tail(1).copy()
    frame['weight'] = 1/frame.groupby('coordinate_year').field_id.transform('nunique')
    frame['weight'] /= frame.groupby('field_id').endpoint_series.transform('nunique')
    if not final:
        frame['weight'] /= frame.groupby(['field_id', 'endpoint_series']).value.transform('size')
    return frame


def statistics(frame):
    # A separate aggregation implementation independently checks the production
    # weighted formula: date -> source leaf -> field -> coordinate/year.
    error = frame.predicted_percent-frame.value
    values = frame.assign(mse=error**2, absolute=abs(error), error=error, square=frame.value**2)
    columns = ['mse', 'absolute', 'error', 'value', 'square']
    leaf = values.groupby(['coordinate_year', 'field_id', 'endpoint_series'])[columns].mean()
    field = leaf.groupby(level=['coordinate_year', 'field_id']).mean()
    mean = field.groupby(level='coordinate_year').mean().mean()
    variance = mean.square-mean.value**2
    return dict(n=len(frame), field_seasons=int(frame.field_id.nunique()),
        coordinate_years=int(frame.coordinate_year.nunique()), locations=int(frame.site_id.nunique()),
        rmse=float(np.sqrt(mean.mse)), mae=float(mean.absolute), bias=float(mean.error),
        weighted_r2=float(1-mean.mse/variance) if variance > 0 else None)


def paired(frame, reference, draws=5000):
    comparison = reference[KEYS+['predicted_percent']].rename(columns={'predicted_percent':'reference'})
    merged = frame.merge(comparison, on=KEYS, validate='one_to_one')
    if len(merged) != len(frame) or len(merged) != len(reference):
        raise ValueError('Paired endpoint membership differs.')
    error, old = merged.predicted_percent-merged.value, merged.reference-merged.value
    cluster = merged.assign(mse=merged.weight*error**2, ref_mse=merged.weight*old**2,
        absolute=merged.weight*abs(error), ref_absolute=merged.weight*abs(old)).groupby('site_id')[
            ['weight','mse','ref_mse','absolute','ref_absolute']].sum()
    indices = np.random.default_rng(20261006).integers(0,len(cluster),(draws,len(cluster)))
    sampled = cluster.to_numpy()[indices].sum(axis=1)
    rmse = np.sqrt(sampled[:,1]/sampled[:,0])-np.sqrt(sampled[:,2]/sampled[:,0])
    mae = (sampled[:,3]-sampled[:,4])/sampled[:,0]
    point = cluster.sum()
    return dict(cluster_unit='complete_location_history',cluster_count=len(cluster),bootstrap_draws=draws,
        rmse_difference=float(np.sqrt(point.mse/point.weight)-np.sqrt(point.ref_mse/point.weight)),
        rmse_difference_lower=float(np.quantile(rmse,.025)),rmse_difference_upper=float(np.quantile(rmse,.975)),
        mae_difference=float((point.absolute-point.ref_absolute)/point.weight),
        mae_difference_lower=float(np.quantile(mae,.025)),mae_difference_upper=float(np.quantile(mae,.975)))


def onset(data, expressed, model, fold, kind):
    records = []
    for (field, series), group in data.targets.groupby(['field_index','endpoint_series']):
        row = group.iloc[0]
        meta = data.metadata.loc[data.metadata.field_index.eq(field)].iloc[0]
        score = 100*expressed[field,1:int(meta.forcing_days)+1,int(row.leaf_index)]
        for cutoff in (.1,1.,5.):
            ordered = group.sort_values('date')
            positive = ordered.loc[ordered.value.ge(cutoff),'date']
            upper = None if positive.empty else pd.Timestamp(positive.min())
            negative = ordered.loc[ordered.value.lt(cutoff),'date']
            if upper is not None:
                negative = negative.loc[negative.lt(upper)]
            lower = None if negative.empty else pd.Timestamp(negative.max())
            crossings = np.flatnonzero(score >= cutoff)
            predicted = None if not len(crossings) else pd.Timestamp(meta.sowing_date)+pd.Timedelta(days=int(crossings[0]))
            if predicted is None:
                compatible, distance = upper is None, 0 if upper is None else None
            elif lower is not None and predicted <= lower:
                compatible, distance = False, (predicted-lower).days-1
            elif upper is not None and predicted > upper:
                compatible, distance = False, (predicted-upper).days
            else:
                compatible, distance = True, 0
            records.append(dict(evaluation_kind=kind,outer_fold=fold,model=model,field_id=row.field_id,
                source=row.source,coordinate_year=row.coordinate_year,site_id=row.site_id,
                endpoint_series=series,leaf_index=row.leaf_index,cutoff_percent=cutoff,
                censoring='right' if upper is None else 'left' if lower is None else 'interval',
                last_negative=lower,first_positive=upper,predicted_visible_date=predicted,
                compatible=compatible,delta_days=distance,missing_prediction=predicted is None))
    return pd.DataFrame(records)


def main():
    if DEST.exists():
        raise FileExistsError('Use a new immutable statistics archive.')
    DEST.mkdir()
    archives = {name:BASE/f'nested_canopy_{tag}_20261006'
        for name,tag in [('daily_or','daily'),('duration_proxy','duration')]}
    identities = {str(path.relative_to(ROOT)):sha(path) for archive in archives.values()
        for path in [archive/'receipt.json', archive/'all_out_of_fold_predictions.parquet',
            archive/'nested_membership_before_fitting.csv',archive/'all_outer_fits_frozen_before_scoring.json']}
    identities[str((HERE/'ensemble_protocol_before_nested_scoring.json').relative_to(ROOT))] = sha(HERE/'ensemble_protocol_before_nested_scoring.json')
    (DEST/'configuration_before_statistics.json').write_text(json.dumps(dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),input_sha256=identities,
        independent_statistics=True,all_outcomes_previously_exposed=True,untouched_test=False,
        component_weights_fixed_before_pooled_nested_scores=True,bootstrap_draws=5000),indent=2)+'\n')
    frames = {}
    maximum, checks = 0.,0
    for name, archive in archives.items():
        receipt=json.loads((archive/'receipt.json').read_text())
        assert receipt['status']=='complete' and not receipt['untouched_test_evaluated']
        for filename,digest in receipt['output_sha256'].items():
            assert sha(archive/filename)==digest
        frames[name]=pd.read_parquet(archive/'all_out_of_fold_predictions.parquet')
        stored=pd.read_csv(archive/'severity_metrics.csv')
        for row in stored.itertuples():
            part=frames[name].loc[frames[name].evaluation_kind.eq(row.evaluation_kind)&frames[name].model.eq(row.model)]
            if row.report_scope in ('BASF','Corteva'):part=part.loc[part.source.eq(row.report_scope)]
            elif row.report_scope!='pooled':part=part.loc[part.outer_fold.eq(row.report_scope)]
            metrics=statistics(selected(part,row.endpoint=='final_numeric_assessment',row.leaf_scope=='upper_three'))
            for metric in ('rmse','mae','bias','weighted_r2'):
                delta=abs(metrics[metric]-getattr(row,metric));maximum=max(maximum,delta);checks+=1
                assert delta<1e-9
    daily, duration=frames['daily_or'],frames['duration_proxy']
    comparator_columns=[column for column in daily.columns if column!='fit_sha256']
    pd.testing.assert_frame_equal(daily.loc[daily.model.eq('original_seir'),comparator_columns].sort_values(KEYS).reset_index(drop=True),
        duration.loc[duration.model.eq('original_seir'),comparator_columns].sort_values(KEYS).reset_index(drop=True))
    component=daily.loc[daily.model.eq('structural_canopy')].merge(
        duration.loc[duration.model.eq('structural_canopy'),KEYS+['predicted_percent']],
        on=KEYS,validate='one_to_one',suffixes=('','_duration'))
    component['predicted_percent']=.5*(component.predicted_percent+component.predicted_percent_duration)
    component['model']='fixed_structural_ensemble'
    daily.loc[daily.model.eq('structural_canopy'),'model']='daily_or_canopy'
    duration.loc[duration.model.eq('structural_canopy'),'model']='duration_canopy'
    combined=pd.concat([daily,duration.loc[duration.model.eq('duration_canopy')],component],ignore_index=True)
    combined.to_parquet(DEST/'all_out_of_fold_predictions.parquet',index=False)
    rows,differences,high=[] ,[],[]
    for kind,frame in combined.groupby('evaluation_kind'):
        for scope in ['pooled','BASF','Corteva',*sorted(frame.outer_fold.unique())]:
            part=frame if scope=='pooled' else frame.loc[frame.source.eq(scope)] if scope in ('BASF','Corteva') else frame.loc[frame.outer_fold.eq(scope)]
            for final in (False,True):
                for upper in (False,True):
                    endpoints={model:selected(group,final,upper) for model,group in part.groupby('model')}
                    reference=endpoints['original_seir']
                    common=dict(evaluation_kind=kind,report_scope=scope,endpoint='final_numeric_assessment' if final else 'all_assessments',
                        leaf_scope='upper_three' if upper else 'all_ordinal_leaves')
                    for model,group in endpoints.items():
                        rows.append(dict(**common,model=model,**statistics(group)))
                        if model!='original_seir':differences.append(dict(**common,model=model,**paired(group,reference)))
            if scope=='pooled':
                for model,group in part.groupby('model'):
                    terminal=selected(group)
                    for cutoff in (50,80):
                        subset=terminal.loc[terminal.value.ge(cutoff)].copy()
                        # Reweight within the observed-score stratum explicitly.
                        if len(subset):high.append(dict(evaluation_kind=kind,model=model,observed_cutoff=cutoff,**statistics(subset)))
    pd.DataFrame(rows).to_csv(DEST/'severity_metrics.csv',index=False)
    pd.DataFrame(differences).to_csv(DEST/'paired_location_bootstrap.csv',index=False)
    pd.DataFrame(high).to_csv(DEST/'high_score_metrics.csv',index=False)
    data,weather,accumulation,thresholds=load_inputs(DEFAULT_PATHS)
    membership=pd.read_csv(archives['daily_or']/'nested_membership_before_fitting.csv')
    assert (archives['daily_or']/'nested_membership_before_fitting.csv').read_bytes()==(archives['duration_proxy']/'nested_membership_before_fitting.csv').read_bytes()
    onset_records=[]
    for fold,part in membership.loc[membership.level.eq('outer')].groupby('outer_fold'):
        train=part.loc[part.role.eq('training')];test=part.loc[part.role.eq('validation')]
        assert not set(train.site_id)&set(test.site_id)
        indices=np.sort(test.target_index.unique())
        fields=np.sort(data.targets.iloc[indices].field_index.unique())
        redacted=subset_fields(data,indices,redact_values=True)
        if fold.startswith('forward'):
            assert train.season_year.max()<test.season_year.min()
        trajectories={}
        for name,archive in archives.items():
            fitted=json.loads((archive/'outer_folds'/fold/'frozen_structural_canopy_outer_fit.json').read_text())['fitted']
            forecast,trajectory=predict_canopy(redacted,accumulation[fields],thresholds,fitted)
            saved=frames[name].loc[frames[name].outer_fold.eq(fold)&frames[name].model.isin(['structural_canopy','daily_or_canopy','duration_canopy'])].sort_values('global_target_index')
            delta=np.max(abs(forecast-saved.predicted_percent.to_numpy()));maximum=max(maximum,delta);checks+=1
            assert delta<1e-9
            trajectories[name]=trajectory.expressed
        original=json.loads((archives['daily_or']/'outer_folds'/fold/'frozen_original_seir_outer_fit.json').read_text())['fitted']
        _,trajectory=predict_seir(redacted,Parameters(**original['parameters']))
        trajectories['original_seir']=trajectory.damage
        trajectories['fixed_structural_ensemble']=.5*(trajectories['daily_or']+trajectories['duration_proxy'])
        # Outcome attachment occurs after all forecasts for the outer fold.
        redacted.targets['value']=data.targets.iloc[indices].value.to_numpy()
        for name,trajectory in trajectories.items():
            onset_records.append(onset(redacted,trajectory,name,fold,'spatial' if fold.startswith('spatial') else 'forward'))
    onsets=pd.concat(onset_records,ignore_index=True)
    onsets.to_parquet(DEST/'onset_predictions.parquet',index=False)
    onset_metrics=[]
    for labels,group in onsets.groupby(['evaluation_kind','model','cutoff_percent','censoring']):
        means=group.groupby(['coordinate_year','field_id']).compatible.mean().groupby(level='coordinate_year').mean().mean()
        error=group.delta_days.dropna()
        onset_metrics.append(dict(evaluation_kind=labels[0],model=labels[1],cutoff_percent=labels[2],censoring=labels[3],
            n=len(group),coordinate_years=int(group.coordinate_year.nunique()),locations=int(group.site_id.nunique()),
            weighted_compatible=float(means),missing_predictions=int(group.missing_prediction.sum()),
            median_distance_days=float(error.median()) if len(error) else None))
    pd.DataFrame(onset_metrics).to_csv(DEST/'onset_metrics.csv',index=False)
    for path,digest in identities.items():assert sha(ROOT/path)==digest
    (DEST/'verification.json').write_text(json.dumps(dict(status='passed',independent_metric_and_prediction_checks=checks,
        maximum_difference=maximum,source_receipts_and_output_hashes_verified=True,
        comparator_predictions_identical_between_operators=True,location_and_time_separation_verified=True,
        untouched_test=False,output_sha256={p.name:sha(p) for p in sorted(DEST.glob('*')) if p.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(status='passed',checks=checks,maximum_difference=maximum)))


if __name__=='__main__':
    main()
