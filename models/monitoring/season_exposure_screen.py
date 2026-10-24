"""Temporal screen of cumulative exposure in the first-field head only."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval,score_predictions,score_subset
from .field_history_models import fit_ensemble,predict_ensemble
from .infection_event_experiment import populations
from .inputs import sha
from .season_exposure_features import build_season_exposure

ROOT=Path(__file__).resolve().parents[2]
NAME='season_exposure_first'


def run(root=ROOT,output=None):
    root=Path(root).resolve()
    parent=root/'data/final_monitoring/cohort'
    weather_data=root/'data/disease_improvement/station_preferred_ie_20261022'
    fields_file=root/'data/disease_improvement/infection_events_20261022/fields.csv'
    out=Path(output or root/'outputs/disease_improvement/season_exposure_screen_20261022').resolve()
    if out.exists():
        raise FileExistsError('A new output directory is required')
    obs=pd.read_csv(parent/'training_observation_ledger.csv')
    arrays=np.load(parent/'predictors.npz')
    bundle=joblib.load(parent/'fitted_candidate.joblib')
    history=pd.DataFrame(arrays['history'],columns=bundle['base']['history_names'])
    fields=pd.read_csv(fields_file)
    weather=pd.read_parquet(weather_data/'daily_weather.parquet')
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    frame=pd.read_csv(parent/'assessment_predictions.csv')
    splits=[s for s in json.loads((parent/'split_membership.json').read_text()) if s['fold'].startswith('temporal')]
    sources=[parent/n for n in ['training_observation_ledger.csv','predictors.npz','fitted_candidate.joblib',
               'fitted_candidate_lock.json','assessment_predictions.csv','split_membership.json','verification.json']]
    sources += [weather_data/'daily_weather.parquet',weather_data/'manifest.json',fields_file]
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['season_exposure_features','models',
           'field_history_models','infection_event_experiment','evaluation']]
    codes += [root/'model/seasonal_septoria/wetness.py',root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Seasonal weather history is compressed into three fixed-default process states; six explicit cumulative proxies may improve initial field assessments',
        changes='Only the first-field53-column head receives six cumulative exposure summaries; process component and all earlier-field predictions stay fixed',
        feature_windows='Existing scenario sowing through day before assessment; early exposure excludes the latest42days',
        no_actual_sowing_or_initial_disease_recovered=True,
        model=dict(structure='field_split',members=20,fraction=.8,seed=20261022),
        temporal_screen_to_spatial='Strictly lower overall and first-field RMSE; integral noninferiority; severe absolute bias noninferiority; cold/warm/BASF/Corteva RMSE changes no greater than1point',
        posthoc_development=True,hyperparameter_search=False,independent_external_validation=False,publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)):sha(p) for p in codes})
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for p in codes:
        dest=out/'code_snapshot'/p.relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
    extra,names,provenance=build_season_exposure(weather,fields,obs.drop(columns='observed_percent'))
    first=np.column_stack([arrays['station'][:,:-3],extra,arrays['station'][:,-3:]])
    assert first.shape==(2304,59)
    np.savez_compressed(out/'predictors.npz',target_ids=arrays['target_ids'],first=first,season_exposure=extra,history=arrays['history'])
    provenance.to_csv(out/'feature_provenance.csv',index=False)
    (out/'feature_schema.json').write_text(json.dumps(dict(new_feature_names=names,new_features=6,first_field_columns=59,
        unchanged_original_columns=53,history_columns=35,weather_cutoff_days=1,
        interpretation='Cumulative observed/phase-assisted weather and modeled exposure proxies; scenario sowing, no measured initial disease or canopy wetness'),indent=2)+'\n')
    parts,support=[],[]
    has_history=history.prior_assessment_available.eq(1)|history.other_leaf_available.eq(1)
    for split in splits:
        train,test=[position.loc[split[key]].to_numpy() for key in ['train','test']]
        assert obs.iloc[train].season_year.max()<obs.iloc[test].season_year.min()
        fitted=fit_ensemble(first[train],history.iloc[train],obs.iloc[train],dict(structure='field_split'))
        joblib.dump(fitted,out/f"{split['fold']}_first.joblib")
        query=obs.iloc[test].drop(columns='observed_percent')
        prediction=predict_ensemble(fitted,first[test],history.iloc[test],query)
        part=frame.set_index('global_target_index').loc[split['test']].reset_index().copy()
        cold=~has_history.iloc[test].to_numpy()
        np.testing.assert_array_equal(cold,~part.ie_uses_event_head.to_numpy())
        part['new_first_component']=prediction
        part[NAME]=np.where(cold,.5*(part.candidate_process.to_numpy()+prediction),part.irish_station_candidate)
        parts.append(part)
        # Descriptive support audit; no observation is excluded or used to tune the model.
        training=obs.iloc[train].loc[~has_history.iloc[train].to_numpy()]
        for row in part.loc[cold].itertuples(index=False):
            category=training.loc[training.source.eq(row.source)&training.leaf_index.eq(row.leaf_index)]
            support.append(dict(global_target_index=row.global_target_index,fold=split['fold'],source=row.source,
                leaf_index=row.leaf_index,observed_percent=row.observed_percent,
                prior_first_field_rows=len(training),prior_first_field_severe_rows=int(training.observed_percent.gt(75).sum()),
                prior_same_source_code_rows=len(category),prior_same_source_code_max_percent=float(category.observed_percent.max()) if len(category) else None,
                prior_same_source_code_severe_rows=int(category.observed_percent.gt(75).sum())))
        print('fitted',split['fold'],flush=True)
    result=pd.concat(parts,ignore_index=True)
    result.to_csv(out/'assessment_predictions.csv',index=False)
    pd.DataFrame(support).to_csv(out/'training_support_audit.csv',index=False)
    metrics=pd.DataFrame([dict(model=name,**score_predictions(result,name)) for name in
                         [NAME,'irish_station_candidate','reference_replay','current']])
    metrics.to_csv(out/'metrics.csv',index=False)
    low,high=paired_interval(result,NAME,'irish_station_candidate')
    (out/'paired_interval.json').write_text(json.dumps(dict(lower=low,upper=high,reference='irish_station_candidate'),indent=2)+'\n')
    h=history.iloc[position.loc[result.global_target_index]].reset_index(drop=True)
    masks=populations(result,h)
    groups=pd.DataFrame([dict(population=key,model=name,**score_subset(result,mask,name))
        for key,mask in masks.items() for name in [NAME,'irish_station_candidate']])
    groups.to_csv(out/'subgroup_metrics.csv',index=False)
    old,new=[score_predictions(result,name) for name in ['irish_station_candidate',NAME]]
    gates=dict(overall_rmse_improves=new['rmse']<old['rmse'],
        first_field_rmse_improves=score_subset(result,masks['first_field'],NAME)['rmse']<score_subset(result,masks['first_field'],'irish_station_candidate')['rmse'],
        integral_noninferiority=new['integral_rmse']<=old['integral_rmse'],
        severe_bias_noninferiority=abs(score_subset(result,masks['severe_above75'],NAME)['bias'])<=abs(score_subset(result,masks['severe_above75'],'irish_station_candidate')['bias']))
    for key in ['cold','warm','BASF','Corteva']:
        gates[key+'_rmse_noninferiority']=score_subset(result,masks[key],NAME)['rmse']-score_subset(result,masks[key],'irish_station_candidate')['rmse']<=1.
    (out/'screen_decision.json').write_text(json.dumps(dict(proceed_to_spatial=all(gates.values()),gates=gates,
        independent_external_validation=False,publication_model_promoted=False),indent=2)+'\n')
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output).to_string(index=False))
