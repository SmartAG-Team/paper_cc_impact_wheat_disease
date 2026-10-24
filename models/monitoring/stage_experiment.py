"""Fixed tests of recovered growth stage under explicit information conditions."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .evaluation import score_predictions,score_subset,paired_interval
from .inputs import sha,WEATHER
from .models import fit_candidate,predict_candidate
from .robust_models import fit_cluster_ensemble,predict_cluster_ensemble
from .stage_features import build_stage_features,contemporaneous_features,PRIOR_NAMES,CLOCK_NAMES

ROOT=Path(__file__).resolve().parents[2]
BASE='outputs/disease_improvement/station_repairs_20261022_v4'
META='outputs/raw_metadata_recovery_20261022'
RECIPES=['reference_replay','shared_replay','station_replay','station_weather_clock',
         'station_prior_stage','station_stage_clock','reanalysis_stage_clock','current_stage_conditioned']
SPEC=dict(structure='process',process_baseline_column=-3,crossleaf=True)


def augment(matrix,features):
    return np.column_stack([matrix[:,:-3],features,matrix[:,-3:]])


def _fit_predict(x,history,obs,train,test,spec):
    if spec.get('ensemble',False):
        fitted=fit_cluster_ensemble(x[train],history.iloc[train],obs.iloc[train],spec)
        prediction=predict_cluster_ensemble(fitted,x[test],history.iloc[test],obs.iloc[test])
    else:
        fitted=fit_candidate(x[train],history.iloc[train],obs.iloc[train],spec)
        prediction=predict_candidate(fitted,x[test],history.iloc[test],obs.iloc[test])
    return fitted,prediction


def run(root=ROOT,output=None,robust=False,history_anchor=False,forest_probe=False):
    root=Path(root);base=root/BASE;meta=root/META
    label='forest_probe' if forest_probe else 'history_anchor' if history_anchor else 'cluster_robust' if robust else 'stage_context'
    out=Path(output or root/f'outputs/disease_improvement/{label}_20261022')
    if out.exists():raise FileExistsError('A new registered output directory is required')
    obs=pd.read_csv(base/'observation_ledger.csv')
    fields=pd.read_csv(root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv')
    calendar=pd.read_csv(meta/'stage_calendar.csv')
    ledger=pd.read_csv(meta/'metadata_ledger.csv')
    if set(ledger.global_target_index)!=set(obs.global_target_index):
        raise ValueError('Recovered metadata and benchmark identities differ')
    original=pd.read_csv(base/'assessment_predictions.csv')
    schema=json.loads((root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json').read_text())
    arrays=np.load(base/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    history=pd.DataFrame(arrays['history'],columns=schema['history_names'])
    weather=pd.read_parquet(base/'station_preferred_daily_weather.parquet')
    reanalysis=pd.read_parquet(root/WEATHER)
    stage,names,provenance=build_stage_features(weather,fields,obs,calendar)
    stage_reanalysis,_,_=build_stage_features(reanalysis,fields,obs,calendar)
    current,current_names=contemporaneous_features(obs,calendar)
    prior_columns=[names.index(name) for name in PRIOR_NAMES]
    clock_columns=[names.index(name) for name in CLOCK_NAMES]
    designs=dict(reference_replay=arrays['reference'],shared_replay=arrays['reanalysis'],station_replay=arrays['station_all'],
        station_weather_clock=augment(arrays['station_all'],stage[:,clock_columns]),
        station_prior_stage=augment(arrays['station_all'],stage[:,prior_columns]),
        station_stage_clock=augment(arrays['station_all'],stage),
        reanalysis_stage_clock=augment(arrays['reanalysis'],stage_reanalysis),
        current_stage_conditioned=augment(arrays['station_all'],np.column_stack([stage,current])))
    specs={name:SPEC.copy() for name in RECIPES}
    if robust:
        designs={name:designs[name] for name in ['reference_replay','shared_replay','station_replay']}|dict(
            regularized_shared=arrays['reanalysis'],
            regularized_station_stage=augment(arrays['station_all'],stage),
            bagged_shared=arrays['reanalysis'],bagged_station_stage=augment(arrays['station_all'],stage))
        specs={name:dict(SPEC,**(dict(max_depth=2,min_samples_leaf=30,l2_regularization=10.)
            if name.startswith('regularized_') else {}),ensemble=name.startswith('bagged_')) for name in designs}
    if history_anchor:
        designs={name:designs[name] for name in ['reference_replay','shared_replay','station_replay']}|dict(
            bagged_shared=arrays['reanalysis'],bagged_history_residual=arrays['reanalysis'],
            bagged_history_split=arrays['reanalysis'],bagged_station_history_split=augment(arrays['station_all'],stage))
        specs={name:dict(SPEC,structure='split' if 'split' in name else 'residual' if 'residual' in name else 'process',
            ensemble=name.startswith('bagged_')) for name in designs}
    if forest_probe:
        designs={name:designs[name] for name in ['reference_replay','shared_replay','station_replay']}|dict(
            rf_shared_history=arrays['reanalysis'],rf_shared_direct=arrays['reanalysis'],
            rf_station_history=augment(arrays['station_all'],stage),rf_station_direct=augment(arrays['station_all'],stage))
        specs={name:dict(SPEC,structure='direct' if name.endswith('_direct') else 'residual' if name.startswith('rf_') else 'process',
            **(dict(estimator='rf',min_samples_leaf=5) if name.startswith('rf_') else {})) for name in designs}
    recipes=list(designs)
    out.mkdir(parents=True)
    source_paths=[base/name for name in ['observation_ledger.csv','assessment_predictions.csv','predictors.npz',
        'spatial_predictions.csv','spatial_membership.json','station_preferred_daily_weather.parquet']]
    source_paths += [meta/name for name in ['stage_calendar.csv','metadata_ledger.csv','field_treatment_ledger.csv','coverage.json']]
    source_paths += [root/WEATHER,root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv',
                     root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json']
    code_paths=[Path(__file__),Path(__file__).with_name('stage_features.py'),Path(__file__).with_name('models.py'),Path(__file__).with_name('robust_models.py'),
                Path(__file__).with_name('evaluation.py'),root/'analysis/final_study/metrics.py']
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),recipes=specs,
        tuning=False,independent_validation=False,cohort='unchanged qualified2286, temporal/spatial evaluation2017',
        stage_primary='usable untreated source stages with sowing_scenario<=date<assessmentdate only; no learned stage model',
        source_stage_interval_conflicts='excluded from usable stage context; disease observations retained',
        clock='latitude,longitude,astronomical daylength,temperature-prefix summaries through previous day; sowing is a scenario',
        conditional_only=['current_stage_conditioned'] if 'current_stage_conditioned' in recipes else [],
        regularization_probe='Depth2/minleaf30/l2=10 or mean of20 full-coordinate-year80%subsamples; all choices fixed before fitting' if robust else None,
        history_anchor_probe='Same ensemble and predictors; replace generic process offset by strictly earlier same-leaf severity (shrunk training category mean when unavailable), optionally separate first/subsequent heads' if history_anchor else None,
        forest_probe='Same predictors/information boundaries; fixed existing160-tree depth8/minleaf5 RF recipe, direct or history residual; complementary estimator test, no hyperparameter search' if forest_probe else None,
        conditional_information='same-date observed untreated field BBCH, potentially simultaneous with severity; not a pre-visit prediction',
        source_leaf_equivalence_verified=False,primary_reference='shared_replay',
        source_sha256={str(p.relative_to(root)):sha(p) for p in source_paths},
        code_sha256={str(p.relative_to(root)):sha(p) for p in code_paths},
        environment={name:version(name) for name in ['numpy','pandas','scikit-learn','joblib']},
        adoption='untouched external evaluation plus prior RMSE/source/history/severe/newsite/integral guards',
        original_outcomes_previously_examined=True)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for path in code_paths:
        dest=out/'code_snapshot'/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    for name in ['observation_ledger.csv','spatial_membership.json']:
        (out/name).write_bytes((base/name).read_bytes())
    provenance.to_csv(out/'stage_provenance.csv',index=False)
    (out/'feature_schema.json').write_text(json.dumps(dict(stage_names=names,current_names=current_names,
        design_widths={name:x.shape[1] for name,x in designs.items()},process_baseline_column=-3),indent=2)+'\n')
    np.savez_compressed(out/'predictors.npz',target_ids=obs.global_target_index,history=history.to_numpy(),**designs)
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    temporal=[]
    for year,block in original.groupby('season_year'):
        train=np.flatnonzero(obs.season_year.lt(year));test=position.loc[block.global_target_index].to_numpy()
        part=block.copy()
        for name in recipes:
            x=designs[name];fitted,prediction=_fit_predict(x,history,obs,train,test,specs[name])
            part[name]=prediction
            joblib.dump(fitted,out/f'{year}_{name}.joblib')
        temporal.append(part);print('temporal',year,flush=True)
    temporal=pd.concat(temporal,ignore_index=True)
    for new,old in [('reference_replay','reference_replay'),('shared_replay','shared_code'),('station_replay','station_all_shared_code')]:
        np.testing.assert_allclose(temporal[new],original.set_index('global_target_index').loc[temporal.global_target_index,old],atol=1e-10)
    temporal.to_csv(out/'assessment_predictions.csv',index=False)
    spatial=[]
    for split in json.loads((base/'spatial_membership.json').read_text()):
        train=position.loc[split['train']].to_numpy();test=position.loc[split['test']].to_numpy()
        if not len(train):continue
        part=obs.iloc[test].copy();part['outer_fold']=split['fold']
        for name in recipes:
            x=designs[name];fitted,prediction=_fit_predict(x,history,obs,train,test,specs[name])
            part[name]=prediction
        spatial.append(part)
    spatial=pd.concat(spatial,ignore_index=True);spatial.to_csv(out/'spatial_predictions.csv',index=False)
    metrics=[];intervals=[]
    for label,frame in [('temporal',temporal),('spatial_50km',spatial)]:
        for name in recipes:
            metrics.append(dict(evaluation=label,model=name,conditional_only=name=='current_stage_conditioned',**score_predictions(frame,name)))
            if name not in ['reference_replay','shared_replay','station_replay']:
                for reference in ['reference_replay','shared_replay','station_replay']:
                    ci=paired_interval(frame,name,reference)
                    intervals.append(dict(evaluation=label,model=name,reference=reference,lower=ci[0],upper=ci[1]))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    masks=dict(cold=~temporal.prior_assessment_available,warm=temporal.prior_assessment_available,
        BASF=temporal.source.eq('BASF'),Corteva=temporal.source.eq('Corteva'),
        severe_above75=temporal.observed_percent.gt(75),low_atmost5=temporal.observed_percent.le(5),new_site=temporal.new_site)
    diagnostics=[]
    for population,mask in masks.items():
        for name in recipes:diagnostics.append(dict(population=population,model=name,**score_subset(temporal,mask,name)))
    pd.DataFrame(diagnostics).to_csv(out/'subgroup_metrics.csv',index=False)
    temporal.loc[temporal.global_target_index.isin([479,1750,413,159,1231]),
        ['global_target_index','observed_percent']+recipes].to_csv(out/'hard_cases.csv',index=False)
    return pd.DataFrame(metrics)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    parser.add_argument('--robust',action='store_true')
    parser.add_argument('--history-anchor',action='store_true')
    parser.add_argument('--forest-probe',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):print(run(output=args.output,robust=args.robust,history_anchor=args.history_anchor,forest_probe=args.forest_probe).to_string(index=False))
