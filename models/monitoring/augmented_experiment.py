"""Training augmentation from official corrected coordinates, fixed evaluation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .evaluation import distances_km,score_predictions,paired_interval,score_subset
from .inputs import sha
from .stage_experiment import augment,_fit_predict,SPEC

ROOT=Path(__file__).resolve().parents[2]
COMPONENTS=['aug_bagged_shared','aug_bagged_history_residual','aug_bagged_station_history_split']
BLENDS={'aug_process_history_mean':COMPONENTS[:2],
    'aug_process_station_history_mean':[COMPONENTS[0],COMPONENTS[2]],'aug_all_three_mean':COMPONENTS}


def run(root=ROOT,output=None):
    root=Path(root);base=root/'outputs/disease_improvement/station_repairs_20261022_v4'
    stagebase=root/'outputs/disease_improvement/stage_context_20261022'
    prior=root/'outputs/disease_improvement/history_anchor_20261022'
    extra=root/'data/disease_improvement/coordinate_augmentation_20261022'
    out=Path(output or root/'outputs/disease_improvement/coordinate_augmentation_20261022')
    if out.exists():raise FileExistsError('New output directory required')
    original=pd.read_csv(base/'observation_ledger.csv');added=pd.read_csv(extra/'observations.csv')
    obs=pd.concat([original,added],ignore_index=True)
    assert obs.global_target_index.is_unique and set(added.global_target_index).isdisjoint(original.global_target_index)
    old=np.load(base/'predictors.npz');new=np.load(extra/'predictors.npz');stage=np.load(stagebase/'predictors.npz')
    np.testing.assert_array_equal(old['target_ids'],original.global_target_index)
    np.testing.assert_array_equal(stage['target_ids'],original.global_target_index)
    np.testing.assert_array_equal(new['target_ids'],added.global_target_index)
    names=json.loads((root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json').read_text())['history_names']
    history=pd.DataFrame(np.vstack([old['history'],new['history']]),columns=names)
    shared=np.vstack([old['reanalysis'],new['matrix']])
    station=np.vstack([stage['station_stage_clock'],augment(new['matrix'],new['stage'])])
    designs={COMPONENTS[0]:shared,COMPONENTS[1]:shared,COMPONENTS[2]:station}
    specs={name:dict(SPEC,structure='split' if name.endswith('_split') else 'residual' if 'residual' in name else 'process',ensemble=True) for name in COMPONENTS}
    sites=sorted(original.site_id.unique());ordered=sorted(sites,key=lambda s:hashlib.sha256(str(s).encode()).hexdigest())
    assignment={site:i%3 for i,site in enumerate(ordered)}
    for site in sorted(set(added.site_id)-set(sites)):
        assignment[site]=int(hashlib.sha256(str(site).encode()).hexdigest(),16)%3
    source_paths=[base/'observation_ledger.csv',base/'predictors.npz',base/'spatial_membership.json',
        stagebase/'predictors.npz',prior/'assessment_predictions.csv',prior/'spatial_predictions.csv',
        extra/'observations.csv',extra/'predictors.npz',extra/'manifest.json']
    code_paths=[Path(__file__),Path(__file__).with_name('models.py'),Path(__file__).with_name('robust_models.py'),
        Path(__file__).with_name('stage_experiment.py'),Path(__file__).with_name('evaluation.py'),root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),recipes=specs,blends={k:dict(components=v,weights=[1/len(v)]*len(v)) for k,v in BLENDS.items()},
        tuning=False,independent_external_validation=False,cohort='Original2017evaluationrows unchanged;18newrecords trainingonly',
        additional_training='OfficialCortevav1.1 positive-longitude corrections verified by French Geo API; no disease outcome filtering',
        temporal_rule='All training assessment years earlier than testyear',
        spatial_rule='Keep old site-fold assignments; assign a new site by SHA256mod3 without moving old sites; exclude heldfold and50km buffer',
        site_assignment=assignment,source_sha256={str(p.relative_to(root)):sha(p) for p in source_paths},
        code_sha256={str(p.relative_to(root)):sha(p) for p in code_paths},publication_model_promoted=False)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for path in code_paths:
        dst=out/'code_snapshot'/path.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(path.read_bytes())
    obs.to_csv(out/'training_observation_ledger.csv',index=False)
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    np.savez_compressed(out/'predictors.npz',target_ids=obs.global_target_index,history=history.to_numpy(),shared=shared,station=station)
    temporal=[];spatial=[];memberships=[]
    for year,block in pd.read_csv(prior/'assessment_predictions.csv').groupby('season_year'):
        train=np.flatnonzero(obs.season_year.lt(year));test=position.loc[block.global_target_index].to_numpy();part=block.copy()
        for name in COMPONENTS:
            fitted,prediction=_fit_predict(designs[name],history,obs,train,test,specs[name])
            part[name]=prediction;joblib.dump(fitted,out/f'{year}_{name}.joblib')
        temporal.append(part);memberships.append(dict(fold=f'temporal{year}',train=obs.iloc[train].global_target_index.tolist(),test=part.global_target_index.tolist(),added_train_ids=obs.iloc[train].loc[lambda a:a.global_target_index.isin(added.global_target_index),'global_target_index'].tolist()))
        print('temporal',year,flush=True)
    old_spatial=pd.read_csv(prior/'spatial_predictions.csv').set_index('global_target_index')
    for split in json.loads((base/'spatial_membership.json').read_text()):
        test=position.loc[split['test']].to_numpy();old_train=position.loc[split['train']].to_numpy()
        if not len(old_train):continue
        year=int(obs.iloc[test].season_year.iloc[0]);fold=int(re.search(r'sitefold(\d+)',split['fold']).group(1))
        eligible=added.season_year.lt(year)&added.site_id.map(assignment).ne(fold)
        candidates=position.loc[added.loc[eligible,'global_target_index']].to_numpy()
        if len(candidates):
            distance=distances_km(obs.iloc[candidates][['latitude','longitude']],obs.iloc[test][['latitude','longitude']])
            candidates=candidates[distance.min(axis=1)>=50]
        train=np.r_[old_train,candidates]
        part=old_spatial.loc[split['test']].reset_index().copy()
        for name in COMPONENTS:
            fitted,prediction=_fit_predict(designs[name],history,obs,train,test,specs[name]);part[name]=prediction
        spatial.append(part);memberships.append(dict(fold=split['fold'],train=obs.iloc[train].global_target_index.tolist(),test=part.global_target_index.tolist(),added_train_ids=obs.iloc[candidates].global_target_index.tolist()))
    (out/'split_membership.json').write_text(json.dumps(memberships,indent=2)+'\n')
    metrics=[];intervals=[];diagnostics=[]
    for label,frame,filename in [('temporal',pd.concat(temporal,ignore_index=True),'assessment_predictions.csv'),('spatial_50km',pd.concat(spatial,ignore_index=True),'spatial_predictions.csv')]:
        for name,components in BLENDS.items():frame[name]=frame[components].mean(axis=1)
        frame['unaugmented_process_station_history_mean']=frame[['bagged_shared','bagged_station_history_split']].mean(axis=1)
        frame.to_csv(out/filename,index=False)
        for name in COMPONENTS+list(BLENDS):
            metrics.append(dict(evaluation=label,model=name,**score_predictions(frame,name)))
            for reference in ['reference_replay','unaugmented_process_station_history_mean']:
                ci=paired_interval(frame,name,reference);intervals.append(dict(evaluation=label,model=name,reference=reference,lower=ci[0],upper=ci[1]))
        if label=='temporal':
            for population,mask in dict(cold=~frame.prior_assessment_available,warm=frame.prior_assessment_available,BASF=frame.source.eq('BASF'),Corteva=frame.source.eq('Corteva'),severe_above75=frame.observed_percent.gt(75),new_site=frame.new_site).items():
                for name in BLENDS:diagnostics.append(dict(population=population,model=name,**score_subset(frame,mask,name)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'subgroup_metrics.csv',index=False)
    return pd.DataFrame(metrics)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):print(run(output=args.output).to_string(index=False))
