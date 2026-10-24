"""Fixed tests of same-field cross-leaf anchors on the retained benchmark."""
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
from .field_history_models import field_history_anchor,fit_ensemble,predict_ensemble
from .inputs import sha

ROOT=Path(__file__).resolve().parents[2]
RECIPES={name:dict(structure=name) for name in ['field_residual','field_split']}


def run(root=ROOT,output=None):
    root=Path(root).resolve();out=Path(output or root/'outputs/disease_improvement/field_history_20261022').resolve()
    if out.exists():raise FileExistsError('A new output directory is required')
    base=root/'outputs/disease_improvement/french_station_20261022'
    obs=pd.read_csv(base/'observation_ledger.csv');arrays=np.load(base/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    schema_path=root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json'
    names=json.loads(schema_path.read_text())['history_names']
    history=pd.DataFrame(arrays['history'],columns=names)
    x=arrays['mf_all_history_split']
    anchor,features=field_history_anchor(history,obs)
    sources=[base/name for name in ['observation_ledger.csv','predictors.npz','registration.json',
        'assessment_predictions.csv','spatial_predictions.csv','spatial_membership.json',
        'mf_all_history_split_feature_provenance.csv','mf_all_history_split_stage_provenance.csv']]+[schema_path]
    codes=[Path(__file__),Path(__file__).with_name('field_history_models.py'),Path(__file__).with_name('models.py'),
           Path(__file__).with_name('evaluation.py'),Path(__file__).with_name('history.py'),root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),recipes=RECIPES,
        training='All2286 original qualified records; same earlier-year and50km buffered folds',
        ensemble=dict(members=20,fraction=.8,seed=20261022,groups='complete coordinate-year groups'),
        baseline='Latest same recorded leaf; otherwise nearest numeric code among most recent other leaves, average ties; source/code training mean if no history',
        anatomical_direction_assumed=False,severity_used_to_choose_anchor=False,
        blend=dict(process='unchanged bagged_shared',weights=[.5,.5]),
        tuning=False,adaptive_retrospective_development=True,independent_external_validation=False,
        source_sha256={str(p.relative_to(root)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)):sha(p) for p in codes},
        environment={n:version(n) for n in ['numpy','pandas','scikit-learn','joblib']},publication_model_promoted=False)
    (out/'registration.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for path in codes:
        dest=out/'code_snapshot'/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    for name in ['observation_ledger.csv','spatial_membership.json']:
        (out/name).write_bytes((base/name).read_bytes())
    np.savez_compressed(out/'predictors.npz',target_ids=obs.global_target_index,matrix=x,history=history.to_numpy(),anchor=anchor,anchor_features=features)
    pd.DataFrame(dict(global_target_index=obs.global_target_index,anchor_percent=100*anchor,
        same_leaf=history.prior_assessment_available,other_leaf=history.other_leaf_available,
        available=features[:,1],age_scaled=features[:,2],signed_code_delta_scaled=features[:,3],
        absolute_code_distance_scaled=features[:,4])).to_csv(out/'anchor_ledger.csv',index=False)
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    temporal,spatial=[],[]
    for year,block in pd.read_csv(base/'assessment_predictions.csv').groupby('season_year'):
        train=np.flatnonzero(obs.season_year.lt(year));test=position.loc[block.global_target_index].to_numpy()
        part=block.copy()
        for name,spec in RECIPES.items():
            fitted=fit_ensemble(x[train],history.iloc[train],obs.iloc[train],spec)
            part[name]=predict_ensemble(fitted,x[test],history.iloc[test],part.drop(columns='observed_percent'))
            joblib.dump(fitted,out/f'{year}_{name}.joblib')
        temporal.append(part);print('temporal',year,flush=True)
    old_spatial=pd.read_csv(base/'spatial_predictions.csv').set_index('global_target_index')
    for number,split in enumerate(json.loads((base/'spatial_membership.json').read_text())):
        if not len(split['train']):continue
        train=position.loc[split['train']].to_numpy();test=position.loc[split['test']].to_numpy()
        part=old_spatial.loc[split['test']].reset_index().copy()
        for name,spec in RECIPES.items():
            fitted=fit_ensemble(x[train],history.iloc[train],obs.iloc[train],spec)
            part[name]=predict_ensemble(fitted,x[test],history.iloc[test],part.drop(columns='observed_percent'))
            joblib.dump(fitted,out/f'spatial{number}_{name}.joblib')
        spatial.append(part)
    metrics,intervals,groups=[],[],[]
    for label,frame,filename in [('temporal',pd.concat(temporal,ignore_index=True),'assessment_predictions.csv'),
                                ('spatial_50km',pd.concat(spatial,ignore_index=True),'spatial_predictions.csv')]:
        for name in RECIPES:frame[name+'_mean']=.5*(frame.bagged_shared+frame[name])
        frame.to_csv(out/filename,index=False)
        for name in list(RECIPES)+[n+'_mean' for n in RECIPES]+['mf_all_mean','reference_replay']:
            metrics.append(dict(evaluation=label,model=name,**score_predictions(frame,name)))
            if name.startswith('field_'):
                ci=paired_interval(frame,name,'mf_all_mean')
                intervals.append(dict(evaluation=label,model=name,reference='mf_all_mean',lower=ci[0],upper=ci[1]))
        if label=='temporal':
            h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
            populations=dict(cold=~frame.prior_assessment_available,warm=frame.prior_assessment_available,
                first_field=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(0),
                first_leaf_other_history=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(1),
                BASF=frame.source.eq('BASF'),Corteva=frame.source.eq('Corteva'),
                severe_above75=frame.observed_percent.gt(75),new_site=frame.new_site)
            for population,mask in populations.items():
                for name in list(RECIPES)+[n+'_mean' for n in RECIPES]+['mf_all_mean']:
                    groups.append(dict(population=population,model=name,**score_subset(frame,mask,name)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'subgroup_metrics.csv',index=False)
    frame=pd.read_csv(out/'assessment_predictions.csv')
    frame.loc[frame.global_target_index.isin([479,1750,413,159,1231,122,71,1748]),
        ['global_target_index','observed_percent','mf_all_mean']+list(RECIPES)+[n+'_mean' for n in RECIPES]].to_csv(out/'hard_cases.csv',index=False)
    return pd.DataFrame(metrics)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):print(run(output=args.output).to_string(index=False))
