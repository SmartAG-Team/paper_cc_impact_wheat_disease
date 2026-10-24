"""Replay saved stage/cluster models and check source and information boundaries."""
import argparse
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .evaluation import score_predictions
from .inputs import sha


def verify(root,output):
    root,output=Path(root),Path(output)
    protocol=json.loads((output/'registration.json').read_text())
    for path,digest in protocol['source_sha256'].items():
        if sha(root/path)!=digest:raise ValueError('Source hash mismatch: '+path)
    for path,digest in protocol['code_sha256'].items():
        if sha(output/'code_snapshot'/path)!=digest:raise ValueError('Frozen code hash mismatch: '+path)
    spec=importlib.util.spec_from_file_location('frozen_stage_models',output/'code_snapshot/analysis/disease_improvement/models.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    obs=pd.read_csv(output/'observation_ledger.csv')
    original=root/'outputs/disease_improvement/station_repairs_20261022_v4'
    old=pd.read_csv(original/'assessment_predictions.csv');old_spatial=pd.read_csv(original/'spatial_predictions.csv')
    temporal=pd.read_csv(output/'assessment_predictions.csv');spatial=pd.read_csv(output/'spatial_predictions.csv')
    metrics=pd.read_csv(output/'metrics.csv');stage=pd.read_csv(output/'stage_provenance.csv')
    history_names=json.loads((root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json').read_text())['history_names']
    arrays=np.load(output/'predictors.npz');np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    h=pd.DataFrame(arrays['history'],columns=history_names)
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    n_models=0;n_members=0
    for year,part in temporal.groupby('season_year'):
        ids=set(obs.loc[obs.season_year.lt(year),'global_target_index'])
        selected=position.loc[part.global_target_index].to_numpy()
        for name in protocol['recipes']:
            model=joblib.load(output/f'{year}_{name}.joblib')
            assert set(model['training_ids'])==ids
            models=model.get('members',[model]);predictions=[]
            for member in models:
                assert set(member['training_ids'])<=ids
                if 'members' in model:
                    train=obs.loc[obs.global_target_index.isin(member['training_ids'])]
                    eligible=obs.loc[obs.season_year.lt(year)&obs.coordinate_year.isin(train.coordinate_year)]
                    assert set(train.global_target_index)==set(eligible.global_target_index)
                predictions.append(frozen.predict_candidate(member,arrays[name][selected],h.iloc[selected],part.drop(columns='observed_percent')))
                n_members+=1
            replay=np.mean(predictions,axis=0)
            np.testing.assert_allclose(replay,part[name],atol=1e-10)
            n_models+=1
    for label,frame,prior in [('temporal',temporal,old),('spatial_50km',spatial,old_spatial)]:
        assert set(frame.global_target_index)==set(prior.global_target_index) and not frame.global_target_index.duplicated().any()
        indexed=prior.set_index('global_target_index').loc[frame.global_target_index]
        for name,reference in [('reference_replay','reference_replay'),('shared_replay','shared_code'),('station_replay','station_all_shared_code')]:
            np.testing.assert_allclose(frame[name],indexed[reference],atol=1e-10)
        for name in protocol['recipes']:
            assert frame[name].between(0,100).all()
            recalculated=score_predictions(frame,name)
            saved=metrics.loc[metrics.evaluation.eq(label)&metrics.model.eq(name)].iloc[0]
            for key in ['rmse','mae','bias','r2','integral_rmse','n','sites','fields','windows']:
                np.testing.assert_allclose(recalculated[key],saved[key],atol=1e-9)
    keyed=obs.set_index('global_target_index').loc[stage.global_target_index]
    now=pd.to_datetime(keyed.date).reset_index(drop=True)
    latest=pd.to_datetime(stage.latest_stage_date)
    assert (latest.isna()|latest.lt(now)).all()
    assert pd.to_datetime(stage.weather_end).lt(now).all()
    assert (output/'spatial_membership.json').read_bytes()==(original/'spatial_membership.json').read_bytes()
    result=dict(verified=True,assessments=len(temporal),spatial_assessments=len(spatial),saved_models_replayed=n_models,
        component_models_replayed=n_members,source_hashes=len(protocol['source_sha256']),
        frozen_code_hashes=len(protocol['code_sha256']),metrics_recomputed=len(metrics),
        stage_history_strictly_prior=True,weather_strictly_prior=True,new_observation_exclusions=0,
        independent_external_validation=False,publication_model_promoted=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):print(json.dumps(verify(Path(__file__).resolve().parents[2],args.output),indent=2))
