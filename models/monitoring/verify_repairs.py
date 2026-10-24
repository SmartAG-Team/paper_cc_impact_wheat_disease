"""Independently recompute metrics and replay saved temporal repair models."""
from __future__ import annotations
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


def verify(root, output):
    root,output=Path(root),Path(output)
    registration=json.loads((output/'registration.json').read_text())
    for name,digest in registration['source_sha256'].items():
        if sha(root/name)!=digest:
            raise ValueError(f'Source hash mismatch: {name}')
    for name,digest in registration['code_sha256'].items():
        if sha(output/'code_snapshot'/name)!=digest:
            raise ValueError(f'Frozen code hash mismatch: {name}')
    path=output/'code_snapshot/analysis/disease_improvement/models.py'
    spec=importlib.util.spec_from_file_location('frozen_repair_models',path)
    models=importlib.util.module_from_spec(spec);spec.loader.exec_module(models)
    obs=pd.read_csv(output/'observation_ledger.csv')
    base=root/'outputs/disease_improvement/quality_weather_20261021_v2'
    schema=json.loads((base/'feature_schema.json').read_text())
    old=pd.read_csv(base/'assessment_predictions.csv')
    temporal=pd.read_csv(output/'assessment_predictions.csv')
    spatial=pd.read_csv(output/'spatial_predictions.csv')
    metrics=pd.read_csv(output/'metrics.csv')
    arrays=np.load(output/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    history=pd.DataFrame(arrays['history'],columns=schema['history_names'])
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    mapping=dict(reference_replay='reference',shared_code='reanalysis',station_rain_shared_code='station_rain',
        station_all_shared_code='station_all',shared_code_direct='reanalysis',station_all_shared_code_direct='station_all')
    replays=0
    for year,part in temporal.groupby('season_year'):
        test=position.loc[part.global_target_index].to_numpy()
        expected_train=obs.loc[obs.season_year.lt(year),'global_target_index'].astype(int).tolist()
        for name,key in mapping.items():
            fitted=joblib.load(output/f'{year}_{name}.joblib')
            if fitted['training_ids']!=expected_train:
                raise ValueError('Temporal training identities differ')
            replay=models.predict_candidate(fitted,arrays[key][test],history.iloc[test],part.drop(columns='observed_percent'))
            np.testing.assert_allclose(replay,part[name],atol=1e-10)
            replays+=1
    for label,frame in [('temporal',temporal),('spatial_50km',spatial)]:
        if frame.global_target_index.duplicated().any() or set(frame.global_target_index)!=set(old.global_target_index):
            raise ValueError('Evaluation cohort changed')
        for name in mapping:
            if not frame[name].between(0,100).all():
                raise ValueError('Invalid severity prediction')
            actual=score_predictions(frame,name)
            recorded=metrics.loc[metrics.evaluation.eq(label)&metrics.model.eq(name)].iloc[0]
            for key in ['rmse','mae','bias','r2','integral_rmse','n','fields','sites','windows']:
                np.testing.assert_allclose(actual[key],recorded[key],atol=1e-9)
    np.testing.assert_allclose(temporal.reference_replay,temporal.selected_candidate,atol=1e-10)
    membership=json.loads((output/'spatial_membership.json').read_text())
    indexed=obs.set_index('global_target_index')
    for split in membership:
        if not split['train']:continue
        train=indexed.loc[split['train']];test=indexed.loc[split['test']]
        assert train.season_year.max()<test.season_year.min()
        assert not set(train.site_id)&set(test.site_id)
    weather=pd.read_parquet(output/'station_preferred_daily_weather.parquet')
    assert not weather.duplicated(['location_id','date']).any()
    required=['tmean_c','tmin_c','tmax_c','rh_mean_pct','rain_mm','precipitation_mm','rain_hours_gt_0_1mm','rh_hours_ge_90pct']
    assert np.isfinite(weather[required].to_numpy(float)).all()
    assert weather.rain_mm.ge(0).all() and weather.precipitation_mm.ge(0).all()
    receipt=dict(verified=True,temporal_assessments=len(temporal),spatial_assessments=len(spatial),
        saved_model_replays=replays,metric_rows=len(metrics),source_hashes=len(registration['source_sha256']),
        frozen_code_hashes=len(registration['code_sha256']),raw_observations_changed=False,
        additional_exclusions=[],independent_external_validation=False,
        publication_model_promoted=False)
    (output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(Path(__file__).resolve().parents[2],args.output),indent=2))
