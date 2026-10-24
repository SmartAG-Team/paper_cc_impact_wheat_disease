"""Verify French station reconstruction, frozen predictions and fixed splits."""
import argparse
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .evaluation import score_predictions,score_subset,paired_interval
from .inputs import WEATHER,sha
from .meteo_france_station import ROOT,OUTPUT,CACHE
from .repairs import dwd_elevation_screen,hybrid_weather


def load_frozen(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write_verification_receipt(output,result,code_path,spatial=False):
    """Append versioned evidence without changing export-bound receipt bytes."""
    output,code_path = Path(output),Path(code_path)
    digest = sha(code_path)
    assert result['verifier_sha256']==digest
    directory = output/'verification_receipts'/digest
    directory.mkdir(parents=True,exist_ok=True)
    content = (json.dumps(result,indent=2)+'\n').encode()
    target = directory/('spatial_refit.json' if spatial else 'temporal_replay.json')
    if target.exists():
        assert target.read_bytes()==content
    else:
        target.write_bytes(content)
    snapshot = directory/'verifier.py'
    if snapshot.exists():
        assert sha(snapshot)==digest
    else:
        snapshot.write_bytes(code_path.read_bytes())
    if not (output/'verification.json').exists():
        (output/'verification.json').write_bytes(content)
        (output/'verification_code_snapshot.py').write_bytes(code_path.read_bytes())
    if spatial and not (output/'spatial_refit_verification.json').exists():
        (output/'spatial_refit_verification.json').write_bytes(content)
        (output/'spatial_verification_code_snapshot.py').write_bytes(code_path.read_bytes())
    return target


def verify(root,output,refit_spatial=False):
    root,output = Path(root),Path(output)
    protocol = json.loads((output/'registration.json').read_text())
    for path,digest in protocol['source_sha256'].items():
        assert sha(root/path)==digest,path
    for path,digest in protocol['code_sha256'].items():
        assert sha(output/'code_snapshot'/path)==digest,path
    metrics_path = 'analysis/final_study/metrics.py'
    assert sha(root/metrics_path)==protocol['code_sha256'][metrics_path]
    qa = root/protocol.get('station_validation_directory',OUTPUT)
    for filename in ['receipt.json','timing_receipt.json']:
        receipt = json.loads((qa/filename).read_text())
        for path,digest in receipt['source_sha256'].items():
            assert sha(root/path)==digest,path
        for archive in receipt['archives']:
            assert sha(root/archive['path'])==archive['sha256'],archive['path']
    approval = pd.read_csv(qa/'timing_approval.csv',dtype={'station_id':str})
    matches = pd.read_csv(qa/'qualified_station_matches.csv',dtype={'station_id':str})
    assert set(matches.station_id)<=set(approval.loc[approval.timing_approved,'station_id'])
    assert matches.site_id.is_unique and matches.distance_km.le(50).all()
    assert matches.elevation_difference_m.le(300).all() and matches.coverage.ge(.8).all()
    # Confirm complete 24-hour provider-approved observations create each selected
    # station's rainfall totals. These saved hours remain hash-traceable to archives.
    collector = load_frozen(output/'code_snapshot/analysis/disease_improvement/meteo_france_station.py',
                            'analysis.disease_improvement.frozen_mf_collector')
    station_daily = pd.read_parquet(qa/'station_daily.parquet')
    for station in sorted(matches.station_id.unique()):
        hourly = pd.read_parquet(qa/f'{station}_hourly.parquet')
        for name in ['RR1','T','U']:
            assert hourly.loc[hourly[name].notna(),'Q'+name].isin([0,1]).all()
        rebuilt = collector.daily_aggregate(hourly)
        saved = station_daily.loc[station_daily.station_id.eq(station)].reset_index(drop=True)
        pd.testing.assert_frame_equal(rebuilt,saved,check_dtype=False)
    base = root/'outputs/disease_improvement/station_repairs_20261022_v4'
    previous = root/'outputs/disease_improvement/history_anchor_20261022'
    blendbase = root/'outputs/disease_improvement/ensemble_blend_20261022'
    obs = pd.read_csv(output/'observation_ledger.csv')
    pd.testing.assert_frame_equal(obs,pd.read_csv(base/'observation_ledger.csv'),check_dtype=False)
    assert (output/'spatial_membership.json').read_bytes()==(base/'spatial_membership.json').read_bytes()
    arrays = np.load(output/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    names = json.loads((root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json').read_text())['history_names']
    history = pd.DataFrame(arrays['history'],columns=names)
    position = pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    model = load_frozen(output/'code_snapshot/analysis/disease_improvement/models.py','frozen_mf_model')
    metrics = pd.read_csv(output/'metrics.csv')
    intervals = pd.read_csv(output/'paired_intervals.csv')
    subgroups = pd.read_csv(output/'subgroup_metrics.csv')
    temporal_count = component_count = spatial_count = 0
    frames = {}
    for label,filename in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]:
        frame = pd.read_csv(output/filename)
        original = pd.read_csv(previous/filename).set_index('global_target_index').loc[frame.global_target_index].reset_index()
        pd.testing.assert_frame_equal(frame[original.columns],original,check_dtype=False)
        old_blend = pd.read_csv(blendbase/filename).set_index('global_target_index').loc[frame.global_target_index]
        np.testing.assert_allclose(frame.previous_blend,old_blend.process_station_history_mean,atol=1e-10)
        for name,component in protocol['blends'].items():
            np.testing.assert_allclose(frame[name],.5*(frame.bagged_shared+frame[component]),atol=1e-10)
        for name in list(protocol['recipes'])+list(protocol['blends'])+['previous_blend']:
            saved = metrics.loc[metrics.evaluation.eq(label)&metrics.model.eq(name)].iloc[0]
            for key,value in score_predictions(frame,name).items():
                np.testing.assert_allclose(value,saved[key],atol=1e-9)
            if name in protocol['blends']:
                saved = intervals.loc[intervals.evaluation.eq(label)&intervals.model.eq(name)].iloc[0]
                np.testing.assert_allclose(paired_interval(frame,name,'previous_blend'),saved[['lower','upper']].to_numpy(float),atol=1e-9)
        frames[label] = frame
        if label!='temporal':
            continue
        for year,part in frame.groupby('season_year'):
            eligible = set(obs.loc[obs.season_year.lt(year),'global_target_index'])
            selected = position.loc[part.global_target_index].to_numpy()
            for name in protocol['recipes']:
                fitted = joblib.load(output/f'{year}_{name}.joblib')
                assert set(fitted['training_ids'])==eligible
                predictions = []
                for member in fitted['members']:
                    ids = set(member['training_ids']);assert ids<=eligible
                    train = obs.loc[obs.global_target_index.isin(ids)]
                    whole = obs.loc[obs.season_year.lt(year)&obs.coordinate_year.isin(train.coordinate_year)]
                    assert ids==set(whole.global_target_index)
                    predictions.append(model.predict_candidate(member,arrays[name][selected],history.iloc[selected],part.drop(columns='observed_percent')))
                    component_count += 1
                np.testing.assert_allclose(np.mean(predictions,axis=0),part[name],atol=1e-10)
                temporal_count += 1
    for row in subgroups.itertuples(index=False):
        frame = frames['temporal']
        masks = dict(cold=~frame.prior_assessment_available,warm=frame.prior_assessment_available,
            BASF=frame.source.eq('BASF'),Corteva=frame.source.eq('Corteva'),
            severe_above75=frame.observed_percent.gt(75),new_site=frame.new_site)
        recalculated = score_subset(frame,masks[row.population],row.model)
        for key,value in recalculated.items():
            if value is not None:
                np.testing.assert_allclose(value,getattr(row,key),atol=1e-9)
    for name in protocol['recipes']:
        for filename,columns in [(f'{name}_feature_provenance.csv',['weather_end']),
                                 (f'{name}_stage_provenance.csv',['weather_end','latest_stage_date'])]:
            provenance = pd.read_csv(output/filename)
            dates = pd.to_datetime(obs.set_index('global_target_index').loc[provenance.global_target_index,'date']).reset_index(drop=True)
            for column in columns:
                used = pd.to_datetime(provenance[column]);assert (used.isna()|used.lt(dates)).all()
    if refit_spatial:
        # Fit only with the frozen estimator implementation and the registered
        # whole-group subsampling rule; no outcomes from held sets enter fitting.
        for split in json.loads((output/'spatial_membership.json').read_text()):
            if not len(split['train']):
                continue
            train = position.loc[split['train']].to_numpy();test = position.loc[split['test']].to_numpy()
            train_frame = obs.iloc[train]
            clusters = np.asarray(sorted(train_frame.coordinate_year.unique()))
            count = max(1,int(np.ceil(len(clusters)*.8)))
            expected = frames['spatial_50km'].set_index('global_target_index').loc[split['test']]
            for name,spec in protocol['recipes'].items():
                rng = np.random.default_rng(20261022);predictions = []
                for _ in range(20):
                    chosen = rng.choice(clusters,size=count,replace=False)
                    selected = train[train_frame.coordinate_year.isin(chosen).to_numpy()]
                    fitted = model.fit_candidate(arrays[name][selected],history.iloc[selected],obs.iloc[selected],spec)
                    assert set(fitted['training_ids'])<=set(split['train'])
                    predictions.append(model.predict_candidate(fitted,arrays[name][test],history.iloc[test],obs.iloc[test].drop(columns='observed_percent')))
                np.testing.assert_allclose(np.mean(predictions,axis=0),expected[name],atol=1e-10)
                spatial_count += 1
            print('verified spatial',split['fold'],flush=True)
    result = dict(verified=True,assessments=len(frames['temporal']),source_archives_checked=32,
        saved_temporal_ensembles_replayed=temporal_count,component_models_replayed=component_count,
        spatial_ensembles_independently_refitted=spatial_count,metrics_recomputed=len(metrics),
        paired_intervals_recomputed=len(intervals),subgroup_rows_recomputed=len(subgroups),
        original_observations_and_splits_unchanged=True,strictly_prior_weather_and_stage=True,
        independent_external_validation=False,publication_model_promoted=False,
        verifier_sha256=sha(Path(__file__)))
    write_verification_receipt(output,result,Path(__file__),spatial=refit_spatial)
    return result


def export_preferred(root,output,destination=None):
    root,output = Path(root).resolve(),Path(output).resolve()
    folder = Path(destination or root/'data/disease_improvement/station_preferred_france_20261022').resolve()
    if (folder/'manifest.json').exists():
        raise FileExistsError('Use a new output directory to preserve the archived preferred forcing')
    protocol = json.loads((output/'registration.json').read_text())
    qa = root/protocol.get('station_validation_directory',OUTPUT)
    from .inputs import INPUT
    forcing = pd.read_parquet(root/WEATHER)
    pairs = pd.read_parquet(root/'outputs/weather_validation_20261022/dwd/paired_daily.parquet')
    eligible = dwd_elevation_screen(root)
    rain,first = hybrid_weather(forcing,pairs,eligible['RR'],rainfall=True,meteorology=False)
    station,second = hybrid_weather(rain,pairs,eligible['TU'])
    france = pd.read_parquet(qa/'qualified_paired_daily.parquet')
    weather,third = hybrid_weather(station,france,set(france.site_id),rainfall=True,
                                  meteorology=True,source='METEO_FRANCE')
    fitted_weather = pd.read_parquet(output/'mf_all_history_split_daily_weather.parquet')
    pd.testing.assert_frame_equal(weather[fitted_weather.columns],fitted_weather,check_dtype=False)
    folder.mkdir(parents=True,exist_ok=True)
    weather.to_parquet(folder/'daily_weather.parquet',index=False)
    pd.concat([first,second,third],ignore_index=True).to_csv(folder/'replacement_coverage.csv',index=False)
    provenance_files = {}
    for recipe in ['mf_rain_history_split','mf_all_history_split']:
        filename = f'{recipe}_feature_provenance.csv'
        provenance = pd.read_csv(output/filename)
        provenance['wetness_measurement'] = 'RH_and_precipitation_duration_inputs_plus_modeled_exposure_proxy; not_measured_canopy_leaf_wetness'
        provenance.to_csv(folder/filename,index=False)
        provenance_files[filename] = sha(folder/filename)
    origin_columns = [c for c in weather if c.startswith('origin_')]
    sources = [root/WEATHER,root/'outputs/weather_validation_20261022/dwd/paired_daily.parquet',
        qa/'qualified_paired_daily.parquet',qa/'receipt.json',qa/'timing_receipt.json',
        output/'registration.json',output/'verification.json',output/'mf_all_history_split_daily_weather.parquet']
    for name in ['spatial_refit_verification.json','spatial_verification_code_snapshot.py']:
        if (output/name).exists():
            sources.append(output/name)
    manifest = dict(rows=len(weather),sites=weather.location_id.nunique(),
        dataset_sha256=sha(folder/'daily_weather.parquet'),source_sha256={str(p.relative_to(root)):sha(p) for p in sources},
        origin_counts={c:weather[c].value_counts(dropna=False).to_dict() for c in origin_columns},
        rainfall='Measured total precipitation; liquid phase remains estimated, with explicit warm-day or reanalysis phase assumptions',
        rain_hours='Explicit origins; thresholded hourly precipitation on warm-liquid days, or measured dry total, otherwise original input',
        observed_canopy_leaf_wetness=False,
        feature_provenance_files_sha256=provenance_files,
        fallback='Original reanalysis for missing or unqualified values; NOAA rainfall remains disabled',
        temporal_use='Strictly earlier UTC calendar dates; historical final QC, not verified real-time publication latency',
        model_run_numerical_inputs_exactly_preserved=True,origin_annotations_added_after_model_run=True,
        source_preference=True,independent_external_validation=False,publication_model_promoted=False,
        export_code_sha256=sha(Path(__file__)),hybrid_code_sha256=sha(Path(__file__).with_name('repairs.py')))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'export_code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return dict(path=str(folder.relative_to(root)),rain_origins=manifest['origin_counts']['origin_precipitation_mm'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--refit-spatial',action='store_true')
    parser.add_argument('--export-preferred',action='store_true')
    parser.add_argument('--preferred-output',type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(ROOT,args.output,args.refit_spatial),indent=2))
        if args.export_preferred:
            print(json.dumps(export_preferred(ROOT,args.output,args.preferred_output),indent=2))
