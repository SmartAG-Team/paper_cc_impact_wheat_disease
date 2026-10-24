"""Fixed station-forcing extensions of the existing disease-history blend."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .dwd_station import join_station_daily
from .evaluation import score_predictions,score_subset,paired_interval
from .features import build_features
from .inputs import sha
from .meteo_france_station import ROOT,OUTPUT
from .repairs import hybrid_weather,recorded_leaf_features
from .stage_experiment import augment,_fit_predict,SPEC
from .stage_features import build_stage_features

RECIPES = {'mf_rain_history_split':dict(SPEC,structure='split',ensemble=True),
           'mf_all_history_split':dict(SPEC,structure='split',ensemble=True)}
BLENDS = {'mf_rain_mean':'mf_rain_history_split','mf_all_mean':'mf_all_history_split'}


def qualified_pairs(root,folder=None):
    folder = Path(folder or Path(root)/OUTPUT)
    options = pd.read_csv(folder/'station_candidates.csv',dtype={'station_id':str})
    coverage = pd.read_csv(folder/'candidate_coverage.csv',dtype={'station_id':str})
    approval = pd.read_csv(folder/'timing_approval.csv',dtype={'station_id':str})
    approved = set(approval.loc[approval.timing_approved,'station_id'])
    candidates = options.merge(coverage[['site_id','station_id','coverage']],on=['site_id','station_id'],validate='one_to_one')
    candidates = candidates.loc[candidates.station_id.isin(approved)&candidates.coverage.ge(.8)]
    chosen = candidates.sort_values(['site_id','distance_km','station_id']).drop_duplicates('site_id')
    if not len(chosen):
        raise ValueError('No rainfall station passes timing and coverage qualification')
    from .inputs import WEATHER
    weather = pd.read_parquet(Path(root)/WEATHER);weather['date'] = pd.to_datetime(weather.date)
    station = pd.read_parquet(folder/'station_daily.parquet')
    rows = []
    for match in chosen.itertuples(index=False):
        daily = station.loc[station.station_id.eq(match.station_id)].drop(columns='station_id')
        pair = join_station_daily(weather.loc[weather.location_id.eq(match.site_id)],daily)
        pair['site_id'] = match.site_id;pair['station_id_RR'] = match.station_id
        pair['distance_km_RR'] = match.distance_km
        rows.append(pair)
    pairs = pd.concat(rows,ignore_index=True)
    if (folder/'qualified_station_matches.csv').exists():
        saved = pd.read_csv(folder/'qualified_station_matches.csv',dtype={'station_id':str})
        pd.testing.assert_frame_equal(chosen.reset_index(drop=True),saved,check_dtype=False)
    else:
        chosen.to_csv(folder/'qualified_station_matches.csv',index=False)
    if (folder/'qualified_paired_daily.parquet').exists():
        saved = pd.read_parquet(folder/'qualified_paired_daily.parquet')
        comparison = pairs.copy()
        for frame in [comparison,saved]:
            frame['solid_precipitation_observed'] = frame.solid_precipitation_observed.astype('boolean')
        pd.testing.assert_frame_equal(comparison,saved,check_dtype=False)
    else:
        pairs.to_parquet(folder/'qualified_paired_daily.parquet',index=False)
    return pairs


def run(root=ROOT,output=None,station_validation=None):
    root = Path(root)
    out = Path(output or root/'outputs/disease_improvement/french_station_20261022')
    if out.exists():
        raise FileExistsError('A new output directory is required')
    base = root/'outputs/disease_improvement/station_repairs_20261022_v4'
    historybase = root/'outputs/disease_improvement/history_anchor_20261022'
    blendbase = root/'outputs/disease_improvement/ensemble_blend_20261022'
    meta = root/'outputs/raw_metadata_recovery_20261022'
    qa = Path(station_validation or root/OUTPUT).resolve()
    pairs = qualified_pairs(root,qa)
    obs = pd.read_csv(base/'observation_ledger.csv')
    fields = pd.read_csv(root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv')
    schema = json.loads((root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json').read_text())
    arrays = np.load(base/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    history = pd.DataFrame(arrays['history'],columns=schema['history_names'])
    oldweather = pd.read_parquet(base/'station_preferred_daily_weather.parquet')
    calendar = pd.read_csv(meta/'stage_calendar.csv')
    prior = pd.read_json(root/'outputs/disease_improvement/quality_weather_20261021_v2/history_provenance.json')
    config = dict(weather_cutoff_days=1,rain_variable='rain_mm',prior_dates=dict(zip(prior.global_target_index,prior.latest_same_date)))
    source_paths = [base/'observation_ledger.csv',base/'predictors.npz',base/'spatial_membership.json',base/'station_preferred_daily_weather.parquet',
        historybase/'assessment_predictions.csv',historybase/'spatial_predictions.csv',meta/'stage_calendar.csv',
        blendbase/'assessment_predictions.csv',blendbase/'spatial_predictions.csv',
        qa/'qualified_station_matches.csv',qa/'qualified_paired_daily.parquet',qa/'receipt.json',qa/'timing_receipt.json',
        qa/'timing_approval.csv',qa/'timing_policy.json',qa/'station_daily.parquet',
        root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv',
        root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json',
        root/'outputs/disease_improvement/quality_weather_20261021_v2/history_provenance.json']
    code_paths = [Path(__file__)]+[Path(__file__).with_name(name+'.py') for name in
        ['meteo_france_station','meteo_france_timing','models','robust_models','stage_experiment','stage_features',
         'repairs','features','evaluation']]+[root/name for name in ['analysis/final_study/metrics.py',
         'model/seasonal_septoria/core.py','model/seasonal_septoria/wetness.py']]
    out.mkdir(parents=True)
    protocol = dict(registered_utc=datetime.now(timezone.utc).isoformat(),recipes=RECIPES,blends=BLENDS,
        station_validation_directory=str(qa.relative_to(root)),
        blend_weights=[.5,.5],process_component='Unchanged bagged_shared reanalysis process ensemble',
        history_component='Existing20-member split history head with shared code, earlier stages, and extended station forcing',
        forcing='DWD unchanged; add qualified French rain only or French rain/temperature/humidity; missing fallback',
        tuning=False,adaptive_retrospective_development=True,independent_external_validation=False,
        source_sha256={str(p.relative_to(root)):sha(p) for p in source_paths},
        code_sha256={str(p.relative_to(root)):sha(p) for p in code_paths},
        publication_model_promoted=False)
    (out/'registration.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for path in code_paths:
        destination = out/'code_snapshot'/path.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes())
    (out/'spatial_membership.json').write_bytes((base/'spatial_membership.json').read_bytes())
    obs.to_csv(out/'observation_ledger.csv',index=False)
    designs,ledgers = {},[]
    for name in RECIPES:
        forcing,ledger = hybrid_weather(oldweather,pairs,set(pairs.site_id),rainfall=True,
            meteorology=name=='mf_all_history_split',source='METEO_FRANCE')
        ledger['recipe'] = name;ledgers.append(ledger)
        forcing.to_parquet(out/f'{name}_daily_weather.parquet',index=False)
        events,names,provenance = build_features(forcing,fields,obs,config)
        provenance.to_csv(out/f'{name}_feature_provenance.csv',index=False)
        stage,stage_names,stage_provenance = build_stage_features(forcing,fields,obs,calendar)
        stage_provenance.to_csv(out/f'{name}_stage_provenance.csv',index=False)
        wc = [i for i,n in enumerate(names) if n.startswith('weather_')]
        pc = [i for i,n in enumerate(names) if n.startswith('process_')]
        assert len(wc)==12 and len(pc)==3
        category,_ = recorded_leaf_features(obs)
        designs[name] = augment(np.column_stack([events[:,wc],category,events[:,pc]]),stage)
    pd.concat(ledgers,ignore_index=True).to_csv(out/'replacement_coverage.csv',index=False)
    np.savez_compressed(out/'predictors.npz',target_ids=obs.global_target_index,history=history.to_numpy(),**designs)
    position = pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    temporal,spatial = [],[]
    for year,block in pd.read_csv(historybase/'assessment_predictions.csv').groupby('season_year'):
        train = np.flatnonzero(obs.season_year.lt(year));test = position.loc[block.global_target_index].to_numpy()
        part = block.copy()
        for name,spec in RECIPES.items():
            fitted,part[name] = _fit_predict(designs[name],history,obs,train,test,spec)
            joblib.dump(fitted,out/f'{year}_{name}.joblib')
        temporal.append(part);print('temporal',year,flush=True)
    old_spatial = pd.read_csv(historybase/'spatial_predictions.csv').set_index('global_target_index')
    for split in json.loads((base/'spatial_membership.json').read_text()):
        train = position.loc[split['train']].to_numpy();test = position.loc[split['test']].to_numpy()
        if not len(train):
            continue
        part = old_spatial.loc[split['test']].reset_index().copy()
        for name,spec in RECIPES.items():
            _,part[name] = _fit_predict(designs[name],history,obs,train,test,spec)
        spatial.append(part)
    metrics,intervals,diagnostics = [],[],[]
    for label,frame,filename in [('temporal',pd.concat(temporal,ignore_index=True),'assessment_predictions.csv'),
                                  ('spatial_50km',pd.concat(spatial,ignore_index=True),'spatial_predictions.csv')]:
        old_blend = pd.read_csv(blendbase/filename).set_index('global_target_index')
        frame['previous_blend'] = old_blend.loc[frame.global_target_index,'process_station_history_mean'].to_numpy()
        for name,component in BLENDS.items():
            frame[name] = .5*(frame.bagged_shared+frame[component])
        frame.to_csv(out/filename,index=False)
        for name in list(RECIPES)+list(BLENDS)+['previous_blend']:
            metrics.append(dict(evaluation=label,model=name,**score_predictions(frame,name)))
            if name in BLENDS:
                ci = paired_interval(frame,name,'previous_blend')
                intervals.append(dict(evaluation=label,model=name,reference='previous_blend',lower=ci[0],upper=ci[1]))
        if label=='temporal':
            for population,mask in dict(cold=~frame.prior_assessment_available,warm=frame.prior_assessment_available,
                BASF=frame.source.eq('BASF'),Corteva=frame.source.eq('Corteva'),
                severe_above75=frame.observed_percent.gt(75),new_site=frame.new_site).items():
                for name in list(BLENDS)+['previous_blend']:
                    diagnostics.append(dict(population=population,model=name,**score_subset(frame,mask,name)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'subgroup_metrics.csv',index=False)
    return pd.DataFrame(metrics)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--station-validation',type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output,station_validation=args.station_validation).to_string(index=False))
