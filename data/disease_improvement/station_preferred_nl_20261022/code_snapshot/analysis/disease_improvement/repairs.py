"""Fixed recorded-leaf repair and station-preferred weather development checks.

Station total precipitation remains measured total. Liquid input is explicitly
phase-assisted, never mislabeled as a measured liquid amount. Stations and
fallback policies are selected without disease outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .dwd_station import parse_inventory, pair_metrics
from .evaluation import paired_interval, score_predictions, score_subset
from .features import build_features
from .inputs import WEATHER, sha
from .models import fit_candidate, predict_candidate

ROOT = Path(__file__).resolve().parents[2]
BASE = 'outputs/disease_improvement/quality_weather_20261021_v2'
SPEC = dict(structure='process', crossleaf=True, process_baseline_column=-3)


def recorded_leaf_features(frame, shared=True):
    """Retain source interactions while making recorded codes transferable.

    Code order is an empirical predictor, not verified anatomical leaf age.
    """
    if not frame.source.isin(['BASF', 'Corteva']).all() or not frame.leaf_index.isin(range(7)).all():
        raise ValueError('Declared source and recorded leaf code required')
    names = [f'{source}:code{code}' for source in ['BASF', 'Corteva'] for code in range(7)]
    arrays = [(frame.source.eq(source)&frame.leaf_index.eq(code)).to_numpy(float)
              for source in ['BASF', 'Corteva'] for code in range(7)]
    if shared:
        names += [f'shared_recorded_code{code}' for code in range(7)]+['recorded_code_scaled']
        arrays += [frame.leaf_index.eq(code).to_numpy(float) for code in range(7)]
        arrays += [frame.leaf_index.to_numpy(float)/6]
    return np.column_stack(arrays), names


def hybrid_weather(weather, pairs, eligible_sites, *, rainfall=False, meteorology=True,
                   source='DWD'):
    """Prefer valid eligible station values; missing days retain original forcing.

    A measured zero needs no phase inference. Positive total is treated as liquid
    at Tmin>2C; otherwise the original reanalysis liquid/total ratio estimates
    phase. If neither is supported, liquid input retains its original value.
    """
    first = weather.copy(); first['date'] = pd.to_datetime(first.date)
    observed = pairs.loc[pairs.site_id.isin(eligible_sites)].copy()
    observed['date'] = pd.to_datetime(observed.date)
    if observed.duplicated(['site_id','date']).any() or first.duplicated(['location_id','date']).any():
        raise ValueError('Unique site/date weather required')
    observed = observed.set_index(['site_id','date']).reindex(
        pd.MultiIndex.from_frame(first[['location_id','date']]))
    observed.index = first.index
    ledger = []
    names = ['tmean_c','tmin_c','tmax_c','rh_mean_pct','rh_hours_ge_90pct'] if meteorology else []
    for name in names:
        key = name+'_station'
        if key not in observed or name not in first:
            continue
        mask = np.isfinite(observed[key])
        first.loc[mask, name] = observed.loc[mask,key]
        if mask.any():
            if 'origin_'+name not in first:
                first['origin_'+name]='reanalysis'
            first.loc[mask,'origin_'+name]=source
        ledger.append(dict(source=source, variable=name, replaced_days=int(mask.sum()),
                           policy='valid station measurement; remaining days original forcing'))
    if rainfall:
        total = observed.precipitation_mm_station
        valid = np.isfinite(total)&total.ge(0)
        original_total = weather.precipitation_mm.to_numpy(float)
        ratio = np.divide(weather.rain_mm.to_numpy(float), original_total,
                          out=np.full(len(first), np.nan), where=original_total>0)
        ratio = np.clip(ratio,0,1)
        minimum = observed['tmin_c_station'].fillna(weather.tmin_c) if 'tmin_c_station' in observed else weather.tmin_c
        warm = minimum.gt(2).to_numpy()
        if 'solid_precipitation_observed' in observed:
            warm = warm & ~observed.solid_precipitation_observed.eq(True).fillna(False).to_numpy(bool)
        ratio[warm] = 1.
        ratio[total.eq(0).to_numpy()] = 0.
        liquid_valid = valid&np.isfinite(ratio)
        first.loc[valid, 'precipitation_mm'] = total.loc[valid]
        first.loc[liquid_valid, 'rain_mm'] = total.loc[liquid_valid]*ratio[liquid_valid]
        for name,mask,label in [('precipitation_mm',valid,source),('rain_mm',liquid_valid,source+'_phase_assisted')]:
            if mask.any():
                if 'origin_'+name not in first:
                    first['origin_'+name]='reanalysis'
                first.loc[mask,'origin_'+name]=label
        first[f'{source}_total_precip_used'] = valid.to_numpy()
        first[f'{source}_liquid_phase_assisted'] = liquid_valid.to_numpy()
        first[f'{source}_warm_liquid_assumption'] = (liquid_valid&warm&total.gt(0)).to_numpy()
        first.loc[valid&total.eq(0), 'rain_hours_gt_0_1mm'] = 0
        use_hours = valid&total.eq(0)
        duration_name=next((name for name in ['station_precip_hours_gt_0_1mm','rain_hours_gt_0_1mm_station'] if name in observed),None)
        if duration_name is not None:
            duration = observed[duration_name]
            use_hours = valid&((warm&np.isfinite(duration))|total.eq(0))
            first.loc[use_hours, 'rain_hours_gt_0_1mm'] = duration.loc[use_hours].fillna(0)
        if use_hours.any():
            if 'origin_rain_hours_gt_0_1mm' not in first:
                first['origin_rain_hours_gt_0_1mm']='reanalysis'
            first.loc[use_hours,'origin_rain_hours_gt_0_1mm']=source
        ledger.extend([
            dict(source=source, variable='precipitation_mm', replaced_days=int(valid.sum()),
                 policy='measured complete UTC-day total precipitation'),
            dict(source=source, variable='rain_mm', replaced_days=int(liquid_valid.sum()),
                 policy='station total; Tmin>2C liquid assumption or original reanalysis liquid/total phase ratio; otherwise fallback'),
            dict(source=source, variable='rain_hours_gt_0_1mm', replaced_days=int(use_hours.sum()),
                 policy='complete measured hourly support under warm-liquid assumption, or zero measured total; otherwise original duration'),
        ])
    return first, pd.DataFrame(ledger)


def noaa_rain_usable(pairs):
    """Reject a weather-dependent dry-only sample as a general rainfall source."""
    return bool(pairs.precipitation_mm_station.gt(0).any())


def dwd_elevation_screen(root):
    folder = root/'outputs/weather_validation_20261022/dwd'
    matches = pd.read_csv(folder/'station_matches.csv', dtype={'station_id':str})
    elevation = pd.read_csv(root/'outputs/weather_validation_20261022/noaa/field_elevation.csv')
    screened = {}
    for variable in ['TU','RR']:
        inventory = parse_inventory((root/f'data/external_weather_validation_20261022/dwd/{variable}_stations.txt').read_bytes().decode('latin1'))
        block = matches.loc[matches.variable_group.eq(variable)].merge(
            inventory[['station_id','elevation_m']], on='station_id', validate='many_to_one').merge(
            elevation[['site_id','target_dem_elevation_m']], on='site_id', validate='one_to_one')
        block['elevation_difference_m'] = block.elevation_m-block.target_dem_elevation_m
        block['elevation_eligible'] = block.elevation_difference_m.abs().le(300)&block.within_distance
        block.to_csv(folder/f'{variable}_elevation_matches.csv', index=False)
        screened[variable] = set(block.loc[block.elevation_eligible,'site_id'])
    return screened


def run(root=ROOT, output=None, require_noaa=True):
    root = Path(root); base = root/BASE
    output = Path(output or root/'outputs/disease_improvement/station_repairs_20261022')
    if output.exists():
        raise FileExistsError('A new experiment output is required')
    output.mkdir(parents=True)
    noaa_path = root/'outputs/weather_validation_20261022/noaa/hourly_rain/paired_daily.parquet'
    if require_noaa and not noaa_path.exists():
        raise FileNotFoundError('UTC-aligned NOAA hourly precipitation is required')
    obs = pd.read_csv(base/'observation_ledger.csv')
    fields = pd.read_csv(base/'field_metadata_ledger.csv')
    old_predictions = pd.read_csv(base/'assessment_predictions.csv')
    schema = json.loads((base/'feature_schema.json').read_text())
    arrays = np.load(base/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    history = pd.DataFrame(arrays['history'], columns=schema['history_names'])
    weather = pd.read_parquet(root/WEATHER)
    dwd = pd.read_parquet(root/'outputs/weather_validation_20261022/dwd/paired_daily.parquet')
    eligible = dwd_elevation_screen(root)
    rain, ledgers = hybrid_weather(weather, dwd, eligible['RR'], rainfall=True, meteorology=False)
    ledger = [ledgers]
    source_paths = [root/WEATHER, base/'observation_ledger.csv', base/'predictors.npz',
                    base/'field_metadata_ledger.csv',base/'feature_schema.json',base/'history_provenance.json',
                    base/'assessment_predictions.csv',base/'spatial_membership.json',
                    root/'outputs/weather_validation_20261022/dwd/station_matches.csv',
                    root/'data/external_weather_validation_20261022/dwd/TU_stations.txt',
                    root/'data/external_weather_validation_20261022/dwd/RR_stations.txt',
                    root/'outputs/weather_validation_20261022/dwd/paired_daily.parquet',
                    root/'outputs/weather_validation_20261022/noaa/field_elevation.csv']
    if noaa_path.exists():
        noaa = pd.read_parquet(noaa_path)
        usable = noaa_rain_usable(noaa)
        summary_path=noaa_path.parent/'summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            usable = usable and summary.get('usable_for_substitution',True)
            source_paths.append(summary_path)
        if usable:
            rain, used = hybrid_weather(rain, noaa, set(noaa.site_id), rainfall=True,
                                        meteorology=False, source='NOAA_ISD')
            ledger.append(used)
        else:
            ledger.append(pd.DataFrame([dict(source='NOAA_ISD',variable='precipitation_mm',replaced_days=0,
                policy='blocked: dry-only complete-day sample and/or provider condition-code ambiguity; retain reanalysis')]))
        source_paths.append(noaa_path)
    station, used = hybrid_weather(rain, dwd, eligible['TU'])
    ledger.append(used)
    pd.concat(ledger, ignore_index=True).to_csv(output/'station_replacement_coverage.csv', index=False)
    station.to_parquet(output/'station_preferred_daily_weather.parquet', index=False)
    # All choices fixed before fitting; no selection among outcomes is implemented.
    recipes = ['reference_replay','shared_code','station_rain_shared_code','station_all_shared_code',
               'shared_code_direct','station_all_shared_code_direct']
    specs={name:dict(SPEC,structure='direct' if name.endswith('_direct') else 'process') for name in recipes}
    code_paths=list(Path(__file__).parent.glob('*.py'))+[root/name for name in [
        'model/seasonal_septoria/core.py','model/seasonal_septoria/wetness.py','analysis/final_study/metrics.py']]
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(), recipes=specs,
        independent_validation=False, tuning=False, target_filter='unchanged qualified2286; evaluation2017',
        station_policy='<=50km, absolute elevation difference<=300m, valid complete UTC precipitation coverage; missing fallback',
        liquid_phase='station total multiplied by reanalysis liquid/total fraction; Tmin>2C assumes liquid; no phase information retains forcing liquid',
        causal_weather_cutoff='previous day; precipitation report at next00UTC completes that day',
        noaa_mean_temperature_substitution=False, noaa_gsod_precipitation_substitution=False,
        noaa_hourly_precipitation_used=bool(noaa_path.exists() and usable),
        anatomical_code_equivalence_verified=False, process_baseline_column=-3,
        interpretation='Adaptive retrospective development; spatial integral noninferiority required; untouched cohort needed before scientific adoption',
        source_sha256={str(path.relative_to(root)):sha(path) for path in source_paths},
        environment={name:version(name) for name in ['numpy','pandas','scikit-learn','scipy','joblib','numba']},
        code_sha256={str(path.relative_to(root)):sha(path) for path in code_paths})
    (output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    (output/'spatial_membership.json').write_bytes((base/'spatial_membership.json').read_bytes())
    obs.to_csv(output/'observation_ledger.csv',index=False)
    for path in code_paths:
        snapshot=output/'code_snapshot'/path.relative_to(root)
        snapshot.parent.mkdir(parents=True,exist_ok=True); snapshot.write_bytes(path.read_bytes())
    prior = pd.read_json(base/'history_provenance.json')
    config = dict(weather_cutoff_days=1, rain_variable='rain_mm',
                  prior_dates=dict(zip(prior.global_target_index, prior.latest_same_date)))
    designs = {}
    for label, forcing in [('reanalysis',weather),('station_rain',rain),('station_all',station)]:
        if label == 'reanalysis':
            events = arrays['events']; names = schema['event_names']
        else:
            events, names, provenance = build_features(forcing, fields, obs, config)
            provenance.to_csv(output/f'{label}_feature_provenance.csv', index=False)
        weather_columns = [i for i,name in enumerate(names) if name.startswith('weather_')]
        process_columns = [i for i,name in enumerate(names) if name.startswith('process_')]
        if len(weather_columns)!=12 or len(process_columns)!=3:
            raise ValueError('Original weather/process schema required')
        category, category_names = recorded_leaf_features(obs)
        designs[label] = np.column_stack([events[:,weather_columns],category,events[:,process_columns]])
        if label=='reanalysis':
            source_category, _ = recorded_leaf_features(obs,shared=False)
            designs['reference'] = np.column_stack([events[:,weather_columns],source_category,events[:,process_columns]])
    mapping = dict(zip(recipes,['reference','reanalysis','station_rain','station_all','reanalysis','station_all']))
    (output/'feature_schema.json').write_text(json.dumps(dict(recorded_leaf_features=category_names,
        weather_columns=weather_columns,process_columns=process_columns,process_baseline_column=-3),indent=2)+'\n')
    np.savez_compressed(output/'predictors.npz',target_ids=obs.global_target_index,
                        history=history.to_numpy(),**designs)
    position = pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    temporal=[]
    for year, block in old_predictions.groupby('season_year'):
        train=np.flatnonzero(obs.season_year.lt(year)); test=position.loc[block.global_target_index].to_numpy()
        part=block.copy()
        for name in recipes:
            x=designs[mapping[name]]
            fitted=fit_candidate(x[train],history.iloc[train],obs.iloc[train],specs[name])
            part[name]=predict_candidate(fitted,x[test],history.iloc[test],part)
            joblib.dump(fitted,output/f'{year}_{name}.joblib')
        temporal.append(part);print('temporal',year,flush=True)
    temporal=pd.concat(temporal,ignore_index=True)
    np.testing.assert_allclose(temporal.reference_replay,temporal.selected_candidate,atol=1e-10)
    temporal.to_csv(output/'assessment_predictions.csv',index=False)
    spatial=[]
    membership=json.loads((base/'spatial_membership.json').read_text())
    for split in membership:
        train=position.loc[split['train']].to_numpy();test=position.loc[split['test']].to_numpy()
        if not len(train):continue
        part=obs.iloc[test].copy();part['outer_fold']=split['fold']
        for name in recipes:
            x=designs[mapping[name]]
            fitted=fit_candidate(x[train],history.iloc[train],obs.iloc[train],specs[name])
            part[name]=predict_candidate(fitted,x[test],history.iloc[test],part)
        spatial.append(part)
    spatial=pd.concat(spatial,ignore_index=True)
    spatial.to_csv(output/'spatial_predictions.csv',index=False)
    metrics=[];intervals=[]
    for label,frame in [('temporal',temporal),('spatial_50km',spatial)]:
        for name in recipes:
            metrics.append(dict(evaluation=label,model=name,**score_predictions(frame,name)))
            if name!='reference_replay':
                for reference in ['reference_replay','shared_code']:
                    if reference==name:continue
                    ci=paired_interval(frame,name,reference)
                    intervals.append(dict(evaluation=label,model=name,reference=reference,lower=ci[0],upper=ci[1]))
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(output/'paired_intervals.csv',index=False)
    temporal['country']=temporal.field_id.map(fields.set_index('field_id').country)
    subgroups=[]
    for label,mask in dict(Germany=temporal.country.eq('GERMANY'),non_Germany=~temporal.country.eq('GERMANY'),
        severe_above75=temporal.observed_percent.gt(75),cold=~temporal.prior_assessment_available,
        warm=temporal.prior_assessment_available,new_site=temporal.new_site).items():
        for name in recipes:
            subgroups.append(dict(population=label,model=name,**score_subset(temporal,mask,name)))
    pd.DataFrame(subgroups).to_csv(output/'subgroup_metrics.csv',index=False)
    temporal.loc[temporal.global_target_index.isin([479,1750,413,159,1231]),
        ['global_target_index','observed_percent']+recipes].to_csv(output/'hard_cases.csv',index=False)
    return pd.DataFrame(metrics)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--allow-dwd-only',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output,require_noaa=not args.allow_dwd_only).to_string(index=False))
