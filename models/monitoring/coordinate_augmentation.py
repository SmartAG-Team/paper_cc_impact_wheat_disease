"""Prepare training-only records recovered by the official coordinate revision."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import build_features
from .history import build_history
from .inputs import WEATHER,sha
from .repairs import recorded_leaf_features
from .source_metadata import normalize_source,stage_calendar
from .stage_features import build_stage_features


def hourly_to_daily(payload,site_id):
    """Match the original forcing's UTC definitions, including rain end times."""
    h=pd.DataFrame(payload['hourly']);h['time']=pd.to_datetime(h.time)
    if h.time.duplicated().any():raise ValueError('Unique hourly observations required')
    h=h.sort_values('time').reset_index(drop=True)
    if not h.time.diff().dropna().eq(pd.Timedelta(hours=1)).all():raise ValueError('Consecutive hourly forcing required')
    h['date']=h.time.dt.normalize()
    h['precipitation']=h.precipitation.shift(-1);h['rain']=h.rain.shift(-1)
    h['wet']=h.rain.gt(.1).where(h.rain.notna())
    h['humid']=h.relative_humidity_2m.ge(90).where(h.relative_humidity_2m.notna())
    g=h.groupby('date')
    out=g.agg(tmean_c=('temperature_2m','mean'),tmin_c=('temperature_2m','min'),tmax_c=('temperature_2m','max'),
        rh_mean_pct=('relative_humidity_2m','mean'),dewpoint_mean_c=('dew_point_2m','mean'),
        precipitation_mm=('precipitation',lambda a:a.sum(min_count=24)),rain_mm=('rain',lambda a:a.sum(min_count=24)),
        rain_hours_gt_0_1mm=('wet',lambda a:a.sum(min_count=24)),rh_hours_ge_90pct=('humid',lambda a:a.sum(min_count=24)),
        wind_mean_m_s=('wind_speed_10m','mean'),shortwave_mj_m2=('shortwave_radiation',lambda a:a.sum(min_count=24)*.0036),
        hour_count=('temperature_2m','count'))
    out=out.loc[out.hour_count.eq(24)&out.notna().all(axis=1)].reset_index()
    out.insert(0,'location_id',site_id)
    for name in ['rain_hours_gt_0_1mm','rh_hours_ge_90pct']:out[name]=pd.to_numeric(out[name])
    return out


def prepare(root):
    root=Path(root);folder=root/'data/disease_improvement/coordinate_augmentation_20261022'
    source=root/'outputs/source_definition_audit_20261022'
    obs=pd.read_csv(source/'recovered_coordinate_fields_observations.csv')
    base=root/'outputs/disease_improvement/station_repairs_20261022_v4'
    original=pd.read_csv(base/'observation_ledger.csv')
    old_fields=pd.read_csv(root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv')
    obs['global_target_index']=1000000+np.arange(len(obs))
    obs['partition']='training_augmentation_only'
    assert set(obs.global_target_index).isdisjoint(original.global_target_index)
    assert set(obs.field_id).isdisjoint(original.field_id)
    rows=[]
    for index,(field,part) in enumerate(obs.groupby('field_id',sort=True)):
        first=part.iloc[0];same=old_fields.loc[old_fields.latitude.eq(first.latitude)&old_fields.longitude.eq(first.longitude)]
        site=same.site_id.iloc[0] if len(same) else f'corteva6390211_{first.latitude:g}_{first.longitude:g}'
        calendars=old_fields.loc[old_fields.country.eq(first.country)&old_fields.season_year.eq(first.season_year)].sowing_date.unique()
        if len(calendars)!=1:raise ValueError('One existing country/year calendar scenario required')
        rows.append(dict(field_id=field,field_index=int(old_fields.field_index.max())+1+index,site_id=site,
            season_year=int(first.season_year),latitude=first.latitude,longitude=first.longitude,country=first.country,
            source='Corteva',source_unit=first.source_trial_id,sowing_date=calendars[0],
            sowing_basis='calendar_scenario_not_observed',wheat_type='winter_wheat_scenario'))
    fields=pd.DataFrame(rows);indexed=fields.set_index('field_id')
    for name in ['site_id','field_index','sowing_date']:obs[name]=obs.field_id.map(indexed[name])
    obs['coordinate_year']=obs.site_id+'|'+obs.season_year.astype(str)
    existing=pd.read_parquet(root/WEATHER)
    existing['date']=pd.to_datetime(existing.date)
    new=fields.loc[~fields.site_id.isin(existing.location_id)]
    for row in new.itertuples(index=False):
        if (row.latitude,row.longitude)!=(49.7,.4):raise ValueError('Unregistered new coordinate')
        path=folder/'era5_49p7_0p4_hourly.json'
        daily=hourly_to_daily(json.loads(path.read_text()),row.site_id)
        existing=pd.concat([existing,daily],ignore_index=True)
    history,hp=build_history(obs,obs,{})
    config=dict(weather_cutoff_days=1,rain_variable='rain_mm',prior_dates=dict(zip(hp.global_target_index,hp.latest_same_date)))
    events,event_names,provenance=build_features(existing,fields,obs,config)
    category,_=recorded_leaf_features(obs)
    weather_columns=[i for i,n in enumerate(event_names) if n.startswith('weather_')]
    process_columns=[i for i,n in enumerate(event_names) if n.startswith('process_')]
    matrix=np.column_stack([events[:,weather_columns],category,events[:,process_columns]])
    raw_path=Path(obs.source_file.iloc[0]);raw=pd.read_csv(raw_path,sep='\t')
    normalized=normalize_source(raw,'Corteva',raw_path,sha(raw_path));normalized=normalized.loc[normalized.field_id.isin(fields.field_id)]
    calendar=stage_calendar(normalized)
    stage,stage_names,stage_provenance=build_stage_features(existing,fields,obs,calendar)
    folder.mkdir(parents=True,exist_ok=True)
    obs.to_csv(folder/'observations.csv',index=False);fields.to_csv(folder/'fields.csv',index=False)
    existing.loc[existing.location_id.isin(fields.site_id)].to_parquet(folder/'daily_weather.parquet',index=False)
    calendar.to_csv(folder/'stage_calendar.csv',index=False)
    provenance.to_csv(folder/'feature_provenance.csv',index=False);stage_provenance.to_csv(folder/'stage_provenance.csv',index=False)
    np.savez_compressed(folder/'predictors.npz',target_ids=obs.global_target_index,matrix=matrix,stage=stage,history=history.to_numpy())
    manifest=dict(observations=len(obs),fields=len(fields),intended_use='additional training only; original evaluation2017unchanged',
        source_sha256={str(p.relative_to(root)):sha(p) for p in [source/'recovered_coordinate_fields_observations.csv',
            source/'recovered_coordinate_fields_metadata.csv',source/'recovered_coordinate_fields_eligibility.json',
            folder/'era5_49p7_0p4_hourly.json',folder/'weather_receipt.json',Path(__file__),root/WEATHER]},
        raw_source_sha256=sha(raw_path),source_doi='10.5281/zenodo.6390211',stage_names=stage_names,
        sowing='Inherited existing France/year calendar scenario; not observed sowing',
        weather='Existing ERA5 at50.2,2.4; explicit ERA5 API at49.7,.4. No qualified local station substitution available.',
        temporal_rule='augmentationseason_year<evaluationyear',spatial_rule='exclude additional same or nearby sites using corrected coordinates and50km buffer',
        raw_repeats_are_replicates=False,publication_model_promoted=False)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    print(json.dumps(prepare(Path(__file__).resolve().parents[2]),indent=2))
