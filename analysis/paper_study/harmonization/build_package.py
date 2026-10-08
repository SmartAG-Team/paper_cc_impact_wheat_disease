#!/usr/bin/env python3
"""Additive paper-study package with the canonical 29-column measurement schema."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/paper_study/harmonized'
OBS=ROOT/'data/paper_study/observations'
SCHEMA=pq.read_schema(ROOT/'data/harmonized/observations.parquet').remove_metadata()
COLS=SCHEMA.names
NUMERIC={f.name for f in SCHEMA if pa.types.is_floating(f.type)}
HASHES={};PARTITIONS=[];COUNTS=[];METRICS=[]
DEFINITIONS=json.loads((HERE/'definitions.json').read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def watch(path):
    path=Path(path);HASHES[str(path.relative_to(ROOT))]=sha(path);return path


def read(path):
    path=watch(path)
    return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)


def emit(frame,view):
    if frame.empty:return
    for c in COLS:
        if c not in frame:frame[c]=None
        frame[c]=pd.to_numeric(frame[c],errors='coerce').astype('float64') if c in NUMERIC else frame[c].astype('string')
    frame=frame[COLS]
    assert frame.observation_id.notna().all() and not frame.observation_id.duplicated().any()
    for dataset,g in frame.groupby('dataset_id',sort=True):
        name=f'measurements/{view}/{dataset}/part-00000.parquet';path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g,schema=SCHEMA,preserve_index=False),path,compression='zstd')
        PARTITIONS.append(name)
        COUNTS.append(dict(view=view,dataset_id=dataset,measurement_cells=len(g),numeric_cells=int(g.value.notna().sum()),
                           coordinate_sites=int(g.site_id.nunique()),source_record_ids=int(g.record_id.nunique()),
                           record_counts_are_independent_replicates=False,path=name))
        for key,q in g.groupby(['metric','unit','measurement_role'],dropna=False):
            METRICS.append(dict(view=view,dataset_id=dataset,metric=key[0],unit=key[1],measurement_role=key[2],rows=len(q),
                                source_files=';'.join(sorted(q.source_file.dropna().unique())),
                                definition=DEFINITIONS['metrics'].get(key[0],'Unresolved original source metric; excluded from normalized percentage targets; original definition remains authoritative.')))


def wide(frame,name):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False,compression='zstd')


def stable(values):
    text=json.dumps([None if pd.isna(v) else str(v) for v in values],ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def derived_assessments(main,external):
    rows=[];metadata=[];lineage=[]
    for label,source,file in [('development',main,OBS/'assessments.csv'),('reserved_external',external,OBS/'corteva_external_assessments.csv')]:
        for index,r in source.iterrows():
            # Tunisian plot-incidence keys contain semicolons inside source_row.
            # Only a delimiter immediately followed by the next dataset prefix separates IDs.
            keys=re.split(r';(?='+re.escape(str(r.dataset_id))+r'\|)',str(r.sourcekeys))
            identity=[r.dataset_id,r.physical_unit,r.season_year,r.get('cultivar'),r.treatment,r.get('date'),r.organ,r.metric,*keys]
            aid=str(r.dataset_id)+'|assessment|'+stable(identity)
            meta=r.to_dict();meta.update(assessment_id=aid,source_inventory_file=str(file.relative_to(ROOT)),source_inventory_row=index+2,
                                        source_scope=label,sourcekey_count=len(keys),observed_sowing_date_available=pd.notna(r.get('sowing_date')))
            meta['reported_numeric_count_basis']='two source plots represented by one author summary cell' if r.organ=='whole_cultivar_visual_summary' else 'source-defined count; see sampling_scope; not independent field sites'
            if label=='reserved_external':meta['reported_numeric_count_basis']='numeric source records including exact duplicates; biological replicate identities unknown'
            metadata.append(meta)
            for k in keys:lineage.append(dict(assessment_id=aid,source_observation_id=k,dataset_id=r.dataset_id))
            source_table=r.get('source_table');source_table='assessment_inventory' if pd.isna(source_table) else source_table
            common=dict(dataset_id=r.dataset_id,source_file=str(file.relative_to(ROOT)),source_table=source_table,source_row=str(index+2),
                        record_id=str(r.physical_unit),plot_id=r.get('plot_id'),site_id=r.site_id,country=r.country,latitude=r.get('latitude'),
                        longitude=r.get('longitude'),season_year=r.season_year,date=r.get('date'),
                        time_basis='calendar' if pd.notna(r.get('date')) else 'relative_study_day' if pd.notna(r.get('time_index')) else 'undated_summary',
                        cultivar=r.get('cultivar'),pathogen='Zymoseptoria tritici',treatment=r.treatment,organ=r.organ)
            valid=bool(r.get('assessment_eligible',True))
            quality='source_excluded_assessment' if not valid else 'derived_numeric_assessment' if pd.notna(r.value) else 'no_numeric_score_senescent_or_missing'
            measures=[('value',r.metric,r.unit,r.value,'derived_source_assessment',quality),
                      ('n_numeric','assessment_reported_numeric_count','source_defined_count',r.get('n_numeric',r.get('n_source_numeric_records')),'assessment_denominator_metadata','source_defined_count_not_independent_replicates'),
                      ('n_senescent','assessment_senescent_record_count','source_records',r.get('n_senescent'),'assessment_denominator_metadata','source_record_count_not_independent_replicates'),
                      ('n_missing','assessment_missing_record_count','source_records',r.get('n_missing'),'assessment_denominator_metadata','source_record_count_not_independent_replicates'),
                      ('stage_from','crop_stage_BBCH_from','BBCH',r.get('stage_from'),'source_linked_crop_stage','source_stage_definition'),
                      ('stage_to','crop_stage_BBCH_to','BBCH',r.get('stage_to'),'source_linked_crop_stage','source_stage_definition')]
            for column,metric,unit,value,role,status in measures:
                if column!='value' and pd.isna(value):continue
                rows.append(dict(common,observation_id=aid+'|'+column,source_column=column,metric=metric,unit=unit,value=value,
                                 raw_value=None if pd.isna(value) else str(value),quality_status=status,measurement_role=role))
    meta=pd.DataFrame(metadata);wide(meta,'metadata/assessment_registry.parquet')
    pd.DataFrame(lineage).to_csv(OUT/'assessment_source_lineage.csv',index=False)
    emit(pd.DataFrame(rows),'source_assessments')
    # A unit registry is a clustering registry; source identifiers do not prove independence.
    unit=[]
    for key,g in meta.groupby(['dataset_id','physical_unit','site_id','season_year'],dropna=False):
        dataset,physical,site,year=key
        kind='regional_survey_aggregate' if dataset=='adas-uk-regional-survey' else 'cultivar_summary_not_identified_plot' if 'whole_cultivar_visual_summary' in set(g.organ) else 'source_field_or_plot_season'
        unit.append(dict(unit_id=dataset+'|unit|'+stable(key),dataset_id=dataset,physical_unit=physical,site_id=site,season_year=year,
                         source_cluster_type=kind,plot_id=';'.join(sorted(g.plot_id.dropna().astype(str).unique())),
                         cultivars=';'.join(sorted(g.cultivar.dropna().astype(str).unique())),assessment_rows=len(g),
                         source_leaf_series=g.get('endpoint_series',pd.Series(dtype=str)).nunique(),coordinate_year=str(site)+'|'+str(int(year)),
                         observed_sowing_date_available=bool(g.observed_sowing_date_available.any()),
                         independence_guaranteed=False))
    pd.DataFrame(unit).to_csv(OUT/'physical_unit_registry.csv',index=False)
    return meta


def official_survey(frame):
    rows=[]
    for r in frame.itertuples():
        metric={'disease_severity_percent':'regional_mean_disease_severity_percent','plant_incidence_percent':'regional_plant_incidence_percent',
                'crop_incidence_percent':'regional_crop_incidence_percent'}[r.metric]
        oid=f'defra-official-regional-survey|{r.source_sheet}|{r.source_excel_row}|{r.source_excel_column}'
        rows.append(dict(observation_id=oid,dataset_id=r.dataset_id,source_file=r.source_file,source_table=r.source_sheet,
            source_row=str(r.source_excel_row),source_column=str(r.source_excel_column),record_id=f'{r.season_year}|{r.region}',
            site_id='UK_REGION:'+str(r.region),country='UNITED KINGDOM',season_year=r.season_year,time_basis='annual_summer_survey_no_exact_date',
            pathogen='Zymoseptoria tritici',treatment=r.management_status,organ=r.organ,metric=metric,unit=r.unit,value=r.value,
            raw_value=r.raw_value,quality_status='valid_regional_aggregate' if pd.notna(r.value) else 'missing_source_cell',
            measurement_role='regional_aggregate_same_ADAS_survey'))
    q=pd.DataFrame(rows);emit(q,'official_survey_alternative');return q


def field_forcing():
    original=read(ROOT/'data/paper_study/field_weather/daily_weather.parquet')
    receipts=[]
    for folder in ['field_weather','field_weather_extension','field_weather_completed']:
        path=ROOT/f'data/paper_study/{folder}/receipt.json'
        if path.exists():
            receipts.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(watch(path)),content=json.loads(path.read_text())))
    completed=ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet'
    extended=ROOT/'data/paper_study/field_weather_extension/daily_weather_with_extension.parquet'
    chosen=completed if completed.exists() else extended if extended.exists() else ROOT/'data/paper_study/field_weather/daily_weather.parquet'
    weather=read(chosen);assert not weather.duplicated(['location_id','date']).any()
    matched=original.merge(weather,on=['location_id','date'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(matched)==len(original)
    for column in original.columns.difference(['location_id','date']):assert np.allclose(matched[column+'_old'],matched[column+'_new'],equal_nan=True,rtol=0,atol=0)
    wide(weather,'wide/field_daily_weather.parquet')
    requests=read(ROOT/'data/paper_study/field_weather/requests.csv')
    for receipt in receipts:
        r=receipt['content'].get('request')
        if r:requests=pd.concat([requests,pd.DataFrame([dict(r,latitude=r['requested_latitude'],longitude=r['requested_longitude'],country=None)])],ignore_index=True)
    requests=requests.drop_duplicates(['location_id','season_year','window_start','window_end'])
    requests.to_csv(OUT/'field_weather_request_registry.csv',index=False)
    locations=requests.groupby('location_id',sort=True).agg(latitude=('requested_latitude','first'),longitude=('requested_longitude','first'),country=('country','first')).reset_index()
    locations['coordinate_basis']='requested original source coordinates; ERA5 uses provider grid selection, not surveyed station coordinates'
    locations.to_csv(OUT/'field_forcing_location_registry.csv',index=False)
    pd.DataFrame(receipts).assign(content=lambda d:d.content.map(json.dumps)).to_csv(OUT/'field_weather_provenance.csv',index=False)
    units={'tmean_c':'degC','tmin_c':'degC','tmax_c':'degC','rh_mean_pct':'percent','dewpoint_mean_c':'degC',
           'precipitation_mm':'mm_per_day','rain_mm':'mm_per_day','rain_hours_gt_0_1mm':'hours_per_day','rh_hours_ge_90pct':'hours_per_day',
           'wind_mean_m_s':'m_per_s','shortwave_mj_m2':'MJ_per_m2_per_day','hour_count':'hourly_records_per_day'}
    frames=[]
    for metric,unit in units.items():
        frame=pd.DataFrame(dict(site_id=weather.location_id,date=weather.date,value=weather[metric]))
        frame['record_id']=frame.site_id+'|'+frame.date
        frame['observation_id']='paper_study_era5_fields|'+frame.record_id+'|'+metric
        frame=frame.assign(dataset_id='era5_paper_study_fields',source_file=str(chosen.relative_to(ROOT)),source_table='daily',
            source_row=np.arange(len(frame)).astype(str),source_column=metric,
            time_basis='UTC_accumulation_D01_to_Dplus1_00' if metric in ['precipitation_mm','rain_mm','rain_hours_gt_0_1mm'] else 'UTC_reported_hours_D00_to_D23',
            metric=metric,unit=unit,raw_value=frame.value.astype(str),quality_status='validated_reanalysis_daily_aggregate',measurement_role='reanalysis_weather')
        frames.append(frame)
    emit(pd.concat(frames,ignore_index=True),'field_daily_forcing')
    return weather,dict(status='complete_with_both_extensions' if completed.exists() else 'partial_pending_additional_coordinate_season',
                        source_path=str(chosen.relative_to(ROOT)),source_sha256=sha(chosen),original_rows=len(original),completed_rows=len(weather),
                        added_rows=len(weather)-len(original),original_values_preserved=True,receipts=receipts)


def area_and_calendars():
    base=ROOT/'data/paper_study/wheat_area'
    cells=read(base/'europe_wheat_cells_025.parquet');wide(cells,'metadata/europe_wheat_cells_025.parquet')
    countries=read(base/'europe_cell_country_wheat_areas.parquet');wide(countries,'metadata/europe_cell_country_wheat_areas.parquet')
    calendar=read(base/'europe_wheat_calendar_scenarios.parquet');wide(calendar,'metadata/europe_calendar_scenarios.parquet')
    points=read(base/'trial_point_calendar_scenarios.parquet');wide(points,'metadata/point_calendar_scenarios.parquet')
    rows=[]
    metrics=['harvested_total_ha','harvested_irrigated_ha','harvested_rainfed_ha','physical_total_ha','physical_irrigated_ha','physical_rainfed_ha',
             'grid_cell_geometric_area_ha','wheat_area_weight','physical_wheat_area_weight']
    for i,r in cells.iterrows():
        for metric in metrics:
            rows.append(dict(observation_id='spam2020_wheat|'+r.cell_id+'|'+metric,dataset_id='spam2020_europe_wheat_area',source_file=str((base/'europe_wheat_cells_025.parquet').relative_to(ROOT)),
                source_table='positive_wheat_cells',source_row=str(i),source_column=metric,record_id=r.cell_id,site_id=r.cell_id,
                country=str(r.dominant_source_country).upper(),latitude=r.latitude,longitude=r.longitude,season_year=r.area_reference_year,
                time_basis='area_reference_year',organ='all_wheat_unallocated_winter_spring',metric=metric,
                unit='hectares' if metric.endswith('_ha') else 'fraction',value=r[metric],raw_value=str(r[metric]),
                quality_status='source_spatial_aggregation',measurement_role='spatial_wheat_area_reference'))
    emit(pd.DataFrame(rows),'wheat_area_metadata')
    rows=[]
    for label,frame,file in [('europe',calendar,base/'europe_wheat_calendar_scenarios.parquet'),('trial_points',points,base/'trial_point_calendar_scenarios.parquet')]:
        for i,r in frame.iterrows():
            site=r.get('cell_id',r.get('point_id'));rid=str(site)+'|'+str(r.crop_season)+'|'+str(r.water_system)
            if label=='trial_points':rid=str(r.dataset_id)+'|'+rid
            for metric,unit in [('planting_doy','day_of_year'),('maturity_doy','day_of_year'),('growing_season_length_days','days')]:
                rows.append(dict(observation_id='ggcmi_calendar|'+label+'|'+rid+'|'+metric,dataset_id='ggcmi_wheat_calendar_'+label,source_file=str(file.relative_to(ROOT)),
                    source_table='calendar_scenarios',source_row=str(i),source_column=metric,record_id=rid,site_id=site,
                    latitude=r.get('latitude'),longitude=r.get('longitude'),time_basis='climatological_calendar_scenario',organ=r.crop_season,
                    metric=metric,unit=unit,value=r[metric],raw_value=None if pd.isna(r[metric]) else str(r[metric]),
                    quality_status='valid_source_calendar_scenario' if r.calendar_valid else 'missing_source_calendar',measurement_role='crop_calendar_scenario'))
    emit(pd.DataFrame(rows),'crop_calendar_metadata')
    return cells,points


def climate_registry(cells):
    path=ROOT/'analysis/paper_study/climate/actual_coverage_validation.json'
    if not path.exists():return dict(status='partial_not_coverage_audited',full_download_and_validation_complete=False)
    raw=path.read_bytes();receipt=json.loads(raw);saved=OUT/'metadata/climate_coverage_snapshot.json';saved.write_bytes(raw)
    contract=ROOT/'analysis/paper_study/climate/climate_registry_contract.json'
    if contract.exists():(OUT/'metadata/climate_registry_contract_snapshot.json').write_bytes(contract.read_bytes())
    rows=[]
    for source,files in receipt['files'].items():
        for entry in files:
            if entry['status']!='valid':continue
            file=ROOT/entry['path'];meta_path=file.with_suffix('.json');meta=json.loads(meta_path.read_text())
            assert meta['registry_sha256']==sha(ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet')
            footer=pq.ParquetFile(file);quality=meta['quality'];assert footer.metadata.num_rows==quality['rows']
            assert all(c in footer.schema_arrow.names for c in ['date','cell_id','tmean_c','tmax_c','precipitation_mm','rh_mean_pct'])
            model=meta.get('model');scenario=meta.get('scenario')
            rows.append(dict(forcing_partition_id=source+'|'+str(model)+'|'+str(scenario)+'|'+file.stem,source=source,path=str(file.relative_to(ROOT)),
                provenance_path=str(meta_path.relative_to(ROOT)),provenance_sha256=sha(meta_path),parquet_sha256=meta['parquet_sha256'],
                checksum_verified_against_bytes=False,rows=footer.metadata.num_rows,date_min=quality['date_min'],date_max=quality['date_max'],
                cell_count=quality['cells'],grid_registry_sha256=meta['registry_sha256'],model=model,scenario=scenario,
                ensemble_member=None if source=='era5' else 'r1i1p1f1',source_version=None if source=='era5' else '2.0',
                derived_member_id=None if source=='era5' else 'nex_gddp_cmip6_v2:'+str(model)+':r1i1p1f1',
                validation_scope='climate_agent_metadata_coverage_audit_and_package_footer_recheck',download_status='available_partial_ensemble'))
    pd.DataFrame(rows).to_csv(OUT/'continental_forcing_partition_registry.csv',index=False)
    summary=receipt['summary'];complete=all(s['missing']==0 and s['invalid']==0 and s['valid']==s['expected'] for s in summary.values()) and receipt.get('deep',False)
    return dict(status='complete' if complete else 'partial_download_or_pending_deep_coverage_validation',full_download_and_validation_complete=complete,
                audit_as_of=receipt['checked_utc'],audit_snapshot_sha256=hashlib.sha256(raw).hexdigest(),summary=summary,
                partitions_registered=len(rows),source_hashes_are_provenance_reported=True,continental_daily_long_rows_generated=0)


def main():
    HERE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);(OUT/'metadata').mkdir(exist_ok=True)
    assert len(COLS)==29
    protected=[ROOT/'data/harmonized/manifest.json',ROOT/'data/harmonized/observations.parquet',ROOT/'data/harmonized/common_metric_dictionary.csv']
    for p in protected:watch(p)
    raw=read(OBS/'observations.parquet');external=read(OBS/'corteva_external_observations.parquet')
    combined=pd.concat([raw,external],ignore_index=True)
    invalid=combined.unit.eq('percent')&combined.value.notna()&~combined.value.between(0,100)
    events=combined.loc[invalid,['observation_id','dataset_id','value','raw_value','quality_status']].copy()
    events['normalization']='percentage_outside_0_100_to_missing_without_clipping';events.to_csv(OUT/'normalization_events.csv',index=False)
    combined.loc[invalid,'value']=np.nan
    combined.loc[invalid,'quality_status']=combined.loc[invalid,'quality_status']+'; outside_percent_range_raw_retained'
    emit(combined,'source_cells')
    main_assess=read(OBS/'assessments.csv');corteva=read(OBS/'corteva_external_assessments.csv');meta=derived_assessments(main_assess,corteva)
    bad_ids=set(events.observation_id)
    links=pd.read_csv(OUT/'assessment_source_lineage.csv')
    bad_counts=links.source_observation_id.isin(bad_ids).groupby(links.assessment_id).sum()
    meta['out_of_range_source_percentage_cells']=meta.assessment_id.map(bad_counts).fillna(0).astype(int)
    wide(meta,'metadata/assessment_registry.parquet')
    official=official_survey(read(OBS/'defra_official_regional_target_inventory.csv'))
    # Both survey releases are alternative views of the same observed regional survey.
    adas=raw[raw.dataset_id.eq('adas-uk-regional-survey')];links=adas.merge(official,on=['site_id','season_year','organ','metric'],suffixes=('_adas','_official'))
    comparable=links.value_adas.notna()&links.value_official.notna();assert np.allclose(links.loc[comparable,'value_adas'],links.loc[comparable,'value_official'])
    links[['observation_id_adas','observation_id_official','site_id','season_year','organ','metric','value_adas','value_official']].to_csv(OUT/'survey_source_overlap.csv',index=False)
    weather,field_status=field_forcing();cells,points=area_and_calendars();climate_status=climate_registry(cells)
    # Metadata snapshots retain upstream flags and supply separately derived availability.
    endpoints=[]
    for name in ['season_endpoints.csv','corteva_external_season_endpoints.csv']:
        q=read(OBS/name);q['observed_sowing_date_available']=q.sowing_date.notna()
        q['source_sowing_flag_conflict']=q.get('known_sowing_date',False).eq(True)&~q.observed_sowing_date_available
        q['metadata_source_file']=str((OBS/name).relative_to(ROOT));endpoints.append(q)
    end=pd.concat(endpoints,ignore_index=True);wide(end,'metadata/season_endpoints.parquet')
    inventory=read(ROOT/'analysis/paper_study/observations/target_inventory.csv');inventory.to_csv(OUT/'source_target_inventory.csv',index=False)
    for name in ['corteva_organ_dictionary.csv','corteva_BASF_geographic_overlap.csv']:
        q=read(OBS/name);q.to_csv(OUT/name,index=False)
    # Calendar availability is a scenario; it never supplies an observed sowing date.
    calendar_links=meta[['dataset_id','assessment_id','site_id','season_year','date','observed_sowing_date_available']].merge(
        points[['dataset_id','point_id','crop_season','water_system','calendar_valid']],left_on=['dataset_id','site_id'],right_on=['dataset_id','point_id'],how='left')
    wide(calendar_links,'metadata/assessment_calendar_links.parquet')
    # Exact source-cell overlap with the old package is metadata, not an additional cohort.
    canonical=pd.read_parquet(ROOT/'data/harmonized/observations.parquet',columns=['observation_id','dataset_id','value','metric','measurement_role'])
    overlap=pd.concat([raw,external])[['observation_id','dataset_id','value','metric','measurement_role']].merge(canonical,on=['observation_id','dataset_id'],suffixes=('_paper','_canonical'))
    overlap.to_csv(OUT/'canonical_source_overlap.csv',index=False)
    pd.DataFrame(COUNTS).to_csv(OUT/'measurement_counts.csv',index=False);pd.DataFrame(METRICS).to_csv(OUT/'common_metric_dictionary.csv',index=False)
    pd.DataFrame([dict(column=f.name,arrow_type=str(f.type),definition=DEFINITIONS['columns'][f.name]) for f in SCHEMA]).to_csv(OUT/'schema_dictionary.csv',index=False)
    (OUT/'schema.json').write_text(json.dumps(dict(columns=COLS,types={f.name:str(f.type) for f in SCHEMA},canonical_source='data/harmonized/observations.parquet'),indent=2)+'\n')
    # Original source hashes are inherited and checked against the actual archive bytes.
    source_receipt=json.loads(watch(ROOT/'analysis/paper_study/observations/validation.json').read_text())
    for path,expected in source_receipt['source_sha256'].items():assert sha(watch(ROOT/path))==expected
    corteva_receipt=json.loads(watch(ROOT/'analysis/paper_study/observations/corteva_external_validation.json').read_text())
    assert sha(watch(ROOT/corteva_receipt['source_file']))==corteva_receipt['source_sha256']
    assert all(sha(ROOT/path)==expected for path,expected in HASHES.items())
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),common_schema_partitions=PARTITIONS,common_schema_verified=True,
        common_records=int(sum(r['measurement_cells'] for r in COUNTS)),biological_replicates_are_not_common_record_count=True,
        view_selection_required=True,views={view:[r['path'] for r in COUNTS if r['view']==view] for view in sorted({r['view'] for r in COUNTS})},
        schema_compatibility='exact canonical Arrow field names/order/types; strings normalized from large_string only',
        original_files_preserved=True,source_sha256=HASHES,field_forcing_status=field_status,continental_climate_status=climate_status,
        exact_canonical_observation_id_overlap=len(overlap),ADAS_Defra_common_survey_cells=len(links),
        upstream_sowing_availability_flag_conflicts=int(end.source_sowing_flag_conflict.sum()),explicit_numeric_normalizations=len(events),
        Corteva_model_predictions_accessed=False,model_imported_or_fitted=False,old_harmonized_partitions_modified=False,
        source_interpretation='selected source cells and derived assessments are alternative scientific views; raw and derived cells must not be pooled as independent replicates',
        continental_forcing_contract='wide_forcing_contract.json')
    manifest['files']={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['common_records','exact_canonical_observation_id_overlap','ADAS_Defra_common_survey_cells','upstream_sowing_availability_flag_conflicts']},indent=2),flush=True)
    print('field_forcing_status',field_status['status'],'continental_status',climate_status['status'],flush=True)


if __name__=='__main__':main()
