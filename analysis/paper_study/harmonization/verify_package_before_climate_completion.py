#!/usr/bin/env python3
"""Independent schema, source, lineage, forcing and metadata integrity checks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/paper_study/harmonized'
checks=[]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check(name,passed,details=None):
    checks.append(dict(check=name,passed=bool(passed),details=details))
    if not passed:raise AssertionError(name)


def main():
    manifest=json.loads((OUT/'manifest.json').read_text());schema=pq.read_schema(ROOT/'data/harmonized/observations.parquet').remove_metadata()
    for source,expected in manifest['source_sha256'].items():check('preserved source '+source,sha(ROOT/source)==expected)
    all_ids=[];actual_rows=0
    for name in manifest['common_schema_partitions']:
        path=OUT/name;table=pq.ParquetFile(path);check('exact29-column schema '+name,table.schema_arrow.remove_metadata().equals(schema))
        frame=pd.read_parquet(path,columns=['observation_id','value','unit','quality_status'])
        check('cell identifiers present/unique '+name,frame.observation_id.notna().all() and not frame.observation_id.duplicated().any())
        q=frame[frame.unit.eq('percent')&frame.value.notna()];check('normalized percentage domain '+name,q.value.between(0,100).all())
        all_ids.extend(frame.observation_id);actual_rows+=len(frame)
    check('all common measurement identifiers unique within this package',len(all_ids)==len(set(all_ids)))
    check('common record count',actual_rows==manifest['common_records'])
    input_raw=pd.concat([pd.read_parquet(ROOT/'data/paper_study/observations/observations.parquet'),
                         pd.read_parquet(ROOT/'data/paper_study/observations/corteva_external_observations.parquet')],ignore_index=True)
    source_paths=manifest['views']['source_cells'];actual=pd.concat([pd.read_parquet(OUT/name) for name in source_paths],ignore_index=True).set_index('observation_id').sort_index()
    expected=input_raw.set_index('observation_id').sort_index().copy()
    invalid=expected.unit.eq('percent')&expected.value.notna()&~expected.value.between(0,100)
    expected.loc[invalid,'value']=np.nan;expected.loc[invalid,'quality_status']=expected.loc[invalid,'quality_status']+'; outside_percent_range_raw_retained'
    check('all source cells retained once',actual.index.equals(expected.index) and len(actual)==88715)
    for name in schema.names:
        if name=='observation_id':continue
        if name in ['value','latitude','longitude','season_year','days_post_inoculation']:
            passed=np.allclose(actual[name],expected[name],equal_nan=True,rtol=0,atol=1e-12)
        else:passed=actual[name].fillna('').astype(str).equals(expected[name].fillna('').astype(str))
        check('source cell preservation '+name,passed)
    events=pd.read_csv(OUT/'normalization_events.csv');check('all24 numerical normalization events explicit',len(events)==24 and set(events.observation_id)==set(expected.index[invalid]))
    source_lookup=input_raw.set_index('observation_id')
    lineage=pd.read_csv(OUT/'assessment_source_lineage.csv');metadata=pd.read_parquet(OUT/'metadata/assessment_registry.parquet')
    check('lineage has no orphan source IDs',lineage.source_observation_id.isin(source_lookup.index).all())
    check('lineage has no orphan assessments',lineage.assessment_id.isin(metadata.assessment_id).all())
    grouped={key:g.source_observation_id.tolist() for key,g in lineage.groupby('assessment_id',sort=False)}
    means=pd.concat([pd.read_parquet(OUT/name) for name in manifest['views']['source_assessments']],ignore_index=True)
    means=means[means.measurement_role.eq('derived_source_assessment')].set_index('observation_id')
    input_assess={'development':pd.read_csv(ROOT/'data/paper_study/observations/assessments.csv'),
                  'reserved_external':pd.read_csv(ROOT/'data/paper_study/observations/corteva_external_assessments.csv')}
    max_error=0.;count_basis_exceptions=0
    for r in metadata.itertuples():
        ids=grouped[r.assessment_id]
        check('intact sourcekey serialization '+r.assessment_id,';'.join(ids)==r.sourcekeys and len(ids)==r.sourcekey_count)
        original=input_assess[r.source_scope].iloc[r.source_inventory_row-2]
        value=means.loc[r.assessment_id+'|value','value']
        check('assessment source value retained '+r.assessment_id,np.isclose(value,original.value,equal_nan=True,rtol=0,atol=1e-12))
        numeric=source_lookup.loc[ids,'value'].dropna()
        # Whole excluded Corteva groups retain a missing summary, not a repaired mean.
        eligible=bool(original.get('assessment_eligible',True))
        expected_mean=numeric.mean() if eligible and len(numeric) else np.nan
        error=abs(value-expected_mean) if pd.notna(value) and pd.notna(expected_mean) else 0.
        max_error=max(max_error,float(error))
        check('assessment numeric aggregation '+r.assessment_id,np.isclose(value,expected_mean,equal_nan=True,rtol=0,atol=1e-9))
        if r.organ=='whole_cultivar_visual_summary':count_basis_exceptions+=1
    check('author summaries have explicit two-plot count basis',count_basis_exceptions==1005)
    # Source date/coordinate coverage is assessed independently of model predictions.
    weather=pd.read_parquet(OUT/'wide/field_daily_weather.parquet');old=pd.read_parquet(ROOT/'data/paper_study/field_weather/daily_weather.parquet')
    check('wide forcing unique daily location keys',not weather.duplicated(['location_id','date']).any())
    matched=old.merge(weather,on=['location_id','date'],validate='one_to_one',suffixes=('_old','_new'))
    check('all original weather days preserved',len(matched)==len(old))
    for column in old.columns.difference(['location_id','date']):check('original weather value preserved '+column,np.allclose(matched[column+'_old'],matched[column+'_new'],rtol=0,atol=0,equal_nan=True))
    original_receipt=json.loads((ROOT/'data/paper_study/field_weather_extension/receipt.json').read_text())
    complete_receipt=json.loads((ROOT/'data/paper_study/field_weather_completed/receipt.json').read_text())
    check('weather extension checksum chain',original_receipt['original_weather_sha256']==sha(ROOT/'data/paper_study/field_weather/daily_weather.parquet')
          and original_receipt['merged_weather_sha256']==complete_receipt['source_weather_sha256']
          and complete_receipt['merged_weather_sha256']==sha(ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet'))
    weather_records=pd.concat([pd.read_parquet(OUT/name) for name in manifest['views']['field_daily_forcing']],ignore_index=True)
    for metric,g in weather_records.groupby('metric'):
        q=g.merge(weather[['location_id','date',metric]],left_on=['site_id','date'],right_on=['location_id','date'],validate='one_to_one')
        check('common forcing matches wide source '+metric,len(q)==len(weather) and np.allclose(q.value,q[metric],rtol=0,atol=0))
    available=pd.MultiIndex.from_frame(weather[['location_id','date']])
    coverage=metadata[['assessment_id','dataset_id','site_id','date','source_scope']].copy()
    coverage['calendar_date_available']=coverage.date.notna();coverage['forcing_day_available']=pd.MultiIndex.from_frame(coverage[['site_id','date']]).isin(available)
    coverage['source_eligible']=metadata.get('assessment_eligible',True).fillna(True)
    coverage.to_csv(OUT/'assessment_forcing_coverage.csv',index=False)
    external=coverage[coverage.source_scope.eq('reserved_external')&coverage.source_eligible]
    check('all source-eligible Corteva assessment dates have forcing',external.forcing_day_available.all())
    endpoints=pd.read_parquet(OUT/'metadata/season_endpoints.parquet')
    check('observed sowing availability follows actual source dates',endpoints.observed_sowing_date_available.equals(endpoints.sowing_date.notna()))
    check('upstream contradictory sowing flags retained and identified',int(endpoints.source_sowing_flag_conflict.sum())==4347)
    check('infection events remain unobserved',not endpoints.infection_event_observed.fillna(False).any())
    cells=pd.read_parquet(OUT/'metadata/europe_wheat_cells_025.parquet');calendar=pd.read_parquet(OUT/'metadata/europe_calendar_scenarios.parquet')
    check('14941 unique European grid cells',len(cells)==14941 and not cells.cell_id.duplicated().any())
    check('calendar-grid referential integrity',calendar.cell_id.isin(cells.cell_id).all() and not calendar.duplicated(['cell_id','crop_season','water_system']).any())
    check('area weights normalized',np.isclose(cells.wheat_area_weight.sum(),1) and np.isclose(cells.physical_wheat_area_weight.sum(),1))
    registry=pd.read_csv(OUT/'continental_forcing_partition_registry.csv');integrity=[]
    for r in registry.itertuples():
        actual_hash=sha(ROOT/r.path);check('continental partition checksum '+r.path,actual_hash==r.parquet_sha256)
        check('continental adjacent provenance checksum '+r.path,sha(ROOT/r.provenance_path)==r.provenance_sha256)
        integrity.append(dict(forcing_partition_id=r.forcing_partition_id,path=r.path,sha256=actual_hash,checksum_verified_against_bytes=True))
    pd.DataFrame(integrity).to_csv(OUT/'continental_partition_integrity.csv',index=False)
    check('partial continental climate status is explicit',not manifest['continental_climate_status']['full_download_and_validation_complete'])
    per_source=manifest['continental_climate_status'].get('per_source_status',{})
    if per_source:
        era5=per_source['era5'];deep_path=OUT/era5['deep_validation_snapshot']
        deep=json.loads(deep_path.read_text());summary=deep['summary']['era5']
        check('pinned deep ERA5 receipt checksum',sha(deep_path)==era5['deep_validation_sha256'])
        check('complete deep ERA5 validation recorded',deep['deep'] and deep['coverage_status']=='Complete'
              and summary['valid']==summary['expected']==372 and summary['invalid']==summary['missing']==0)
        check('ERA5 registry matches every deeply audited partition',
              set(registry.loc[registry.source.eq('era5'),'path'])=={r['path'] for r in deep['files']['era5'] if r['status']=='valid'})
        check('complete ERA5 and partial NASA distinguished',era5['full_download_and_validation_complete']
              and not per_source['nasa']['full_download_and_validation_complete'])
    check('no continental long daily expansion',manifest['continental_climate_status']['continental_daily_long_rows_generated']==0)
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),status='passed',check_count=len(checks),failed_checks=0,checks=checks,
                 scientific_measurement_rows=actual_rows,normalized_percentage_exceptions=24,assessment_aggregation_max_difference=max_error,
                 field_forcing_rows=len(weather),all_original_sources_preserved=True,Corteva_model_predictions_accessed=False,
                 model_imported_or_fitted=False,continental_daily_long_rows_generated=0,
                 continental_partitions_byte_checksum_verified=len(integrity),
                 independent_executable_sha256=sha(Path(__file__)))
    (HERE/'independent_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='checks'},indent=2),flush=True)


if __name__=='__main__':main()
