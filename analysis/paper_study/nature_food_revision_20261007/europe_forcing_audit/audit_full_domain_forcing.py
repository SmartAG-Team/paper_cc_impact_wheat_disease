"""Read-only cache and engineering audit; never imports or runs a biological model."""
from pathlib import Path
from datetime import datetime, timezone
import calendar, gc, hashlib, importlib.util, json, os, resource, subprocess, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE_CODE = ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/climate/cache_weather.py'
FIELDS = ['tmean_c','tmax_c','rh_mean_pct','precipitation_mm']
GIB = 1024**3

def digest_small(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rss_bytes():
    # macOS ru_maxrss is bytes, unlike Linux's KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def ranges(values):
    vals=sorted(set(map(int,values))); blocks=[]
    for v in vals:
        if not blocks or v>blocks[-1][-1]+1: blocks.append([v])
        else: blocks[-1].append(v)
    return ';'.join(str(b[0]) if len(b)==1 else f'{b[0]}-{b[-1]}' for b in blocks)

def source_inventory():
    # This module has no model imports. Call its existing registration function only.
    spec=importlib.util.spec_from_file_location('readonly_weather_cache_registration',CACHE_CODE)
    cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
    registered=cache.registered_sources()
    frozen=json.loads((CACHE_CODE.parent/'weather_cache_configuration_before_extraction.json').read_text())
    assert registered==frozen['sources']
    rows=[];schemas={}
    nasa_set={str(ROOT/r['path']) for r in registered}
    actual_nasa=set(map(str,(ROOT/'data/paper_study/climate/nasa_v2/daily').rglob('*.parquet')))
    assert nasa_set==actual_nasa
    source_paths=[('nasa',ROOT/r['path'],r) for r in registered]
    source_paths += [('era5',x,None) for x in sorted((ROOT/'data/paper_study/climate/era5_daily').glob('*.parquet'))]
    for provider,path,reg in source_paths:
        meta=json.loads(path.with_suffix('.json').read_text());q=meta['quality'];pf=pq.ParquetFile(path)
        names=pf.schema_arrow.names;schema_key=provider+'|'+','.join(names)
        schemas.setdefault(schema_key,dict(provider=provider,columns=[dict(name=f.name,type=str(f.type)) for f in pf.schema_arrow],count=0,example=str(path.relative_to(ROOT))))['count']+=1
        model=meta.get('model','ERA5');scenario=meta.get('scenario','reference');year=int(meta.get('year',meta.get('window_start','')[:4]))
        present=path.exists() and path.with_suffix('.json').exists()
        assert present and pf.metadata.num_rows==q['rows']==q['days']*q['cells'] and q['cells']==14941
        assert all(q['missing_values'].get(f,1)==0 and f in names and pf.schema_arrow.field(f).type==pa.float32() for f in FIELDS)
        if provider=='nasa':
            assert q['days']==(366 if calendar.isleap(year) else 365)
            assert q['date_min']==f'{year}-01-01' and q['date_max']==f'{year}-12-31'
        else:
            month=int(meta['window_start'][5:7]); assert q['days']==calendar.monthrange(year,month)[1]
            assert q['date_min']==meta['window_start'] and pd.Timestamp(q['date_max'])+pd.Timedelta(days=1)==pd.Timestamp(meta['window_end_exclusive'])
        assert path.stat().st_size==meta['parquet_bytes']
        colbytes={n:sum(pf.metadata.row_group(j).column(i).total_compressed_size for j in range(pf.metadata.num_row_groups)) for i,n in enumerate(names)}
        rows.append(dict(provider=provider,model=model,scenario=scenario,year=year,path=str(path.relative_to(ROOT)),receipt_path=str(path.with_suffix('.json').relative_to(ROOT)),file_bytes=path.stat().st_size,rows=pf.metadata.num_rows,cells=q['cells'],days=q['days'],date_min=q['date_min'],date_max=q['date_max'],row_groups=pf.metadata.num_row_groups,columns=';'.join(names),radiation_columns=';'.join(n for n in names if any(s in n.lower() for s in ['rad','rsds','rlds','solar'])),wind_columns=';'.join(n for n in names if any(s in n.lower() for s in ['wind','uas','vas'])),selected_six_column_compressed_bytes=sum(colbytes[n] for n in ['date','cell_id',*FIELDS]),registered_parquet_sha256=meta['parquet_sha256'],parquet_rehashed=False,receipt_sha256=digest_small(path.with_suffix('.json')),registry_sha256=meta['registry_sha256'],missing_required_values=sum(q['missing_values'][f] for f in FIELDS),prior_deep_check_verified=True,metadata_and_footer_verified_now=True))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'full_domain_forcing_inventory.csv',index=False)
    return frame,list(schemas.values()),frozen

def benchmark(sample,registry_ids):
    before=rss_bytes();start=time.perf_counter();table=pq.read_table(sample,columns=['date','cell_id',*FIELDS],use_threads=True);seconds=time.perf_counter()-start
    loaded_rss=rss_bytes();arrow_bytes=table.nbytes;first=table.slice(0,14941).column('cell_id').to_pylist();last=table.slice(table.num_rows-14941,14941).column('cell_id').to_pylist()
    assert first==registry_ids and last==registry_ids
    unique_ids=pc.unique(table['cell_id']);unique_dates=pc.unique(table['date']);assert len(unique_ids)==14941 and len(unique_dates)==365 and table.num_rows==14941*365
    result=dict(path=str(sample.relative_to(ROOT)),operation='single full-domain annual Arrow load of date,cell_id and four float32 forcing fields',wall_seconds=seconds,rows=table.num_rows,cells=len(unique_ids),days=len(unique_dates),arrow_buffer_bytes=arrow_bytes,process_peak_rss_before_bytes=before,process_peak_rss_after_load_bytes=loaded_rss,peak_rss_increment_after_load_bytes=max(0,loaded_rss-before),arrow_threads=pa.cpu_count(),cold_cache_forced=False,filesystem_cache_state='not controlled; elapsed time is a local measurement, not a guaranteed cold-disk throughput',ordered_first_and_last_days_match_registry=True,model_imported_or_run=False)
    del table;gc.collect();pa.default_memory_pool().release_unused()
    start=time.perf_counter();sample_hash=digest_small(sample);result['single_sample_hash_seconds']=time.perf_counter()-start;result['single_sample_hash_matches_registered']=sample_hash==json.loads(sample.with_suffix('.json').read_text())['parquet_sha256'];assert result['single_sample_hash_matches_registered']
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();inventory,schemas,frozen=source_inventory()
    area=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet');calpath=ROOT/'data/paper_study/wheat_area/europe_wheat_calendar_scenarios.parquet';cal=pd.read_parquet(calpath)
    assert len(area)==14941 and area.cell_id.is_unique
    valid=cal.loc[cal.crop_season.eq('winter_wheat')&cal.water_system.eq('rainfed')&cal.calendar_valid].copy();assert len(valid)==14932 and valid.cell_id.is_unique
    eligible=area.merge(valid[['cell_id','planting_doy','maturity_doy']],on='cell_id',validate='one_to_one');missing=area.loc[~area.cell_id.isin(valid.cell_id)]
    missing.to_csv(OUT/'missing_winter_rainfed_calendar_cells.csv',index=False)
    calrows=[]
    for (crop,water,status),g in cal.groupby(['crop_season','water_system','calendar_valid']):calrows.append(dict(crop_season=crop,water_system=water,calendar_valid=bool(status),cells=len(g),planting_doy_min=g.planting_doy.min(),planting_doy_max=g.planting_doy.max(),maturity_doy_min=g.maturity_doy.min(),maturity_doy_max=g.maturity_doy.max(),provider_files=';'.join(sorted(g.source_file.dropna().unique()))))
    pd.DataFrame(calrows).to_csv(OUT/'calendar_inventory.csv',index=False)
    summaries=[]
    for (provider,model,scenario),g in inventory.groupby(['provider','model','scenario']):summaries.append(dict(provider=provider,model=model,scenario=scenario,partitions=len(g),calendar_years=ranges(g.year),date_min=g.date_min.min(),date_max=g.date_max.max(),file_bytes=int(g.file_bytes.sum()),rows=int(g.rows.sum()),unique_full_domain_cells=14941,minimum_days_per_partition=int(g.days.min()),maximum_days_per_partition=int(g.days.max()),radiation_stored=bool(g.radiation_columns.ne('').any()),wind_stored=bool(g.wind_columns.ne('').any())))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'forcing_source_period_summary.csv',index=False)
    stores=[]
    for folder in ['data/paper_study/climate/nasa_v2/daily','data/paper_study/climate/nasa_v2/variables','data/paper_study/climate/nasa_v2/ncss_cache','data/paper_study/climate/era5_daily','analysis/paper_study/overwinter_leaf_model_20261006/climate/annual_weather_cache']:
        files=[x for x in (ROOT/folder).rglob('*') if x.is_file()];stores.append(dict(folder=folder,files=len(files),total_file_bytes=sum(x.stat().st_size for x in files),parquet_files=sum(x.suffix=='.parquet' for x in files),parquet_bytes=sum(x.stat().st_size for x in files if x.suffix=='.parquet'),npz_files=sum(x.suffix=='.npz' for x in files),metadata_only_size_scan=True,full_domain=not folder.endswith('annual_weather_cache')))
    pd.DataFrame(stores).to_csv(OUT/'weather_store_size_inventory.csv',index=False)
    bench=benchmark(ROOT/'data/paper_study/climate/nasa_v2/daily/ACCESS-CM2/historical/2001.parquet',area.cell_id.tolist())
    replay=json.loads((OUT/'frozen_replay_timing.json').read_text());seconds_per_season=replay['wall_seconds']*len(valid)/replay['draw_seasons'];nseasons=3*3*90
    allocation=[];days=replay['days'];leaves=8
    # Only array dimensions of the already-frozen implementation; no parameter estimation.
    fit=json.loads((ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/disease/overwinter_source_model/frozen_selected_fit.json').read_text());state_slots=int(fit['fitted']['parameters']['latent_stages'])+4
    for cells in [64,128,256,512,1024,14932]:allocation.append(dict(cells=cells,days=days,state_history_bytes=cells*(days+1)*leaves*state_slots*8,flow_history_bytes=4*cells*days*leaves*8,residue_history_bytes=cells*(days+1)*2*8,one_damage_history_bytes=cells*(days+1)*leaves*8,four_float64_forcing_bytes=4*cells*days*8,host_area_and_renewal_float64_bytes=2*cells*days*leaves*8,host_active_bool_bytes=cells*days*leaves,lower_bound_only=True))
    alloc=pd.DataFrame(allocation);numeric=[x for x in alloc if x.endswith('_bytes')];alloc['listed_allocation_bytes']=alloc[numeric].sum(axis=1);alloc.to_csv(OUT/'frozen_array_memory_estimates.csv',index=False)
    coefficients=[]
    for model in ['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']:
        for scenario in ['ssp126','ssp245','ssp585']:
            path=ROOT/'analysis/paper_study/regional_v1/bias_alignment'/model/scenario/'coefficients.parquet';co=pd.read_parquet(path);assert len(co)==14941*12 and not co.duplicated(['cell_id','month']).any();assert np.isfinite(co[['temperature_offset','humidity_logit_offset','precipitation_ratio']].to_numpy()).all();coefficients.append(dict(model=model,scenario=scenario,path=str(path.relative_to(ROOT)),rows=len(co),cells=co.cell_id.nunique(),months=co.month.nunique(),file_bytes=path.stat().st_size,coefficient_sha256=digest_small(path),complete_four_field_alignment=True,radiation_wind_alignment_present=False))
    pd.DataFrame(coefficients).to_csv(OUT/'alignment_coefficient_inventory.csv',index=False)
    mem=int(subprocess.check_output(['sysctl','-n','hw.memsize']).decode());cpus=int(subprocess.check_output(['sysctl','-n','hw.logicalcpu']).decode());disk=os.statvfs(ROOT)
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='read-only source inventories, Parquet footer checks and one annual Arrow load; no simulations or dependency changes',registered_sources_function='analysis.paper_study.overwinter_leaf_model_20261006.climate.cache_weather.registered_sources',registered_NEX_sources_match_frozen_configuration=True,source_partition_counts=inventory.groupby('provider').size().to_dict(),source_parquet_bytes=inventory.groupby('provider').file_bytes.sum().astype(int).to_dict(),source_rows=inventory.groupby('provider').rows.sum().astype(int).to_dict(),schemas=schemas,grid=dict(registry_cells=len(area),eligible_winter_rainfed_cells=len(valid),missing_calendar_cells=len(missing),total_harvested_ha=float(area.harvested_total_ha.sum()),eligible_harvested_ha=float(eligible.harvested_total_ha.sum()),calendar_provider_resolution_degrees=float(cal.calendar_resolution_deg.iloc[0]),calendar_terminal_stage=cal.calendar_terminal_stage.unique().tolist(),winter_spring_area_allocation=cal.winter_spring_area_allocation.unique().tolist(),calendar_source_sha256=digest_small(calpath)),sample_cache=dict(annual_sources=frozen['annual_source_count'],unique_spatial_cells=frozen['cells'],draws=frozen['draws'],full_domain_census=False),benchmark=bench,source_integrity_policy=dict(metadata_and_receipt_SHA256_and_footer_checked=True,prior_deep_validation='analysis/paper_study/climate/actual_coverage_validation_era5.json and actual_coverage_validation_nasa.json',large_Parquet_hashes_reused_from_prior_verified_registration=True,large_Parquet_files_rehashed_now=1),full_domain_compute_extrapolation=dict(reference_receipt='frozen_replay_timing.json',reference_seconds=replay['wall_seconds'],reference_draw_seasons=replay['draw_seasons'],days=days,eligible_cell_seasons=nseasons*len(valid),GCM_SSP_harvest_year_cases=nseasons,estimated_seconds_per_case_linear_only=seconds_per_season,estimated_serial_hours_linear_only=seconds_per_season*nseasons/3600,estimated_three_worker_hours_ideal_only=seconds_per_season*nseasons/10800,not_a_full_domain_benchmark=True,field_sample_timing_does_not_establish_prediction_skill=True),machine=dict(logical_cpus=cpus,physical_RAM_bytes=mem,filesystem_available_bytes=disk.f_bavail*disk.f_frsize,simultaneous_other_workload_not_reserved=True),proposed_resumable_configuration=dict(GCMs=['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0'],SSPs=['ssp126','ssp245','ssp585'],harvest_periods=['1991-2020','2031-2060','2071-2100'],cell_batch_size=256,worker_processes=3,Arrow_threads_per_worker=2,CPU_numerical_library_threads_per_worker=1,task_unit='GCM/SSP/harvest-year/calendar/forcing-version',within_task_batches=int(np.ceil(len(valid)/256)),output_grain='one scalar summary row per registered cell/year; missing-calendar status for9cells',checkpoint='atomic annual summary Parquet plus receipt/configuration hash; existing valid completed annual cases skipped',source_IO='reuse previous/current year arrays across adjacent harvest years; do not load whole century or create duplicate full-domain compressed source caches',source_integrity='reuse existing verified source registration; detect changed bytes/receipts and check selected sample checksum; final results tied to the frozen source/configuration identities',estimated_worker_RAM='listed 256-cell trajectory allocations~0.2GB plus two-year full-domain raw arrays~0.17GB; Arrow and alignment temporaries require additional headroom; reserve2–3GB per worker initially',forecasting_diseases_added=False,full_domain_job_started=False),limitations=['Existing full-domain cache has four meteorological fields only; radiation, wind, daily minimum temperature and solar-energy inputs are not cached.','Catalog band availability is not evidence of stored full-domain coverage.','GGCMI calendars are scenario metadata, not observed field sowing/harvest dates; area registry has no winter/spring allocation.','Original whole-domain compact driver batches1024cells, but is a different model archive from the currently frozen overwinter driver; a new engineering wrapper is required to batch the current unchanged simulator.','Current overwinter climate cache and outputs contain64registered draws at62unique cells; they cannot be relabelled as full-grid results.','Existing main overwinter runner refuses an already-existing configuration at main entry; inner annual-case resume logic exists, but restart entry needs a dedicated new wrapper/namespace.','Compute estimates linearly scale one64-draw measurement and exclude full-domain I/O, startup, concurrency, metadata and uncertain local workload.'])
    receipt['artifact_row_counts']={x.name:len(pd.read_csv(x)) for x in OUT.glob('*inventory.csv')};receipt['artifact_row_counts']['forcing_source_period_summary.csv']=len(summary);receipt['artifact_row_counts']['frozen_array_memory_estimates.csv']=len(alloc)
    receipt['audit_code_sha256']=digest_small(Path(__file__));receipt['audit_wall_seconds']=time.perf_counter()-start
    (OUT/'full_domain_forcing_audit.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(summary[['provider','model','scenario','partitions','calendar_years','file_bytes']].to_string(index=False));print('Arrow annual load:',json.dumps(bench));print('Memory lower bounds:',alloc[['cells','listed_allocation_bytes']].to_dict(orient='records'));print('Linear compute extrapolation:',receipt['full_domain_compute_extrapolation']);print('PASS: inventories and source/coverage registration; one sample checksum; no simulation.')

if __name__=='__main__':main()
