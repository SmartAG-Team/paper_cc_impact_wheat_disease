"""Append phenology simulations with the common schema and seal the data package."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/harmonized'
schema=pq.read_schema(OUT/'observations.parquet')
previous_manifest=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else {}
# Common taxon names and country labels; original EPPO codes and source labels
# remain in immutable files and the complete raw-cell archive.
crosswalk=pd.read_csv(ROOT/'analysis/disease-code-crosswalk.csv')
taxa=crosswalk.set_index('Code').EPPOName.to_dict()
study=pd.read_parquet(OUT/'observations.parquet')
is_basf=study.dataset_id.eq('basf-wheat-diseases')
study.loc[is_basf,'pathogen']=study.loc[is_basf,'pathogen'].map(lambda x:taxa.get(x,x) if x!='NNNNN' else None)
countries={'DK':'DENMARK','FI':'FINLAND','LT':'LITHUANIA','NO':'NORWAY','SE':'SWEDEN','SLOWAKIA':'SLOVAKIA','CZECH REPUBLIC':'CZECHIA'}
study['country']=study.country.map(lambda x:countries.get(str(x).upper(),str(x).upper()) if pd.notna(x) else None)
pq.write_table(pa.Table.from_pandas(study,schema=schema,preserve_index=False),OUT/'observations.parquet',compression='zstd')
study.to_csv(OUT/'observations.csv.gz',index=False,compression='gzip')
crosswalk.to_csv(OUT/'pathogen_code_crosswalk.csv',index=False)
source=ROOT/'analysis/phenology/field_transfer_era5/simulation/daily_features.csv'
f=pd.read_csv(source);meta=pd.read_csv(ROOT/'analysis/phenology/field_transfer_era5/case_metadata.csv')
f=f.merge(meta[['PEP_ID','season_year','wheat_type']],on='PEP_ID',validate='many_to_one')
units={'daylength':'hours','Cumulative_GDD':'degC_day','ppfun':'fraction','t_pp_GDD':'effective_degC_day',
    'Cumulative_t_pp_GDD':'effective_degC_day','VERDAY':'equivalent_vernalization_day',
    'CUMVER':'equivalent_vernalization_day','verfun':'fraction','t_pp_v_GDD':'effective_degC_day',
    'Cumulative_t_pp_v_GDD':'effective_degC_day'}
parts=[]
for col,unit in units.items():
    x=pd.DataFrame(index=f.index,columns=schema.names)
    x['dataset_id']='tpv_frozen_parameter_transfer';x['source_file']=str(source.relative_to(ROOT));x['source_table']='daily_features'
    x['source_row']=(f.index+2).astype(str);x['source_column']=col;x['record_id']=f.case_id+'|'+f.DATE
    x['observation_id']='tpv|'+x.record_id+'|'+col;x['plot_id']=f.case_id;x['site_id']=f.location_id
    x['latitude']=f.LAT;x['longitude']=f.LON;x['season_year']=f.season_year;x['date']=f.DATE
    x['time_basis']='simulation_calendar';x['metric']=col;x['unit']=unit;x['value']=f[col]
    x['raw_value']=f[col].astype(str);x['quality_status']='unvalidated_region_and_cultivar_transfer'
    x['measurement_role']='phenology_simulation';parts.append(x)
out=pd.concat(parts,ignore_index=True)
for field in schema:
    if pa.types.is_floating(field.type):out[field.name]=pd.to_numeric(out[field.name],errors='coerce')
    else:out[field.name]=out[field.name].map(lambda v:None if pd.isna(v) else str(v))
assert out.observation_id.is_unique
pq.write_table(pa.Table.from_pandas(out,schema=schema,preserve_index=False),OUT/'phenology_simulations.parquet',compression='zstd')
out.to_csv(OUT/'phenology_simulations.csv.gz',index=False,compression='gzip')

def normalize_and_write(frame,name):
    for field in schema:
        if field.name not in frame:frame[field.name]=None
        frame[field.name]=pd.to_numeric(frame[field.name],errors='coerce') if pa.types.is_floating(field.type) else frame[field.name].map(lambda v:None if pd.isna(v) else str(v))
    frame['country']=frame.country.map(lambda x:countries.get(str(x).upper(),str(x).upper()) if x else None)
    assert frame.observation_id.is_unique
    pq.write_table(pa.Table.from_pandas(frame[schema.names],schema=schema,preserve_index=False),OUT/name,compression='zstd')

# Evaluation predictions retain fold/model keys and are never extra observations.
p=ROOT/'analysis/disease_evaluation/predictions.csv'
if p.exists():
    f=pd.read_csv(p);x=pd.DataFrame(index=f.index,columns=schema.names)
    x['dataset_id']='conditional_seir_evaluation';x['source_file']=str(p.relative_to(ROOT));x['source_table']='predictions'
    x['source_row']=(f.index+2).astype(str);x['source_column']='predicted_percent';x['record_id']=f.series_id+'|'+f.target_date
    x['observation_id']='disease_prediction|'+f.fold+'|'+f.model+'|'+x.record_id
    x['plot_id']=f.TrialId.astype(str);x['site_id']=f.location_id;x['country']=f.Country
    x['latitude']=f.Latitude;x['longitude']=f.Longitude;x['season_year']=f.year;x['date']=f.target_date
    x['time_basis']='conditional_hindcast_calendar';x['organ']=f.PlantPart;x['metric']='predicted_infection_percent_proxy'
    x['treatment']=f.fold+'|'+f.model;x['unit']='percent';x['value']=f.predicted_percent;x['raw_value']=f.predicted_percent.astype(str)
    x['quality_status']=np.where(f.is_conditioning_observation,'conditioning_value_not_scored','heldout_prediction')
    x['measurement_role']='conditional_model_prediction';normalize_and_write(x,'disease_predictions.parquet')

# Continental climate and one-day forcing remain separate provider/grid products.
eu=ROOT/'data/era5/europe';p=eu/'era5_fixed_intervals_2017_2019_native025_grid.parquet'
if p.exists():
    f=pd.read_parquet(p);parts=[]
    for year in [2017,2018,2019]:
        for col,unit in [(f'tmean_{year}_c','degC'),(f'precipitation_{year}_mm','mm')]:
            x=pd.DataFrame(index=f.index,columns=schema.names);x['dataset_id']='era5_gee_monthly_context'
            x['source_file']=str(p.relative_to(ROOT));x['source_table']='continental_grid';x['source_row']=(f.index+2).astype(str)
            x['source_column']=col;x['site_id']=f.grid_cell_id;x['record_id']=f.grid_cell_id+'|'+str(year)
            x['observation_id']='europe_monthly|'+x.record_id+'|'+col;x['country']=f.country
            x['latitude']=f.latitude;x['longitude']=f.longitude;x['season_year']=year;x['date']=f'{year}-09-30'
            x['time_basis']='monthly_aggregated_September_to_September';x['metric']=col;x['unit']=unit
            x['value']=f[col];x['raw_value']=f[col].astype(str);x['quality_status']='native_monthly_grid_context'
            x['measurement_role']='regional_reanalysis_context';parts.append(x)
    normalize_and_write(pd.concat(parts,ignore_index=True),'europe_monthly_context.parquet')
p=eu/'era5_hourly_forcing_2019-05-01_2019-05-02_grid.parquet'
if p.exists():
    f=pd.read_parquet(p);parts=[]
    units={'tmean_c':'degC','rh_mean_pct':'percent','wind_mean_m_s':'m_s-1','rh_hours_ge90':'hours','rain_mm':'mm',
           'rain_hours_gt0_1mm':'hours','precipitation_mm':'mm','shortwave_mj_m2':'MJ_m-2'}
    for col,unit in units.items():
        x=pd.DataFrame(index=f.index,columns=schema.names);x['dataset_id']='era5_gee_hourly_forcing_smoke'
        x['source_file']=str(p.relative_to(ROOT));x['source_table']='continental_daily_grid';x['source_row']=(f.index+2).astype(str)
        x['source_column']=col;x['site_id']=f.grid_cell_id;x['record_id']=f.grid_cell_id+'|2019-05-01'
        x['observation_id']='europe_hourly_daily|'+x.record_id+'|'+col;x['country']=f.country
        x['latitude']=f.latitude;x['longitude']=f.longitude;x['date']='2019-05-01';x['time_basis']='UTC_daily_from_24_hours'
        x['metric']=col;x['unit']=unit;x['value']=f[col];x['raw_value']=f[col].astype(str)
        x['quality_status']='genuine_one_day_forcing_not_full_season';x['measurement_role']='regional_reanalysis_forcing_test';parts.append(x)
    normalize_and_write(pd.concat(parts,ignore_index=True),'europe_hourly_forcing_test.parquet')
files=[]
for p in sorted(OUT.glob('*')):
    if p.is_file() and p.name!='manifest.json':
        record=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        if p.suffix=='.parquet':record['rows']=pq.ParquetFile(p).metadata.num_rows
        files.append(record)
partitions=['observations.parquet','era5_hourly_observations.parquet','era5_daily_observations.parquet','phenology_simulations.parquet']
partitions.extend(p.name for p in [OUT/'disease_predictions.parquet',OUT/'europe_monthly_context.parquet',OUT/'europe_hourly_forcing_test.parquet'] if p.exists())
partitions.extend(p.name for p in [OUT/'primary_process_observations.parquet',OUT/'primary_secondary_predictions.parquet'] if p.exists())
assert all(pq.read_schema(OUT/p).equals(schema) for p in partitions)
manifest=dict(completed_utc=datetime.now(timezone.utc).isoformat(),common_schema_partitions=partitions,
    common_schema_verified=True,common_records=sum(pq.ParquetFile(OUT/p).metadata.num_rows for p in partitions),
    files=files,biological_replicates_are_not_common_record_count=True)
manifest['raw_cell_partitions']=[p.name for p in [OUT/'raw_table_cells.parquet',OUT/'raw_primary_process_table_cells.parquet'] if p.exists()]
manifest['raw_cell_records']=sum(pq.ParquetFile(OUT/p).metadata.num_rows for p in manifest['raw_cell_partitions'])
for key in ['additional_process_sources','forecast_source_sha256']:
    if key in previous_manifest:manifest[key]=previous_manifest[key]
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
readme='''Septoria research data package

Original source files are preserved in data/public_septoria and the original BASF files in data/. The original39 downloaded public files have verified repository MD5 and local SHA256 checksums. Original metadata, licence and download URLs remain in download-manifest.json. Additional acquisitions retain separate source manifests. Prospective sources without downloaded raw observations are catalogued separately and do not contribute observations.

Common Parquet partitions listed in manifest.json use the same29-column scientific schema and can be concatenated without schema conversion. observations.parquet contains field, laboratory, ancillary weather and yield cells; weather, phenology and model predictions retain separate partitions and measurement roles. observations.csv.gz and phenology_simulations.csv.gz are text equivalents. The ERA5 acquisition directory also contains conventional wide CSV/Parquet tables. common_metric_dictionary.csv defines study, metric, unit and role across all partitions. An observation is one measurement cell, not an independent biological replicate.

raw_table_cells.parquet preserves the initial spreadsheet/CSV archive; raw_primary_process_table_cells.parquet preserves the used rectangular cells of four additional original workbooks when present. Both retain original table, row and column keys for headers, measurements and source-derived views. Original byte files are authoritative for formatting and metadata. The packed Nordic weather records retain their complete original payload; seven unambiguous daily fields are decoded into scientific observations. Binary code and prose files remain in their original archived formats.

observation_id is unique within each common partition. record_id is a source-record or simulation-day key; plot_id and replicate identify clustering when available. source_file may include !archive/member syntax. date is YYYY-MM-DD. Controlled challenges retain days_post_inoculation and may lack calendar dates or field coordinates. Leaf sampling labels are not interpreted as identity of a persistent physical leaf. ERA5 site_id maps to location_registry.csv; daily weather has no single season_year because overlapping September windows share days. coordinate_seasons.csv and source link tables define those windows.

pathogen uses common EPPO scientific names across BASF and public experiments. pathogen_code_crosswalk.csv preserves the original codes, EPPO labels and source URLs. Non-disease yield rows have no pathogen. Common country labels use uppercase names, including Slovakia for the source label SLOWAKIA, Czechia for Czech Republic and full names for Nordic country codes; original labels remain in source files and the raw-cell archive.

value is the normalized numeric observation; raw_value preserves source interpretation. S means senescent in French field scores and is not zero infection. Missing, unresolved, percentage-range and calendar-quality states remain explicit. Percent-of-control yield is unbounded above100 and distinct from area percentages. Relative Tunisian indices are archived but are not labelled absolute percentages. Four absolute Tunisian PLACL values slightly exceed100; normalized percentage values are missing, without clipping. The 2019 Tunisian weather calendar remains inconsistent and is excluded from dated weather joins. Pycnidia density on zero lesion area is undefined. Nordic RH values above100 remain flagged. BASF absolute yield units remain unknown; source magnitude differences are not corrected.

Pycnidia-covered area does not directly measure emitted spores or infectious intensity. BASF P%INF retains an unspecified percentage-infection basis; it is not automatically a compartment fraction. Germination, mortality and in-vitro growth are process experiments rather than latent-period observations. Candidate fit rows and already-derived summaries are not additional experimental replicates. French F5 greenhouse assays retain their laboratory inoculation/scoring dates; field-origin weather is not assigned as their experimental forcing.

Phenology records are frozen-parameter transfer simulations, not observations. BASF has no supplied sowing dates; no BASF field-specific phenology is imputed. ERA5 uses models=era5, UTC, nearest0.25degree grid and elevation=nan. Rain/precipitation are preceding-hour sums. RH>=90percent hours are humidity exposure, not measured leaf wetness. ERA5 data served by Open-Meteo: https://open-meteo.com/en/docs/historical-weather-api .

Reproduction:
uv run --no-project --python 3.12 --with pandas --with pyarrow --with openpyxl --with xlrd python analysis/harmonize_septoria.py
uv run --no-project --python 3.12 --with pandas --with pyarrow python analysis/assemble_data_package.py

The additional disease_predictions.parquet, europe_monthly_context.parquet and europe_hourly_forcing_test.parquet partitions use the same scientific schema. Predictions retain fold/model identifiers and distinguish first-observation conditioning values from scored held-out predictions. Monthly European climate layers retain separate native-grid identifiers; they are not hourly forcing. The continental hourly example covers only1May2019, rather than a complete epidemic season.

Additional source process measurements are in primary_process_observations.parquet when present. These include greenhouse phenotyping, source-derived isolate classifications, residue-discharge plate colony counts and moisture-treatment pycnidial coverage. Plate colonies do not measure natural airborne ascospore concentrations. Field-isolate origin years are distinct from assay calendars.

Primary–secondary v2 forecasts are in primary_secondary_predictions.parquet when present. Predictions and source-labelled compartment states are derived model outputs rather than additional field observations. The prediction-key dictionary retains endpoints, issue dates, lead times, model/fold keys and realized-weather context. Origin-state percentages include latent and visible infection and are not visible-disease source contributions. French transfer comprises one independent station across two seasons. Diagnostic severity cutoffs do not establish economic injury or spray thresholds.

Uniform loading:
import json
import pyarrow.dataset as ds
package=json.load(open("data/harmonized/manifest.json"))
files=["data/harmonized/"+name for name in package["common_schema_partitions"]]
dataset=ds.dataset(files,format="parquet")
field_data=dataset.to_table(filter=ds.field("measurement_role")=="observed")
'''
(OUT/'README.txt').write_text(readme)
definitions=[]
for name in partitions:
    for batch in pq.ParquetFile(OUT/name).iter_batches(batch_size=150000,columns=['dataset_id','metric','unit','measurement_role']):
        definitions.extend(batch.to_pandas().drop_duplicates().to_dict('records'))
pd.DataFrame(definitions).drop_duplicates().sort_values(['dataset_id','metric','unit']).to_csv(OUT/'common_metric_dictionary.csv',index=False)
# Hash the final README after writing it, and include all common partitions.
manifest['files']=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**({'rows':pq.ParquetFile(p).metadata.num_rows} if p.suffix=='.parquet' else {}))
    for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json']
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
