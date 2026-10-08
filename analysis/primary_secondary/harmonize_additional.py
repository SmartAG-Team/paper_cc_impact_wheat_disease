"""Append public crop-protection process data using the existing common schema."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/harmonized'
BASE=ROOT/'data/public_septoria/primary_evidence'
SCHEMA=pq.read_schema(OUT/'observations.parquet')


def value_string(value):
    if pd.isna(value):return None
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    return str(value)


def common(source,dataset,sheet,frame,column,metric,unit,metadata=None,role='observed',binary=False):
    x=pd.DataFrame(index=frame.index,columns=SCHEMA.names)
    x['dataset_id']=dataset;x['source_file']=str(source.relative_to(ROOT));x['source_table']=sheet
    x['source_row']=(frame.index+2).astype(str);x['source_column']=column
    x['record_id']=dataset+'|'+source.name+'|'+sheet+'|'+x.source_row
    x['observation_id']=x.record_id+'|'+column
    x['pathogen']='Zymoseptoria tritici'
    for key,value in (metadata or {}).items():
        x[key]=frame[value].to_numpy() if isinstance(value,str) and value in frame else value
    raw=frame[column]
    number=raw.map({'Y':1.,'N':0.}) if binary else pd.to_numeric(raw,errors='coerce')
    x['raw_value']=raw.map(value_string);x['value']=number
    x['metric']=metric;x['unit']=unit;x['measurement_role']=role;x['quality_status']='valid'
    missing=raw.isna()|raw.astype(str).str.strip().str.lower().isin(['na','nan','-','*',''])
    x.loc[missing,'quality_status']='missing'
    x.loc[number.isna()&~missing,'quality_status']='unresolved_text'
    if unit=='percent':
        outside=number.notna()&~number.between(0,100)
        x.loc[outside,'quality_status']='outside_0_100';x.loc[outside,'value']=np.nan
    if unit=='colony_count':
        negative=number.lt(0)
        x.loc[negative,'quality_status']='negative_count';x.loc[negative,'value']=np.nan
    return x


def write(frame,path):
    for field in SCHEMA:
        if field.name not in frame:frame[field.name]=None
        frame[field.name]=pd.to_numeric(frame[field.name],errors='coerce') if pa.types.is_floating(field.type) else frame[field.name].map(value_string)
    assert frame.observation_id.is_unique
    pq.write_table(pa.Table.from_pandas(frame[SCHEMA.names],schema=SCHEMA,preserve_index=False),path,compression='zstd')


def main():
    parts=[];inventories=[];sources={}
    inter=BASE/'orellana-2022-interseason'
    source=inter/'F1_Disease_score_population-phenotyping_rawdata-1.xlsx'
    f=pd.read_excel(source,sheet_name='dataset')
    f['dpi']=(pd.to_datetime(f.scor_date)-pd.to_datetime(f.inoc_date)).dt.days
    f['score_date']=pd.to_datetime(f.scor_date).dt.strftime('%Y-%m-%d')
    f['replication']=f.block.astype(str)+'|'+f.plant.astype(str)
    meta={'season_year':'year','date':'score_date','time_basis':'greenhouse_calendar',
        'days_post_inoculation':'dpi','cultivar':'var_test','isolate':'iso','replicate':'replication',
        'organ':'leaf','treatment':'mixture'}
    parts.append(common(source,'orellana-2022-interseason','dataset',f,'score','pycnidial_coverage_percent','percent',meta))
    parts.append(common(source,'orellana-2022-interseason','dataset',f,'ddpi','thermal_time_post_inoculation','degC_day',meta,'experimental_ancillary'))
    source=inter/'F2_Ascospores_discharged_score_rawdata.xlsx';f=pd.read_excel(source,sheet_name='dataset')
    f['score_date']=pd.to_datetime(f.date).dt.strftime('%Y-%m-%d')
    f['replication']=f.block.astype(str)+'|dish'+f.dish.astype(str)
    meta={'season_year':'year','date':'score_date','time_basis':'residue_discharge_assay_calendar',
        'cultivar':'var_origin','replicate':'replication','treatment':'mixture','organ':'wheat_residue_discharge_assay'}
    for column in [c for c in f if str(c).startswith('spores_')]+['tot_dish','tot_mod']:
        metric='residue_discharge_blastospore_colonies_'+column
        role='source_derived_summary' if column=='tot_mod' else 'observed'
        parts.append(common(source,'orellana-2022-interseason','dataset',f,column,metric,'colony_count',meta,role))
    source=inter/'F3_List_of_isolates-1.xlsx';f=pd.read_excel(source,sheet_name='dataset')
    f['score_date']=pd.to_datetime(f.scor_date).dt.strftime('%Y-%m-%d')
    meta={'season_year':'year','date':'score_date','time_basis':'greenhouse_isolate_phenotyping',
        'cultivar':'var_origin','isolate':'iso','replicate':'bloc','treatment':'mixture'}
    for column in ('patho_APA','vir_CEL'):
        parts.append(common(source,'orellana-2022-interseason','dataset',f,column,column+'_indicator',
                            'binary',meta,'derived_isolate_classification',binary=True))
    source=BASE/'boixel-moisture/data_moisture_regimes_Zymoseptoria_tritici_Boixel_et_al._2020.xlsx'
    f=pd.read_excel(source,sheet_name='data_globales_R')
    f['replication']='series'+f.serie.astype(str)+'|block'+f.bloc.astype(str)+'|'+f.rep.astype(str)
    meta={'days_post_inoculation':'DPI','time_basis':'days_post_inoculation','isolate':'iso',
          'treatment':'duration_HR','replicate':'replication','organ':'inoculated_seedling_area'}
    parts.append(common(source,'boixel-moisture-process','data_globales_R',f,'PYC',
                        'pycnidial_coverage_percent','percent',meta))
    records=pd.concat(parts,ignore_index=True)
    write(records,OUT/'primary_process_observations.parquet')
    for source in sorted(BASE.rglob('*.xlsx')):
        sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
        for sheet in pd.ExcelFile(source).sheet_names:
            raw=pd.read_excel(source,sheet_name=sheet,header=None)
            inventories.append({'source_file':str(source.relative_to(ROOT)),'sheet':sheet,
                'rows_including_header':len(raw),'columns':raw.shape[1]})
    pd.DataFrame(inventories).to_csv(OUT/'primary_process_source_inventory.csv',index=False)
    counts=records.groupby(['dataset_id','metric','measurement_role','quality_status'],dropna=False).size().rename('cells').reset_index()
    counts.to_csv(OUT/'primary_process_observation_counts.csv',index=False)
    manifest_path=OUT/'manifest.json';manifest=json.loads(manifest_path.read_text())
    name='primary_process_observations.parquet'
    if name not in manifest['common_schema_partitions']:manifest['common_schema_partitions'].append(name)
    definitions=[]
    for name in manifest['common_schema_partitions']:
        assert pq.read_schema(OUT/name).equals(SCHEMA)
        for batch in pq.ParquetFile(OUT/name).iter_batches(batch_size=150000,
                       columns=['dataset_id','metric','unit','measurement_role']):
            definitions.append(batch.to_pandas().drop_duplicates())
    pd.concat(definitions).drop_duplicates().sort_values(['dataset_id','metric','unit']).to_csv(OUT/'common_metric_dictionary.csv',index=False)
    manifest['common_records']=sum(pq.ParquetFile(OUT/p).metadata.num_rows for p in manifest['common_schema_partitions'])
    manifest['additional_process_sources']=sources
    manifest['updated_utc']=datetime.now(timezone.utc).isoformat()
    manifest['files']=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        **({'rows':pq.ParquetFile(p).metadata.num_rows} if p.suffix=='.parquet' else {}))
        for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json']
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    receipt={'original_workbooks_sha256':sources,'common_schema_equals_original':True,
        'new_measurement_cells':len(records),'new_partition_observation_ids_unique':True,
        'classification':'laboratory/residue discharge measurements; no natural airborne time series',
        'field_origin_year_is_not_assay_calendar_year':True,
        'residue_plate_count_is_not_ascospore_air_concentration':True}
    (ROOT/'analysis/primary_secondary/additional_harmonization.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
