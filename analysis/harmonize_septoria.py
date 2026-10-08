"""Traceable common tables for original trials, public Septoria studies and ERA5.

uv run --no-project --python 3.12 --with pandas --with pyarrow --with openpyxl
       --with xlrd python analysis/harmonize_septoria.py
Original files are immutable. Source row/column identifiers remain explicit.
"""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import gzip
import hashlib
import json
import re
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hafeez_metadata import isolate_metadata, leaf_organ_metadata

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'data/public_septoria'
OUT = ROOT / 'data/harmonized'
OUT.mkdir(parents=True, exist_ok=True)
COLS = ['observation_id','dataset_id','source_file','source_table','source_row',
        'source_column','record_id','plot_id','site_id','country','latitude','longitude',
        'season_year','date','time_utc','time_basis','days_post_inoculation','cultivar',
        'pathogen','isolate','treatment','replicate','organ','metric','unit','value',
        'raw_value','quality_status','measurement_role']
NUMERIC = {'latitude','longitude','season_year','days_post_inoculation','value'}
SCHEMA = pa.schema([(c, pa.float64() if c in NUMERIC else pa.string()) for c in COLS])
parts, dictionaries = [], {}


def loc(lat, lon):
    return 'era5loc_' + hashlib.sha256(f'{lat:.6f},{lon:.6f}'.encode()).hexdigest()[:12]


def scalar(v):
    if pd.isna(v): return None
    if isinstance(v, (datetime, pd.Timestamp)): return v.isoformat()
    return str(v)


def canonical(frame, dataset, filename, table, values, units, metadata=None,
              role='observed', default_quality='valid'):
    """One source cell per observation; values maps source column to metric."""
    frame = frame.copy().reset_index(drop=True)
    if '_row' not in frame: frame['_row'] = np.arange(2, len(frame)+2)
    metadata = metadata or {}
    for column, metric in values.items():
        unit = units.get(column, 'source_unit_unspecified')
        dictionaries[(dataset, metric, unit)] = {'dataset_id':dataset,'metric':metric,'unit':unit,
            'source_column':column,'source_file':filename,'source_table':table}
        out = pd.DataFrame(index=frame.index, columns=COLS)
        out['dataset_id'],out['source_file'],out['source_table'] = dataset,filename,table
        out['source_row'],out['source_column'] = frame['_row'].astype(str),column
        out['observation_id'] = dataset+'|'+filename+'|'+table+'|'+out.source_row+'|'+column
        out['record_id'] = dataset+'|'+table+'|'+out.source_row
        for key, value in metadata.items():
            out[key] = frame[value].to_numpy() if isinstance(value,str) and value in frame else value
        raw = frame[column]
        out['raw_value'] = raw.map(scalar)
        number = pd.to_numeric(raw, errors='coerce')
        out['value'],out['metric'],out['unit'],out['measurement_role'] = number,metric,unit,role
        out['quality_status'] = default_quality
        missing = raw.isna() | raw.astype(str).str.strip().isin(['','-','*','NA','xx','.'])
        out.loc[missing,'quality_status'] = 'missing'
        out.loc[number.isna() & ~missing,'quality_status'] = 'unresolved_text'
        if unit=='percent':
            invalid = number.notna() & ~number.between(0,100)
            out.loc[invalid,'quality_status'] = 'outside_0_100'
            out.loc[invalid,'value'] = np.nan
        if '_quality' in frame:
            special = frame['_quality'].notna()
            out.loc[special,'quality_status'] = frame.loc[special,'_quality']
        parts.append(out)


def write_common(frame, destination):
    for c in COLS:
        if c not in frame: frame[c]=None
        if c in NUMERIC: frame[c]=pd.to_numeric(frame[c],errors='coerce')
        else: frame[c]=frame[c].map(scalar)
    pq.write_table(pa.Table.from_pandas(frame[COLS],schema=SCHEMA,preserve_index=False),destination,
                   compression='zstd',use_dictionary=True)


# Original trials include every disease and yield record, not only Septoria.
basf=pd.read_csv(ROOT/'data/basf-wheat-diseases.txt',sep='\t')
basf['_row']=np.arange(2,len(basf)+2)
basf['date_iso']=pd.to_datetime(basf.Date).dt.strftime('%Y-%m-%d')
basf['season']=pd.to_datetime(basf.Date).dt.year
basf['weather_site']=[loc(a,b) for a,b in zip(basf.Latitude,basf.Longitude)]
taxa=pd.read_csv(ROOT/'analysis/disease-code-crosswalk.csv').set_index('Code').EPPOName
basf['canonical_pathogen']=basf.Organism.map(taxa)
basf['_quality']=None
basf.loc[basf.Clarifier.eq('CROP INJURY'),'_quality']='crop_injury_annotation'
for param,group in basf.groupby('Method'):
    metric={'P%INF':'infection_percent_unspecified_basis','MWCGS':'absolute_grain_yield',
            'MCALCR':'relative_grain_yield_percent'}[param]
    canonical(group,'basf-wheat-diseases','data/basf-wheat-diseases.txt','trials',{'Value':metric},
        {'Value':'source_mass_unit_unspecified' if param=='MWCGS' else 'percent_of_untreated_yield' if param=='MCALCR' else 'percent'},
        dict(plot_id='TrialId',site_id='weather_site',country='Country',latitude='Latitude',
            longitude='Longitude',season_year='season',date='date_iso',time_basis='calendar',
            pathogen='canonical_pathogen',treatment='Treatment',organ='PlantPart'))

# All five regularly structured controlled inoculation sheets.
p=PUBLIC/'hafeez-2025-infection/Hafeez et al. Stb15, Nature Plants - pathology data.xlsx'
for sheet in pd.ExcelFile(p).sheet_names:
    if sheet=='ArinaEMS_IPO88004':continue
    f=pd.read_excel(p,sheet_name=sheet)
    f['_declared_isolate']=isolate_metadata(f,sheet).to_numpy()
    for col in [c for c in f if re.fullmatch(r'[pd]\d+',str(c))]:
        canonical(f,'hafeez-2025-infection',str(p.relative_to(ROOT)),sheet,
            {col:'pycnidial_coverage_percent' if col[0]=='p' else 'leaf_damage_necrosis_and_chlorosis_percent'},
            {col:'percent'},dict(days_post_inoculation=int(col[1:]),time_basis='days_post_inoculation',
                pathogen='Zymoseptoria tritici',isolate='_declared_isolate',cultivar='Name' if 'Name' in f else ('Line' if 'Line' in f else 'Variety_newname'),
                replicate='Rep',organ='seedling_leaf',treatment='controlled_conidial_inoculation'))
f=pd.read_excel(p,sheet_name='ArinaEMS_IPO88004',header=None)
for r in range(2,7):
    for c in range(2,14):
        small=pd.DataFrame({'score':[f.iloc[r,c]],'_row':[r+1]})
        canonical(small,'hafeez-2025-infection',str(p.relative_to(ROOT)),'ArinaEMS_IPO88004',
            {'score':'pycnidial_coverage_percent' if c%2 else 'necrosis_percent'}, {'score':'percent'},
            dict(days_post_inoculation=21,time_basis='days_post_inoculation',cultivar=str(f.iloc[r,0]),
                replicate=str((c-2)//2+1),pathogen='Zymoseptoria tritici',isolate='IPO88004',organ=leaf_organ_metadata('ArinaEMS_IPO88004'),
                record_id=f'ArinaEMS|{r+1}|plant{(c-2)//2+1}'))
        # Actual Excel column avoids colliding N and P scores.
        parts[-1]['source_column']=f'column_{c+1}'
        parts[-1]['observation_id']=parts[-1].observation_id+'|'+str(c+1)

# Field pycnidia coverage; senescent and unassessed leaves remain explicit.
p=PUBLIC/'orellana-torrejon-2022-field/F1_Field_disease_severity_rawdata.csv'
f=pd.read_csv(p,sep=';',comment='#');f['_row']=np.arange(4,len(f)+4)
f['date_iso']=pd.to_datetime(f.date,dayfirst=True).dt.strftime('%Y-%m-%d')
f['season']=pd.to_datetime(f.date,dayfirst=True).dt.year
f['plot']=f.mixture+'|'+f.rep+'|'+f.var_origin
for c in [c for c in f if c.startswith('%spor_')]:
    g=f.copy();g['_quality']=None
    g.loc[g[c].astype(str).str.strip().eq('S'),'_quality']='senescent'
    g.loc[g[c].isna()|g[c].astype(str).str.strip().eq('NA'),'_quality']='not_assessed'
    canonical(g,'orellana-torrejon-2022-field',str(p.relative_to(ROOT)),'F1_field',
        {c:'pycnidial_coverage_percent'},{c:'percent'},dict(date='date_iso',season_year='season',
        time_basis='calendar',plot_id='plot',site_id=loc(48.8469444444444,1.94277777777778),
        country='FRANCE',latitude=48.8469444444444,longitude=1.94277777777778,
        cultivar='var_origin',pathogen='Zymoseptoria tritici',replicate='rep',treatment='mixture',organ=c[6:]))
    if g['_quality'].eq('senescent').any():
        s=g[g['_quality'].eq('senescent')].copy();s[c]=1;s['_quality']='valid_leaf_state'
        canonical(s,'orellana-torrejon-2022-field',str(p.relative_to(ROOT)),'F1_field_senescence',
            {c:'leaf_senescent'},{c:'boolean'},dict(date='date_iso',season_year='season',plot_id='plot',
                site_id=loc(48.8469444444444,1.94277777777778),time_basis='calendar',cultivar='var_origin',organ=c[6:]),role='leaf_state')

# Population-phenotyping greenhouse assays; field-origin weather is not forcing.
p=PUBLIC/'orellana-torrejon-2022-field/F5_Disease_score_population-phenotyping_rawdata.xlsx'
f=pd.read_excel(p,sheet_name='dataset');f['score_date']=pd.to_datetime(f.scor_date).dt.strftime('%Y-%m-%d')
f['dpi']=(pd.to_datetime(f.scor_date)-pd.to_datetime(f.inoc_date)).dt.days
canonical(f,'orellana-torrejon-2022-field',str(p.relative_to(ROOT)),'F5_greenhouse',{'score':'pycnidial_coverage_percent'},
    {'score':'percent'},dict(date='score_date',days_post_inoculation='dpi',time_basis='controlled_calendar_and_dpi',
        season_year='year',cultivar='var_test',isolate='iso',pathogen='Zymoseptoria tritici',organ='leaf',
        replicate='plant',treatment='controlled_conidial_inoculation'))
p=PUBLIC/'orellana-torrejon-2022-field/F4_List_of_isolates.xlsx'
f=pd.read_excel(p,sheet_name='dataset')
for c,m in [('patho_APA','pathogenic_on_Apache'),('vir_CEL','virulent_on_Cellule')]:
    g=f.copy();g['raw_binary']=g[c];g[c]=g[c].map({'Y':1,'N':0})
    canonical(g,'orellana-torrejon-2022-field',str(p.relative_to(ROOT)),'F4_isolates', {c:m},{c:'boolean'},
        dict(isolate='iso',season_year='year',pathogen='Zymoseptoria tritici',time_basis='season',treatment='period'),role='derived_classification')
    parts[-1]['raw_value']=g['raw_binary'].map(scalar)

# Swiss leaf scans: two destructive collections, no longitudinal leaf identity.
p=PUBLIC/'karisto-2018-field/Output_c1_c3.csv'
f=pd.read_csv(ROOT/'analysis/public_septoria/karisto_field_leaf_scores.csv')
f['_row']=f.source_row
image_metrics={'leafArea':'leaf_area','necrosisArea':'necrosis_area','PLACL':'necrotic_leaf_area_percent',
               'pycnidiaCount':'pycnidia_count','meanPycnidiaArea':'mean_pycnidium_area',
               'pycnidiaPerCm2Leaf':'pycnidia_density_leaf','pycnidiaPerCm2Lesion':'pycnidia_density_lesion',
               'pycnidiaGreyValue':'pycnidium_greyscale'}
image_units={'leafArea':'mm2','necrosisArea':'mm2','PLACL':'percent','pycnidiaCount':'count',
             'meanPycnidiaArea':'mm2','pycnidiaPerCm2Leaf':'count_cm-2','pycnidiaPerCm2Lesion':'count_cm-2',
             'pycnidiaGreyValue':'greyscale_0_255'}
canonical(f,'karisto-2018-field',str(p.relative_to(ROOT)),'leaf_scans',image_metrics,image_units,
    dict(record_id='Picture',plot_id='sowing_number',site_id=loc(47.449,8.682),country='SWITZERLAND',
        latitude=47.449,longitude=8.682,season_year=2016,date='date_from_repository_notes',time_basis='calendar_nominal_collection',
        cultivar='Gen',pathogen='Zymoseptoria tritici',organ='leaf'))
for x in parts[-len(image_metrics):]:
    x.loc[f.duplicate_leaf_label.astype(bool),'quality_status']='duplicate_label_retained'
    if x.metric.iloc[0]=='pycnidia_density_lesion':
        x.loc[f.necrosisArea.eq(0),['value','quality_status']]=[np.nan,'undefined_zero_denominator']

# Tunisian absolute leaf measurements. Relative and mean views are archived below.
p=PUBLIC/'durum-mixtures-2020/Durum_mixtures_Data_out.zip'
with zipfile.ZipFile(p) as z:
    for year in [2018,2019]:
        for suffix in ['flag','flag-1']:
            member=f'out/{year}_Output_final_{suffix}.xls';f=pd.read_excel(BytesIO(z.read(member)))
            f['plot']=f.treatment.astype(str)+'|'+f.replicate.astype(str)
            f['composition']=f[['INRAT100','karim','salim','monastir']].astype(str).agg('|'.join,axis=1)
            dates={(2018,'flag'):'2018-05-09',(2018,'flag-1'):'2018-04-22',
                   (2019,'flag'):'2019-04-25',(2019,'flag-1'):'2019-04-17'}
            canonical(f,'durum-mixtures-2020',str(p.relative_to(ROOT))+'!'+member,'absolute_leaf_scans',image_metrics,image_units,
                dict(record_id='Picture',plot_id='plot',site_id=loc(36.5477472222222,9.01131388888889),country='TUNISIA',
                    latitude=36.5477472222222,longitude=9.01131388888889,season_year=year,date=dates[(year,suffix)],
                    time_basis='calendar',cultivar='composition',pathogen='Zymoseptoria tritici',organ=suffix,
                    replicate='replicate',treatment='treatment'),default_quality='conditional_pycnidia_positive_leaf_sample')
            # Incidence is a plot-level fraction repeated per scanned leaf.
            canonical(f,'durum-mixtures-2020',str(p.relative_to(ROOT))+'!'+member,'incidence_repeated_per_leaf',
                {'incidence':'plot_incidence_repeated'},{'incidence':'fraction'},dict(plot_id='plot',season_year=year,date=dates[(year,suffix)]),role='repeated_plot_metadata')
        member=f'out/{year}_Tunisia_Cultivar_Mix_Yield_Data.xls';f=pd.read_excel(BytesIO(z.read(member)))
        f['plot']=f.treatment.astype(str)+'|'+f.replicate.astype(str)
        canonical(f,'durum-mixtures-2020',str(p.relative_to(ROOT))+'!'+member,'yield',{'TKW':'thousand_kernel_weight','kg_per_ha':'grain_yield'},
            {'TKW':'g','kg_per_ha':'kg_ha-1'},dict(plot_id='plot',season_year=year,time_basis='season',treatment='treatment',replicate='replicate'))
        member=f'out/{year}_Tunisia_cultivar_mixtures_weather.xlsx';f=pd.read_excel(BytesIO(z.read(member)))
        f['date_iso']=pd.to_datetime(f.Date).dt.strftime('%Y-%m-%d')
        if year==2019:f['_quality']='calendar_mismatch_source_year'
        canonical(f,'durum-mixtures-2020',str(p.relative_to(ROOT))+'!'+member,'station_weather',
            {'Rainfall':'precipitation_daily','TempMinC':'temperature_daily_min','TempMaxC':'temperature_daily_max'},
            {'Rainfall':'mm','TempMinC':'degC','TempMaxC':'degC'},dict(season_year=year,date='date_iso',time_basis='calendar_source',
                site_id=loc(36.5477472222222,9.01131388888889)),role='observed_weather')

# Thermal process experiments: fits and summary rows do not become replicates.
p=PUBLIC/'chaloner-2019-weather-response/rstb20180266_si_002.xlsx'
for sheet in ['1h Thermal death','Germination','Growth (sqrt(#pixels))']:
    f=pd.read_excel(p,sheet_name=sheet)
    f['_row']=np.arange(2,len(f)+2)
    f=f[pd.to_numeric(f.temperature,errors='coerce').notna()].copy()
    metrics={c:'process_'+c for c in f if c not in ['temperature','bio_replicate','tech_replicate','treatment']}
    units={c:('percent' if c.startswith('percentage') else 'count' if c in ['cells','hyphae'] else 'source_process_unit') for c in metrics}
    canonical(f,'chaloner-2019-weather-response',str(p.relative_to(ROOT)),sheet,metrics,units,
        dict(pathogen='Zymoseptoria tritici',replicate='bio_replicate',time_basis='experimental_endpoint'),role='process_experiment')
    for x in parts[-len(metrics):]:
        x['treatment']='temperature='+f.temperature.astype(str)+'degC'+(' '+f.treatment.astype(str) if 'treatment' in f else '')
        if sheet=='1h Thermal death':
            raw=pd.to_numeric(x.raw_value,errors='coerce');neg=raw.lt(0)
            x.loc[neg,'value']=raw[neg];x.loc[neg,'quality_status']='negative_baseline_corrected_estimate'

# Ancillary Nordic yields and unambiguous daily weather fields. Packed hourly
# payloads remain untouched in the raw-table archive and original text file.
p=PUBLIC/'nordic-baltic-2012-2016/Yield.xlsx';f=pd.read_excel(p,header=3)
f['_row']=np.arange(5,len(f)+5);f=f[pd.to_numeric(f.Year,errors='coerce').between(1900,2100)]
yield_cols={c:'ancillary_'+str(c) for c in f if str(c).startswith('Yield') and pd.to_numeric(f[c],errors='coerce').notna().any()}
canonical(f,'nordic-baltic-2012-2016',str(p.relative_to(ROOT)),'Yield',yield_cols,
    {c:'kg_ha-1' for c in yield_cols},dict(season_year='Year',country='Country',plot_id='SpotIT_ID',
        treatment='Treatment_code**',time_basis='season'),role='ancillary_mixed_pathogen_yield')
p=PUBLIC/'nordic-baltic-2012-2016/Weather data.txt';weather=[];packed_rows=[]
for block,line in enumerate(p.read_text(encoding='cp1252').splitlines(),1):
    if not line.strip():continue
    chunks=line.split('\\');station,year=chunks[0].split(';');year=int(year);header=chunks[1].split(';')
    for record,chunk in enumerate(chunks[2:],1):
        if not chunk.strip():continue
        fields=chunk.split('//',1)[0].split(';')
        d={k:v for k,v in zip(header,fields)}
        # Exactly the first sixteen simple fields end before the packed payload.
        d.update(station=station,year=year,_row=f'{block}:{record}',packed_payload=chunk)
        day=int(d['yDay']);date=pd.Timestamp(year=year,month=1,day=1)+pd.Timedelta(days=day-1)
        assert date.year==year and day>=1
        d['date_iso']=date.strftime('%Y-%m-%d');weather.append(d)
        packed_rows.append(dict(dataset_id='nordic-baltic-2012-2016',source_file=str(p.relative_to(ROOT)),
            source_table='packed_weather',source_row=f'{block}:{record}',source_column='packed_record',raw_value=chunk))
f=pd.DataFrame(weather)
for c in ['avgT','minT','maxT','avgRh','minRh','maxRh','mm']:
    f[c]=f[c].str.replace(',','.',regex=False)
canonical(f,'nordic-baltic-2012-2016',str(p.relative_to(ROOT)),'weather_daily_simple',
    dict(zip(['avgT','minT','maxT','avgRh','minRh','maxRh','mm'],
             ['temperature_daily_mean','temperature_daily_min','temperature_daily_max','rh_daily_mean','rh_daily_min','rh_daily_max','precipitation_daily'])),
    {c:'degC' if c.endswith('T') else 'percent' if c.endswith('Rh') else 'mm' for c in ['avgT','minT','maxT','avgRh','minRh','maxRh','mm']},
    dict(site_id='station',season_year='year',date='date_iso',time_basis='calendar'),role='ancillary_observed_weather')

observations=pd.concat(parts,ignore_index=True)
assert observations.observation_id.is_unique, observations[observations.observation_id.duplicated()].head().to_dict('records')
write_common(observations,OUT/'observations.parquet')
observations[COLS].to_csv(OUT/'observations.csv.gz',index=False,compression='gzip')
pd.DataFrame(dictionaries.values()).to_csv(OUT/'metric_dictionary.csv',index=False)
scientific_counts=observations.groupby(['dataset_id','measurement_role','metric'],dropna=False).agg(
    records=('value','size'),numeric_values=('value','count')).reset_index()
scientific_counts.to_csv(OUT/'observation_counts.csv',index=False)

# Identical schema for actual ERA5 hourly and daily forcing. Partitions remain
# independently readable and can be concatenated with observations.parquet.
for resolution in ['hourly','daily']:
    f=pd.read_parquet(ROOT/f'data/era5/{resolution}_weather.parquet')
    key='time_utc' if resolution=='hourly' else 'date'
    writer=pq.ParquetWriter(OUT/f'era5_{resolution}_observations.parquet',SCHEMA,compression='zstd')
    ignore={'location_id','date','time_utc','season_year'}
    units={'temperature_2m_c':'degC','relative_humidity_2m_pct':'percent','dew_point_2m_c':'degC',
        'wind_speed_10m_m_s':'m_s-1','shortwave_radiation_w_m2':'W_m-2',
        'tmean_c':'degC','tmin_c':'degC','tmax_c':'degC','rh_mean_pct':'percent','dewpoint_mean_c':'degC',
        'precipitation_mm':'mm','rain_mm':'mm','rain_hours_gt_0_1mm':'hours','rh_hours_ge_90pct':'hours',
        'wind_mean_m_s':'m_s-1','shortwave_mj_m2':'MJ_m-2','hour_count':'hours'}
    count=0
    for c in [c for c in f if c not in ignore and pd.api.types.is_numeric_dtype(f[c])]:
        if c not in units:raise ValueError(f'Undocumented weather field {c}')
        for start in range(0,len(f),100000):
            chunk=f.iloc[start:start+100000];out=pd.DataFrame(index=chunk.index,columns=COLS)
            out['dataset_id']='era5_open_meteo';out['source_file']=f'data/era5/{resolution}_weather.parquet'
            out['source_table']=resolution;out['source_row']=chunk.index.astype(str);out['source_column']=c
            out['site_id']=chunk.location_id;out['record_id']=chunk.location_id+'|'+chunk[key].astype(str)
            out['observation_id']='era5|'+resolution+'|'+out.record_id+'|'+c
            out['date']=chunk['date'] if 'date' in chunk else chunk.time_utc.astype(str).str[:10]
            if resolution=='hourly':out['time_utc']=chunk.time_utc.astype(str)
            out['time_basis']='UTC_'+resolution;out['metric']=c;out['unit']=units[c];out['value']=chunk[c]
            out['raw_value']=chunk[c].map(scalar);out['quality_status']='validated_reanalysis'
            out['measurement_role']='reanalysis_weather';out=out[COLS]
            for col in COLS:
                out[col]=pd.to_numeric(out[col],errors='coerce') if col in NUMERIC else out[col].map(scalar)
            writer.write_table(pa.Table.from_pandas(out,schema=SCHEMA,preserve_index=False));count+=len(out)
    writer.close()
    pd.DataFrame([dict(resolution=resolution,records=count)]).to_csv(OUT/f'era5_{resolution}_counts.csv',index=False)

# Complete cell archive for every downloaded spreadsheet and CSV, including
# original headers, alternate derived views, numerical fits and metadata sheets.
raw_schema=pa.schema([(c,pa.string()) for c in ['dataset_id','source_file','source_table','source_row','source_column','raw_value']])
raw_writer=pq.ParquetWriter(OUT/'raw_table_cells.parquet',raw_schema,compression='zstd')
tables=[];raw_count=0

def archive_table(df,dataset,path,table):
    global raw_count
    df=df.copy();df.index=np.arange(1,len(df)+1);df.columns=np.arange(1,len(df.columns)+1)
    long=df.stack(future_stack=True).reset_index();long.columns=['source_row','source_column','raw_value']
    long['dataset_id'],long['source_file'],long['source_table']=dataset,path,table
    for c in raw_schema.names:long[c]=long[c].map(scalar)
    raw_writer.write_table(pa.Table.from_pandas(long[raw_schema.names],schema=raw_schema,preserve_index=False))
    raw_count+=len(long);tables.append(dict(dataset_id=dataset,source_file=path,source_table=table,rows=len(df),columns=len(df.columns),cells=len(long)))

for folder in sorted(PUBLIC.iterdir()):
    if not folder.is_dir():continue
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in ['.xlsx','.xls']:
            for sheet in pd.ExcelFile(p).sheet_names:archive_table(pd.read_excel(p,sheet_name=sheet,header=None),folder.name,str(p.relative_to(ROOT)),sheet)
        elif p.suffix.lower()=='.csv':
            archive_table(pd.read_csv(p,sep=';' if folder.name.startswith('orellana') else ',',header=None,comment='#' if folder.name.startswith('orellana') else None),folder.name,str(p.relative_to(ROOT)),'csv')
        elif p.name=='Durum_mixtures_Data_out.zip':
            with zipfile.ZipFile(p) as z:
                for member in z.namelist():
                    if member.endswith(('.xlsx','.xls')):
                        buf=BytesIO(z.read(member))
                        for sheet in pd.ExcelFile(buf).sheet_names:
                            buf.seek(0);archive_table(pd.read_excel(buf,sheet_name=sheet,header=None),folder.name,str(p.relative_to(ROOT))+'!'+member,sheet)
archive_table(pd.read_csv(ROOT/'data/basf-wheat-diseases.txt',sep='\t',header=None),'basf-wheat-diseases','data/basf-wheat-diseases.txt','tsv')
raw=pd.DataFrame(packed_rows)
raw_writer.write_table(pa.Table.from_pandas(raw[raw_schema.names],schema=raw_schema,preserve_index=False));raw_count+=len(raw)
raw_writer.close()
pd.DataFrame(tables).to_csv(OUT/'source_table_inventory.csv',index=False)
counts={'generated_utc':datetime.now(timezone.utc).isoformat(),'canonical_observations':len(observations),
        'canonical_unique_ids':int(observations.observation_id.nunique()),'raw_cells_including_packed_weather':raw_count,
        'archived_tables':len(tables),'nordic_daily_records':len(weather),'observation_schema':str(SCHEMA),
        'schema_matches':all(pq.read_schema(OUT/f'era5_{r}_observations.parquet').equals(SCHEMA) for r in ['hourly','daily']),
        'quality_counts':observations.quality_status.value_counts().to_dict()}
assert counts['schema_matches']
(OUT/'validation.json').write_text(json.dumps(counts,indent=2)+'\n')
print(json.dumps(counts,indent=2))
