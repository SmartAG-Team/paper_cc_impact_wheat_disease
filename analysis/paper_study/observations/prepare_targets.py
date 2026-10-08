#!/usr/bin/env python3
"""Inventory source-backed wheat Septoria targets without disease fitting.

Original sources and canonical harmonized partitions are immutable. Outputs use
the existing 29-column observation schema and a separate rich endpoint schema.
Dependencies: pandas, numpy, pyarrow, openpyxl, xlrd.
"""
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hashlib
import json
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parent
OUT = ROOT / "data/paper_study/observations"
PUBLIC = ROOT / "data/public_septoria"
COLS = ['observation_id','dataset_id','source_file','source_table','source_row',
        'source_column','record_id','plot_id','site_id','country','latitude','longitude',
        'season_year','date','time_utc','time_basis','days_post_inoculation','cultivar',
        'pathogen','isolate','treatment','replicate','organ','metric','unit','value',
        'raw_value','quality_status','measurement_role']
NUMERIC = {'latitude','longitude','season_year','days_post_inoculation','value'}
OBS, ASSESS, INVENTORY, HASHES = [], [], [], {}
LEAVES = {'LEAF, 1ST / FLAG LEAF':1,'LEAF, FLAG':1,'LEAF, 2ND':2,'LEAF, 3RD':3,
          'LEAF, 4TH':4,'LEAF, 5TH':5,'LEAF, 6ST':6,'LEAF, 7ST':7}


def watch(path):
    p=Path(path);HASHES.setdefault(str(p.relative_to(ROOT)),hashlib.sha256(p.read_bytes()).hexdigest());return p


def source_key(file,table,row,column):
    return f'{file}|{table}|{row}|{column}'


def observation(dataset,file,table,row,column,raw,metadata,quality='valid',role='observed_field'):
    item={c:None for c in COLS};item.update(metadata)
    value=pd.to_numeric(raw,errors='coerce')
    item.update(dataset_id=dataset,source_file=file,source_table=table,source_row=str(row),
                source_column=column,observation_id=dataset+'|'+source_key(file,table,row,column),
                raw_value=None if pd.isna(raw) else str(raw),value=None if pd.isna(value) else float(value),
                quality_status=quality,measurement_role=role)
    OBS.append(item);return item['observation_id']


def assessment(meta,date,value,keys,n_numeric=1,n_senescent=0,n_missing=0,time_index=None):
    ASSESS.append({**meta,'date':date,'time_index':time_index,'value':value,'sourcekeys':';'.join(keys),
                   'n_numeric':n_numeric,'n_senescent':n_senescent,'n_missing':n_missing,
                   'source_file':keys[0].split('|')[1] if keys else None,
                   'measurement_role':'derived_plot_cultivar_leaf_numeric_mean' if meta['dataset_id']=='orellana-torrejon-2022-field' else 'source_unit_assessment',
                   'source_table':'F1' if meta['dataset_id']=='orellana-torrejon-2022-field' else None})


def basf():
    path=watch(ROOT/'data/basf-wheat-diseases.txt');file=str(path.relative_to(ROOT))
    raw=pd.read_csv(path,sep='\t');raw['source_row']=np.arange(2,len(raw)+2)
    f=raw.loc[raw.Treatment.eq('Untreated') & raw.Organism.eq('SEPTTR')].copy()
    links=pd.read_csv(watch(ROOT/'data/era5/basf_trial_weather_links.csv')).set_index('TrialId')
    trial_occurrence=[]
    for row in f.itertuples():
        part='unspecified' if pd.isna(row.PlantPart) else row.PlantPart
        year=int(str(row.Date)[:4]);location=links.loc[row.TrialId,'location_id']
        quality='crop_injury_annotation' if row.Clarifier=='CROP INJURY' else 'valid'
        if not np.isfinite(row.Value) or not 0<=row.Value<=100:quality='invalid_percent'
        metadata=dict(record_id=str(row.TrialId),plot_id=str(row.TrialId),site_id=location,country=row.Country,
                      latitude=row.Latitude,longitude=row.Longitude,season_year=year,date=row.Date,
                      time_basis='calendar',pathogen='Zymoseptoria tritici',treatment='Untreated',organ=part,
                      metric='infection_percent_unspecified_basis',unit='percent')
        key=observation('basf-wheat-diseases',file,'trials',row.source_row,'Value',row.Value,metadata,quality)
        if quality=='valid':
            meta=dict(dataset_id='basf-wheat-diseases',endpoint_series=f'{row.TrialId}|{part}',physical_unit=str(row.TrialId),
                      plot_id=str(row.TrialId),site_id=location,country=row.Country,season_year=year,
                      cultivar=None,treatment='Untreated',organ=part,leaf_numbering='explicit source label; unknown where not ordinal',
                      metric=metadata['metric'],unit='percent',sowing_date=None,sowing_date_source=None,
                      trial_design='untreated natural field; sowing and cultivar absent',
                      sampling_scope='trial assessment; P%INF denominator unspecified',
                      occurrence_identifiable='observed_window_only',date_source='original Date field',
                      stage_from=row.GsFrom,stage_to=row.GsTo)
            assessment(meta,row.Date,float(row.Value),[key])
    included=f.loc[f.Clarifier.ne('CROP INJURY') & f.Value.between(0,100)]
    for trial,g in included.groupby('TrialId'):
        dates=g.groupby('Date').Value.max().sort_index()
        positive=dates[dates.gt(0)];first=positive.index[0] if len(positive) else None
        zeros=dates[(dates.eq(0)) & (dates.index<first)] if first else dates[dates.eq(0)]
        trial_occurrence.append(dict(dataset_id='basf-wheat-diseases',TrialId=trial,year=int(str(dates.index[0])[:4]),
                                     first_assessment=dates.index[0],last_assessment=dates.index[-1],
                                     first_positive=first,last_zero_before_positive=zeros.index[-1] if len(zeros) else None,
                                     onset_censoring='interval' if len(positive) and len(zeros) else 'left' if len(positive) else 'right',
                                     observed_window_occurrence=int(len(positive)>0),full_season_absence_supported=False,
                                     infection_event_observed=False,aggregation='any positive among assessed organs; no mixed-organ severity average'))
    pd.DataFrame(trial_occurrence).to_csv(OUT/'basf_trial_occurrence.csv',index=False)
    INVENTORY.append(dict(dataset_id='basf-wheat-diseases',source_file=file,source_design='untreated natural field',
                         source_rows=len(f),usable_numeric_rows=len(included),physical_trial_seasons=included.TrialId.nunique(),
                         known_sowing=False,calendar_dates=True,occurrence_target='all 76 positive within assessed window; no whole-season negatives',
                         onset_target='observed first-positive symptom/assessment intervals; predominantly left-censored',
                         final_target='last numeric score separately by source organ; not necessarily harvest severity',
                         limitations='6 crop-injury records retained as annotations but excluded from disease targets; P%INF denominator unspecified'))


def french():
    path=watch(PUBLIC/'orellana-torrejon-2022-field/F1_Field_disease_severity_rawdata.csv');file=str(path.relative_to(ROOT))
    watch(path.parent/'README.md');f=pd.read_csv(path,sep=';',comment='#',keep_default_na=False)
    columns=[c for c in f if c.startswith('%spor_')];grouped=defaultdict(list)
    sow={2018:'2017-10-17',2019:'2018-11-15'}
    methods='https://bsppjournals.onlinelibrary.wiley.com/doi/10.1111/ppa.13458 (sections 2.3–2.4)'
    for data in f.to_dict('records'):
        date=pd.to_datetime(data['date'],dayfirst=True).date().isoformat();year=int(date[:4])
        plot=f'{year}|{data["mixture"]}|{data["rep"]}'
        unit=plot+'|'+data['var_origin']
        for column in columns:
            raw=data[column];text=str(raw).strip();value=pd.to_numeric(raw,errors='coerce');organ=column[6:]
            quality='senescent' if text=='S' else 'not_assessed' if pd.isna(raw) or text in ('NA','nan','') else 'valid'
            if quality=='valid' and (pd.isna(value) or not 0<=value<=100):quality='unresolved_or_invalid'
            key=observation('orellana-torrejon-2022-field',file,'F1',data['id'],column,raw,
                            dict(record_id=str(data['id']),plot_id=plot,site_id='era5loc_ed480fff4117',country='FRANCE',
                                 season_year=year,date=date,time_basis='calendar',cultivar=data['var_origin'],
                                 pathogen='Zymoseptoria tritici',treatment='no_fungicide; '+data['mixture'],
                                 replicate=data['rep']+'|'+data['plant'],organ=organ,metric='pycnidial_coverage_percent',unit='percent'),quality)
            grouped[(unit,organ,date)].append((key,value,quality,data))
    for (unit,organ,date),records in grouped.items():
        values=[float(v) for _,v,q,_ in records if q=='valid'];sample=records[0][3];year=int(date[:4])
        meta=dict(dataset_id='orellana-torrejon-2022-field',endpoint_series=unit+'|'+organ,
                  physical_unit=f'{year}|{sample["mixture"]}|{sample["rep"]}',plot_id=f'{year}|{sample["mixture"]}|{sample["rep"]}',site_id='era5loc_ed480fff4117',
                  country='FRANCE',season_year=year,cultivar=sample['var_origin'],treatment='no_fungicide; '+sample['mixture'],
                  organ=organ,leaf_numbering='juvenile L: five low seedling leaves' if organ.startswith('L') else 'adult F: top-down flag-relative numbering',
                  metric='pycnidial_coverage_percent',unit='percent',sowing_date=sow[year],sowing_date_source=methods,
                  trial_design='natural field; no fungicide; insecticide seed treatment reported',
                  sampling_scope='destructive plants; plot×cultivar×leaf-category mean over numeric scores; plant labels do not identify the same leaf over time',
                  occurrence_identifiable='observed_window_only',date_source='original F1 day/month/year; differing dates within same visit retained',
                  stage_from=15 if organ.startswith('L') else None,stage_to=None)
        assessment(meta,date,sum(values)/len(values) if values else np.nan,[r[0] for r in records],
                   len(values),sum(r[2]=='senescent' for r in records),sum(r[2]=='not_assessed' for r in records))
    # Author late-epidemic SEV summaries are different endpoint definitions, not duplicate raw F-rank means.
    for year in (2018,2019):
        p=watch(path.parent/f'F{2 if year==2018 else 3}_t2_{year}_field-severity_means.txt')
        table=pd.read_csv(p,sep='\t',comment='#');table['source_row']=np.arange(4,len(table)+4)
        for row in table.itertuples():
            plot=f'{year}|{row.mixture}|{row.rep}';key=observation('orellana-torrejon-2022-field',str(p.relative_to(ROOT)),f'F{2 if year==2018 else 3}',
                row.source_row,'mean',row.mean,dict(record_id=plot+'|'+row.var_origin,plot_id=plot,site_id='era5loc_ed480fff4117',
                 country='FRANCE',season_year=year,time_basis='late_epidemic_summary',cultivar=row.var_origin,
                 pathogen='Zymoseptoria tritici',treatment='no_fungicide; '+row.mixture,replicate=row.rep,
                 organ='author_representative_leaf_layers',metric='author_late_epidemic_SEV_percent',unit='percent'),role='author_summary_not_extra_independent_observation')
    INVENTORY.append(dict(dataset_id='orellana-torrejon-2022-field',source_file=file,source_design='natural field; no fungicide; F5 is a separate controlled challenge',
                         source_rows=len(f),usable_numeric_rows=sum(pd.to_numeric(f[c],errors='coerce').notna().sum() for c in columns),
                         physical_trial_seasons=30,known_sowing=True,calendar_dates=True,
                         occurrence_target='plot×cultivar observed-window positives; juvenile leaves preserved',
                         onset_target='leaf-category observation-window censoring; earliest seasonal symptom often predates first seedling assessment',
                         final_target='latest numeric score for each L/F category plus separate author F2/F3 SEV summary',
                         limitations='L and F ranks are different numbering systems; senescent S and NA are not zero; one station across two years'))


def swiss():
    folder=PUBLIC/'karisto-2018-field';p=watch(folder/'Output_c1_c3.csv');file=str(p.relative_to(ROOT))
    f=pd.read_csv(p);cult=pd.read_csv(watch(folder/'cultivars_sowingnumbers.csv')).set_index('Sow_Nr')
    aggregates=defaultdict(list)
    nominal={'c1':'2016-05-25','c3':'2016-07-04'}
    for idx,row in f.iterrows():
        m=re.fullmatch(r'(c\d+)_sn(\d+)_(\d+)',row.Picture);collection,plot,leaf=m.groups();plot=int(plot)
        genotype=cult.loc[plot,'Gen']
        for column,metric,unit in [('PLACL','necrotic_leaf_area_percent','percent'),('pycnidiaCount','pycnidia_count','count')]:
            key=observation('karisto-2018-field',file,'leaf_scans',idx+2,column,row[column],dict(record_id=row.Picture,plot_id=str(plot),
                site_id='era5loc_eccd4a216326',country='SWITZERLAND',season_year=2016,date=nominal[collection],time_basis='calendar_collection_date',
                cultivar=genotype,pathogen='Zymoseptoria tritici',treatment='natural_infection_reference_trial; fungicide_status_requires_repository_methods',
                organ='destructive_leaf_sample',metric=metric,unit=unit),quality='conditional_diseased_leaf_collection')
            aggregates[(plot,column,collection,genotype)].append((key,float(row[column])))
    for (plot,column,collection,genotype),records in aggregates.items():
        meta=dict(dataset_id='karisto-2018-field',endpoint_series=f'{plot}|{column}',physical_unit=str(plot),plot_id=str(plot),
                  site_id='era5loc_eccd4a216326',country='SWITZERLAND',season_year=2016,cultivar=genotype,
                  treatment='natural_infection_reference_trial; untreated status pending source methods',organ='destructive_leaf_sample',
                  leaf_numbering='leaf index within collection is sample identity, not leaf rank',
                  metric='necrotic_leaf_area_percent' if column=='PLACL' else 'pycnidia_count',unit='percent' if column=='PLACL' else 'count',
                  sowing_date=None,sowing_date_source=None,trial_design='natural field; sampled-diseased-leaf reference collection',
                  sampling_scope='conditional leaf collection; plot means are not unconditional crop incidence',
                  occurrence_identifiable='conditional_sample_only',date_source='https://doi.org/10.5061/dryad.171q4 usage notes: c1 2016-05-25; c3 2016-07-04',
                  stage_from=None,stage_to=None)
        assessment(meta,nominal[collection],sum(v for _,v in records)/len(records),[k for k,_ in records],len(records))
    p=watch(folder/'2018_Karisto_et_al_Cultivar_rankings_AUDPC_data.xlsx');f=pd.read_excel(p)
    for idx,row in f.iterrows():
        for day in (0,14,44,52):
            column=f'score_Day{day}';quality='assumed_zero_not_observed' if day==0 else 'valid_source_visual_score'
            key=observation('karisto-2018-field',str(p.relative_to(ROOT)),'Sheet1',idx+2,column,row[column],dict(record_id=str(row.gen_id),
                site_id='era5loc_eccd4a216326',country='SWITZERLAND',season_year=2016,time_basis='relative_study_day',cultivar=row.cultivar,
                pathogen='Zymoseptoria tritici',organ='whole_cultivar_visual_summary',metric='visual_STB_score',unit='source_visual_score'),quality,
                role='assumed_AUDPC_anchor' if day==0 else 'cultivar_mean_over_two_plots')
            if day:
                meta=dict(dataset_id='karisto-2018-field',endpoint_series=f'visual|{row.gen_id}',physical_unit='cultivar_summary_not_independent_plot',
                          plot_id=None,site_id='era5loc_eccd4a216326',country='SWITZERLAND',season_year=2016,cultivar=row.cultivar,
                          treatment='natural_infection_reference_trial; untreated status pending source methods',organ='whole_cultivar_visual_summary',
                          leaf_numbering='not ranked',metric='visual_STB_score',unit='source_visual_score',sowing_date=None,sowing_date_source=None,
                          trial_design='natural field',sampling_scope='two-plot cultivar summary; Day0 assumed and excluded',
                          occurrence_identifiable='observed_relative_window_only',date_source='relative Day14/44/52; no invented calendar anchor',stage_from=None,stage_to=None)
                assessment(meta,None,float(row[column]),[key],2,time_index=day)
    INVENTORY.append(dict(dataset_id='karisto-2018-field',source_file=file,source_design='natural field; untreated status pending primary repository methods',
                         source_rows=len(pd.read_csv(folder/'Output_c1_c3.csv')),usable_numeric_rows=len(aggregates),physical_trial_seasons='plot identifiers, one station-year',
                         known_sowing=False,calendar_dates='repository c1/c3 dates; visual series relative dates only',
                         occurrence_target='conditional leaf samples; positive-only sampling limits crop absence labels',
                         onset_target='not exact; visual first assessment left-censored; assumed Day0 excluded',
                         final_target='last collection PLACL or pycnidia count; last observed visual score in separate domain',
                         limitations='same Picture label may identify distinct rows; keep source rows; no physical-leaf tracking'))


def tunis():
    p=watch(PUBLIC/'durum-mixtures-2020/Durum_mixtures_Data_out.zip');file=str(p.relative_to(ROOT))
    watch(PUBLIC/'durum-mixtures-2020/Durum_mixtures_Source_code.zip.zip')
    with zipfile.ZipFile(p) as z:
        for year in (2018,2019):
            for suffix in ('flag','flag-1'):
                member=f'out/{year}_Output_final_{suffix}.xls';f=pd.read_excel(BytesIO(z.read(member)))
                for (treatment,rep),g in f.groupby(['treatment','replicate']):
                    composition='|'.join(f'{name}:{g[name].iloc[0]}' for name in ('INRAT100','karim','salim','monastir'))
                    plot=f'{year}|{treatment}|{rep}'
                    for column,metric,unit in [('PLACL','necrotic_leaf_area_percent','percent'),('pycnidiaCount','pycnidia_count','count')]:
                        keys=[]
                        for idx,row in g.iterrows():
                            keys.append(observation('durum-mixtures-2020',file+'!'+member,'leaf_scans',idx+2,column,row[column],dict(record_id=row.Picture,
                             plot_id=plot,site_id='tunis_field_station',country='TUNISIA',season_year=year,time_basis='late_season_endpoint',cultivar=composition,
                             pathogen='Zymoseptoria tritici',treatment=str(treatment),replicate=str(rep),organ=suffix,metric=metric,unit=unit),
                             quality='conditional_infected_leaf_sample; challenge_no_fungicide_design'))
                        meta=dict(dataset_id='durum-mixtures-2020',endpoint_series=plot+'|'+suffix+'|'+column,physical_unit=plot,plot_id=plot,
                         site_id='tunis_field_station',country='TUNISIA',season_year=year,cultivar=composition,treatment=str(treatment),organ=suffix,
                         leaf_numbering='adult flag or flag-minus-one; not juvenile L',metric=metric,unit=unit,sowing_date=None,sowing_date_source=None,
                         trial_design='field challenge/no-fungicide cohort indicated by source code; not verified untreated natural epidemic',
                         sampling_scope='infected-leaf sample; values are conditional severity, not population incidence',
                         occurrence_identifiable='conditional_sample_only',date_source='raw sheets have no calendar Date; source dates not invented',stage_from=None,stage_to=None)
                        assessment(meta,None,float(g[column].mean()),keys,len(g))
                    # Incidence appears once per scanned leaf; one plot/rank endpoint prevents pseudo-replication.
                    unique=g.incidence.dropna().unique();assert len(unique)==1
                    key=observation('durum-mixtures-2020',file+'!'+member,'incidence_plot_metadata',';'.join(str(i+2) for i in g.index),'incidence',unique[0],
                        dict(plot_id=plot,site_id='tunis_field_station',country='TUNISIA',season_year=year,time_basis='late_season_endpoint',cultivar=composition,
                             pathogen='Zymoseptoria tritici',treatment=str(treatment),replicate=str(rep),organ=suffix,metric='plot_incidence_fraction',unit='fraction'),
                             role='plot_metadata_repeated_in_original_leaf_table')
                    incidence_meta={**meta,'endpoint_series':plot+'|'+suffix+'|incidence','metric':'plot_incidence_fraction','unit':'fraction',
                                    'sampling_scope':'plot incidence denominator is not provided by the scanned-leaf row count',
                                    'occurrence_identifiable':'endpoint_plot_incidence_only'}
                    assessment(incidence_meta,None,float(unique[0]),[key],n_numeric=None)
    INVENTORY.append(dict(dataset_id='durum-mixtures-2020',source_file=file,source_design='field challenge/no-fungicide versus protected yield cohorts; not untreated natural onset data',
                         source_rows=3004,usable_numeric_rows=3004,physical_trial_seasons='42 plot-seasons in 2018; 40 scanned plot-seasons in 2019',
                         known_sowing=False,calendar_dates='not encoded in raw severity workbooks',occurrence_target='late plot incidence; no natural-season absence or onset',
                         onset_target='unavailable; one date per leaf layer, infected-leaf sampling',final_target='absolute PLACL/pycnidia only; relative sheets excluded',
                         limitations='2019 weather workbook carries 2017–2018 dates; no automatic year replacement; no-fungicide challenge is distinct from untreated natural field'))


def nordic():
    for crop,filename in [('winter','Winter_Wheat_sasdataset_extr.xls'),('spring','Spring_Wheat_sasdataset_extr.xls')]:
        p=watch(PUBLIC/'nordic-baltic-2012-2016'/filename);f=pd.read_excel(p,header=5)
        INVENTORY.append(dict(dataset_id='nordic-baltic-2012-2016-'+crop,source_file=str(p.relative_to(ROOT)),
                         source_design='fungicide management and mixed leaf-blotch outcomes',source_rows=len(f),usable_numeric_rows=0,
                         physical_trial_seasons=f.SpotID.replace('.',np.nan).dropna().nunique(),known_sowing=False,calendar_dates=False,
                         occurrence_target='no wheat-specific Septoria occurrence cells in deposited analysis table',onset_target='no disease assessment dates',
                         final_target='yield and model treatment recommendations; not disease severity',
                         limitations='TrtCode A means untreated; recommendation counts are model outputs; 2022 stage dates are day-of-year templates, not trial dates'))
    for filename in ['Development stage.xlsx','Weather stations.xlsx','Yield.xlsx','Weather data.txt']:
        watch(PUBLIC/'nordic-baltic-2012-2016'/filename)


def adas_regional():
    p=OUT/'new_sources/ADAS_data_pest_data.csv'
    if not p.exists():return
    watch(p);watch(OUT/'new_sources/ADAS_README.md');f=pd.read_csv(p)
    file=str(p.relative_to(ROOT));columns=[c for c in f if 'Zymoseptoria_tritici' in c]
    for idx,row in f.iterrows():
        for column in columns:
            value=pd.to_numeric(row[column],errors='coerce');organ=column.split('_')[0]
            metric='regional_crop_incidence_percent' if column.endswith('Crop_Incidence') else 'regional_mean_disease_severity_percent'
            quality='valid_regional_aggregate' if np.isfinite(value) and 0<=value<=100 else 'future_placeholder' if value==-9999 else 'missing'
            key=observation('adas-uk-regional-survey',file,'pest_data',idx+2,column,row[column],dict(
                record_id=f'{row.Year}|{row.Region}',site_id='UK_REGION:'+row.Region,country='UNITED KINGDOM',season_year=row.Year,
                time_basis='annual_regional_aggregate; exact assessment dates absent',pathogen='Zymoseptoria tritici',
                treatment='survey crops; fungicide management unknown',organ=organ,metric=metric,unit='percent'),quality,
                role='regional_aggregate' if quality=='valid_regional_aggregate' else 'placeholder_or_missing')
            if quality!='valid_regional_aggregate':OBS[-1]['value']=None;continue
            meta=dict(dataset_id='adas-uk-regional-survey',endpoint_series=f'{row.Year}|{row.Region}|{column}',
                physical_unit='regional aggregate; constituent fields unidentified',plot_id=None,site_id='UK_REGION:'+row.Region,
                country='UNITED KINGDOM',season_year=row.Year,cultivar=None,treatment='survey crops; fungicide management unknown',organ=organ,
                leaf_numbering='source leaves 1 and 2; no juvenile numbering mapping inferred',metric=metric,unit='percent',
                sowing_date=None,sowing_date_source=None,trial_design='official ADAS survey aggregate released in SPHERE-PPL forecasting repository',
                sampling_scope='regional mean severity / proportion of surveyed crops affected; field denominators absent',
                occurrence_identifiable='regional_survey_only; untreated status unknown',
                date_source='source Year only; exact assessment date not encoded',stage_from=None,stage_to=None)
            assessment(meta,None,float(value),[key],n_numeric=None)
    valid=f[columns].apply(pd.to_numeric,errors='coerce');count=int(valid.apply(lambda s:s.between(0,100)).sum().sum())
    INVENTORY.append(dict(dataset_id='adas-uk-regional-survey',source_file=file,source_design='natural commercial crop survey; management unknown; regional aggregates',
        source_rows=len(f),usable_numeric_rows=count,physical_trial_seasons='463 region-years; constituent field counts unknown',
        known_sowing=False,calendar_dates='annual years only',occurrence_target='regional crop-incidence percentages including genuine zero cells; not untreated binary seasons',
        onset_target='unavailable; annual aggregate without date sequence',final_target='regional severity on source leaves 1 and 2; separate from plot targets',
        limitations='2026 -9999 placeholders excluded; no exact locations, field identities, denominators or cultivar; historical management is unknown'))


def endpoints():
    f=pd.DataFrame(ASSESS);result=[]
    for (dataset,series),g in f.groupby(['dataset_id','endpoint_series'],dropna=False):
        g=g.copy();g['sort_time']=g.date.where(g.date.notna(),g.time_index.astype(str))
        if g.date.notna().any():g=g.sort_values('date');time_kind='calendar'
        elif g.time_index.notna().any():g=g.sort_values('time_index');time_kind='relative_study_day'
        else:time_kind='undated_endpoint'
        g=g.reset_index(drop=True)
        meta=g.iloc[0].to_dict();valid=g.loc[pd.to_numeric(g.value,errors='coerce').notna()].reset_index(drop=True)
        positive=valid.loc[valid.value.gt(0)];first=positive.iloc[0] if len(positive) else None
        initial=valid.iloc[0] if len(valid) else None;last=valid.iloc[-1] if len(valid) else None
        zeros=valid.loc[valid.value.eq(0)]
        if first is not None:
            before=valid.iloc[:int(first.name)].loc[lambda x:x.value.eq(0)]
            previous=before.iloc[-1] if len(before) else None;censor='interval' if previous is not None else 'left'
        else:previous=zeros.iloc[-1] if len(zeros) else None;censor='right' if len(valid) else 'unobserved'
        if time_kind=='undated_endpoint':censor='date_unavailable_positive_endpoint' if len(positive) else 'date_unavailable_zero_endpoint' if len(valid) else 'unobserved'
        record={k:meta.get(k) for k in ['dataset_id','endpoint_series','physical_unit','plot_id','site_id','country','season_year','cultivar','treatment','organ',
            'leaf_numbering','metric','unit','sowing_date','sowing_date_source','trial_design','sampling_scope','occurrence_identifiable','date_source']}
        record.update(endpoint_id=dataset+'|'+str(series),time_basis=time_kind,
            untreated_natural_field_status='confirmed' if dataset in ('basf-wheat-diseases','orellana-torrejon-2022-field') else
                'survey fungicide management unknown' if dataset=='adas-uk-regional-survey' else
                'source-code field challenge; no-fungicide cohort' if dataset=='durum-mixtures-2020' else 'natural field; fungicide status not verified',
            first_assessment=g.iloc[0].date,last_assessment=g.iloc[-1].date,
            first_numeric_assessment=initial.date if initial is not None else None,
            first_value=initial.value if initial is not None else None,
            first_n_numeric=initial.n_numeric if initial is not None else None,
            first_sourcekeys=initial.sourcekeys if initial is not None else None,
            last_numeric_assessment=last.date if last is not None else None,
            first_time_index=initial.time_index if initial is not None else None,last_time_index=last.time_index if last is not None else None,
            last_zero_before_first_positive=previous.date if previous is not None else None,first_positive=first.date if first is not None else None,
            onset_lower_time_index=previous.time_index if previous is not None else None,onset_upper_time_index=first.time_index if first is not None else None,
            onset_censoring=censor,infection_event_observed=False,exact_onset_observed=False,
            observed_window_occurrence=int(len(positive)>0) if len(valid) else None,full_season_absence_supported=False,
            n_assessment_dates=g.date.nunique(),n_numeric_assessments=len(valid),n_senescent=sum(g.n_senescent),n_missing=sum(g.n_missing),
            final_score=last.value if last is not None else None,final_date=last.date if last is not None else None,
            final_n_numeric=last.n_numeric if last is not None else None,
            final_sourcekeys=last.sourcekeys if last is not None else None,
            first_positive_sourcekeys=first.sourcekeys if first is not None else None,last_zero_sourcekeys=previous.sourcekeys if previous is not None else None,
            sourcekeys=';'.join(g.sourcekeys),final_score_is_season_end=False,
            first_stage_from=meta.get('stage_from'),last_stage_from=last.stage_from if last is not None else None,
            last_stage_to=last.stage_to if last is not None else None,
            source_unit=str(series),onset_domain='first positive score in specified organ/metric observation window; infection event is unobserved',
            denominator_definition='numeric assessed plant leaves; percentage within leaf area' if dataset=='orellana-torrejon-2022-field' else
                'not provided in P%INF source' if dataset=='basf-wheat-diseases' else
                'surveyed crops/fields; numeric sample count unavailable' if dataset=='adas-uk-regional-survey' else
                'leaf area per scanned leaf; equal leaf mean; conditional diseased-leaf sample' if meta.get('occurrence_identifiable')=='conditional_sample_only' and meta.get('unit')=='percent' else
                'conditional scanned leaves; not crop incidence denominator' if meta.get('occurrence_identifiable')=='conditional_sample_only' else
                'original incidence denominator unavailable' if meta.get('metric')=='plot_incidence_fraction' else 'source visual summary, denominator not preserved',
            known_sowing_date=bool(meta.get('sowing_date')),
            last_scheduled_assessment_has_numeric=bool(pd.notna(g.iloc[-1].value)))
        if record['sowing_date'] and record['first_positive']:
            record['delta_sowing_to_onset_upper_days']=(pd.Timestamp(record['first_positive'])-pd.Timestamp(record['sowing_date'])).days
            record['delta_sowing_to_onset_lower_days']=(pd.Timestamp(record['last_zero_before_first_positive'])-pd.Timestamp(record['sowing_date'])).days if record['last_zero_before_first_positive'] else None
        else:record['delta_sowing_to_onset_upper_days']=None;record['delta_sowing_to_onset_lower_days']=None
        result.append(record)
    return f,pd.DataFrame(result)


def canopy_occurrence(assessment_table):
    # Any-organ positives identify window occurrence; organs are never averaged into a severity target.
    rows=[]
    fields=['dataset_id','physical_unit','site_id','season_year','cultivar','treatment']
    for group,g in assessment_table.loc[assessment_table.dataset_id.isin(['basf-wheat-diseases','orellana-torrejon-2022-field'])].groupby(fields,dropna=False):
        times=[]
        for date,q in g.groupby('date'):
            valid=q.loc[q.value.notna()]
            if not len(valid):continue
            times.append(dict(date=date,positive=bool(valid.value.gt(0).any()),n_numeric=int(valid.n_numeric.sum()),
                              organ_categories=';'.join(sorted(valid.organ.unique())),sourcekeys=';'.join(valid.sourcekeys)))
        dates=pd.DataFrame(times).sort_values('date').reset_index(drop=True)
        if not len(dates):continue
        pos=dates.loc[dates.positive];first=pos.iloc[0] if len(pos) else None
        zeros=dates.iloc[:int(first.name)].loc[lambda q:~q.positive] if first is not None else dates.loc[~dates.positive]
        lastzero=zeros.iloc[-1] if len(zeros) else None
        meta=dict(zip(fields,group));sample=g.iloc[0]
        meta.update(source_unit='|'.join(str(s) for s in group),plot_id=sample.plot_id,
            sowing_date=sample.sowing_date,sowing_date_source=sample.sowing_date_source,
            first_assessment=dates.iloc[0].date,last_assessment=dates.iloc[-1].date,
            first_value=int(dates.iloc[0].positive),first_positive=first.date if first is not None else None,
            last_zero_before_first_positive=lastzero.date if lastzero is not None else None,
            onset_censoring='interval' if first is not None and lastzero is not None else 'left' if first is not None else 'right',
            observed_window_occurrence=int(len(pos)>0),full_season_absence_supported=False,
            infection_event_observed=False,metric='any_assessed_organ_positive',unit='binary',
            organ='assessed canopy categories',final_score=None,final_date=None,
            first_positive_sourcekeys=first.sourcekeys if first is not None else None,
            last_zero_sourcekeys=lastzero.sourcekeys if lastzero is not None else None,
            assessment_scope='juvenile L and adult F remain distinct in organ endpoints; any-category occurrence only',
            onset_domain='visible positivity across assessed categories; unsampled crop periods remain unknown')
        rows.append(meta)
    return pd.DataFrame(rows)


def add_coordinates(frame):
    registry=pd.read_csv(watch(ROOT/'data/era5/location_registry.csv')).set_index('location_id')
    frame=frame.copy()
    frame['latitude']=frame.site_id.map(registry.requested_latitude)
    frame['longitude']=frame.site_id.map(registry.requested_longitude)
    frame['coordinate_basis']=frame.site_id.map(registry.coordinate_sources)
    return frame


def main():
    OUT.mkdir(parents=True,exist_ok=True);ANALYSIS.mkdir(parents=True,exist_ok=True)
    basf();french();swiss();tunis();nordic();adas_regional()
    observed=pd.DataFrame(OBS,columns=COLS)
    locations=pd.read_csv(watch(ROOT/'data/era5/location_registry.csv')).set_index('location_id')
    observed['latitude']=observed.latitude.fillna(observed.site_id.map(locations.requested_latitude))
    observed['longitude']=observed.longitude.fillna(observed.site_id.map(locations.requested_longitude))
    for c in COLS:
        observed[c]=pd.to_numeric(observed[c],errors='coerce').astype(float) if c in NUMERIC else observed[c].astype('string')
    assert len(observed.columns)==29 and observed.observation_id.is_unique
    observed.to_parquet(OUT/'observations.parquet',index=False)
    assessment_table,endpoint_table=endpoints()
    assessment_table=add_coordinates(assessment_table);endpoint_table=add_coordinates(endpoint_table)
    canopy=add_coordinates(canopy_occurrence(assessment_table))
    canopy.to_csv(OUT/'canopy_observed_window_occurrence.csv',index=False)
    assessment_table.to_csv(OUT/'assessments.csv',index=False)
    endpoint_table.to_csv(OUT/'season_endpoints.csv',index=False)
    inventory=pd.DataFrame(INVENTORY)
    inventory['observation_schema_cells']=inventory.dataset_id.map(observed.groupby('dataset_id').size()).fillna(0).astype(int)
    inventory['aggregated_assessment_rows']=inventory.dataset_id.map(assessment_table.groupby('dataset_id').size()).fillna(0).astype(int)
    inventory['numeric_assessment_rows']=inventory.dataset_id.map(assessment_table[assessment_table.value.notna()].groupby('dataset_id').size()).fillna(0).astype(int)
    inventory.to_csv(ANALYSIS/'target_inventory.csv',index=False)
    watch(Path(__file__))
    unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in HASHES.items());assert unchanged
    audit={'created_utc':datetime.now(timezone.utc).isoformat(),'schema_columns':COLS,'source_sha256':HASHES,'original_files_preserved':unchanged,
           'rows_29_column_observations':len(observed),'wide_endpoints':len(endpoint_table),
           'no_inferred_infection_dates':True,'no_fitted_model':True,'full_season_negative_labels_inferred':False,
           'endpoint_censoring_counts':endpoint_table.groupby(['dataset_id','onset_censoring']).size().to_dict(),
           'source_dataset_counts':observed.dataset_id.value_counts().to_dict()}
    audit['endpoint_censoring_counts']={str(k):v for k,v in audit['endpoint_censoring_counts'].items()}
    (ANALYSIS/'validation.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,default=lambda x:int(x) if isinstance(x,np.integer) else str(x))+'\n')
    print(pd.DataFrame(INVENTORY)[['dataset_id','source_rows','usable_numeric_rows','occurrence_target']].to_string(index=False))
    print('Observation cells',len(observed),'wide endpoints',len(endpoint_table))


if __name__=='__main__':main()
