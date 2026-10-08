#!/usr/bin/env python3
"""Source audit and external targets for public Corteva wheat observations.

No model is imported or fitted. Disease values remain separated from BASF
development tables. Original row identities survive repeated identical records.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/paper_study/observations'
HERE=Path(__file__).resolve().parent
SOURCE=OUT/'new_sources/corteva-2014-2018/corteva-wheat-diseases.txt'
DATASET='corteva-2014-2018-external'
DOI='https://doi.org/10.5281/zenodo.6352615'
COLS=['observation_id','dataset_id','source_file','source_table','source_row','source_column','record_id','plot_id','site_id',
      'country','latitude','longitude','season_year','date','time_utc','time_basis','days_post_inoculation','cultivar','pathogen','isolate',
      'treatment','replicate','organ','metric','unit','value','raw_value','quality_status','measurement_role']
NUMERIC={'latitude','longitude','season_year','days_post_inoculation','value'}


def location_id(lat,lon):
    return 'era5loc_'+hashlib.sha256(f'{lat:.6f},{lon:.6f}'.encode()).hexdigest()[:12]


def make_endpoint(g,organ,metric='infection_percent_unspecified_basis'):
    g=g.sort_values('date').reset_index(drop=True);valid=g.loc[g.value.notna()].reset_index(drop=True)
    pos=valid.loc[valid.value.gt(0)];first=pos.iloc[0] if len(pos) else None
    zeros=valid.iloc[:int(first.name)].loc[lambda x:x.value.eq(0)] if first is not None else valid.loc[valid.value.eq(0)]
    previous=zeros.iloc[-1] if len(zeros) else None
    initial=valid.iloc[0] if len(valid) else None;final=valid.iloc[-1] if len(valid) else None;s=g.iloc[0]
    return dict(dataset_id=DATASET,endpoint_id=f'{DATASET}|{s.source_unit}|{organ}',endpoint_series=f'{s.source_unit}|{organ}',
        source_unit=s.source_unit,physical_unit=s.source_unit,plot_id=None,cultivar=None,treatment='Untreated',
        site_id=s.location_id,location_id=s.location_id,latitude=s.latitude,longitude=s.longitude,country=s.country,season_year=int(s.season_year),
        coordinate_source='published rounded coordinates to 0.1 degree; '+DOI,
        sowing_date=None,known_sowing_date=False,sowing_date_source=None,season_start_date=None,
        organ=organ,leaf_numbering='source SamplingUnit; numerical leaf rank not equated to juvenile sequence',metric=metric,unit='percent',
        time_basis='calendar',first_assessment=g.iloc[0].date,last_assessment=g.iloc[-1].date,
        first_numeric_assessment=initial.date if initial is not None else None,first_value=initial.value if initial is not None else None,
        first_sourcekeys=initial.sourcekeys if initial is not None else None,
        first_positive=first.date if first is not None else None,last_zero_before_first_positive=previous.date if previous is not None else None,
        onset_censoring='interval' if first is not None and previous is not None else 'left' if first is not None else 'right' if len(valid) else 'unobserved',
        onset_domain='first positive percentage in specified source organ; infection event is unobserved',
        onset_lower_bound_exclusive=True,observed_window_occurrence=int(len(pos)>0) if len(valid) else None,
        full_season_absence_supported=False,infection_event_observed=False,exact_onset_observed=False,
        final_date=final.date if final is not None else None,last_numeric_assessment=final.date if final is not None else None,
        final_score=final.value if final is not None else None,final_n_numeric=final.n_source_numeric_records if final is not None else None,
        final_sourcekeys=final.sourcekeys if final is not None else None,first_positive_sourcekeys=first.sourcekeys if first is not None else None,
        last_zero_sourcekeys=previous.sourcekeys if previous is not None else None,sourcekeys=';'.join(g.sourcekeys),
        final_score_is_season_end=False,n_assessment_dates=len(g),n_numeric_assessments=len(valid),
        excluded_assessment_count=int((~g.assessment_eligible).sum()),n_senescent=0,n_missing=0,
        first_stage_from=initial.stage_from if initial is not None else None,last_stage_from=final.stage_from if final is not None else None,
        last_stage_to=final.stage_to if final is not None else None,
        denominator_definition='source numeric record count; biological replicate/plant count unavailable; INFECT percentage denominator unspecified',
        sampling_scope='equal raw-record mean within field×date×source organ; unidentified replicate/control multiplicity retained',
        untreated_spray_basis='Treatment=Untreated AND NumSprays=0; nonempty Applications is not interpreted as actual application',
        seed_treatment_status='unreported',inoculation_status='unreported; no challenge status inferred',
        external_status='prospective external evaluation; source quality audit before frozen-model evaluation',
        measurement_role='derived_untreated_field_organ_record_mean',source_file=str(SOURCE.relative_to(ROOT)))


def main():
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest();md5=hashlib.md5(SOURCE.read_bytes()).hexdigest()
    meta=json.loads((OUT/'new_sources/zenodo6352615_metadata.json').read_text())
    expected=next(q['checksum'] for q in meta['files'] if q['key']==SOURCE.name)
    assert expected=='md5:'+md5
    raw=pd.read_csv(SOURCE,sep='\t');raw['source_row']=np.arange(2,len(raw)+2)
    raw['calendar_year']=raw.Date.str[:4].astype(int)
    raw['field_label_year']=raw.Field.str.extract(r'^(\d{4})-')[0].astype(int)
    raw['season_year']=raw.field_label_year
    assert pd.to_datetime(raw.Date,errors='coerce').notna().all()
    assert raw.groupby('Field')[['Latitude','Longitude','Country','season_year']].nunique().max().eq(1).all()
    all_sept=raw.loc[raw.Treatment.eq('Untreated') & raw.PestCode.eq('SEPTTR')].copy()
    candidate=all_sept.loc[all_sept.EvaluationType.eq('INFECT')].copy()
    assert candidate.NumSprays.eq(0).all()
    candidate['in_percent_range']=candidate.Value.between(0,100)
    grouping=['Field','SamplingUnit','Date']
    group_eligible=candidate.groupby(grouping).in_percent_range.all()
    bad_groups=group_eligible.loc[~group_eligible].index
    candidate['assessment_eligible']=(~pd.MultiIndex.from_frame(candidate[grouping]).isin(bad_groups)) & candidate.calendar_year.eq(candidate.field_label_year)
    candidate.to_csv(OUT/'corteva_external_raw_INFECT_audit.csv',index=False)
    quality_map=candidate.set_index('source_row').assessment_eligible.to_dict()
    observations=[]
    for row in all_sept.itertuples():
        quality='unresolved_source_metric_excluded_from_percent_target'
        if row.EvaluationType=='INFECT':
            quality='valid_percent_assessment' if quality_map[row.source_row] else 'assessment_excluded_percent_range_or_calendar_year_discordance'
        oid=f'{DATASET}|{SOURCE.name}|{row.source_row}|Value'
        item={c:None for c in COLS};item.update(observation_id=oid,dataset_id=DATASET,source_file=str(SOURCE.relative_to(ROOT)),
            source_table='wheat_trials',source_row=str(row.source_row),source_column='Value',record_id=row.Field,
            site_id=location_id(row.Latitude,row.Longitude),country=row.Country,latitude=row.Latitude,longitude=row.Longitude,
            season_year=row.season_year,date=row.Date,time_basis='calendar',pathogen='Zymoseptoria tritici',treatment='Untreated',
            organ=row.SamplingUnit,metric='infection_percent_unspecified_basis' if row.EvaluationType=='INFECT' else 'source_'+row.EvaluationType,
            unit='percent' if row.EvaluationType=='INFECT' else 'unresolved_source_unit',value=float(row.Value),raw_value=str(row.Value),
            quality_status=quality,measurement_role='observed_external_untreated_field_record')
        observations.append(item)
    obs=pd.DataFrame(observations,columns=COLS)
    for c in COLS:obs[c]=pd.to_numeric(obs[c],errors='coerce').astype(float) if c in NUMERIC else obs[c].astype('string')
    assert obs.observation_id.is_unique
    obs.to_parquet(OUT/'corteva_external_observations.parquet',index=False)
    # Only exact same-field, same-date crop stage records are attached. No stage/date interpolation.
    crop=raw.loc[raw.Treatment.eq('Untreated') & raw.PestCode.isna() & raw.GsFrom.notna()]
    stages={key:dict(stage_from=float(g.GsFrom.min()),stage_to=float(g.GsTo.max()),
                    stage_sourcekeys=';'.join(f'{SOURCE.name}|{r}|GsFrom/GsTo' for r in g.source_row),
                    stage_unique_values=int(g.GsFrom.nunique())) for key,g in crop.groupby(['Field','Date'])}
    assessments=[]
    for (field,organ,date),g in candidate.groupby(grouping,dropna=False):
        s=g.iloc[0];eligible=bool(g.assessment_eligible.all())
        assessment=dict(dataset_id=DATASET,source_unit=field,physical_unit=field,plot_id=None,cultivar=None,
            treatment='Untreated',date=date,organ=organ,metric='infection_percent_unspecified_basis',unit='percent',
            season_year=int(s.season_year),country=s.Country,latitude=s.Latitude,longitude=s.Longitude,
            location_id=location_id(s.Latitude,s.Longitude),site_id=location_id(s.Latitude,s.Longitude),
            value=float(g.Value.mean()) if eligible else None,assessment_eligible=eligible,
            n_source_numeric_records=len(g),n_source_exact_duplicate_rows=int(g.drop(columns=['source_row']).duplicated().sum()),
            sourcekeys=';'.join(f'{DATASET}|{SOURCE.name}|{r}|Value' for r in g.source_row),
            treatment_code_values=';'.join(sorted(g.TreatmentCode.dropna().unique())),
            applications_values=';'.join(sorted(g.Applications.dropna().unique())),
            daft_values=';'.join(str(x) for x in sorted(g.Daft.unique())),
            stage_from=None,stage_to=None,stage_sourcekeys=None,stage_unique_values=0,
            calendar_year=int(s.calendar_year),field_label_year=int(s.field_label_year),
            calendar_year_consistent_with_field_label=bool(g.calendar_year.eq(g.field_label_year).all()),
            stage_basis='same field and exact same date, Untreated crop record; disease rows have no stage',
            denominator_definition='numeric source records, not identified biological replicates',
            source_file=str(SOURCE.relative_to(ROOT)),source_table='wheat_trials',
            measurement_role='derived_untreated_field_organ_record_mean',
            external_status='prospective external evaluation; no model fitted in source audit')
        assessment.update(stages.get((field,date),{}));assessments.append(assessment)
    a=pd.DataFrame(assessments).sort_values(['source_unit','organ','date'])
    a.to_csv(OUT/'corteva_external_assessments.csv',index=False)
    endpoint=pd.DataFrame([make_endpoint(g,organ) for (_,organ),g in a.groupby(['source_unit','organ'])])
    endpoint.to_csv(OUT/'corteva_external_season_endpoints.csv',index=False)
    canopy=[]
    for field,g in a.groupby('source_unit'):
        dates=[]
        for date,q in g.groupby('date'):
            valid=q.loc[q.assessment_eligible & q.value.notna()]
            if not len(valid):continue
            s=valid.iloc[0].copy();s['value']=int(valid.value.gt(0).any());s['sourcekeys']=';'.join(valid.sourcekeys)
            s['n_source_numeric_records']=int(valid.n_source_numeric_records.sum());dates.append(s)
        if not dates:continue
        e=make_endpoint(pd.DataFrame(dates),'any_assessed_organ_positive','any_assessed_organ_positive')
        e.update(unit='binary',final_score=None,final_date=None,final_n_numeric=None,final_sourcekeys=None,
                 measurement_role='any_category_occurrence_only; no mixed-organ severity',
                 sampling_scope='any valid observed organ positive; assessed organ coverage changes with date')
        canopy.append(e)
    canopy=pd.DataFrame(canopy);canopy.to_csv(OUT/'corteva_external_occurrence.csv',index=False)
    metadata=raw.groupby('Field').agg(country=('Country','first'),latitude=('Latitude','first'),longitude=('Longitude','first'),
        season_year=('season_year','first'),first_source_date=('Date','min'),last_source_date=('Date','max'),n_source_rows=('Field','size')).reset_index().rename(columns={'Field':'source_unit'})
    fields=set(a.loc[a.assessment_eligible].source_unit)
    metadata['dataset_id']=DATASET;metadata['location_id']=[location_id(lat,lon) for lat,lon in zip(metadata.latitude,metadata.longitude)]
    metadata['eligible_untreated_septoria_percent']=metadata.source_unit.isin(fields)
    metadata['coordinate_source']='published rounded to 0.1 degree; '+DOI
    metadata['sowing_date']=None;metadata['sowing_known']=False;metadata['cultivar']=None
    metadata['season_year_basis']='explicit year prefix of source Field identifier; discordant source dates flagged and not shifted'
    mismatches=raw.loc[raw.calendar_year.ne(raw.field_label_year)]
    metadata['n_calendar_year_discordant_source_rows']=metadata.source_unit.map(mismatches.groupby('Field').size()).fillna(0).astype(int)
    metadata['untreated_seed_treatment_status']='unreported';metadata['inoculation_status']='unreported'
    metadata['external_status']='prospective external evaluation; data-quality audit precedes fitting'
    metadata.to_csv(OUT/'corteva_source_trial_metadata.csv',index=False)
    metadata.loc[metadata.eligible_untreated_septoria_percent].to_csv(OUT/'corteva_weather_requests.csv',index=False)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==digest
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_url=DOI,source_file=str(SOURCE.relative_to(ROOT)),
        source_sha256=digest,preparation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),repository_md5=md5,repository_checksum_verified=True,source_preserved=True,
        source_title_trial_count=168,distinct_source_Field_ids=len(metadata),source_trial_count_discrepancy_retained=True,
        source_rows=len(raw),untreated_SEPTTR_rows=len(all_sept),untreated_INFECT_rows=len(candidate),
        out_of_percent_range_records=int((~candidate.in_percent_range).sum()),
        excluded_out_of_percent_range_assessments=len(bad_groups),excluded_entire_assessments=int((~a.assessment_eligible).sum()),
        excluded_entire_assessment_records=int((~candidate.assessment_eligible).sum()),
        discordant_calendar_year_source_rows=len(mismatches),discordant_calendar_year_untreated_INFECT_rows=int(candidate.calendar_year.ne(candidate.field_label_year).sum()),
        eligible_percent_records=int(candidate.assessment_eligible.sum()),eligible_assessments=int(a.assessment_eligible.sum()),
        eligible_trial_seasons=len(fields),unique_published_coordinates=metadata.loc[metadata.eligible_untreated_septoria_percent,['latitude','longitude']].drop_duplicates().shape[0],
        organ_endpoints=len(endpoint),canopy_occurrence_seasons=len(canopy),canopy_censoring=canopy.onset_censoring.value_counts().to_dict(),
        no_sowing_or_infection_dates_inferred=True,source_audit_completed_before_model_evaluation=True,no_model_imported_or_fitted=True,
        calibration_partition_modified=False,
        preprocessing_rules=['Untreated and SEPTTR source fields identify candidate pathogen/treatment',
            'INFECT is the published percentage-domain score; other source metrics remain unresolved',
            'Exclude complete field/date/organ assessment when any original score is outside 0–100; preserve raw scores',
            'Exclude disease assessment dates that disagree with explicit source Field year prefix; preserve original dates without replacement',
            'Retain every original row; missing replicate identifiers prevent inferred plots or deduplication',
            'Source rows are averaged per field/date/source organ; counts are source-record counts',
            'Exact same-date untreated crop stage only; no future stage interpolation',
            'Last valid numeric severity is an observation-window endpoint, not necessarily season-end severity'],
        eligibility_limitations=['No sowing dates, cultivars, biological replicate identifiers or seed-treatment details',
            'Inoculation or natural-infection status is not stated in deposited row metadata',
            'Published 0.1-degree coordinates identify weather requests, not exact trial plots',
            'Nonempty Applications fields in untreated records are not interpreted as actual sprays',
            'Exact infection dates and complete-season absence labels are unobserved'])
    (HERE/'corteva_external_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['source_rows','distinct_source_Field_ids','untreated_SEPTTR_rows','eligible_percent_records','eligible_trial_seasons','organ_endpoints','canopy_censoring']},indent=2))


if __name__=='__main__':main()
