#!/usr/bin/env python3
"""Independent read-only reconstruction of disease source targets; no model imports."""
from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/paper_study/observations'
checks=[]


def check(name,condition,detail=None):
    checks.append({'name':name,'passed':bool(condition),'detail':detail})


def reconstruct_endpoint_scores(a,e,idcol='endpoint_series'):
    failures=[]
    for _,r in e.iterrows():
        if idcol=='endpoint_series':q=a[(a.dataset_id==r.dataset_id)&(a.endpoint_series==r.endpoint_series)].copy()
        else:q=a[(a.source_unit==r.source_unit)&(a.organ==r.organ)].copy()
        if not len(q):failures.append([r.endpoint_id,'absent assessments']);continue
        q=q.loc[q.value.notna()]
        if 'assessment_eligible' in q:q=q.loc[q.assessment_eligible]
        if q.date.notna().any():q=q.sort_values('date')
        elif 'time_index' in q and q.time_index.notna().any():q=q.sort_values('time_index')
        if not len(q):
            if pd.notna(r.final_score):failures.append([r.endpoint_id,'unobserved has score'])
            continue
        if not np.isclose(q.iloc[-1].value,r.final_score,atol=1e-10):failures.append([r.endpoint_id,'final score'])
        if not np.isclose(q.iloc[0].value,r.first_value,atol=1e-10):failures.append([r.endpoint_id,'first value'])
        if q.date.notna().all():
            if q.iloc[-1].date!=r.final_date:failures.append([r.endpoint_id,'final date'])
            pos=q.loc[q.value.gt(0)]
            if len(pos):
                date=pos.iloc[0].date
                if date!=r.first_positive:failures.append([r.endpoint_id,'first positive date'])
                zero=q.loc[q.value.eq(0)&q.date.lt(date)]
                if len(zero) and zero.iloc[-1].date!=r.last_zero_before_first_positive:failures.append([r.endpoint_id,'lower bound'])
                if (len(zero)>0)!=(r.onset_censoring=='interval'):failures.append([r.endpoint_id,'censoring'])
            elif r.onset_censoring!='right':failures.append([r.endpoint_id,'all-zero censoring'])
    check('independent_endpoint_reconstruction_'+idcol,len(failures)==0,{'failures':failures[:15],'endpoint_count':len(e)})


def main():
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()}
    tree=ast.parse((ROOT/'analysis/harmonize_septoria.py').read_text())
    columns=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='COLS' for t in n.targets))
    o=pd.read_parquet(OUT/'observations.parquet');external=pd.read_parquet(OUT/'corteva_external_observations.parquet')
    for label,t in [('development_inventory',o),('reserved_corteva',external)]:
        check(label+'_29_column_schema',t.columns.tolist()==columns)
        check(label+'_observation_ids_unique',t.observation_id.is_unique)
    check('Corteva_absent_from_development_observation_table',not o.dataset_id.str.contains('corteva').any())
    a=pd.read_csv(OUT/'assessments.csv');e=pd.read_csv(OUT/'season_endpoints.csv')
    check('endpoints_unique',e.endpoint_id.is_unique)
    check('infection_dates_not_invented',not e.infection_event_observed.any() and not e.exact_onset_observed.any())
    check('no_full_season_negative_inferred',not e.full_season_absence_supported.any())
    # BASF original untreated disease cells, including annotated exclusions.
    source=pd.read_csv(ROOT/'data/basf-wheat-diseases.txt',sep='\t');b=source[(source.Treatment=='Untreated')&(source.Organism=='SEPTTR')]
    ob=o.loc[o.dataset_id=='basf-wheat-diseases'];indices=pd.to_numeric(ob.source_row).astype(int)-2
    check('BASF_all_609_untreated_SEPTTR_preserved',len(ob)==len(b)==609)
    check('BASF_raw_values_preserved',np.allclose(ob.value.to_numpy(),source.iloc[indices].Value.to_numpy()))
    check('BASF_six_crop_injury_annotations_excluded',ob.quality_status.eq('crop_injury_annotation').sum()==6 and len(a[a.dataset_id=='basf-wheat-diseases'])==603)
    cb=pd.read_csv(OUT/'basf_trial_occurrence.csv')
    check('BASF_trial_windows_76_not_leaf_pseudoreplicates',len(cb)==76 and cb.observed_window_occurrence.eq(1).all())
    check('BASF_first_date_any_organ_74_positive',cb.onset_censoring.eq('left').sum()==74)
    # French source identities and values are reconstructed independently by melting the original raw F1.
    f=pd.read_csv(ROOT/'data/public_septoria/orellana-torrejon-2022-field/F1_Field_disease_severity_rawdata.csv',sep=';',comment='#',keep_default_na=False)
    leaves=[c for c in f if c.startswith('%spor_')]
    m=f.melt(id_vars=['id','date','mixture','rep','var_origin'],value_vars=leaves,var_name='source_column',value_name='source_raw')
    m['source_row']=m.id.astype(str);of=o[(o.dataset_id=='orellana-torrejon-2022-field')&(o.source_table=='F1')]
    joined=of.merge(m,on=['source_row','source_column'],how='outer',indicator=True,validate='one_to_one')
    check('French_all_2400_plants_x_11_leaf_categories',len(of)==len(m)==26400 and joined._merge.eq('both').all())
    numeric=pd.to_numeric(joined.source_raw,errors='coerce');check('French_numeric_cells_preserved',np.allclose(joined.value,numeric,equal_nan=True))
    check('French_S_is_senescent_not_zero',of.quality_status.eq('senescent').sum()==2366 and of.loc[of.quality_status=='senescent','value'].isna().all())
    check('French_juvenile_and_adult_categories_distinct',set(of.organ)=={'L1','L2','L3','L4','L5','F1','F2','F3','F4','F5','F6'})
    af=a.loc[a.dataset_id=='orellana-torrejon-2022-field'];check('French_F1_only_in_assessments',af.source_table.eq('F1').all() and af.source_file.str.endswith('F1_Field_disease_severity_rawdata.csv').all())
    check('French_physical_plots_not_split_by_mixed_cultivar',af.physical_unit.nunique()==30 and af.plot_id.nunique()==30)
    f['date']=pd.to_datetime(f.date,dayfirst=True).dt.strftime('%Y-%m-%d');f['season_year']=f.date.str[:4].astype(int)
    raw_groups={}
    for row in f.to_dict('records'):
        unit=f'{row["season_year"]}|{row["mixture"]}|{row["rep"]}|{row["var_origin"]}'
        for col in leaves:raw_groups.setdefault((unit+'|'+col[6:],row['date']),[]).append(row[col])
    disagreements=[]
    for r in af.itertuples():
        vals=pd.to_numeric(pd.Series(raw_groups[(r.endpoint_series,r.date)]),errors='coerce').dropna()
        expected=vals.mean() if len(vals) else np.nan
        if not np.isclose(r.value,expected,equal_nan=True,atol=1e-10) or len(vals)!=r.n_numeric:disagreements.append([r.endpoint_series,r.date])
    check('French_raw_numeric_only_group_means_and_denominators',not disagreements,disagreements[:12])
    check('French_senescence_count_retained_in_assessments',af.n_senescent.sum()==2366)
    summaries=o[(o.dataset_id=='orellana-torrejon-2022-field')&o.source_table.isin(['F2','F3'])]
    summary_n=sum(len(pd.read_csv(ROOT/f'data/public_septoria/orellana-torrejon-2022-field/F{number}_t2_{year}_field-severity_means.txt',sep='\t',comment='#')) for year,number in [(2018,2),(2019,3)])
    check('French_author_summary_separate_role',len(summaries)==summary_n==66 and summaries.measurement_role.eq('author_summary_not_extra_independent_observation').all())
    # Assumed Swiss AUDPC anchors never enter observation onset bounds.
    sw=o[o.dataset_id=='karisto-2018-field'];assumed=sw.quality_status.eq('assumed_zero_not_observed')
    check('Swiss_335_assumed_Day0_not_observed_zero',assumed.sum()==335 and sw.loc[assumed,'measurement_role'].eq('assumed_AUDPC_anchor').all())
    check('Swiss_observed_visual_first_relative_day14',a.loc[(a.dataset_id=='karisto-2018-field')&a.organ.eq('whole_cultivar_visual_summary')].time_index.min()==14)
    region=o[o.dataset_id=='adas-uk-regional-survey'];check('ADAS_future_placeholders_excluded',region.quality_status.eq('future_placeholder').sum()==36 and region.loc[region.quality_status=='future_placeholder','value'].isna().all())
    check('ADAS_1816_historical_numeric_cells',region.quality_status.eq('valid_regional_aggregate').sum()==1816)
    check('ADAS_84_true_zero_metric_cells_preserved',region.value.eq(0).sum()==84)
    reconstruct_endpoint_scores(a,e)
    # External eligibility is reconstructed directly from original source values and year identifiers.
    raw=pd.read_csv(OUT/'new_sources/corteva-2014-2018/corteva-wheat-diseases.txt',sep='\t');raw['source_row']=np.arange(2,len(raw)+2)
    q=raw[(raw.Treatment=='Untreated')&(raw.PestCode=='SEPTTR')&(raw.EvaluationType=='INFECT')].copy()
    ext=pd.read_csv(OUT/'corteva_external_assessments.csv');end=pd.read_csv(OUT/'corteva_external_season_endpoints.csv')
    check('Corteva_all_9436_untreated_pathogen_rows_preserved',len(external)==9436)
    check('Corteva_source_value_identity',np.allclose(external.value,raw.iloc[pd.to_numeric(external.source_row).astype(int)-2].Value))
    bad=[]
    for key,g in q.groupby(['Field','SamplingUnit','Date']):
        expected=bool(g.Value.between(0,100).all() and g.Date.str[:4].eq(g.Field.str[:4]).all())
        r=ext[(ext.source_unit==key[0])&(ext.organ==key[1])&(ext.date==key[2])].iloc[0]
        if expected!=r.assessment_eligible or (expected and not np.isclose(g.Value.mean(),r.value)):bad.append(key)
    check('Corteva_source_defined_eligibility_and_raw_record_means',not bad,bad[:12])
    check('Corteva_1785_eligible_assessments_and_7789_records',ext.assessment_eligible.sum()==1785 and ext.loc[ext.assessment_eligible,'n_source_numeric_records'].sum()==7789)
    check('Corteva_no_invented_biological_replicates',external.replicate.isna().all() and external.plot_id.isna().all())
    check('Corteva_valid_scores_bounded',ext.loc[ext.assessment_eligible,'value'].between(0,100).all())
    c=pd.read_csv(OUT/'corteva_external_occurrence.csv');check('Corteva_148_windows_and_147_left_censored',len(c)==148 and c.observed_window_occurrence.eq(1).all() and c.onset_censoring.eq('left').sum()==147)
    check('Corteva_no_unsupported_season_absence',not end.full_season_absence_supported.any() and not end.infection_event_observed.any())
    reconstruct_endpoint_scores(ext,end,'source_unit')
    receipt=json.loads((HERE/'validation.json').read_text());check('original_source_hashes_match',all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in receipt['source_sha256'].items()))
    extreceipt=json.loads((HERE/'corteva_external_validation.json').read_text());check('Corteva_archive_exact_hash',hashlib.sha256((ROOT/extreceipt['source_file']).read_bytes()).hexdigest()==extreceipt['source_sha256'])
    check('verification_preserves_all_target_and_source_files',all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in before.items()))
    failed=[q for q in checks if not q['passed']]
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'n_checks':len(checks),'n_failed':len(failed),'passed':not failed,
            'model_imported_or_fitted':False,'independent_source_reconstruction':True,'source_or_target_files_modified':False}
    (HERE/'independent_validation.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({'checks':len(checks),'failed':len(failed),'failures':failed},indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
