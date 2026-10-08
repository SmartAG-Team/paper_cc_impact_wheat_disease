"""Audit and aggregate complete regional periods with true country areas."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/regional_reporting';ANNUAL=ROOT/'analysis/paper_study/regional_v1/annual'
CONTRACT=json.loads((HERE/'aggregation_contract.json').read_text())
CUTOFFS=CONTRACT['exceedance_cutoffs_percent']
STATUS=['complete','missing_calendar','missing_weather','soft_dough_not_reached','unsupported_mean_temperature_above40C']
OFFSETS=['flag_onset5_from_sowing_days','flag_onset5_from_heading_days','upper3_any_leaf_onset5_from_sowing_days','upper3_any_leaf_onset5_from_heading_days','heading_from_sowing_days','softdough_from_sowing_days']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def weighted_metrics(d,weight):
    w=d[weight].to_numpy(float);complete=d.status.eq('complete').to_numpy();y=d.upper3_final_damage_percent.to_numpy(float)
    total=float(w.sum());valid=float(w[complete].sum());numerator=float(np.sum(w[complete]*y[complete]));missing=total-valid
    out=dict(expected_weighted_exposure=total,valid_weighted_exposure=valid,valid_area_year_fraction=valid/total if total else np.nan,
        mean_final_damage_percent=numerator/valid if valid else np.nan,
        missing_data_damage_lower_percent=numerator/total if total else np.nan,
        missing_data_damage_upper_percent=(numerator+100*missing)/total if total else np.nan,
        cell_year_records=len(d),completed_cell_year_records=int(complete.sum()))
    for status in STATUS[1:]:out[status+'_weighted_exposure']=float(w[d.status.eq(status)].sum())
    for cutoff in CUTOFFS:
        hit=float(w[complete&(y>=cutoff)].sum())
        out[f'exceedance_{cutoff}pct_fraction']=hit/valid if valid else np.nan
        out[f'exceedance_{cutoff}pct_missing_lower_fraction']=hit/total if total else np.nan
        out[f'exceedance_{cutoff}pct_missing_upper_fraction']=(hit+missing)/total if total else np.nan
    for key in OFFSETS:
        a=d[key].to_numpy(float);keep=complete&np.isfinite(a);den=float(w[keep].sum())
        out['mean_'+key]=float(np.sum(w[keep]*a[keep])/den) if den else np.nan
        out[key+'_valid_weighted_exposure']=den
    return out

def audit_and_derive(path,registry,checks,hashes,forcing_expected):
    year=int(path.stem);receipt_path=path.with_suffix('.json');r=json.loads(receipt_path.read_text())
    def check(name,condition,details=None):
        if not bool(condition):raise AssertionError(str(path)+': '+name)
        checks.append(dict(year=year,check=name,status='passed',details=details))
    filehash=sha(path);check('annual_parquet_checksum',filehash==r['parquet_sha256'])
    hashes[str(path.relative_to(ROOT))]=filehash;hashes[str(receipt_path.relative_to(ROOT))]=sha(receipt_path)
    d=pd.read_parquet(path)
    check('unique_cell_year_and_full_membership',d.cell_id.is_unique and len(d)==len(registry) and set(d.cell_id)==set(registry.cell_id))
    d=d.set_index('cell_id').loc[registry.cell_id].reset_index()
    check('declared_year',d.harvest_year.eq(year).all() and r['harvest_year']==year)
    check('allowed_status',d.status.isin(STATUS).all())
    for key in ['harvested_total_ha','physical_total_ha','latitude','longitude']:
        check('source_registry_'+key,np.array_equal(d[key].to_numpy(),registry[key].to_numpy()))
    complete=d.status.eq('complete');leaves=[f'leaf{x}_final_damage_percent' for x in range(1,8)]
    check('complete_severity_finite_and_bounded',np.isfinite(d.loc[complete,leaves]).all().all() and d.loc[complete,leaves].ge(0).all().all() and d.loc[complete,leaves].le(100+1e-10).all().all())
    check('missing_terminal_severity_not_zero',d.loc[~complete,leaves+['upper3_final_damage_percent']].isna().all().all())
    check('upper3_operator_exact',np.array_equal(d.loc[complete,leaves[:3]].to_numpy().mean(axis=1),d.loc[complete,'upper3_final_damage_percent'].to_numpy()))
    check('complete_endpoint_is_soft_dough',d.loc[complete,'endpoint_date'].notna().all() and d.loc[complete,'endpoint_date'].equals(d.loc[complete,'bbch85_date']))
    check('incomplete_terminal_date_missing',d.loc[~complete,'endpoint_date'].isna().all())
    previous=d.sowing_date
    for stage in [10,31,51,85]:
        event=d[f'bbch{stage}_date'];check('ordered_stage_'+str(stage),(event[complete]>=previous[complete]).all());previous=event
    check('status_counts_match_receipt',d.status.value_counts().to_dict()==r['status_counts'])
    check('total_area_matches_receipt',abs(d.harvested_total_ha.sum()-r['total_harvested_ha'])<1e-7)
    check('valid_area_matches_receipt',abs(d.loc[complete,'harvested_total_ha'].sum()-r['complete_harvested_ha'])<1e-7)
    check('numerical_mass_and_positivity',max(x['mass_error'] for x in r['numerical'])<1e-12 and min(x['minimum_state'] for x in r['numerical'])>=-1e-13)
    for name in [x for x in d.columns if '_onset_' in x and x.endswith('_date')]+['primary_exposure_date']:
        valid=complete&d[name].notna()
        check('onset_inside_crop_'+name,((d.loc[valid,name]>=d.loc[valid,'sowing_date'])&(d.loc[valid,name]<=d.loc[valid,'endpoint_date'])).all())
    for item in r['source_forcing']:
        if item['path'] in forcing_expected:check('consistent_forcing_version_'+item['path'],forcing_expected[item['path']]==item['sha256'])
        forcing_expected[item['path']]=item['sha256']
    for leaf in range(1,8):
        cols=[f'leaf{leaf}_onset_{cutoff:g}pct_date' for cutoff in [.1,1,5]]
        low=d[cols[0]];middle=d[cols[1]];high=d[cols[2]]
        check('nested_leaf_onset_'+str(leaf),((middle.isna()|low.notna())&(high.isna()|middle.notna())&((middle.isna())|(middle>=low))&((high.isna())|(high>=middle))).all())
    # Retain offsets only for valid complete crop scenarios, separately from
    # archived dates in incomplete windows.
    flag=d.leaf1_onset_5pct_date
    upper=d[[f'leaf{i}_onset_5pct_date' for i in range(1,4)]].min(axis=1)
    for prefix,event in [('flag_onset5',flag),('upper3_any_leaf_onset5',upper)]:
        for reference,date in [('sowing',d.sowing_date),('heading',d.bbch51_date)]:
            d[prefix+'_from_'+reference+'_days']=(event-date).dt.total_seconds().div(86400).where(complete)
    d['heading_from_sowing_days']=(d.bbch51_date-d.sowing_date).dt.total_seconds().div(86400).where(complete)
    d['softdough_from_sowing_days']=(d.endpoint_date-d.sowing_date).dt.total_seconds().div(86400).where(complete)
    d['endpoint_calendar_year_offset']=(d.endpoint_date.dt.year-year).where(complete)
    for cutoff in CUTOFFS:d[f'exceeds_{cutoff}pct']=d.upper3_final_damage_percent.ge(cutoff).astype(float).where(complete)
    fields=['cell_id','harvest_year','status','harvested_total_ha','physical_total_ha','sowing_date','endpoint_date','bbch51_date','leaf1_onset_5pct_date','upper3_final_damage_percent','endpoint_calendar_year_offset',*OFFSETS,*[f'exceeds_{c}pct' for c in CUTOFFS]]
    d=d[fields];d['upper3_any_individual_leaf_onset5_date']=upper
    return d,r

def main():
    HERE.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    area=ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet'
    countries_path=ROOT/'data/paper_study/wheat_area/europe_cell_country_wheat_areas.parquet'
    seasonal_path=ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025_seasonal_sensitivity.parquet'
    registry=pd.read_parquet(area);country=pd.read_parquet(countries_path);seasonal=pd.read_parquet(seasonal_path)
    assert not country.duplicated(['cell_id','ADM0_NAME','FIPS0']).any()
    for key in ['harvested_total_ha','physical_total_ha']:
        bycell=country.groupby('cell_id')[key].sum().reindex(registry.cell_id)
        assert np.max(abs(bycell.to_numpy()-registry[key].to_numpy()))<1e-8
    static_hashes={str(p.relative_to(ROOT)):sha(p) for p in [area,countries_path,seasonal_path,HERE/'aggregation_contract.json']}
    inventory=[];processed=[];allchecks=[];input_hashes={};forcing_expected={}
    directories=sorted({p.parent for p in ANNUAL.rglob('*.parquet')})
    for folder in directories:
        parts=folder.relative_to(ANNUAL).parts
        if len(parts)!=5:raise ValueError('Unexpected annual path layout')
        provider,model,scenario,alignment,calendar=parts
        for period,(first,last) in CONTRACT['periods'].items():
            years=list(range(first,last+1));present=[y for y in years if (folder/f'{y}.parquet').exists() and (folder/f'{y}.json').exists()]
            if not present:continue
            entry=dict(provider=provider,model=model,scenario=scenario,alignment=alignment,calendar=calendar,period=period,expected_years=years,present_years=present,complete_period=present==years)
            inventory.append(entry)
            if present!=years:continue
            output=DATA.joinpath(*parts,period);output.mkdir(parents=True,exist_ok=True)
            checks=[];hashes={};records=[];receipts=[]
            for year in years:
                d,r=audit_and_derive(folder/f'{year}.parquet',registry,checks,hashes,forcing_expected);records.append(d);receipts.append(r)
            annual=pd.concat(records,ignore_index=True);annual.to_parquet(output/'annual_cell_metrics.parquet',index=False)
            grouped=annual.groupby('cell_id',sort=False)
            summary=registry.copy();summary['expected_years']=len(years)
            summary['complete_years']=summary.cell_id.map(grouped.status.apply(lambda x:x.eq('complete').sum())).astype(int)
            for status in STATUS[1:]:summary[status+'_years']=summary.cell_id.map(grouped.status.apply(lambda x:x.eq(status).sum())).astype(int)
            for key in ['upper3_final_damage_percent',*OFFSETS,*[f'exceeds_{c}pct' for c in CUTOFFS]]:
                summary['mean_'+key]=summary.cell_id.map(grouped[key].mean())
                summary['n_'+key]=summary.cell_id.map(grouped[key].count()).astype(int)
            for q in [.1,.5,.9]:summary[f'damage_interannual_p{int(q*100)}_percent']=summary.cell_id.map(grouped.upper3_final_damage_percent.quantile(q))
            summary['complete_year_fraction']=summary.complete_years/len(years)
            summary['all_30_years_valid']=summary.complete_years.eq(len(years))
            summary['period'],summary['calendar_scenario'],summary['forcing_model']=period,calendar,model
            summary.to_parquet(output/'cell_period_summary.parquet',index=False);summary.to_csv(output/'cell_period_summary.csv.gz',index=False)
            country_join=annual.drop(columns=['harvested_total_ha','physical_total_ha']).merge(country[['cell_id','ADM0_NAME','FIPS0','harvested_total_ha','physical_total_ha']],on='cell_id',how='left',validate='many_to_many')
            assert len(country_join)==len(country)*len(years)
            annual_country=[];annual_region=[];period_country=[];area_sensitivity=[]
            for basis,key in [('harvested','harvested_total_ha'),('physical','physical_total_ha')]:
                for (name,fips,year),group in country_join.groupby(['ADM0_NAME','FIPS0','harvest_year'],sort=True):
                    annual_country.append(dict(country=name,source_FIPS0=fips,harvest_year=int(year),area_basis=basis,weight_units='ha',**weighted_metrics(group,key)))
                for year,group in annual.groupby('harvest_year',sort=True):annual_region.append(dict(harvest_year=int(year),area_basis=basis,weight_units='ha',**weighted_metrics(group,key)))
                for (name,fips),group in country_join.groupby(['ADM0_NAME','FIPS0'],sort=True):
                    period_country.append(dict(country=name,source_FIPS0=fips,area_basis=basis,weight_units='ha-years',expected_years=len(years),**weighted_metrics(group,key)))
            regional_period=[dict(area_basis=basis,weight_units='ha-years',expected_years=len(years),**weighted_metrics(annual,key)) for basis,key in [('harvested','harvested_total_ha'),('physical','physical_total_ha')]]
            common_ids=set(summary.loc[summary.all_30_years_valid,'cell_id'])
            common=annual.loc[annual.cell_id.isin(common_ids)]
            regional_period.append(dict(area_basis='harvested_common30validcells',weight_units='ha-years',expected_years=len(years),**weighted_metrics(common,'harvested_total_ha')))
            # These are area-weighting subsets under the same winter process,
            # not additive winter/spring genotype predictions.
            subset=annual.drop(columns=['harvested_total_ha','physical_total_ha']).merge(seasonal[['cell_id',*[g+'_harvested_ha' for g in ['autumn_sowing','spring_sowing','ambiguous','unallocated']]]],on='cell_id',validate='many_to_one')
            for group in ['autumn_sowing','spring_sowing','ambiguous','unallocated']:
                area_sensitivity.append(dict(reference_area_partition=group,applied_phenology_scenario=calendar,weight_units='ha-years',**weighted_metrics(subset,group+'_harvested_ha')))
            pd.DataFrame(annual_country).to_csv(output/'country_annual_metrics.csv',index=False)
            pd.DataFrame(period_country).to_csv(output/'country_period_metrics.csv',index=False)
            pd.DataFrame(annual_region).to_csv(output/'europe_annual_metrics.csv',index=False)
            pd.DataFrame(regional_period).to_csv(output/'europe_period_metrics.csv',index=False)
            pd.DataFrame(area_sensitivity).to_csv(output/'area_weighting_sensitivity.csv',index=False)
            status_area=country_join.groupby(['ADM0_NAME','harvest_year','status'])[['harvested_total_ha','physical_total_ha']].sum().reset_index()
            status_area.to_csv(output/'country_status_area_by_year.csv',index=False)
            entry['output_path']=str(output.relative_to(ROOT));entry['annual_archive_hashes']=hashes
            entry['archived_execution_source_missing_years']=[r['harvest_year'] for r in receipts if 'execution_source_sha256' not in r]
            entry['frozen_parameter_versions']=sorted({r['fitted_parameter_sha256'] for r in receipts})
            entry['phenology_versions']=sorted({r['phenology_sha256'] for r in receipts})
            assert len(entry['frozen_parameter_versions'])==1 and len(entry['phenology_versions'])==1
            entry['checks_passed']=len(checks);processed.append(entry);allchecks.extend(checks);input_hashes.update(hashes)
            print(json.dumps(dict(period=period,provider=provider,model=model,scenario=scenario,calendar=calendar,checks=len(checks),rows=len(annual))),flush=True)
    # Full source-chain byte verification is distinct from sampled numerical replay.
    for name,expected in sorted(forcing_expected.items()):
        actual=sha(ROOT/name)
        if actual!=expected:raise ValueError('Forcing source hash mismatch: '+name)
    for name,value in static_hashes.items():assert sha(ROOT/name)==value
    pd.DataFrame([{k:v for k,v in x.items() if k not in ['annual_archive_hashes']} for x in inventory]).to_json(HERE/'period_inventory.json',orient='records',indent=2)
    receipt=dict(status='completed_periods_audited',completed_utc=datetime.now(timezone.utc).isoformat(),completed_periods=processed,
        partial_periods=[x for x in inventory if not x['complete_period']],checks_passed=len(allchecks),checks=allchecks,
        static_source_hashes=static_hashes,annual_input_hashes=input_hashes,forcing_hashes_verified=forcing_expected,
        country_weighting='Source country-partition hectares within cells; no dominant-country whole-cell assignment',
        higher_cutoff_onset_dates_reconstructed=False,missing_treated_as_zero=False,field_parameters_refit=False)
    (HERE/'aggregation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],complete_periods=len(processed),partial_periods=len(receipt['partial_periods']),checks=len(allchecks),forcing_files_verified=len(forcing_expected))),flush=True)

if __name__=='__main__':main()
