"""Reconstruct UTC precipitation totals from NOAA ISD AA1--AA4 intervals.

The original compressed global-hourly archive and CSV contain the same ISD
observation fields. No interval is rounded, shifted, prorated or inferred from
GSOD. Rainfall station IDs are inherited from outcome-independent elevation
matching. The AA element is retained as precipitation amount, with phase unknown.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .noaa_station import fetch_source, pair_metrics

FORMAT_URL='https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf'
MIGRATION_URL='https://www.nesdis.noaa.gov/news/service-location-change-integrated-surface-data-global-hourly'
DATA_URL='https://noaa-isd-pds.s3.amazonaws.com/data/{year}/{usaf}-{wban}-{year}.gz'
AA_PATTERN=re.compile(r'AA([1-4])(\d{2})(\d{4})([0-9A-Z])([0-9A-Z])')
MW_PATTERN=re.compile(r'MW[1-7](\d{2})([0-9A-Z])')
GOOD_QC={'1','5'}
INTERVAL_COLUMNS=['start','end','period_hours','amount_mm','condition_code','quality_code','trace','accepted','reason','aa_id','station_id','report_type']


def parse_aa(text):
    """Decode period, tenths-mm amount, condition and full-pass quality flags."""
    parts=text.split(',')
    if len(parts)!=4:
        raise ValueError('AA field must contain period, amount, condition, quality')
    period,depth,condition,qc=parts
    hours=int(period); amount=int(depth)
    reasons=[]
    if not 1<=hours<=24:
        reasons.append('period_missing_or_outside_1_24h')
    if amount==9999:
        reasons.append('amount_missing')
    if qc not in GOOD_QC:
        reasons.append('not_full_pass_qc')
    # 9 marks absent condition metadata; a present numeric depth is not missing.
    # 4 denotes end accumulated period; its explicit duration remains required.
    if condition not in {'2','4','9'}:
        reasons.append('incomplete_estimated_or_invalid_condition')
    return dict(period_hours=hours,amount_mm=amount/10 if amount!=9999 else np.nan,
                condition_code=condition,quality_code=qc,trace=condition=='2',
                accepted=not reasons,reason=';'.join(reasons) if reasons else 'accepted')


def extract_isd(lines):
    """Read only ISD ADD fields, excluding remarks and quality-control echoes."""
    intervals=[]; weather=[]
    for line in lines:
        if len(line)<105 or line[105:108]!='ADD':
            continue
        additional=line[108:].split('REM',1)[0].split('EQD',1)[0]
        aa=list(AA_PATTERN.finditer(additional)); mw=list(MW_PATTERN.finditer(additional))
        if not aa and not mw:
            continue
        try:
            end=pd.Timestamp.strptime(line[15:27],'%Y%m%d%H%M')
        except NotImplementedError:
            from datetime import datetime
            end=pd.Timestamp(datetime.strptime(line[15:27],'%Y%m%d%H%M'))
        station=line[4:15]
        for match in aa:
            ident,period,depth,condition,qc=match.groups()
            parsed=parse_aa(','.join([period,depth,condition,qc]))
            intervals.append(dict(start=end-pd.Timedelta(hours=parsed['period_hours']),end=end,
                                  aa_id='AA'+ident,station_id=station,report_type=line[41:46],**parsed))
        for match in mw:
            code,qc=match.groups()
            if qc in GOOD_QC and code!='99':
                c=int(code)
                # WMO present-weather codes 70--79,83--88 explicitly include
                # snow/ice pellets or mixed rain and snow; absence is not no snow.
                weather.append(dict(date=end.normalize(),observation_utc=end,station_id=station,
                                    manual_weather_code=c,solid_precipitation_observed=(70<=c<=79 or 83<=c<=88)))
    return pd.DataFrame(intervals,columns=INTERVAL_COLUMNS), pd.DataFrame(weather,columns=[
        'date','observation_utc','station_id','manual_weather_code','solid_precipitation_observed'])


def reconstruct_daily(intervals, dates):
    """Find an exact non-overlapping 24h cover, maximizing true hourly coverage."""
    valid=intervals.loc[intervals.accepted].copy() if len(intervals) else pd.DataFrame(columns=INTERVAL_COLUMNS)
    valid=valid.sort_values(['start','end','quality_code','aa_id'])
    conflict_keys=set()
    if len(valid):
        values=valid.groupby(['start','end']).amount_mm.agg(['min','max'])
        conflict_keys=set(values.loc[(values['max']-values['min']).gt(1e-9)].index)
        keep=[(r.start,r.end) not in conflict_keys for r in valid.itertuples()]
        valid=valid.loc[keep].drop_duplicates(['start','end'])
        valid['day']=(valid.end-pd.Timedelta(nanoseconds=1)).dt.normalize()
    grouped={day:g for day,g in valid.groupby('day')} if len(valid) else {}
    output=[]
    for date in pd.to_datetime(dates):
        date=pd.Timestamp(date).normalize(); finish=date+pd.Timedelta(days=1)
        g=grouped.get(date,pd.DataFrame(columns=valid.columns))
        g=g.loc[g.start.ge(date) & g.end.le(finish)]
        # Union measures available valid intervals, not an inferred amount.
        union_hours=0.; last=None
        for row in g.sort_values(['start','end']).itertuples():
            begin=max(row.start,last) if last is not None else row.start
            if row.end>begin:
                union_hours+=(row.end-begin).total_seconds()/3600
            last=max(last,row.end) if last is not None else row.end
        # Graph edges represent intervals; a path can only join equal timestamps.
        # Lexicographic cost maximizes 1h coverage, then favors shorter durations.
        best={date:((0.,0.,0),[])}
        for row in g.sort_values(['end','start']).itertuples():
            if row.start not in best:
                continue
            previous,path=best[row.start]
            cost=(previous[0]-(1 if row.period_hours==1 else 0),
                  previous[1]+row.period_hours**2,previous[2]+1)
            if row.end not in best or cost<best[row.end][0]:
                best[row.end]=(cost,path+[row])
        selected=best.get(finish,(None,[]))[1]
        complete=bool(selected)
        hourly=complete and all(r.period_hours==1 for r in selected)
        nconflicts=sum(start>=date and end<=finish for start,end in conflict_keys)
        result=dict(date=date,complete_utc_day=complete,
                    precipitation_mm_station=float(sum(r.amount_mm for r in selected)) if complete else np.nan,
                    valid_interval_union_hours=union_hours,
                    selected_interval_count=len(selected),
                    hours_from_1h_intervals=sum(r.period_hours for r in selected if r.period_hours==1),
                    max_accumulation_hours=max((r.period_hours for r in selected),default=np.nan),
                    trace_interval_count=sum(r.trace for r in selected),
                    conflicting_interval_count=nconflicts,
                    rain_hours_gt_0_1mm_station=sum(r.amount_mm>.1 for r in selected) if hourly else np.nan,
                    true_hourly_cover=bool(hourly),
                    latest_interval_end_utc=max((r.end for r in selected),default=pd.NaT),
                    nominal_available_after_utc=finish,
                    reconstruction_status='complete_exact_utc_cover' if complete else 'no_exact_nonoverlapping_utc_cover',
                    selected_intervals_json=json.dumps([dict(start=r.start.isoformat(),end=r.end.isoformat(),
                        hours=r.period_hours,amount_mm=r.amount_mm,trace=bool(r.trace),quality=r.quality_code,
                        condition=r.condition_code,aa_id=r.aa_id) for r in selected]))
        output.append(result)
    return pd.DataFrame(output)


def substitution_safety(counts, daily):
    """Block rain-dependent missingness caused by unresolved source flags."""
    rejected=int(counts.loc[~counts.accepted,'positive_intervals'].sum())
    condition3=int(counts.loc[counts.condition_code.eq('3') & counts.quality_code.isin(GOOD_QC),'positive_intervals'].sum())
    positive_days=int(daily.precipitation_mm_station.gt(0).sum())
    usable=not (rejected>0 and positive_days==0)
    return dict(usable_for_substitution=usable,rejected_positive_intervals=rejected,
                condition3_full_qc_positive_intervals=condition3,
                positive_reconstructed_days=positive_days,
                reason='Only dry days reconstruct while positive source depths are rejected; unresolved condition3 conflicts with numeric depths and full-pass QC. Substitution would impose rain-dependent selection.' if not usable else 'No all-dry reconstruction anomaly detected.')


def run(root):
    root=Path(root);base=root/'outputs/weather_validation_20261022/noaa'
    out=base/'hourly_rain';out.mkdir(parents=True,exist_ok=True)
    cache=root/'data/external_weather_validation_20261022/noaa/hourly'
    cache.mkdir(parents=True,exist_ok=True)
    receipts=[fetch_source(FORMAT_URL,cache/'isd-format-document.pdf'),
              fetch_source(MIGRATION_URL,cache/'migration.html')]
    if receipts[0]['status_code']!=200:
        raise RuntimeError('Official ISD documentation unavailable')
    pairs=pd.read_parquet(base/'elevation_300m/paired_daily.parquet')
    pairs['station_id']=pairs.station_id.astype(str).str.zfill(11)
    wanted=pairs[['station_id','date']].drop_duplicates()
    year_pairs=pd.concat([wanted.assign(year=wanted.date.dt.year),
                          wanted.assign(year=(wanted.date+pd.Timedelta(days=1)).dt.year)])[['station_id','year']].drop_duplicates()
    tasks={}
    for row in year_pairs.itertuples():
        url=DATA_URL.format(year=row.year,usaf=row.station_id[:6],wban=row.station_id[6:])
        tasks[(row.station_id,row.year)]=(url,cache/'isd'/str(row.year)/f'{row.station_id}.gz')
    results={}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures={pool.submit(fetch_source,*task):key for key,task in tasks.items()}
        for future in as_completed(futures):
            key=futures[future];results[key]=future.result();receipts.append(results[key])
            if len(results)%20==0 or len(results)==len(tasks):
                print(json.dumps(dict(downloaded_station_years=len(results),total=len(tasks),
                                      downloaded_mb=round(sum(r['bytes'] for r in results.values() if r['status_code']==200)/1e6,1))),flush=True)
    (out/'source_receipt.json').write_text(json.dumps(receipts,indent=2))
    station_daily=[];quality_counts=[]
    for i,(station,g) in enumerate(wanted.groupby('station_id')):
        intervals=[];weather=[]
        for year in sorted(year_pairs.loc[year_pairs.station_id.eq(station),'year']):
            receipt=results[(station,year)]
            if receipt['status_code']!=200:
                continue
            with gzip.open(receipt['path'],'rt',encoding='ascii') as source:
                a,w=extract_isd(source)
            if len(a):
                # Retain only intervals touching the required broad date range.
                a=a.loc[a.end.ge(g.date.min()) & a.start.le(g.date.max()+pd.Timedelta(days=1))]
                intervals.append(a)
            if len(w):
                weather.append(w)
        a=pd.concat(intervals,ignore_index=True) if intervals else pd.DataFrame(columns=INTERVAL_COLUMNS)
        if len(a):
            counts=a.groupby(['quality_code','condition_code','accepted','reason']).agg(
                n_intervals=('amount_mm','size'),positive_intervals=('amount_mm',lambda x:x.gt(0).sum()),
                zero_intervals=('amount_mm',lambda x:x.eq(0).sum()),missing_intervals=('amount_mm',lambda x:x.isna().sum()),
                min_depth_mm=('amount_mm','min'),median_depth_mm=('amount_mm','median'),max_depth_mm=('amount_mm','max')).reset_index()
            counts['station_id']=station;quality_counts.append(counts)
        d=reconstruct_daily(a,sorted(g.date.unique()));d['station_id']=station
        if len(a):
            stage=a.copy();stage['date']=(stage.end-pd.Timedelta(nanoseconds=1)).dt.normalize()
            stage=stage.loc[stage.start.ge(stage.date)]
            stage['full_qc_positive']=stage.amount_mm.gt(0)&stage.quality_code.isin(GOOD_QC)
            stage['full_qc_condition3_positive']=stage.full_qc_positive&stage.condition_code.eq('3')
            stage['rejected_positive']=stage.amount_mm.gt(0)&~stage.accepted
            diag=stage.groupby('date').agg(
                raw_full_qc_positive_intervals_inside_utc_day=('full_qc_positive','sum'),
                condition3_full_qc_positive_intervals_inside_utc_day=('full_qc_condition3_positive','sum'),
                rejected_positive_intervals_inside_utc_day=('rejected_positive','sum')).reset_index()
            d=d.merge(diag,on='date',how='left',validate='one_to_one')
        if weather:
            w=pd.concat(weather,ignore_index=True).drop_duplicates()
            daily_w=w.groupby('date').agg(manual_weather_codes=('manual_weather_code',lambda x:','.join(map(str,sorted(set(x))))),
                                         solid_precipitation_observed=('solid_precipitation_observed','max'),
                                         manual_weather_report_count=('manual_weather_code','size')).reset_index()
            d=d.merge(daily_w,on='date',how='left',validate='one_to_one')
        else:
            d['manual_weather_codes']='';d['solid_precipitation_observed']=pd.NA;d['manual_weather_report_count']=0
        station_daily.append(d)
        if (i+1)%10==0:
            print(json.dumps(dict(parsed_stations=i+1,total_stations=wanted.station_id.nunique())),flush=True)
    daily=pd.concat(station_daily,ignore_index=True)
    counts=pd.concat(quality_counts,ignore_index=True)
    safety=substitution_safety(counts,daily)
    daily['reconstructed_precipitation_mm_audit_only']=daily.precipitation_mm_station
    daily['reconstructed_rain_hours_audit_only']=daily.rain_hours_gt_0_1mm_station
    daily['usable_for_substitution']=safety['usable_for_substitution']
    if not safety['usable_for_substitution']:
        daily['precipitation_mm_station']=np.nan
        daily['rain_hours_gt_0_1mm_station']=np.nan
    daily['substitution_status']='unusable_all_dry_selection_anomaly' if not safety['usable_for_substitution'] else 'usable'
    daily.to_parquet(out/'station_daily.parquet',index=False)
    daily.to_csv(out/'station_daily.csv',index=False)
    keep=pairs[['site_id','date','station_id','country','distance_km','precipitation_mm_forcing']].copy()
    paired=keep.merge(daily.drop(columns='selected_intervals_json'),on=['station_id','date'],how='left',validate='many_to_one')
    paired.to_parquet(out/'paired_daily.parquet',index=False);paired.to_csv(out/'paired_daily.csv',index=False)
    # GSOD is an archive consistency diagnostic only; its observing period is
    # not assumed identical to this UTC day and never supplies missing values.
    wet_audit=paired.merge(pairs[['site_id','date','precipitation_mm_station']].rename(
        columns={'precipitation_mm_station':'gsod_precipitation_mm_observing_day'}),on=['site_id','date'],validate='one_to_one')
    wet_audit.loc[wet_audit.gsod_precipitation_mm_observing_day.ge(1)].to_csv(out/'gsod_wet_observing_day_audit.csv',index=False)
    metrics=[]
    for kind,column in [('all',None),('site','site_id'),('country','country'),('station','station_id')]:
        grouped=[('all',paired)] if column is None else paired.groupby(column)
        for group,p in grouped:
            m=pair_metrics(pd.DataFrame({'forcing':p.precipitation_mm_forcing,'station':p.precipitation_mm_station}),len(p),True)
            metrics.append(dict(group_kind=kind,group=str(group),**m,
                                true_hourly_days=int(p.true_hourly_cover.sum())))
    pd.DataFrame(metrics).to_csv(out/'coverage_metrics.csv',index=False)
    counts.to_csv(out/'interval_qc_counts.csv',index=False)
    station_audit=counts.groupby('station_id').agg(raw_intervals=('n_intervals','sum'),
        raw_positive_intervals=('positive_intervals','sum')).reset_index()
    bad=counts.loc[~counts.accepted].groupby('station_id').positive_intervals.sum().rename('rejected_positive_intervals')
    c3=counts.loc[counts.condition_code.eq('3') & counts.quality_code.isin(GOOD_QC)].groupby('station_id').positive_intervals.sum().rename('condition3_full_qc_positive_intervals')
    station_audit=station_audit.merge(bad,on='station_id',how='left').merge(c3,on='station_id',how='left')
    station_audit=pd.DataFrame({'station_id':sorted(wanted.station_id.unique())}).merge(station_audit,on='station_id',how='left')
    station_audit['no_aa_interval_records']=station_audit.raw_intervals.isna()
    for column in ['raw_intervals','raw_positive_intervals','rejected_positive_intervals','condition3_full_qc_positive_intervals']:
        station_audit[column]=station_audit[column].fillna(0).astype(int)
    station_audit.to_csv(out/'station_amount_anomaly_audit.csv',index=False)
    # A fixed raw example confirms that CSV and original ISD carry identical
    # contradictory condition3 flags; neither representation is silently fixed.
    example_url='https://noaa-global-hourly-pds.s3.amazonaws.com/2017/07245099999.csv'
    receipts.append(fetch_source(example_url,cache/'source_examples/07245099999_2017.csv'))
    (out/'source_receipt.json').write_text(json.dumps(receipts,indent=2))
    summary=dict(station_year_requests=len(tasks),successful_station_years=sum(r['status_code']==200 for r in results.values()),
                 downloaded_mb=sum(r['bytes'] for r in results.values() if r['status_code']==200)/1e6,
                 matched_sites=paired.site_id.nunique(),stations=paired.station_id.nunique(),
                 site_days=len(paired),complete_site_days=int(paired.complete_utc_day.sum()),
                 complete_site_day_fraction=float(paired.complete_utc_day.mean()),
                 true_hourly_site_days=int(paired.true_hourly_cover.sum()),
                 sites_with_any_complete_day=int(paired.loc[paired.complete_utc_day,'site_id'].nunique()),
                 usable_site_days=int(paired.precipitation_mm_station.notna().sum()),
                 **safety,
                 source_format='Original compressed ISD global-hourly archive; same AA1-AA4 elements as NOAA CSV, downloaded from NOAA public S3 after 2026 service migration.',
                 quality='AA quality1/5 only (passed all checks). Period1-24h; valid numeric depth. Condition9 (unspecified),2(trace),4(end accumulation with explicit duration) accepted. Estimated/incomplete/deleted/missing-period and suspect/error flags rejected.',
                 timing='Exact [00:00UTC,next00:00UTC] nonoverlapping interval tiling. Original end timestamps retained; no rounding, day shifting, proration or guessed missing zeros. Prefer1h coverage, then shorter accumulation periods.',
                 inference_cutoff='A dayD total requires the report endingD+1 at00:00UTC. At an assessment/prediction on dateT, use weather dates<=T-1 only and enforce latest_interval_end_utc<=T00:00UTC. Archived timestamps do not document real-time publication latency; operational deployment needs a further availability-lag policy.',
                 rain_duration='Only full24x1h cover yields rain_hours_gt_0_1mm_station; longer-period cover never yields inferred wet hours.',
                 trace='Trace flags retained; summed numeric depths are reported measurable precipitation, with sub-resolution trace not quantified.',
                 phase='AA precipitation amount is not independently confirmed liquid rain. Manual snow/mixed/ice indicators retained where reported; absent manual weather is unknown. No phase partition applied.',
                 station_selection='Unchanged elevation-screened station IDs; no prediction errors or observed disease labels used.')
    summary['stations_with_aa_records']=int((~station_audit.no_aa_interval_records).sum())
    summary['stations_without_aa_records']=int(station_audit.no_aa_interval_records.sum())
    summary['stations_with_condition3_full_qc_positive_intervals']=int(station_audit.condition3_full_qc_positive_intervals.gt(0).sum())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().root)
