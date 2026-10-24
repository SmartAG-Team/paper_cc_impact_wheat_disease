"""Reproduce F1 onset information across original calibration/evaluation cohorts.

No parameter is fitted. One-sided observations inform event constraints; only
histories with a zero before the first positive support two-sided timing error.
Calibration is never counted as additional validation.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd


def observation_interval(history,sowing_date):
    history=history.sort_values('date');positive=history[history.observed_percent.gt(0)]
    to_day=lambda date:(pd.Timestamp(date)-pd.Timestamp(sowing_date)).days+1
    if positive.empty:
        return 'no_positive',to_day(history.date.iloc[-1]),np.nan
    first=positive.date.iloc[0]
    zeros=history[history.date.lt(first)&history.observed_percent.eq(0)]
    if zeros.empty:return 'positive_at_first',np.nan,to_day(first)
    return 'zero_to_positive',to_day(zeros.date.iloc[-1]),to_day(first)


def replay(package,output):
    package=Path(package).resolve();output=Path(output).resolve();data=package/'data/process_f1'
    if output==data or data in output.parents:raise ValueError('Preserve original archived inputs')
    checkout=next((parent for parent in [package,*package.parents] if (parent/'models/compat.py').is_file()),None)
    if checkout is not None:
        sys.path.insert(0,str(checkout))
        from models.monitoring.climate_prefix import simulate_prefix
    else:
        sys.path.insert(0,str(package/'code'))
        from analysis.disease_improvement.climate_prefix import simulate_prefix
    source=data/'all115_F1_records_with_replay.csv';records=pd.read_csv(source)
    registry=pd.read_csv(data/'field55_calendar_membership.csv').set_index('field_id')
    weather=pd.read_csv(data/'four_weather_forcing.csv',parse_dates=['date'])
    phenology=json.loads((package/'parameters/phenology_parameters.json').read_text())
    fit=json.loads((package/'parameters/disease_parameters.json').read_text())['fitted']
    thresholds={int(k):v for k,v in json.loads((package/'parameters/stage_thresholds.json').read_text())['all_stage_thresholds'].items()}
    phase_names={'calibration':'calibration','reused_BASF2019':'evaluation'}
    names=['tmean_c','tmax_c','rh_mean_pct','precipitation_mm'];rows=[];max_error=0.
    for field,history in records.groupby('field_id',sort=True):
        history=history.sort_values('date');meta=registry.loc[field]
        assert history.partition.nunique()==1 and history.season_year.nunique()==1
        phase=phase_names[history.partition.iloc[0]]
        status,lower,upper=observation_interval(history,meta.sowing_date)
        part=weather[weather.field_id.eq(field)].sort_values('date');dates=pd.DatetimeIndex(part.date)
        assert dates[0]==pd.Timestamp(meta.sowing_date)
        forcing={key:part[key].to_numpy(float)[None,:] for key in names}
        sow=np.array([meta.sowing_date],dtype='datetime64[D]');latitude=np.array([meta.latitude])
        state=simulate_prefix(dates,forcing,sow,latitude,phenology,fit,thresholds)
        day=int(state['trajectory'].symptom_day[0,0])
        archived=history.frozen_symptom_day.unique();assert len(archived)==1 and day==archived[0]
        model_date=str((pd.Timestamp(meta.sowing_date)+pd.Timedelta(days=day-1)).date()) if day>0 else None
        early=max(lower+1-day,0) if np.isfinite(lower) and day>0 else 0.
        late=max(day-upper,0) if np.isfinite(upper) and day>0 else 0.
        compatible=bool((early+late)==0) if day>0 else status=='no_positive'
        excess=early+late if day>0 else np.nan
        signed=late-early if status=='zero_to_positive' and day>0 else np.nan
        if status=='zero_to_positive':
            assert day>0
            prefix=simulate_prefix(dates[:day],{k:v[:,:day] for k,v in forcing.items()},sow,latitude,phenology,fit,thresholds)
            previous=simulate_prefix(dates[:day-1],{k:v[:,:day-1] for k,v in forcing.items()},sow,latitude,phenology,fit,thresholds)
            assert int(prefix['trajectory'].symptom_day[0,0])==day and int(previous['trajectory'].symptom_day[0,0])==-1
            max_error=max(max_error,float(np.max(abs(prefix['trajectory'].state-state['trajectory'].state[:,:day+1]))))
        rows.append(dict(field_id=field,source=history.source.iloc[0],season_year=int(history.season_year.iloc[0]),phase=phase,
            site_id=history.site_id.iloc[0],assessments=len(history),observation_status=status,
            observed_positive=bool(history.observed_percent.gt(0).any()),first_assessment_date=history.date.iloc[0],
            last_assessment_date=history.date.iloc[-1],last_zero_state_day=lower,first_positive_state_day=upper,
            bracket_width_days=upper-lower if status=='zero_to_positive' else np.nan,
            model_state_day=day,model_onset_date=model_date,model_onset_detected=day>0,
            observational_constraint_compatible=compatible,constraint_excess_days=excess,
            signed_interval_excess_days=signed,interval_excess_days=abs(signed) if np.isfinite(signed) else np.nan))
    histories=pd.DataFrame(rows);assert len(histories)==55 and histories.field_id.nunique()==55
    bracketed=histories[histories.observation_status.eq('zero_to_positive')].copy()
    assert len(bracketed)==7
    original=pd.read_csv(package/'data/onset_evaluation/onset_records.csv').set_index('field_id')
    evaluation=bracketed[bracketed.phase.eq('evaluation')].set_index('field_id')
    assert set(evaluation.index)==set(original.index)
    np.testing.assert_array_equal(evaluation.loc[original.index,'model_state_day'],original.model_state_day)
    np.testing.assert_allclose(evaluation.loc[original.index,'interval_excess_days'],original.model_interval_excess_days,rtol=0,atol=0)
    coverage=[]
    for (phase,year),group in histories.groupby(['phase','season_year'],sort=True):
        for status in ['positive_at_first','zero_to_positive','no_positive']:
            coverage.append(dict(phase=phase,season_year=year,observation_status=status,
                field_seasons=int(group.observation_status.eq(status).sum()),year_total=len(group)))
    metrics=[]
    for phase,group in histories.groupby('phase',sort=True):
        b=group[group.observation_status.eq('zero_to_positive')]
        metrics.append(dict(phase=phase,field_seasons=len(group),positive_field_seasons=int(group.observed_positive.sum()),
            positive_at_first=int(group.observation_status.eq('positive_at_first').sum()),
            no_positive=int(group.observation_status.eq('no_positive').sum()),bracketed_histories=len(b),
            inside_bracket=int(b.interval_excess_days.eq(0).sum()),mean_interval_excess_days=float(b.interval_excess_days.mean()),
            root_mean_square_interval_excess_days=float(np.sqrt(np.mean(b.interval_excess_days**2))),
            minimum_bracket_width_days=float(b.bracket_width_days.min()),maximum_bracket_width_days=float(b.bracket_width_days.max()),
            calibration_is_validation=False,exact_onset_MAE_identified=False))
    output.mkdir(parents=True,exist_ok=True)
    histories.to_csv(output/'field_histories.csv',index=False);bracketed.to_csv(output/'bracketed_histories.csv',index=False)
    pd.DataFrame(coverage).to_csv(output/'coverage_by_year.csv',index=False)
    pd.DataFrame(metrics).to_csv(output/'phase_metrics.csv',index=False)
    inputs=[source,data/'field55_calendar_membership.csv',data/'four_weather_forcing.csv',package/'data/onset_evaluation/onset_records.csv',
        *[package/'parameters'/name for name in ['phenology_parameters.json','disease_parameters.json','stage_thresholds.json']]]
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    checks=dict(status='passed',field_seasons=55,positive_field_seasons=int(histories.observed_positive.sum()),
        calibration_fields=22,evaluation_fields=33,calibration_bracketed_histories=3,evaluation_bracketed_histories=4,
        archived_onset_days_reproduced=55,bracketed_causal_prefix_maximum_state_error=max_error,
        original_four_evaluation_dates_and_errors_unchanged=True,parameters_modified=False,fitting_performed=False,
        phase_metrics=metrics,one_sided_records='coverage and constraints only; not exact-date timing precision',
        source_sha256={str(path.relative_to(package)):sha(path) for path in inputs},replay_code_sha256=sha(Path(__file__)))
    (output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();replay(args.package,args.output)
