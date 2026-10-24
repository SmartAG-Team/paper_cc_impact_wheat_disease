"""Reproduce the qualified four-field symptom-onset check without fitting.

Only anatomically identified BASF flag-leaf histories from the original 2019
population enter. Days are one-based state days since sowing; onset is observed
only within (last zero, first positive]. Saved calendar dates are an archived
comparison; its original fitting implementation is not reconstructed here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd


def interval_excess(lower, upper, predicted):
    """Distance to integer-day onset support lower+1,...,upper."""
    lower,upper,predicted=np.broadcast_arrays(np.asarray(lower,float),np.asarray(upper,float),np.asarray(predicted,float))
    if not (np.isfinite(lower).all() and np.isfinite(upper).all() and np.isfinite(predicted).all()):
        raise ValueError('Finite two-sided onset brackets and observed model crossings required')
    if np.any(upper<=lower) or np.any(predicted<1):raise ValueError('Nonempty brackets and one-based state days required')
    return np.maximum(lower+1-predicted,0)+np.maximum(predicted-upper,0)


def state_day_date(sowing, day):
    return pd.Timestamp(sowing)+pd.Timedelta(days=int(day)-1)


def replay(package,output):
    package=Path(package).resolve();output=Path(output).resolve();data=package/'data/process_f1'
    if output==data or data in output.parents:raise ValueError('Preserve original process inputs')
    checkout=next((parent for parent in [package,*package.parents] if (parent/'models/compat.py').is_file()),None)
    if checkout is not None:
        sys.path.insert(0,str(checkout))
        from models.monitoring.climate_prefix import simulate_prefix
    else:
        sys.path.insert(0,str(package/'code'))
        from analysis.disease_improvement.climate_prefix import simulate_prefix
    source=data/'original2019_F1_two_sided_onsets.csv'
    archived=pd.read_csv(source);records=pd.read_csv(data/'evaluation70_F1_records_with_replay.csv')
    fields=pd.read_csv(data/'field55_calendar_membership.csv').set_index('field_id')
    weather=pd.read_csv(data/'four_weather_forcing.csv',parse_dates=['date'])
    phenology=json.loads((package/'parameters/phenology_parameters.json').read_text())
    fit=json.loads((package/'parameters/disease_parameters.json').read_text())['fitted']
    thresholds={int(k):v for k,v in json.loads((package/'parameters/stage_thresholds.json').read_text())['all_stage_thresholds'].items()}
    assert len(archived)==4 and archived.field_id.nunique()==4
    assert archived.source.eq('BASF').all() and archived.leaf_index.eq(0).all() and archived.season_year.eq(2019).all()
    eligible={}
    for field,part in records.groupby('field_id'):
        part=part.sort_values('date');positive=part[part.observed_percent.gt(0)]
        if positive.empty:continue
        first=positive.date.iloc[0];zero=part[part.date.lt(first)&part.observed_percent.eq(0)]
        if not zero.empty:eligible[field]=(zero.date.iloc[-1],first)
    assert set(eligible)==set(archived.field_id)
    names=['tmean_c','tmax_c','rh_mean_pct','precipitation_mm'];rows=[];max_state_error=0.
    for row in archived.itertuples():
        meta=fields.loc[row.field_id];last_zero,first_positive=eligible[row.field_id]
        lower=(pd.Timestamp(last_zero)-pd.Timestamp(meta.sowing_date)).days+1
        upper=(pd.Timestamp(first_positive)-pd.Timestamp(meta.sowing_date)).days+1
        assert lower==row.lower_day and upper==row.upper_day
        w=weather[weather.field_id.eq(row.field_id)].sort_values('date');dates=pd.DatetimeIndex(w.date)
        forcing={name:w[name].to_numpy(float)[None,:] for name in names}
        sow=np.array([meta.sowing_date],dtype='datetime64[D]');lat=np.array([meta.latitude])
        full=simulate_prefix(dates,forcing,sow,lat,phenology,fit,thresholds)
        k=int(full['trajectory'].symptom_day[0,0]);assert k==row.predicted_symptom_day
        assert np.array_equal(full['trajectory'].damage,full['trajectory'].state[...,-3:].sum(axis=-1))
        hit=np.flatnonzero(full['trajectory'].damage[0,:,0]>=.001);assert len(hit) and int(hit[0])==k
        prefix=simulate_prefix(dates[:k],{name:value[:,:k] for name,value in forcing.items()},sow,lat,phenology,fit,thresholds)
        previous=simulate_prefix(dates[:k-1],{name:value[:,:k-1] for name,value in forcing.items()},sow,lat,phenology,fit,thresholds)
        assert int(prefix['trajectory'].symptom_day[0,0])==k and int(previous['trajectory'].symptom_day[0,0])==-1
        max_state_error=max(max_state_error,float(np.max(abs(prefix['trajectory'].state-full['trajectory'].state[:,:k+1]))))
        model_date=state_day_date(meta.sowing_date,k);assert model_date==pd.Timestamp(row.predicted_symptom_date)
        error=float(interval_excess(lower,upper,k));bench=float(interval_excess(lower,upper,row.predicted_symptom_day_benchmark))
        assert error==row.onset_distance_days and bench==row.onset_distance_days_benchmark
        rows.append(dict(field_id=row.field_id,site_id=row.site_id,source='BASF',season_year=2019,anatomy='F1 flag leaf',
            last_zero_date=last_zero,first_positive_date=first_positive,earliest_admissible_date=str(pd.Timestamp(last_zero).date()+pd.Timedelta(days=1)),
            model_onset_date=str(model_date.date()),archived_calendar_date=str(state_day_date(meta.sowing_date,row.predicted_symptom_day_benchmark).date()),
            last_zero_state_day=lower,first_positive_state_day=upper,model_state_day=k,
            archived_calendar_state_day=row.predicted_symptom_day_benchmark,interval_width_days=upper-lower,
            model_interval_excess_days=error,archived_calendar_interval_excess_days=bench,
            model_inside_interval=error==0,archived_calendar_inside_interval=bench==0,field_weight=.25))
    points=pd.DataFrame(rows);metrics=[]
    for model,column in [('frozen_process','model_interval_excess_days'),('archived_calendar','archived_calendar_interval_excess_days')]:
        values=points[column].to_numpy(float)
        metrics.append(dict(model=model,histories=len(points),fields=points.field_id.nunique(),sites=points.site_id.nunique(),
            weighting='one equal vote per field',mean_interval_excess_days=float(values.mean()),
            root_mean_square_interval_excess_days=float(np.sqrt(np.mean(values**2))),inside_observed_intervals=int((values==0).sum()),
            minimum_bracket_width_days=int(points.interval_width_days.min()),maximum_bracket_width_days=int(points.interval_width_days.max()),
            exact_onset_date_MAE_identified=False,new_independent_validation=False))
    output.mkdir(parents=True,exist_ok=True);points.to_csv(output/'onset_records.csv',index=False)
    pd.DataFrame(metrics).to_csv(output/'onset_metrics.csv',index=False)
    inputs=[source,data/'evaluation70_F1_records_with_replay.csv',data/'field55_calendar_membership.csv',data/'four_weather_forcing.csv',
        *[package/'parameters'/name for name in ['phenology_parameters.json','disease_parameters.json','stage_thresholds.json']]]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    check=dict(status='passed',records=4,eligible_original2019_F1_fields=records.field_id.nunique(),all_eligible_two_sided_histories_retained=True,
        original_symptom_dates_reproduced=True,onset_day_prefix_state_maximum_error=max_state_error,
        no_future_weather_needed_for_replayed_onset=True,detection_fraction=.001,operator='first daily N+I+D crossing; one-based state days',
        observed_interval='(last observed zero, first observed positive], daily support lower+1 through upper',
        original_records_reused=True,models_fitted=False,calendar_comparison='saved dates verified; original fitting implementation not authenticated',
        metrics=metrics,source_sha256={str(q.relative_to(package)):sha(q) for q in inputs},replay_code_sha256=sha(Path(__file__)))
    (output/'checks.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check,indent=2))
    return check


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();replay(args.package,args.output)
