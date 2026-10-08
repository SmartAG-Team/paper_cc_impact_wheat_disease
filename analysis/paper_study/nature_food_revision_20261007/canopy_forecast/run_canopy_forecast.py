"""Frozen-host empirical crop symptom forecasting and retrospective evaluation."""
from pathlib import Path
import hashlib, json, os, sys, time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT/'numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'.mplconfig'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from canopy_response import FEATURES, hierarchical_weights, extract_assessment_features, forecast_percent, fit_response, validate_site_fold
from analysis.paper_study.structural_evaluation.run import load_inputs, DEFAULT_PATHS
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.wetness import duration_exposure
from calibration.seasonal_septoria.infection_events import symptom_brackets, onset_distance

HERE = ROOT/'analysis/paper_study/overwinter_leaf_model_20261006'
FIT = HERE/'disease/overwinter_source_model/frozen_selected_fit.json'
PHENOLOGY = HERE/'phenology/calibrated_stage_thresholds.json'
MEMBERSHIP = HERE/'disease/original_sign_only_target_membership.parquet'
FOLDS = HERE/'disease/inner_location_membership_before_fitting.csv'
EVAL = OUT.parent/'epidemic_evaluation'
REGISTRY = OUT/'candidate_feature_penalty_registry_before_fitting.json'
PARTITIONS = ['calibration', 'reused_BASF2019', 'reused_strict_Corteva']
PRED = {'frozen_raw_damage':'frozen_damage_percent', 'calibration_leaf_mean':'leaf_mean_percent', 'crop_only':'crop_only_percent', 'crop_moisture':'crop_moisture_percent'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def metrics(frame, observed, predicted, **labels):
    q = frame.loc[np.isfinite(frame[observed]) & np.isfinite(frame[predicted])].copy()
    if q.empty:
        return None
    w = hierarchical_weights(q)
    o, p = q[observed].to_numpy(float), q[predicted].to_numpy(float)
    e = p-o
    variance = w@((o-w@o)**2)
    return dict(**labels, n=len(q), fields=int(q.field_id.nunique()), coordinate_years=int(q.coordinate_year.nunique()), source_ordinal_leaf_series=int(q.groupby(['field_id','leaf_index']).ngroups), rmse=float(np.sqrt(w@(e*e))), mae=float(w@abs(e)), bias=float(w@e), observed_mean=float(w@o), predicted_mean=float(w@p), weighted_R2=None if variance<=0 else float(1-(w@(e*e))/variance), weight_sum=float(w.sum()), unit='percentage_points', uncertainty_interval_available=False)


def build_daily(data, accumulation, thresholds, fitted):
    """Weather/development only; disease targets and source stages are unused."""
    days = data.metadata.set_index('field_index').forcing_days.reindex(range(len(accumulation))).to_numpy(int)
    mask = np.arange(accumulation.shape[1])[None,:] < days[:,None]
    host = leaf_host(accumulation, data.temperature, thresholds,
        rank_spacing_units=fitted['rank_spacing_units'], forcing_mask=mask,
        juvenile_policy=fitted.get('juvenile_policy','handover_31_39'))
    active = host.active[:,:,:7]
    exposure = duration_exposure(data.temperature, data.maximum_temperature, data.humidity,
        data.rain, rain_rate_mm_hour=fitted['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    stage = np.zeros_like(accumulation, int)
    for key, value in sorted(thresholds.items()):
        stage = np.where(accumulation>=value, key, stage)
    stage[~mask] = -1
    daily = dict(predicted_stage_code=stage, available=active,
        crop_progress=np.clip((accumulation-thresholds[31])/(thresholds[85]-thresholds[31]),0.,1.),
        leaf_thermal_age=np.cumsum(np.maximum(data.temperature,0.)[:,:,None]*active,axis=1)/200.,
        cum_moisture=np.cumsum(exposure[:,:,None]*active,axis=1)/20.)
    return daily, days, host


def daily_prediction(daily, record):
    a=np.asarray(record['leaf_intercepts']); b=np.asarray(record['nonnegative_slopes'])
    eta=np.broadcast_to(a,daily['available'].shape).copy()
    for k, feature in enumerate(FEATURES[record['variant']]):
        values=daily[feature]
        eta += b[k]*(values[:,:,None] if values.ndim==2 else values)
    from scipy.special import expit
    p=100*daily['available']*expit(eta)
    assert np.isfinite(p).all() and p.min()>=0 and p.max()<=100
    return p


def trajectory_tables(ass):
    ends=ass.sort_values('date').groupby(['partition','field_id','leaf_index'],as_index=False).tail(1).copy()
    increments, windows=[], []
    for _, g in ass.groupby(['partition','field_id','leaf_index']):
        g=g.sort_values('date'); first=g.iloc[0]
        span=(g.date.iloc[-1]-g.date.iloc[0]).days
        for i in range(1,len(g)):
            old, now=g.iloc[i-1], g.iloc[i]
            r={k:now[k] for k in ['partition','field_id','coordinate_year','source','leaf_index','endpoint_series']}
            r.update(previous_date=old.date, date=now.date, gap_days=int((now.date-old.date).days), observed_increment_pp=float(now.observed_percent-old.observed_percent))
            for model,col in PRED.items():
                r[model+'_increment_pp']=float(now[col]-old[col])
            increments.append(r)
        if span>0:
            x=(g.date-g.date.iloc[0]).dt.days.to_numpy()
            r={k:first[k] for k in ['partition','field_id','coordinate_year','source','leaf_index','endpoint_series']}
            r.update(window_start=g.date.iloc[0],window_end=g.date.iloc[-1],span_days=int(span),n_assessments=len(g),observed_integral_percent_days=float(np.trapezoid(g.observed_percent,x)))
            r['observed_window_mean_percent']=r['observed_integral_percent_days']/span
            for model,col in PRED.items():
                r[model+'_integral_percent_days']=float(np.trapezoid(g[col],x))
                r[model+'_window_mean_percent']=r[model+'_integral_percent_days']/span
            windows.append(r)
    return ends, pd.DataFrame(increments), pd.DataFrame(windows)


def magnitude_metrics(ass, ends, increments, windows):
    rows=[]
    for partition in PARTITIONS:
        for name, frame, observed in [('all_assessments',ass,'observed_percent'),('final_numeric_assessment',ends,'observed_percent'),('temporal_increment',increments,'observed_increment_pp'),('sampling_window_mean_severity',windows,'observed_window_mean_percent')]:
            for leafscope in ['all_numbered_leaves','top3']:
                q=frame.loc[frame.partition.eq(partition)]
                if leafscope=='top3': q=q.loc[q.leaf_index.lt(3)]
                for model, col in PRED.items():
                    col=col if name in ['all_assessments','final_numeric_assessment'] else model+'_increment_pp' if name=='temporal_increment' else model+'_window_mean_percent'
                    r=metrics(q,observed,col,partition=partition,endpoint=name,leaf_scope=leafscope,model=model)
                    if r is not None:
                        if name=='temporal_increment':
                            w=hierarchical_weights(q);nz=q[observed].abs().gt(1e-10).to_numpy()
                            r['observed_decreasing_pairs']=int(q[observed].lt(-1e-10).sum())
                            r['direction_accuracy_nonzero_observed_increment']=None if not nz.any() else float(w[nz]@(np.sign(q[col].to_numpy()[nz])==np.sign(q[observed].to_numpy()[nz]))/w[nz].sum())
                        rows.append(r)
    return pd.DataFrame(rows)


def sign_metrics(ass):
    rows=[]
    for partition in PARTITIONS:
        for leafscope in ['all_numbered_leaves','top3']:
            q=ass.loc[ass.partition.eq(partition)]
            if leafscope=='top3':q=q.loc[q.leaf_index.lt(3)]
            w=hierarchical_weights(q);obs=q.observed_percent.gt(0).to_numpy()
            for model,col in PRED.items():
                pred=q[col].ge(.1).to_numpy();tp=float(w@(obs&pred));tn=float(w@(~obs&~pred));fp=float(w@(~obs&pred));fn=float(w@(obs&~pred))
                rows.append(dict(partition=partition,leaf_scope=leafscope,model=model,n=len(q),fields=q.field_id.nunique(),coordinate_years=q.coordinate_year.nunique(),observed_positive_rows=int(obs.sum()),observed_zero_rows=int((~obs).sum()),accuracy=tp+tn,sensitivity=None if tp+fn==0 else tp/(tp+fn),specificity=None if tn+fp==0 else tn/(tn+fp),false_positive_weight=fp,false_negative_weight=fn,forecast_cutoff_percent=.1,observation_positive_rule='source numeric severity >0',uncertainty_interval_available=False))
    return pd.DataFrame(rows)


def timing_tables(ass, daily_predictions, days):
    rows=[]
    for partition in PARTITIONS:
        q=ass.loc[ass.partition.eq(partition)].copy()
        for definition, cutoffs in [('existing_zero_positive_sign',[.1]),('matched_severity_threshold',[.1,1.,5.])]:
            for cutoff in cutoffs:
                q['value']=q.observed_percent if definition=='existing_zero_positive_sign' else q.observed_percent.ge(cutoff).astype(int)
                b=symptom_brackets(q,upper_three=False)
                field,leaf=b.field_index.to_numpy(int),b.leaf_index.to_numpy(int)
                for model,daily in daily_predictions.items():
                    cross=daily>=cutoff;occur=cross.any(axis=1)
                    onset=np.where(occur,cross.argmax(axis=1)+1,-1)
                    g=b.copy();g['predicted_symptom_day']=onset[field,leaf];g['forcing_end_day']=days[field]
                    g['onset_distance_days']=onset_distance(g,g.predicted_symptom_day.to_numpy(),g.forcing_end_day.to_numpy())
                    g['compatible']=g.onset_distance_days.eq(0);g['model']=model;g['partition']=partition;g['timing_definition']=definition;g['cutoff_percent']=cutoff
                    rows.append(g)
    onset=pd.concat(rows,ignore_index=True)
    # Frozen trajectory timing comes from the verified, unchanged raw evaluation.
    old=pd.read_csv(EVAL/'continuous_severity_onset_brackets.csv');old['model']='frozen_raw_damage'
    onset=pd.concat([onset,old],ignore_index=True,sort=False)
    scores=[]
    for (partition,model,definition,cutoff,censor),g in onset.groupby(['partition','model','timing_definition','cutoff_percent','censoring']):
        for leafscope in ['all_numbered_leaves','top3']:
            q=g.loc[g.leaf_index.lt(3)] if leafscope=='top3' else g
            if q.empty:continue
            w=hierarchical_weights(q)
            scores.append(dict(partition=partition,model=model,timing_definition=definition,cutoff_percent=cutoff,censoring=censor,leaf_scope=leafscope,n_leaf_series=len(q),fields=q.field_id.nunique(),coordinate_years=q.coordinate_year.nunique(),equal_hierarchy_distance_days=float(w@q.onset_distance_days),compatible_fraction=float(w@q.compatible),missed_positive=int((q.observed_positive&q.predicted_symptom_day.lt(0)).sum()),observed_persistence_violations=int(q.persistence_violations.sum()),uncertainty_interval_available=False))
    return onset,pd.DataFrame(scores)


def support_tables(ass, features, selected_variant):
    rows=[]
    cal=features.loc[features.partition.eq('calibration')]
    for partition in PARTITIONS:
        for leafscope in ['all_numbered_leaves','top3']:
            q=ass.loc[ass.partition.eq(partition)]
            if leafscope=='top3':q=q.loc[q.leaf_index.lt(3)]
            for status in ['all_assessments','predicted_leaf_available','predicted_leaf_unavailable','same_date_observed_stage_available','same_date_observed_stage_missing']:
                g=q if status=='all_assessments' else q.loc[q.available] if status=='predicted_leaf_available' else q.loc[~q.available] if status=='predicted_leaf_unavailable' else q.loc[q.stage_from.notna()] if status=='same_date_observed_stage_available' else q.loc[q.stage_from.isna()]
                if g.empty:continue
                w=hierarchical_weights(g)
                rows.append(dict(partition=partition,leaf_scope=leafscope,support=status,n=len(g),fields=g.field_id.nunique(),coordinate_years=g.coordinate_year.nunique(),positive_rows=int(g.observed_percent.gt(0).sum()),observed_mean=float(w@g.observed_percent),selected_rmse=float(np.sqrt(w@((g[selected_variant+'_percent']-g.observed_percent)**2))),observed_stage_used_for_forecast=False))
    excursions=[]
    for partition in PARTITIONS:
        for leaf in range(7):
            q=features.loc[features.partition.eq(partition)&features.leaf_index.eq(leaf)]
            train=cal.loc[cal.leaf_index.eq(leaf)]
            if q.empty:continue
            for feature in FEATURES['crop_moisture']:
                lo,hi=float(train[feature].min()),float(train[feature].max())
                excursions.append(dict(partition=partition,source_ordinal_leaf=leaf+1,feature=feature,training_rows=len(train),training_min=lo,training_max=hi,n=len(q),below_training_min=int(q[feature].lt(lo-1e-10).sum()),above_training_max=int(q[feature].gt(hi+1e-10).sum()),validation_min=float(q[feature].min()),validation_max=float(q[feature].max())))
    return pd.DataFrame(rows),pd.DataFrame(excursions)


def render(ass, representative_daily, selection, selected_variant):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,3,figsize=(12,8.6),sharey=True)
    for ax,z in zip(axes.flat,selection.itertuples()):
        q=ass.loc[ass.partition.eq(z.partition)&ass.field_id.eq(z.field_id)&ass.leaf_index.eq(z.leaf_index)].sort_values('date')
        d=representative_daily.loc[representative_daily.partition.eq(z.partition)&representative_daily.field_id.eq(z.field_id)&representative_daily.leaf_index.eq(z.leaf_index)]
        ax.plot(d.date,d.frozen_damage_percent,color='#0072B2',lw=1.6,label='Frozen raw damage')
        ax.plot(d.date,d.crop_only_percent,color='#D18F00',lw=1.6,label='Crop development response')
        ax.plot(d.date,d.crop_moisture_percent,color='#9E4674',lw=1.6,ls='--',label='Crop + moisture response')
        ax.plot(q.date,q.observed_percent,color='#292929',lw=.7,marker='o',ms=4,label='Observed severity')
        ax.set_ylim(-2,102);ax.set_xlim(q.date.min(),q.date.max());ax.grid(axis='y',color='#eeeeee')
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=4));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'));ax.tick_params(axis='x',rotation=25)
        ax.set_title(f'{z.partition}\nField {z.field_id.split("|")[1]}; source rank {z.leaf_index+1}; n={len(q)}',fontsize=9)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False)
    fig.supylabel('Observed severity / predicted response (%)',x=.006)
    fig.text(.5,.012,'Series selected by assessment count, date span and lexical key. Corteva ordinal ranks remain unverified. Connecting lines guide the eye.',ha='center',fontsize=8)
    fig.tight_layout(rect=[.015,.04,1,.95]);fig.savefig(OUT/'representative_canopy_trajectories.png',dpi=180);fig.savefig(OUT/'representative_canopy_trajectories.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharex=True,sharey=True)
    for ax,partition in zip(axes,PARTITIONS):
        q=ass.loc[ass.partition.eq(partition)];ax.scatter(q.observed_percent,q[selected_variant+'_percent'],s=12,color='#9E4674',alpha=.35,edgecolors='none');ax.plot([0,100],[0,100],ls='--',lw=1,color='#666666');ax.set_xlim(0,100);ax.set_ylim(0,100);ax.set_title(f'{partition}\nn={len(q)} / {q.field_id.nunique()} fields');ax.set_xlabel('Observed severity (%)');ax.grid(color='#eeeeee')
    axes[0].set_ylabel('Calibration-CV-selected forecast (%)');fig.tight_layout();fig.savefig(OUT/'observed_predicted_canopy_severity.png',dpi=180);fig.savefig(OUT/'observed_predicted_canopy_severity.pdf');plt.close(fig)


def main():
    started=time.perf_counter();registry=json.loads(REGISTRY.read_text())
    dependencies=[Path(__file__),OUT/'canopy_response.py',REGISTRY,FIT,PHENOLOGY,MEMBERSHIP,FOLDS,*DEFAULT_PATHS.values(),EVAL/'observed_vs_frozen_assessments.parquet',EVAL/'continuous_severity_onset_brackets.csv',EVAL/'representative_daily_predictions.csv',EVAL/'representative_series_selection_before_predictions.csv',ROOT/'model/seasonal_septoria/overwinter.py',ROOT/'model/seasonal_septoria/leaf_phenology.py',ROOT/'model/seasonal_septoria/wetness.py',ROOT/'data/basf-wheat-diseases.txt',ROOT/'data/paper_study/observations/new_sources/corteva-2014-2018/corteva-wheat-diseases.txt']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in dependencies}
    for path,digest in registry['frozen_input_sha256'].items():assert sha(ROOT/path)==digest
    data,weather,accumulation,_=load_inputs(DEFAULT_PATHS)
    membership=pd.read_parquet(MEMBERSHIP);source=data.targets.copy()
    truth=source.loc[source.global_target_index.isin(membership.global_target_index)].merge(membership[['global_target_index','partition']],on='global_target_index',validate='one_to_one')
    truth['observed_percent']=truth.value.astype(float)
    assert truth.groupby('partition').field_id.nunique().to_dict()=={'calibration':28,'reused_BASF2019':45,'reused_strict_Corteva':143}
    assert truth.groupby('partition').size().to_dict()=={'calibration':254,'reused_BASF2019':330,'reused_strict_Corteva':1703}
    # Only this whitelist reaches the forecast feature API.
    coordinates=truth[['global_target_index','field_id','field_index','site_id','source','season_year','coordinate_year','endpoint_series','leaf_index','date','day_index','partition']].copy()
    assert not {'value','observed_percent','stage_from','stage_to'} & set(coordinates)
    fitted=json.loads(FIT.read_text())['fitted'];thresholds={int(k):float(v) for k,v in json.loads(PHENOLOGY.read_text())['all_stage_thresholds'].items()}
    # Prediction boundary excludes outcome and actual-stage tables even in data.targets.
    data.targets=coordinates.copy()
    daily,days,host=build_daily(data,accumulation,thresholds,fitted)
    features=extract_assessment_features(coordinates,daily)
    features['frozen_host_area_fraction']=host.area[features.field_index.to_numpy(int),features.day_index.to_numpy(int)-1,features.leaf_index.to_numpy(int)]
    assert features.available.eq(features.frozen_host_area_fraction.gt(0)).all()
    features.to_csv(OUT/'predicted_crop_weather_features_at_assessments.csv',index=False)
    selection=pd.read_csv(EVAL/'representative_series_selection_before_predictions.csv')
    selection.to_csv(OUT/'representative_series_selection_before_predictions.csv',index=False)
    folds=pd.read_csv(FOLDS);training_ids=features.loc[features.partition.eq('calibration'),'global_target_index'].to_numpy(int)
    records=[];cvrows=[];oofrows=[];allfits={}
    (OUT/'fits').mkdir(exist_ok=True)
    for variant in FEATURES:
        for penalty in registry['ridge_grid']:
            predictions=[]
            for fold, fm in folds.groupby('fold',sort=True):
                train=fm.loc[fm.role.eq('train'),'global_target_index'].to_numpy(int);test=fm.loc[fm.role.eq('validation'),'global_target_index'].to_numpy(int)
                assert set(train)|set(test)==set(training_ids);validate_site_fold(features,train,test)
                record=fit_response(features,truth,train,variant,float(penalty));record.update(fold=fold,registry_sha256=sha(REGISTRY),created_utc=datetime.now(timezone.utc).isoformat())
                save_json(OUT/'fits'/f'{variant}_ridge{penalty:g}_{fold}.json',record)
                q=features.set_index('global_target_index').loc[test].reset_index().copy()
                q['observed_percent']=truth.set_index('global_target_index').loc[test].observed_percent.to_numpy(float)
                q['prediction_percent']=forecast_percent(q,record);q['variant']=variant;q['ridge_penalty']=float(penalty);q['fold']=fold
                predictions.append(q);oofrows.append(q)
                cvrows.append(metrics(q,'observed_percent','prediction_percent',variant=variant,ridge_penalty=float(penalty),fold=fold,scope='held_out_location_fold'))
            oof=pd.concat(predictions,ignore_index=True)
            assert oof.global_target_index.is_unique and set(oof.global_target_index)==set(training_ids)
            score=metrics(oof,'observed_percent','prediction_percent',variant=variant,ridge_penalty=float(penalty),fold='pooled',scope='original28_out_of_fold')
            records.append(score);cvrows.append(score)
    cv=pd.DataFrame(cvrows);pooled=pd.DataFrame(records)
    cv.to_csv(OUT/'inner_location_cv_metrics.csv',index=False);pd.concat(oofrows,ignore_index=True).to_csv(OUT/'all_candidate_out_of_fold_predictions.csv',index=False)
    selected=[]
    for variant,g in pooled.groupby('variant',sort=False):
        q=g.copy();q['tie_rmse']=q.rmse.round(12);best=q.sort_values(['tie_rmse','ridge_penalty'],ascending=[True,False]).iloc[0]
        selected.append(best.to_dict())
        record=fit_response(features,truth,training_ids,variant,float(best.ridge_penalty));record.update(registry_sha256=sha(REGISTRY),created_utc=datetime.now(timezone.utc).isoformat(),inner_cv_rmse=float(best.rmse))
        allfits[variant]=record;save_json(OUT/'fits'/f'{variant}_all_original28_selected_ridge.json',record)
    selection_scores=pd.DataFrame(selected);selection_scores['n_features']=selection_scores.variant.map(lambda x:len(FEATURES[x]));best=selection_scores.sort_values(['tie_rmse','n_features','ridge_penalty'],ascending=[True,True,False]).iloc[0];selected_variant=str(best.variant)
    save_json(OUT/'calibration_only_selection.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selected_variant=selected_variant,selected_ridge_penalty=float(best.ridge_penalty),selected_pooled_cv_rmse=float(best.rmse),within_variant_records=selected,selection_rule=registry['selection'],registry_sha256=sha(REGISTRY),validation_numeric_values_used=False))
    # Numeric withheld outcomes enter the evaluator only after candidate selection is saved.
    baseline=pd.read_parquet(EVAL/'observed_vs_frozen_assessments.parquet')
    ass=truth.merge(features[['global_target_index','predicted_stage_code','available','crop_progress','leaf_thermal_age','cum_moisture','frozen_host_area_fraction']],on='global_target_index',validate='one_to_one').drop(columns='value')
    ass=ass.merge(baseline[['global_target_index','observed_percent','frozen_damage_percent','leaf_mean_percent']],on='global_target_index',suffixes=('','_verified'),validate='one_to_one')
    assert np.array_equal(ass.observed_percent,ass.observed_percent_verified);ass=ass.drop(columns='observed_percent_verified')
    assert np.allclose(ass.frozen_host_area_fraction,baseline.set_index('global_target_index').loc[ass.global_target_index].frozen_host_area_fraction,rtol=0,atol=0)
    daily_predictions={}
    for variant,record in allfits.items():
        ass[variant+'_percent']=forecast_percent(features,record)
        daily_predictions[variant]=daily_prediction(daily,record)
        i,d,l=features.field_index.to_numpy(int),features.day_index.to_numpy(int)-1,features.leaf_index.to_numpy(int)
        np.testing.assert_allclose(ass[variant+'_percent'],daily_predictions[variant][i,d,l],rtol=1e-12,atol=1e-12)
    ass['selected_variant']=selected_variant;ass['selected_percent']=ass[selected_variant+'_percent']
    ass['ordinal_mapping_verified']=ass.source.eq('BASF') & ass.leaf_index.eq(0)
    ass['observed_stage_used_for_forecast']=False
    ass['evaluation_weight']=ass.groupby('partition',group_keys=False).apply(lambda g:pd.Series(hierarchical_weights(g),index=g.index),include_groups=False).sort_index()
    ends,increments,windows=trajectory_tables(ass);scores=magnitude_metrics(ass,ends,increments,windows)
    signs=sign_metrics(ass);onsets,timings=timing_tables(ass,daily_predictions,days)
    supports,excursions=support_tables(ass,features,selected_variant)
    representative=[]
    frozen_daily=pd.read_csv(EVAL/'representative_daily_predictions.csv',parse_dates=['date'])
    for z in selection.itertuples():
        q=ass.loc[ass.partition.eq(z.partition)&ass.field_id.eq(z.field_id)&ass.leaf_index.eq(z.leaf_index)].sort_values('day_index');row=q.iloc[0];start,end=int(q.day_index.min()),int(q.day_index.max());idx=np.arange(start-1,end)
        dates=pd.Timestamp(data.metadata.set_index('field_index').loc[row.field_index].sowing_date)+pd.to_timedelta(idx,unit='D')
        d=pd.DataFrame(dict(partition=z.partition,field_id=z.field_id,leaf_index=z.leaf_index,date=dates,crop_only_percent=daily_predictions['crop_only'][int(row.field_index),idx,z.leaf_index],crop_moisture_percent=daily_predictions['crop_moisture'][int(row.field_index),idx,z.leaf_index],available=daily['available'][int(row.field_index),idx,z.leaf_index],predicted_stage_code=daily['predicted_stage_code'][int(row.field_index),idx]))
        f=frozen_daily.loc[frozen_daily.partition.eq(z.partition)&frozen_daily.field_id.eq(z.field_id)&frozen_daily.leaf_index.eq(z.leaf_index),['date','model_damage_percent']].rename(columns={'model_damage_percent':'frozen_damage_percent'})
        d=d.merge(f,on='date',validate='one_to_one');assert len(d)==len(idx);representative.append(d)
    representative=pd.concat(representative,ignore_index=True)
    outputs={'observed_vs_canopy_forecasts.csv':ass,'final_numeric_leaf_endpoints.csv':ends,'consecutive_severity_increments.csv':increments,'sampling_window_severity_integrals.csv':windows,'severity_trajectory_metrics.csv':scores,'assessment_sign_metrics.csv':signs,'continuous_severity_onset_brackets.csv':onsets,'severity_timing_metrics.csv':timings,'leaf_availability_and_source_stage_support.csv':supports,'feature_range_excursions.csv':excursions,'representative_daily_predictions.csv':representative}
    for name,frame in outputs.items():frame.to_csv(OUT/name,index=False)
    ass.to_parquet(OUT/'observed_vs_canopy_forecasts.parquet',index=False)
    # Support rule was fixed before all empirical fitting.
    checks=[]
    for partition in PARTITIONS[1:]:
        for leafscope in ['all_numbered_leaves','top3']:
            for endpoint,comparators in [('all_assessments',['frozen_raw_damage','calibration_leaf_mean']),('final_numeric_assessment',['frozen_raw_damage'])]:
                q=scores.loc[scores.partition.eq(partition)&scores.leaf_scope.eq(leafscope)&scores.endpoint.eq(endpoint)].set_index('model')
                for comparator in comparators:
                    checks.append(dict(partition=partition,leaf_scope=leafscope,endpoint=endpoint,comparator=comparator,selected_rmse=float(q.loc[selected_variant].rmse),comparator_rmse=float(q.loc[comparator].rmse),passes=bool(q.loc[selected_variant].rmse<q.loc[comparator].rmse)))
    supported=all(x['passes'] for x in checks)
    weather_checks=[]
    for partition in PARTITIONS[1:]:
        for leafscope in ['all_numbered_leaves','top3']:
            q=scores.loc[scores.partition.eq(partition)&scores.leaf_scope.eq(leafscope)&scores.endpoint.eq('all_assessments')].set_index('model')
            weather_checks.append(dict(partition=partition,leaf_scope=leafscope,crop_only_rmse=float(q.loc['crop_only'].rmse),crop_moisture_rmse=float(q.loc['crop_moisture'].rmse),passes=bool(q.loc['crop_moisture'].rmse<q.loc['crop_only'].rmse)))
    support_record=dict(supported_retrospective_severity_forecast=supported,support_rule=registry['transport_record_support_rule'],checks=checks,weather_added_value_supported=selected_variant=='crop_moisture' and all(x['passes'] for x in weather_checks),weather_added_value_checks=weather_checks,selected_variant=selected_variant,scope='retrospective crop severity prediction only; no climate/yield transfer established')
    save_json(OUT/'forecast_support_decision.json',support_record)
    if supported:
        config=dict(schema_version=1,empirical_coefficient_record=allfits[selected_variant],feature_definitions=registry['feature_definitions'],forecast_rule=registry['response'],availability_rule=registry['availability'],leaf_ranks='source ordinal1–7; existing fixed host rank mapping; Corteva identity undocumented',fixed_crop_stage_thresholds=thresholds,fixed_host_rank_spacing_units=fitted['rank_spacing_units'],fixed_juvenile_policy=fitted['juvenile_policy'],fixed_weather_preprocessing=fitted['weather_preprocessing'],source_hashes=hashes,training_feature_ranges=excursions.loc[excursions.partition.eq('calibration')].to_dict(orient='records'),support= support_record,limits=['observational source severity percent; exact denominator unresolved','retrospective reused cohorts; no untouched confirmatory test','crop thresholds already calibrated from original training stages outside empirical inner folds','nonnegative response enforces within-season nondecrease; natural symptom decline/senescence absent','future climate and cultivar transfer unvalidated; no yield relationship or causal management effect','very sparse original training evidence on ordinal ranks6/7'])
        save_json(OUT/'transportable_retrospective_forecast_config.json',config)
    render(ass,representative,selection,selected_variant)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    save_json(OUT/'evaluation_receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,source_sha256=hashes,artifact_rows={k:len(v) for k,v in outputs.items()},calibration_only_selection=True,withheld_numeric_targets_enter_optimizer=False,source_observed_stage_enters_forecast=False,physical_pathogen_equations_or_parameters_changed=False,new_empirical_symptom_response_only=True,ordinal_Corteva_mapping_resolved=False,prior_validation_outcomes_reused=True,independent_biological_location_identity_confirmed=False,confidence_intervals_reported=False,partition_counts=ass.groupby('partition').agg(n=('observed_percent','size'),fields=('field_id','nunique'),coordinate_years=('coordinate_year','nunique')).to_dict(orient='index'),selected_variant=selected_variant,retrospective_forecast_supported=supported,weather_added_value_supported=support_record['weather_added_value_supported'],registry_saved_before_fitting=True,registry_sha256=sha(REGISTRY),inner_fold_disjoint_location_guards_passed=True,forecast_daily_assessment_alignment_passed=True,source_hashes_unchanged_after_evaluation=True))
    print('CALIBRATION CV',pooled[['variant','ridge_penalty','rmse','mae']].to_string(index=False))
    print('SELECTED',selected_variant,'SUPPORTED',supported,'WEATHER VALUE',support_record['weather_added_value_supported'])
    print(scores.loc[scores.endpoint.isin(['all_assessments','final_numeric_assessment'])&scores.leaf_scope.eq('all_numbered_leaves'),['partition','endpoint','model','n','fields','coordinate_years','rmse','mae','bias']].to_string(index=False))
    print('TOP3 TWO-SIDED ONSET',timings.loc[timings.timing_definition.eq('existing_zero_positive_sign')&timings.censoring.eq('two_sided')&timings.leaf_scope.eq('top3'),['partition','model','n_leaf_series','fields','coordinate_years','equal_hierarchy_distance_days']].to_string(index=False))
    print('PASS',len(ass),'actual-date forecasts; source hashes unchanged')


if __name__=='__main__':
    main()
