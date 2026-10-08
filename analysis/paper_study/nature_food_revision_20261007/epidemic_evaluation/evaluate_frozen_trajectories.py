"""Frozen retrospective crop-symptom evaluation; empirical baselines use original training only."""
from pathlib import Path
import os,sys,hashlib,json,time,gc
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR',str(OUT/'numba_cache'));os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'));sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from analysis.paper_study.structural_evaluation.run import load_inputs,DEFAULT_PATHS,subset_fields
from analysis.paper_study.overwinter_leaf_model_20261006.disease.run import predict_record,PHENOLOGY,SIGN_COLUMNS
from calibration.seasonal_septoria.infection_events import symptom_brackets,onset_distance

PARTITIONS=['calibration','reused_BASF2019','reused_strict_Corteva']
PRED={'frozen_raw_damage':'frozen_damage_percent','calibration_leaf_mean':'leaf_mean_percent','calibration_stage_time':'stage_time_percent'}
FIT=ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/disease/overwinter_source_model/frozen_selected_fit.json'
MEMBERSHIP=FIT.parent.parent/'original_sign_only_target_membership.parquet'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def weights(frame):
    c=frame.coordinate_year.nunique();nf=frame.groupby('coordinate_year').field_id.transform('nunique').to_numpy(float);nl=frame.groupby(['coordinate_year','field_id']).leaf_index.transform('nunique').to_numpy(float);nr=frame.groupby(['coordinate_year','field_id','leaf_index']).leaf_index.transform('size').to_numpy(float);w=1/(c*nf*nl*nr);assert np.isclose(w.sum(),1);return w

def scope(frame,name):return frame.loc[frame.leaf_index.lt(3)] if name=='top3' else frame

def stats(frame,observed,predicted,base):
    g=frame.loc[np.isfinite(frame[observed])&np.isfinite(frame[predicted])].copy()
    if g.empty:return None
    w=weights(g);o=g[observed].to_numpy(float);p=g[predicted].to_numpy(float);e=p-o;mo=float(w@o);mp=float(w@p);vo=float(w@((o-mo)**2));vp=float(w@((p-mp)**2));corr=None if vo<=0 or vp<=0 else float(w@((o-mo)*(p-mp))/np.sqrt(vo*vp))
    return dict(**base,n=len(g),fields=g.field_id.nunique(),coordinate_years=g.coordinate_year.nunique(),source_numbered_leaf_series=g.groupby(['field_id','leaf_index']).ngroups,weight_sum=float(w.sum()),observed_mean=mo,predicted_mean=mp,rmse=float(np.sqrt(w@(e*e))),mae=float(w@np.abs(e)),bias=float(w@e),weighted_R2=None if vo<=0 else float(1-(w@(e*e))/vo),weighted_pearson_r=corr,within10percentage_points=float(w@(abs(e)<=10)),unit='percentage_points',uncertainty_interval_available=False)

def metrics_table(ass,ends,increments,windows):
    result=[]
    for partition in PARTITIONS:
      for name,frame,obs in [('all_assessments',ass,'observed_percent'),('final_numeric_assessment',ends,'observed_percent'),('temporal_increment',increments,'observed_increment_pp'),('sampling_window_mean_severity',windows,'observed_window_mean_percent')]:
       raw=frame.loc[frame.partition.eq(partition)]
       for leafscope in ['all_numbered_leaves','top3']:
        full=scope(raw,leafscope)
        for pop,g,models in [('all_eligible_assessments',full,['frozen_raw_damage','calibration_leaf_mean']),('shared_observed_stage_coverage',full.loc[full.stage_time_available],list(PRED))]:
         for model in models:
          col=PRED[model] if name in ['all_assessments','final_numeric_assessment'] else model+'_increment_pp' if name=='temporal_increment' else model+'_window_mean_percent'
          row=stats(g,obs,col,dict(partition=partition,endpoint=name,leaf_scope=leafscope,comparison_population=pop,model=model))
          if row:
           if name=='temporal_increment':
            h=g.loc[np.isfinite(g[obs])&np.isfinite(g[col])];w=weights(h);nonzero=h[obs].abs().gt(1e-10).to_numpy();row['observed_decreasing_pairs']=int(h[obs].lt(-1e-10).sum());row['direction_accuracy_observed_nonzero']=None if not nonzero.any() else float(w[nonzero]@(np.sign(h[col].to_numpy()[nonzero])==np.sign(h[obs].to_numpy()[nonzero]))/w[nonzero].sum())
           result.append(row)
    return pd.DataFrame(result)

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    deps=[Path(__file__),FIT,PHENOLOGY,MEMBERSHIP,*DEFAULT_PATHS.values(),ROOT/'model/seasonal_septoria/overwinter.py',ROOT/'model/seasonal_septoria/leaf_phenology.py',ROOT/'model/seasonal_septoria/wetness.py',ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/disease/run.py',ROOT/'data/basf-wheat-diseases.txt',ROOT/'data/paper_study/observations/new_sources/corteva-2014-2018/corteva-wheat-diseases.txt']
    hashes={str(x.relative_to(ROOT)):sha(x) for x in deps};data,weather,accumulation,_=load_inputs(DEFAULT_PATHS);membership=pd.read_parquet(MEMBERSHIP);truth=data.targets.loc[data.targets.global_target_index.isin(membership.global_target_index)].copy();truth=truth.merge(membership[['global_target_index','partition']],on='global_target_index',validate='one_to_one');truth['observed_percent']=truth.value.astype(float)
    joined=truth[['global_target_index','field_id','leaf_index','date','value']].merge(membership[['global_target_index','field_id','leaf_index','date','value']],on='global_target_index',suffixes=('_numeric','_signed'),validate='one_to_one');assert joined.field_id_numeric.eq(joined.field_id_signed).all() and joined.leaf_index_numeric.eq(joined.leaf_index_signed).all() and joined.date_numeric.eq(joined.date_signed).all() and joined.value_numeric.gt(0).astype(int).eq(joined.value_signed).all()
    assert truth.groupby('partition').field_id.nunique().to_dict()=={'calibration':28,'reused_BASF2019':45,'reused_strict_Corteva':143}
    truth['source_ordinal_leaf']=truth.leaf_index+1;truth['elapsed_field_assessment_days']=(truth.date-truth.groupby('field_id').date.transform('min')).dt.days
    truth['stage_min_observed']=pd.to_numeric(truth.stage_from,errors='coerce');rev=truth.stage_from.notna()&truth.stage_to.notna()&truth.stage_from.gt(truth.stage_to);truth.loc[rev,'stage_min_observed']=np.nan;truth['stage_time_available']=truth.stage_min_observed.notna();truth['leaf_mapping_status']=np.where(truth.source.eq('BASF'),'source rank1 explicitly flag; other source ranks retained','source ordinal final/current ranking unverified')
    training=truth.loc[truth.partition.eq('calibration')].copy();w=weights(training);means={int(leaf):float(np.average(g.observed_percent,weights=w[training.index.get_indexer(g.index)])) for leaf,g in training.groupby('leaf_index')}
    def design(frame):
      return np.column_stack([np.eye(7)[frame.leaf_index.to_numpy(int)],(frame.stage_min_observed.to_numpy()-50)/20,frame.elapsed_field_assessment_days.to_numpy()/30])
    st=training.loc[training.stage_time_available].copy();sw=weights(st);X=design(st);coef=np.linalg.lstsq(X*np.sqrt(sw[:,None]),st.observed_percent.to_numpy()*np.sqrt(sw),rcond=None)[0];assert len(st)>=100
    selection=[]
    for partition in PARTITIONS:
      for leaf in range(3):
       q=truth.loc[truth.partition.eq(partition)&truth.leaf_index.eq(leaf)];counts=q.groupby(['field_id','endpoint_series']).agg(n=('date','nunique'),first=('date','min'),last=('date','max')).reset_index();counts['span_days']=(counts['last']-counts['first']).dt.days;choice=counts.sort_values(['n','span_days','field_id','endpoint_series'],ascending=[False,False,True,True]).iloc[0];selection.append(dict(partition=partition,leaf_index=leaf,field_id=choice.field_id,endpoint_series=choice.endpoint_series,n_assessments=int(choice.n),span_days=int(choice.span_days),criterion='highest assessment count, longest observed span, lexical field/series tie-break; severity and predictions unused'))
    selected=pd.DataFrame(selection);selected.to_csv(OUT/'representative_series_selection_before_predictions.csv',index=False)
    contract=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='retrospective fixed raw-damage symptom/measurement evaluation',partitions=truth.groupby('partition').agg(assessments=('value','size'),fields=('field_id','nunique'),coordinate_years=('coordinate_year','nunique')).to_dict(orient='index'),prediction_mapping='100×trajectory.damage at original field/day/source-numbered-leaf indices; no fitted observation mapping',model_boundary='targets value redacted; only original SIGN_COLUMNS retained; observed stage excluded',weights='equal coordinate-year, then equal field, source-numbered leaf and assessment/pair within leaf',baselines=dict(leaf_mean='original28 numeric calibration only; equal-hierarchy weighted leaf-rank means',stage_time='single prespecified weighted linear least-squares on7leaf indicators, observed minimum stage centered50/scaled20, elapsed days since first observed field assessment/scaled30; original28only; clip predicted percentages0–100; no validation-based selection',stage_feature='same-date source observed minimum stage; reject reversed source ranges; missing upper stage permitted; missing minimum is unavailable; retrospective stage metadata comparator, not a preseason forecast'),severity_thresholds_percent=[.1,1.,5.],symptom_sign_timing='existing frozen0.001fraction definition and observed zero/positive signs; interval-distance scores retain censoring',endpoints='last numeric assessment per source-numbered leaf, not measured harvest-season terminal disease',integrals='trapezoidal observed-assessment window integral in percentage-days; sampled model points at same dates, no extrapolation beyond measured window',representative_selection=selection,Corteva_leaf_number_mapping_fully_verified=False,independent_biological_site_identity_verified=False,confidence_intervals_invented=False,input_sha256=hashes)
    write_json(OUT/'evaluation_contract_before_predictions.json',contract)
    baseline=dict(training_fields=sorted(training.field_id.unique()),training_assessments=len(training),training_coordinate_years=training.coordinate_year.nunique(),leaf_mean_percent={str(k+1):v for k,v in means.items()},leaf_training_counts=training.groupby('source_ordinal_leaf').size().to_dict(),stage_time_training_assessments=len(st),stage_time_training_fields=st.field_id.nunique(),design_columns=[f'source_leaf_{i}' for i in range(1,8)]+['(observed_GSmin-50)/20','elapsed_field_assessment_days/30'],coefficients=coef.tolist(),matrix_rank=int(np.linalg.matrix_rank(X)),condition_number=float(np.linalg.cond(X)),predictions_clipped_at_physical_percent_bounds=True,baseline_selection_uses_validation=False)
    write_json(OUT/'calibration_only_empirical_baselines.json',baseline)
    fitted=json.loads(FIT.read_text())['fitted'];thresholds={int(k):v for k,v in json.loads(PHENOLOGY.read_text())['all_stage_thresholds'].items()};predparts=[];daily=[];onsets=[];checks=[]
    # Keep observed values and observed stages outside the entire biological prediction object.
    forecast_source=data.targets[SIGN_COLUMNS].copy();forecast_source['value']=np.nan;data.targets=forecast_source
    for partition in PARTITIONS:
      ids=membership.loc[membership.partition.eq(partition),'global_target_index'].to_numpy(int);redacted=subset_fields(data,ids,redact_values=True);assert redacted.targets.value.isna().all();assert 'stage_from' not in redacted.targets and 'stage_to' not in redacted.targets
      original_fields=np.sort(membership.loc[membership.partition.eq(partition),'field_index'].unique());trajectory,host=predict_record(redacted,accumulation[original_fields],thresholds,fitted)
      part=redacted.targets.drop(columns='value').merge(truth.drop(columns=['field_index','value']),on=['global_target_index','field_id','site_id','source','dataset_id','physical_unit','season_year','coordinate_year','endpoint_series','leaf_index','date','day_index','metric'],validate='one_to_one')
      fi,di,li=[part[k].to_numpy(int) for k in ['field_index','day_index','leaf_index']];part['frozen_damage_fraction']=trajectory.damage[fi,di,li];part['frozen_damage_percent']=100*part.frozen_damage_fraction;part['frozen_symptom_day']=trajectory.symptom_day[fi,li];part['frozen_host_area_fraction']=host.area[fi,di-1,li];part['leaf_mean_percent']=part.leaf_index.map(means);part['stage_time_percent_unclipped']=np.nan;known=part.stage_time_available;part.loc[known,'stage_time_percent_unclipped']=design(part.loc[known])@coef;part['stage_time_percent']=part.stage_time_percent_unclipped.clip(0,100);part['evaluation_weight']=weights(part);predparts.append(part)
      b=symptom_brackets(part.rename(columns={'observed_percent':'value'}),upper_three=False);b['partition']=partition;f,l=b.field_index.to_numpy(int),b.leaf_index.to_numpy(int);end=b.field_index.map(redacted.metadata.set_index('field_index').forcing_days).to_numpy(int)
      for mode,cutoffs in [('existing_zero_positive_sign',[.1]),('matched_severity_threshold',[.1,1.,5.])]:
       for cutoff in cutoffs:
        target=part.copy();target['value']=target.observed_percent if mode=='existing_zero_positive_sign' else target.observed_percent.ge(cutoff).astype(int);bound=symptom_brackets(target,upper_three=False);bf,bl=bound.field_index.to_numpy(int),bound.leaf_index.to_numpy(int);ending=bound.field_index.map(redacted.metadata.set_index('field_index').forcing_days).to_numpy(int);cross=trajectory.damage>=cutoff/100;occur=cross.any(axis=1);day=np.where(occur,cross.argmax(axis=1),-1);bound['predicted_symptom_day']=day[bf,bl];bound['forcing_end_day']=ending;bound['onset_distance_days']=onset_distance(bound,bound.predicted_symptom_day.to_numpy(),ending);bound['compatible']=bound.onset_distance_days.eq(0);bound['partition']=partition;bound['timing_definition']=mode;bound['cutoff_percent']=cutoff;onsets.append(bound)
      for choice in selection:
       if choice['partition']!=partition:continue
       q=part.loc[part.field_id.eq(choice['field_id'])&part.leaf_index.eq(choice['leaf_index'])].sort_values('day_index');row=q.iloc[0];startday=int(q.day_index.min());endday=int(q.day_index.max());metadata=redacted.metadata.loc[redacted.metadata.field_index.eq(row.field_index)].iloc[0];dates=pd.Timestamp(metadata.sowing_date)+pd.to_timedelta(np.arange(startday,endday+1)-1,unit='D');daily.append(pd.DataFrame(dict(partition=partition,field_id=row.field_id,source=row.source,leaf_index=int(row.leaf_index),date=dates,model_damage_percent=100*trajectory.damage[int(row.field_index),startday:endday+1,int(row.leaf_index)],leaf_mean_percent=means[int(row.leaf_index)])))
      mass=float(np.max(abs(trajectory.state.sum(axis=-1)-1)));assert mass<1e-10;checks.append(dict(partition=partition,redacted_fields=len(redacted.metadata),assessments=len(part),maximum_tissue_mass_error=mass));del trajectory,host;gc.collect()
    ass=pd.concat(predparts,ignore_index=True);assert len(ass)==2287 and ass.global_target_index.is_unique;archived=pd.read_parquet(FIT.parent.parent/'all_assessment_sign_predictions.parquet');archived=archived.loc[archived.model.eq('overwinter_source_model')&archived.scenario.eq('baseline')&archived.detection_fraction.eq(.001)];matched=ass.merge(archived[['global_target_index','predicted_positive','predicted_symptom_day']],on='global_target_index',validate='one_to_one');assert matched.frozen_damage_fraction.ge(.001).eq(matched.predicted_positive).all();assert matched.frozen_symptom_day.eq(matched.predicted_symptom_day).all()
    ends=ass.sort_values('date').groupby(['partition','field_id','leaf_index'],as_index=False).tail(1).copy();inc=[];wins=[]
    for (partition,field,leaf),g in ass.groupby(['partition','field_id','leaf_index']):
      g=g.sort_values('date');z=g.iloc[0];span=(g.date.iloc[-1]-g.date.iloc[0]).days
      for j in range(1,len(g)):
       now,before=g.iloc[j],g.iloc[j-1];rec={k:now[k] for k in ['partition','field_id','coordinate_year','source','leaf_index','endpoint_series']};rec.update(previous_date=before.date,date=now.date,gap_days=(now.date-before.date).days,observed_increment_pp=float(now.observed_percent-before.observed_percent),stage_time_available=bool(now.stage_time_available and before.stage_time_available))
       for model,col in PRED.items():rec[model+'_increment_pp']=float(now[col]-before[col]);rec[model+'_increment_pp_per_day']=rec[model+'_increment_pp']/rec['gap_days']
       rec['observed_increment_pp_per_day']=rec['observed_increment_pp']/rec['gap_days'];inc.append(rec)
      if span>0:
       x=(g.date-g.date.min()).dt.days.to_numpy();rec={k:z[k] for k in ['partition','field_id','coordinate_year','source','leaf_index','endpoint_series']};rec.update(n_assessments=len(g),window_start=g.date.iloc[0],window_end=g.date.iloc[-1],span_days=int(span),observed_window_AUDPC_percent_days=float(np.trapezoid(g.observed_percent,x)),stage_time_available=bool(g.stage_time_available.all()));rec['observed_window_mean_percent']=rec['observed_window_AUDPC_percent_days']/span
       for model,col in PRED.items():rec[model+'_window_AUDPC_percent_days']=float(np.trapezoid(g[col],x));rec[model+'_window_mean_percent']=rec[model+'_window_AUDPC_percent_days']/span
       wins.append(rec)
    increments=pd.DataFrame(inc);windows=pd.DataFrame(wins);metrics=metrics_table(ass,ends,increments,windows);onset=pd.concat(onsets,ignore_index=True);timings=[]
    for (partition,mode,cutoff,censor),g in onset.groupby(['partition','timing_definition','cutoff_percent','censoring']):
      for leafscope in ['all_numbered_leaves','top3']:
       q=scope(g,leafscope)
       if len(q):w=weights(q);timings.append(dict(partition=partition,timing_definition=mode,cutoff_percent=cutoff,censoring=censor,leaf_scope=leafscope,n_leaf_series=len(q),fields=q.field_id.nunique(),coordinate_years=q.coordinate_year.nunique(),equal_hierarchy_distance_days=float(w@q.onset_distance_days),compatible_fraction=float(w@q.compatible),missed_positive=int((q.observed_positive&q.predicted_symptom_day.lt(0)).sum()),observed_persistence_violations=int(q.persistence_violations.sum()),uncertainty_interval_available=False))
    timings=pd.DataFrame(timings);dailies=pd.concat(daily,ignore_index=True)
    outputs={'observed_vs_frozen_assessments.csv':ass,'final_observed_leaf_endpoints.csv':ends,'consecutive_severity_increments.csv':increments,'observed_window_severity_integrals.csv':windows,'severity_trajectory_metrics.csv':metrics,'continuous_severity_onset_brackets.csv':onset,'severity_timing_metrics.csv':timings,'representative_daily_predictions.csv':dailies}
    for name,frame in outputs.items():frame.to_csv(OUT/name,index=False)
    ass.to_parquet(OUT/'observed_vs_frozen_assessments.parquet',index=False)
    plot(ass,dailies,selected)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    write_json(OUT/'evaluation_receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),model='current frozen overwinter_source_model',biological_parameters_refitted=False,observation_mapping_refitted=False,baseline_training_numeric_original28_only=True,ordinal_Corteva_mapping_resolved=False,prior_validation_outcomes_reused=True,independent_biological_site_identity_confirmed=False,confidence_intervals_reported=False,archived_positive_and_symptom_day_predictions_reproduced_exactly=True,checks=checks,source_sha256=hashes,artifact_rows={k:len(v) for k,v in outputs.items()},elapsed_seconds=time.perf_counter()-start,metric_denominators='coordinate-year/field/source-numbered leaf hierarchy within each stated population',baseline='observed-stage/time comparator is retrospective measurement-time information; no future stage interpolation',publication_figure_claim='plot selection uses counts/span/lexical keys only; no prediction fit selection'))
    print(metrics.loc[metrics.endpoint.isin(['all_assessments','final_numeric_assessment'])&metrics.leaf_scope.eq('all_numbered_leaves'),['partition','endpoint','comparison_population','model','n','fields','coordinate_years','rmse','mae','bias','weighted_R2']].to_string(index=False));print('Timing',timings.loc[timings.timing_definition.eq('existing_zero_positive_sign')&timings.censoring.eq('two_sided')&timings.leaf_scope.eq('top3')].to_string(index=False));print('PASS frozen binary replay, source hashes and evaluation counts',len(ass))

def plot(ass,daily,selected):
    colors={'model':'#0072B2','mean':'#D18F00','stage':'#AD5673'};plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':9})
    fig,axes=plt.subplots(3,3,figsize=(12,8.2),sharey=True)
    for ax,z in zip(axes.flat,selected.itertuples()):
      q=ass.loc[ass.partition.eq(z.partition)&ass.field_id.eq(z.field_id)&ass.leaf_index.eq(z.leaf_index)].sort_values('date');d=daily.loc[daily.partition.eq(z.partition)&daily.field_id.eq(z.field_id)&daily.leaf_index.eq(z.leaf_index)];ax.plot(d.date,d.model_damage_percent,c=colors['model'],lw=1.8,label='Frozen damage ×100');ax.plot(d.date,d.leaf_mean_percent,c=colors['mean'],lw=1.3,ls='--',label='Calibration leaf mean');ax.plot(q.date,q.observed_percent,c='#333333',lw=.8,marker='o',ms=4,label='Observed score');stage=q[q.stage_time_available];ax.scatter(stage.date,stage.stage_time_percent,c=colors['stage'],marker='s',s=20,label='Calibration stage/time');ax.set_ylim(-2,102);ax.grid(axis='y',color='#e8e8e8');ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=4));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'));short=z.field_id.split('|')[1];ax.set_title(f'{z.partition}\nField {short}; source leaf {z.leaf_index+1}; n={len(q)}');ax.tick_params(axis='x',rotation=25)
    for ax in axes[:,0]:ax.set_ylabel('Severity score / modeled damage (%)')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False);fig.text(.5,.01,'Longest observed series selected by count, span and lexical key. Corteva leaf ranks remain unverified. Observed connecting lines are guides.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(OUT/'representative_trajectories.png',dpi=180);fig.savefig(OUT/'representative_trajectories.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharex=True,sharey=True)
    for ax,partition in zip(axes,PARTITIONS):
      q=ass.loc[ass.partition.eq(partition)];ax.scatter(q.observed_percent,q.frozen_damage_percent,s=12,c=colors['model'],alpha=.35,edgecolors='none');ax.plot([0,100],[0,100],c='#666666',ls='--',lw=1);ax.set_xlim(0,100);ax.set_ylim(0,100);ax.set_title(f'{partition}\nn={len(q)} assessments / {q.field_id.nunique()} fields');ax.set_xlabel('Observed severity score (%)');ax.grid(color='#eeeeee')
    axes[0].set_ylabel('Frozen raw damage ×100 (%)');fig.tight_layout();fig.savefig(OUT/'observed_predicted_severity.png',dpi=180);fig.savefig(OUT/'observed_predicted_severity.pdf');plt.close(fig)

if __name__=='__main__':main()
