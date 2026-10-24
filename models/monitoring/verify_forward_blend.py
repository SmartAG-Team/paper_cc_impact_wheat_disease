from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from analysis.final_study.metrics import hierarchical_weights
from analysis.disease_improvement.evaluation import score_predictions,score_subset,paired_interval
from analysis.disease_improvement.infection_event_experiment import populations
from analysis.disease_improvement.verify_french_weather import write_verification_receipt
root=Path(__file__).resolve().parents[2]; out=root/'outputs/disease_improvement/forward_blend_screen_20261022';base=root/'outputs/disease_improvement/knmi_station_candidate_20261022';parent=root/'outputs/disease_improvement/history_interval_20261022'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((out/'registration.json').read_text())
for path,digest in json.loads((out/'dependency_manifest.json').read_text())['files_sha256'].items():assert sha(root/path)==digest,path
for path,digest in reg['source_sha256'].items():assert sha(root/path)==digest,path
for path,digest in reg['code_sha256'].items():assert sha(root/path)==digest and sha(out/'code_snapshot'/path)==digest,path
obs=pd.read_csv(base/'training_observation_ledger.csv');old=pd.read_csv(parent/'assessment_predictions.csv');new=pd.read_csv(out/'assessment_predictions.csv');pd.testing.assert_frame_equal(new[old.columns],old)
splits=[s for s in json.loads((base/'split_membership.json').read_text()) if s['fold'].startswith('temporal')];lookup={s['fold']:s for s in splits};index=obs.set_index('global_target_index');arr=np.load(base/'predictors.npz');names=json.loads((root/'data/disease_improvement/infection_events_20261022/feature_schema.json').read_text())
import joblib
h=pd.DataFrame(arr['history'],columns=joblib.load(base/'fitted_candidate.joblib')['history_names']);positions=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
checked=0
for s in splits:
 yr=index.loc[s['test'],'season_year'].unique();assert len(yr)==1;yr=int(yr[0]);expected=old.loc[old.global_target_index.isin(s['train'])&old.season_year.lt(yr)];saved=pd.read_csv(out/(s['fold']+'_calibration_rows.csv'));pd.testing.assert_frame_equal(saved,expected.reset_index(drop=True),check_dtype=False)
 c=json.loads((out/(s['fold']+'_calibration.json')).read_text());assert c['calibration_training_ids']==expected.global_target_index.tolist()
 for cy in expected.season_year.unique():
  earlier=lookup['temporal'+str(cy)];assert index.loc[earlier['train'],'season_year'].max()<cy;assert set(earlier['train'])<=set(s['train']);assert set(expected.loc[expected.season_year.eq(cy),'global_target_index'])<=set(earlier['test'])
 if len(expected):
  hh=h.iloc[positions.loc[expected.global_target_index]].reset_index(drop=True);weight=hierarchical_weights(expected);weight=weight/weight.sum()*expected.coordinate_year.nunique();labels=np.where(hh.prior_assessment_available.eq(1),'same_leaf',np.where(hh.other_leaf_available.eq(1),'other_leaf_only','first_field'))
  for label in ['first_field','same_leaf','other_leaf_only']:
   m=labels==label;delta=(expected.interval_history_component-expected.candidate_process).to_numpy()[m];den=np.sum(weight[m]*delta**2);num=np.sum(weight[m]*delta*(expected.observed_percent-expected.candidate_process).to_numpy()[m]);mass=weight[m].sum();penalty=10*den/mass if mass else 0;v=np.clip((num+.5*penalty)/(den+penalty),0,1) if den else .5;np.testing.assert_allclose(c['groups'][label]['history_weight'],v,atol=1e-12)
 else:assert all(x['history_weight']==.5 for x in c['groups'].values())
 part=new.set_index('global_target_index').loc[s['test']];hh=h.iloc[positions.loc[part.index]];labels=np.where(hh.prior_assessment_available.eq(1),'same_leaf',np.where(hh.other_leaf_available.eq(1),'other_leaf_only','first_field'));weight=np.array([c['groups'][x]['history_weight'] for x in labels]);np.testing.assert_allclose(part.forward_history_blend,(1-weight)*part.candidate_process+weight*part.interval_history_component,atol=1e-10);checked+=len(part)
metrics=pd.read_csv(out/'metrics.csv');assert len(metrics)==4 and set(metrics.model)=={'forward_history_blend','history_interval','reference_replay','current'}
for r in metrics.itertuples(index=False):
 for k,v in score_predictions(new,r.model).items():
  if v is not None:np.testing.assert_allclose(getattr(r,k),v,atol=1e-9)
hh=h.iloc[positions.loc[new.global_target_index]].reset_index(drop=True);masks=populations(new,hh);groups=pd.read_csv(out/'subgroup_metrics.csv');assert len(groups)==18 and set(zip(groups.population,groups.model))=={(p,m) for p in masks for m in ['forward_history_blend','history_interval']}
for r in groups.itertuples(index=False):
 for k,v in score_subset(new,masks[r.population],r.model).items():
  if v is not None:np.testing.assert_allclose(getattr(r,k),v,atol=1e-9)
a,b=paired_interval(new,'forward_history_blend','history_interval');receipt=json.loads((out/'paired_interval.json').read_text());np.testing.assert_allclose([a,b],[receipt['lower'],receipt['upper']],atol=1e-9)
oldscore,newscore=[score_predictions(new,m) for m in ['history_interval','forward_history_blend']];gates=dict(rmse_noninferior=newscore['rmse']<=oldscore['rmse'],integral_noninferior=newscore['integral_rmse']<=oldscore['integral_rmse'])
for key in ['cold','warm','BASF','Corteva']:gates[key+'_noninferior']=score_subset(new,masks[key],'forward_history_blend')['rmse']<=score_subset(new,masks[key],'history_interval')['rmse']
gates['severe_bias_noninferior']=abs(score_subset(new,masks['severe_above75'],'forward_history_blend')['bias'])<=abs(score_subset(new,masks['severe_above75'],'history_interval')['bias']);decision=json.loads((out/'screen_decision.json').read_text());assert gates==decision['gates'] and decision['proceed_to_spatial']==all(gates.values());assert not decision['publication_model_promoted']
result=dict(verified=True,source_and_code_hashes_match=True,outer_fits=5,calibration_only_earlier_training_labels=True,calibration_forecasts_training_strictly_before_query_year=True,calibration_forecast_training_subsets_of_outer_training=True,evaluation_rows=checked,metrics_recomputed=4,subgroup_rows_recomputed=18,paired_interval_recomputed=True,screen_guards_recomputed=7,proceed_to_spatial=False,independent_external_validation=False,publication_model_promoted=False,verifier_sha256=sha(__file__))
write_verification_receipt(out,result,Path(__file__))
print(json.dumps(result,indent=2))
