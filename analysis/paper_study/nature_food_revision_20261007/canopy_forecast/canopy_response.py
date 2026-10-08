"""Calibration-only empirical crop-symptom forecasting, independent of pathogen kinetics."""
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit

FEATURES={'crop_only':['crop_progress','leaf_thermal_age'],'crop_moisture':['crop_progress','leaf_thermal_age','cum_moisture']}

def hierarchical_weights(frame):
 if frame.empty or not {'coordinate_year','field_id','leaf_index'}.issubset(frame):raise ValueError('Nonempty coordinate-year/field/source-leaf metadata required')
 nf=frame.groupby('coordinate_year').field_id.transform('nunique').to_numpy(float);nl=frame.groupby(['coordinate_year','field_id']).leaf_index.transform('nunique').to_numpy(float);nr=frame.groupby(['coordinate_year','field_id','leaf_index']).leaf_index.transform('size').to_numpy(float);w=1/(frame.coordinate_year.nunique()*nf*nl*nr)
 if not np.isclose(w.sum(),1):raise ValueError('Hierarchy weight sum is not one')
 return w

def extract_assessment_features(coordinates,daily):
 forbidden={'value','observed_percent','stage_from','stage_to','GsFrom','GsTo','stage_min_observed'}
 if forbidden&set(coordinates):raise ValueError('Observed stage/outcome columns must stay outside the forecast feature API')
 f,d,l=[coordinates[k].to_numpy(int) for k in ['field_index','day_index','leaf_index']];d=d-1
 if np.any(d<0) or np.any((l<0)|(l>=7)):raise ValueError('Positive1-based days and source leaf ranks0–6 required')
 out=coordinates.copy();out['predicted_stage_code']=daily['predicted_stage_code'][f,d];out['available']=daily['available'][f,d,l];out['crop_progress']=daily['crop_progress'][f,d];out['leaf_thermal_age']=daily['leaf_thermal_age'][f,d,l];out['cum_moisture']=daily['cum_moisture'][f,d,l]
 return out

def _design(features,variant):
 if variant not in FEATURES:raise ValueError('Unregistered empirical feature variant')
 values=features[FEATURES[variant]].to_numpy(float);leaves=features.leaf_index.to_numpy(int);gate=features.available.to_numpy(bool).astype(float)
 if not np.isfinite(values).all() or np.any(values<0) or np.any((leaves<0)|(leaves>=7)):raise ValueError('Finite nonnegative forecast predictors and ordinal ranks0–6 required')
 return np.column_stack([np.eye(7)[leaves],values]),gate

def forecast_percent(features,record):
 X,gate=_design(features,record['variant']);a=np.asarray(record['leaf_intercepts'],float);b=np.asarray(record['nonnegative_slopes'],float)
 if a.shape!=(7,) or b.shape!=(len(FEATURES[record['variant']]),) or not np.isfinite(a).all() or not np.isfinite(b).all() or (b<0).any():raise ValueError('Finite leaf intercepts and nonnegative response slopes required')
 return 100*gate*expit(X@np.concatenate([a,b]))

def fit_response(features,truth,training_ids,variant,penalty):
 ids=np.asarray(training_ids,int)
 if not len(ids) or len(set(ids))!=len(ids):raise ValueError('Unique nonempty calibration fitting identifiers required')
 # Select the authorized rows before touching or checking any numeric target.
 selected=truth.set_index('global_target_index').loc[ids]
 if not selected.partition.eq('calibration').all():raise ValueError('Withheld outcomes cannot enter the original-calibration-only fitter')
 selected_features=features.set_index('global_target_index').loc[ids].reset_index();y=selected.observed_percent.to_numpy(float)/100
 if not np.isfinite(y).all() or np.any((y<0)|(y>1)) or penalty<0:raise ValueError('Valid calibration percentage scores and nonnegative penalty required')
 X,gate=_design(selected_features,variant);w=hierarchical_weights(selected_features);k=len(FEATURES[variant]);beta=np.ones(k);mu=float(w@y);a=np.full(7,float(logit(np.clip(mu,1e-5,1-1e-5))-np.average(X[:,7:],axis=0,weights=w)@beta))
 for leaf in range(7):
  ix=selected_features.leaf_index.eq(leaf).to_numpy()
  if ix.any():a[leaf]=float(logit(np.clip(np.average(y[ix],weights=w[ix]),1e-5,1-1e-5))-np.average(X[ix,7:],axis=0,weights=w[ix])@beta)
 def objective(theta):
  alpha,slopes=theta[:7],theta[7:];sig=expit(X@theta);p=gate*sig;error=p-y;dev=alpha-alpha.mean();loss=float(w@(error*error)+penalty*(np.mean(dev*dev)+np.mean(slopes*slopes)));gradient=2*X.T@(w*error*gate*sig*(1-sig));gradient[:7]+=2*penalty*dev/7;gradient[7:]+=2*penalty*slopes/k;return loss,gradient
 result=minimize(objective,np.concatenate([a,beta]),jac=True,method='L-BFGS-B',bounds=[(None,None)]*7+[(0,None)]*k,options={'maxiter':5000,'ftol':1e-13,'gtol':1e-9,'maxls':40})
 if not result.success:raise RuntimeError('Empirical least-squares optimizer did not converge: '+str(result.message))
 return dict(variant=variant,ridge_penalty=float(penalty),leaf_intercepts=result.x[:7].tolist(),nonnegative_slopes=result.x[7:].tolist(),feature_names=FEATURES[variant],training_rows=len(ids),training_fields=selected_features.field_id.nunique(),training_sites=selected_features.site_id.nunique(),training_coordinate_years=selected_features.coordinate_year.nunique(),training_target_ids=ids.tolist(),optimizer='L-BFGS-B fixed start, analytic fractional-score MSE gradient',iterations=int(result.nit),objective=float(result.fun),converged=bool(result.success),physical_pathogen_parameters_fitted=False)

def validate_site_fold(features,train_ids,test_ids):
 byid=features.set_index('global_target_index');train=byid.loc[train_ids];test=byid.loc[test_ids]
 if set(train_ids)&set(test_ids):raise ValueError('Fold target identifiers overlap')
 if set(train.site_id)&set(test.site_id):raise ValueError('Training and withheld location/site histories overlap')
 if set(train.field_id)&set(test.field_id):raise ValueError('Training and withheld field histories overlap')
