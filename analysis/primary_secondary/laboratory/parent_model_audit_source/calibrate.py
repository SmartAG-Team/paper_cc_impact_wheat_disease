"""Grouped development-era field calibration; independent evidence is separate."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from .core import Parameters, simulate
from .field_data import prepare_basf, hierarchical_weights

ROOT = Path(__file__).resolve().parents[2]
MECHANISTIC = ['primary_secondary_hidden','primary_only_hidden','secondary_only_hidden',
               'primary_secondary_no_hidden','primary_secondary_no_weather']
BASELINES = ['persistence','global_logit_trend','thermal_logit_trend','canopy_logit_trend']


def make_folds(episodes):
    all_ids = set(episodes.series_id)
    folds = []
    for year in (2018,2019):
        held_locations = set(episodes.loc[episodes.year.eq(year),'location_id'])
        train = set(episodes.loc[episodes.year.lt(year) & ~episodes.location_id.isin(held_locations),'series_id'])
        test = set(episodes.loc[episodes.year.eq(year),'series_id'])
        if train and test:
            folds.append({'name':f'forward_{year}','type':'forward_year','train':train,'test':test})
    for country in sorted(episodes.Country.unique()):
        test = set(episodes.loc[episodes.Country.eq(country),'series_id'])
        locations = set(episodes.loc[episodes.Country.eq(country),'location_id'])
        train = set(episodes.loc[~episodes.Country.eq(country) & ~episodes.location_id.isin(locations),'series_id'])
        if train and test:
            folds.append({'name':'country_'+country.lower().replace(' ','_'),
                          'type':'country_holdout','train':train,'test':test})
    return folds


def inner_folds(episodes,k=3):
    locations = sorted(episodes.location_id.unique())
    k = min(k,len(locations))
    if k<2:
        return []
    # Stable location assignment, with all years/leaves at the location together.
    assignment = {location:i%k for i,location in enumerate(locations)}
    group = episodes.location_id.map(assignment)
    return [(set(episodes.loc[group.ne(i),'series_id']),set(episodes.loc[group.eq(i),'series_id']))
            for i in range(k)]


def score(frame):
    error = frame.predicted_percent-frame.observed_percent
    working = frame[['coordinate_year','series_id']].copy()
    working['squared'] = error**2
    working['absolute'] = abs(error)
    working['signed'] = error
    grouped = working.groupby(['coordinate_year','series_id']).mean(numeric_only=True)
    units = grouped.groupby('coordinate_year').mean()
    return {'rmse_pp':float(np.sqrt(units.squared.mean())), 'mae_pp':float(units.absolute.mean()),
            'bias_pp':float(units.signed.mean()), 'n_coordinate_years':len(units),
            'n_series':frame.series_id.nunique(), 'n_targets':len(frame)}


def target_values(batch,trajectory,observation='damage'):
    rows = batch.targets.loc[~batch.targets.conditioning]
    grid = getattr(trajectory,observation)
    return grid[rows.episode_index.to_numpy(), rows.day.to_numpy(), rows.target_leaf_index.to_numpy()]


def mechanism_prediction(batch,fit,time_step=.25):
    p = Parameters(**fit['parameters'])
    return simulate(batch.initial_visible,batch.temperature,batch.humidity_hours,batch.rain,
                    p,initial_latent=fit['initial_latent'],time_step=time_step,
                    leaf_ranks=batch.ranks,leaf_active=batch.active)


def fit_mechanism(batch,name,latent_days=20.,infectious_days=21.,starts=6,seed=20261004,
                  fixed_alpha=None,fixed_beta=None):
    rows = batch.targets.loc[~batch.targets.conditioning]
    weights = hierarchical_weights(rows)
    target = rows.fraction.to_numpy()
    enabled = np.array([name!='secondary_only_hidden', name!='primary_only_hidden',
                        name!='primary_secondary_no_hidden'])
    if fixed_alpha is not None:
        enabled[0] = False
    if fixed_beta is not None:
        enabled[1] = False
    scales = np.array([.25,10.,.8])
    fixed = np.zeros(3)
    if fixed_alpha is not None: fixed[0] = fixed_alpha
    if fixed_beta is not None: fixed[1] = fixed_beta
    def unpack(x):
        values = fixed.copy()
        values[enabled] = np.asarray(x)*scales[enabled]
        p = Parameters(alpha=float(values[0]),beta=float(values[1]),latent_days=latent_days,
                       infectious_days=infectious_days,weather_response=name!='primary_secondary_no_weather')
        return {'parameters':asdict(p),'initial_latent':float(values[2]),'model':name}
    def objective(x):
        prediction = target_values(batch,mechanism_prediction(batch,unpack(x)))
        return float(np.sum(weights*(prediction-target)**2))
    rng = np.random.default_rng(seed)
    initial = [np.array([.004,.1,.1]),np.array([.08,.3,.5]),np.array([.3,.8,.8])]
    initial += [rng.uniform(.001,.95,3) for _ in range(max(0,starts-len(initial)))]
    candidates = []
    for start in initial[:starts]:
        result = minimize(objective,start[enabled],method='L-BFGS-B',bounds=[(0,1)]*int(enabled.sum()),
                          options={'maxiter':250,'ftol':1e-11,'gtol':1e-6})
        candidates.append(result)
    best = min(candidates,key=lambda x:x.fun)
    fitted = unpack(best.x)
    fitted.update(training_rmse_pp=float(100*np.sqrt(best.fun)),objective=float(best.fun),
                  optimizer_success=bool(best.success),optimizer_message=str(best.message),
                  starts=len(candidates),converged_starts=sum(bool(x.success) for x in candidates),
                  boundary_coordinates=int(np.sum(np.isclose(best.x,0,atol=1e-5)|np.isclose(best.x,1,atol=1e-5))))
    return fitted


def baseline_prediction(batch,fit):
    rows = batch.targets.loc[~batch.targets.conditioning]
    episodes = rows.episode_index.to_numpy()
    days = rows.day.to_numpy()
    target_ranks = rows.target_leaf_index.to_numpy()
    initial = batch.initial_visible[episodes,target_ranks]
    name = fit['model']
    if name=='persistence':
        return initial
    exposure = days.astype(float)
    if name=='thermal_logit_trend':
        cumulative = np.column_stack([np.zeros(len(batch.episodes)),
                                      np.cumsum(np.maximum(batch.temperature,0)/18,axis=1)])
        exposure = cumulative[episodes,days]
    slope = fit['slope']
    if name=='canopy_logit_trend':
        canopy = batch.initial_visible.sum(axis=1)/batch.active.sum(axis=1)
        slope = slope+fit['canopy_slope']*canopy[episodes]
    return expit(logit(np.clip(initial,.005,.995))+slope*exposure)


def fit_baseline(batch,name):
    rows = batch.targets.loc[~batch.targets.conditioning]
    weights = hierarchical_weights(rows)
    target = rows.fraction.to_numpy()
    if name=='persistence':
        return {'model':name}
    def objective(x):
        fit = {'model':name,'slope':float(np.atleast_1d(x)[0])}
        if name=='canopy_logit_trend': fit['canopy_slope']=float(np.atleast_1d(x)[1])
        return float(np.sum(weights*(baseline_prediction(batch,fit)-target)**2))
    if name=='canopy_logit_trend':
        results = [minimize(objective,start,method='L-BFGS-B',bounds=[(-.25,.25),(-.5,.5)])
                   for start in ([.05,0],[0,.1],[.1,-.1])]
        result = min(results,key=lambda x:x.fun)
        return {'model':name,'slope':float(result.x[0]),'canopy_slope':float(result.x[1]),
                'training_rmse_pp':float(100*np.sqrt(result.fun)),'optimizer_success':bool(result.success)}
    result = minimize_scalar(objective,bounds=(-.25,.25),method='bounded',options={'xatol':1e-9})
    candidates = [(result.x,result.fun),(-.25,objective(-.25)),(.25,objective(.25))]
    slope,value = min(candidates,key=lambda x:x[1])
    return {'model':name,'slope':float(slope),'training_rmse_pp':float(100*np.sqrt(value)),
            'optimizer_success':bool(result.success)}


def prediction_frame(batch,fit):
    frame = batch.targets.loc[~batch.targets.conditioning].copy()
    frame['observed_percent'] = frame.Value
    if fit['model'] in BASELINES:
        values = baseline_prediction(batch,fit)
    else:
        trajectory = mechanism_prediction(batch,fit)
        values = target_values(batch,trajectory)
        ep,day,leaf = [frame[x].to_numpy(int) for x in ('episode_index','day','target_leaf_index')]
        for origin,label in enumerate(('unresolved_initial','external','secondary')):
            frame[label+'_infected_percent'] = 100*trajectory.origin_total[ep,day,leaf,origin]
        frame['infectious_percent'] = 100*trajectory.infectious[ep,day,leaf]
        frame['pycnidia_proxy_percent'] = 100*trajectory.pycnidia[ep,day,leaf]
    frame['predicted_percent'] = 100*values
    frame['model'] = fit['model']
    return frame


def select_training_hyperparameters(batch,starts=3):
    records = []
    for latent in (10.,20.,30.):
        for infectious in (14.,21.,28.):
            predictions = []
            for i,(train_ids,test_ids) in enumerate(inner_folds(batch.episodes)):
                fit = fit_mechanism(batch.subset(train_ids),'primary_secondary_hidden',latent,infectious,starts)
                predictions.append(prediction_frame(batch.subset(test_ids),fit))
            if predictions:
                metrics = score(pd.concat(predictions,ignore_index=True))
                records.append({'latent_days':latent,'infectious_days':infectious,**metrics})
    if not records:
        return (20.,21.),pd.DataFrame()
    table = pd.DataFrame(records).sort_values(['rmse_pp','latent_days','infectious_days'])
    best = table.iloc[0]
    return (float(best.latent_days),float(best.infectious_days)),table


def cluster_comparisons(predictions,draws=5000):
    """Conditional prediction-error uncertainty; not parameter or process CIs."""
    records = []
    rng = np.random.default_rng(20261004)
    for fold,frame in predictions.groupby('fold'):
        unit_errors = {}
        for model,sub in frame.groupby('model'):
            sub = sub.copy()
            sub['sq'] = (sub.predicted_percent-sub.observed_percent)**2
            unit_errors[model] = sub.groupby(['coordinate_year','series_id']).sq.mean().groupby('coordinate_year').mean()
        for candidate in ('primary_secondary_hidden','primary_secondary_selected_inner'):
            if candidate not in unit_errors: continue
            for baseline in BASELINES:
                shared = unit_errors[candidate].index.intersection(unit_errors[baseline].index)
                a,b = unit_errors[candidate].loc[shared].to_numpy(),unit_errors[baseline].loc[shared].to_numpy()
                choices = rng.integers(0,len(shared),(draws,len(shared)))
                # Improvement is baseline RMSE minus mechanistic RMSE.
                differences = np.sqrt(b[choices].mean(axis=1))-np.sqrt(a[choices].mean(axis=1))
                records.append({'fold':fold,'candidate':candidate,'baseline':baseline,'n_clusters':len(shared),
                    'rmse_improvement_pp':float(np.sqrt(b.mean())-np.sqrt(a.mean())),
                    'ci_lower_pp':float(np.quantile(differences,.025)),
                    'ci_upper_pp':float(np.quantile(differences,.975)),
                    'interval_scope':'coordinate-year bootstrap, conditional on fitted models; not untouched test evidence'})
    return pd.DataFrame(records)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='analysis/primary_secondary/calibration_v1')
    parser.add_argument('--forward-only',action='store_true')
    parser.add_argument('--starts',type=int,default=6)
    parser.add_argument('--inner-selection',action='store_true')
    args = parser.parse_args()
    out = ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    paths = [ROOT/'data/basf-wheat-diseases.txt', ROOT/'data/era5/daily_weather.parquet',
             ROOT/'data/era5/basf_trial_weather_links.csv',ROOT/'analysis/primary_secondary/evaluation_contract.json']
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    batch = prepare_basf(ROOT)
    batch.episodes.to_csv(out/'episodes.csv',index=False)
    batch.targets.to_csv(out/'target_observations.csv',index=False)
    metrics, fits, predictions, assignments, inner_tables = [],[],[],[],[]
    started = time.monotonic()
    for fold in make_folds(batch.episodes):
        if args.forward_only and fold['type']!='forward_year': continue
        train,test = batch.subset(fold['train']),batch.subset(fold['test'])
        assert not set(train.episodes.location_id).intersection(test.episodes.location_id)
        if fold['type']=='forward_year': assert train.episodes.year.max()<test.episodes.year.min()
        for side,ids in (('train',fold['train']),('test',fold['test'])):
            for series in sorted(ids): assignments.append({'fold':fold['name'],'side':side,'series_id':series})
        names = MECHANISTIC+BASELINES
        fold_fits = [fit_mechanism(train,name,starts=args.starts) if name in MECHANISTIC
                     else fit_baseline(train,name) for name in names]
        if args.inner_selection:
            (latent,infectious),table = select_training_hyperparameters(train)
            table['outer_fold'] = fold['name']
            inner_tables.append(table)
            chosen = fit_mechanism(train,'primary_secondary_hidden',latent,infectious,args.starts)
            chosen['model'] = 'primary_secondary_selected_inner'
            fold_fits.append(chosen)
        for fit in fold_fits:
            frame = prediction_frame(test,fit)
            frame['fold'] = fold['name']
            frame['fold_type'] = fold['type']
            predictions.append(frame)
            row = {'fold':fold['name'],'fold_type':fold['type'],'model':fit['model'],**score(frame),
                   'n_trials':test.episodes.TrialId.nunique()}
            metrics.append(row)
            fits.append({'fold':fold['name'],'fold_type':fold['type'],**fit})
            print(json.dumps(row),flush=True)
        pd.DataFrame(metrics).to_csv(out/'fold_metrics.csv',index=False)
        pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
        (out/'fits.json').write_text(json.dumps(fits,indent=2)+'\n')
    predictions = pd.concat(predictions,ignore_index=True)
    pd.DataFrame(assignments).to_csv(out/'fold_assignments.csv',index=False)
    if inner_tables: pd.concat(inner_tables).to_csv(out/'training_inner_selection.csv',index=False)
    cluster_comparisons(predictions).to_csv(out/'paired_cluster_comparisons.csv',index=False)
    # Final deployment fit is development calibration, not a test.
    full = fit_mechanism(batch,'primary_secondary_hidden',starts=args.starts)
    (out/'all_development_fit.json').write_text(json.dumps(full,indent=2)+'\n')
    trajectory = mechanism_prediction(batch,full)
    finer = mechanism_prediction(batch,full,time_step=.125)
    validation = {'created_utc':datetime.now(timezone.utc).isoformat(),'runtime_seconds':time.monotonic()-started,
        'source_sha256':hashes,'sources_immutable':all(digest(ROOT/p)==h for p,h in hashes.items()),
        'episodes':len(batch.episodes),'targets':len(batch.targets),'scored_targets':int((~batch.targets.conditioning).sum()),
        'max_mass_error':float(abs(trajectory.state.sum(axis=-1)-1).max()),'minimum_state':float(trajectory.state.min()),
        'max_quarter_vs_eighth_day_damage_difference_pp':float(100*abs(trajectory.damage-finer.damage).max()),
        'grouped_training_only':True,'test_states_fitted':False,'primary_secondary_identified':False,
        'natural_primary_onset_validated':False,'external_field_validation_complete':False,
        'publication_ready':False,'claim_scope':'conditional development-era BASF transfer; not full disease or Europe validation'}
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation),flush=True)


if __name__=='__main__': main()
