"""First-visit distributional forecasts with fixed reference or phenology monitoring."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .cold_ordinal import fit_cold_ordinal, predict_cold_ordinal
from .phenology_phase_growth import phase_weights, predict_phase_growth
from .phenology_transfer_experiment import TRAIT_DATA, _anchor, _category
from .discrepancy_experiment import forecast_baseline
from .weather_sequence_model import design
from .innovation_common import ROOT, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

PHENO_MODELS = ROOT/'outputs/disease_improvement/phenology_transfer_20261022'
PRIMARY = 'cold_ordinal_phenology'
NAMES = [PRIMARY, 'cold_mean_phenology', 'cold_ordinal_reference', 'cold_mean_reference']
CONTROL = 'cold_ordinal_unfiltered'
PHENO_CONTROL = 'cold_ordinal_phenology_control'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/cold_ordinal_20261022').resolve()
    codes = [Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'cold_ordinal', 'phenology_phase_growth', 'phenology_transfer_experiment',
        'phenology_transfer', 'phenology_transfer_data', 'discrepancy_experiment', 'discrepancy_filter',
        'crossfit_router', 'field_history_models', 'models', 'robust_models', 'history_interval_hybrid',
        'weather_sequence_model', 'dynamic_model']]
    registration = register(out, PRIMARY, dict(
        hypothesis='A jointly learned bounded distribution supplies more useful first-visit severity means than direct squared-error fitting',
        population='Only no-earlier-field-history training observations fit the cold heads; all original assessment records and scientific weights retained',
        predictors='Frozen21traitrepresentation204features plus nominal source-native onehots; no actual sowing/cultivar/anatomical mapping claim',
        encoding='Randomized adjacent-grid severity score at0,5,...100: soft exceedance q=clip((y-(upper_threshold-5))/5,0,1); 5sum(q)=y exactly',
        loss='Joint binary proper logloss;40uniform feature/threshold/binary rows peroriginal target, opposite binary labels carry w*q/20 andw*(1-q)/20; totaloriginalmassexact',
        meaning='Encoded-grid exceedance probabilities, not assumed calibrated continuous severity probabilities; repeated cases encode loss, not new measurements',
        architecture='20members80%coldcoordinateyear, seed20261022; one joint HGBClassifier threshold input constrained decreasing;200iterations/depth3/leaves15/lr.05/L2=3/minleaf80encodedrows',
        control='Directmean HGB200/depth3/leaves15/lr.05/L2=3/minleaf10 onidenticaloriginalsamples; architecture/encodingdiffers, so not a loss-only causal contrast',
        mean='5sum(predicted encoded exceedance probabilities), clipped0..100; per-member directmeans clipped before category-supported ensemble average',
        warm='Two fixed components: original reference or previouslyregistered21phasegrowth, not selected or fitted on current outer predictions',
        fallback='Warm rows preserve selected warm component; cold rows without category-supported members preserve exactoriginalreference; no target deletion',
        information='Dayminusoneweather/traitprefix and strictly earlier field labels in fixed monitoring component; all same-day outcomes withheld',
        selection='Spatial iff anyof4registered formulations improves original temporalreference; original10%andseveregatesunchanged',
        adaptive_extension=True), codes)
    registration['source_sha256'].update(json.loads((TRAIT_DATA/'registration.json').read_text())['source_sha256'])
    for path in [*sorted(TRAIT_DATA.iterdir()), PHENO_MODELS/'registration.json', *sorted(PHENO_MODELS.glob('*_model.joblib'))]:
        if path.is_file():
            registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, arrays, history, frames, splits, position = load_cohort()
    raw, measured, available = design(arrays, history, obs.drop(columns='observed_percent'))
    with np.load(TRAIT_DATA/'predictors.npz') as data:
        np.testing.assert_array_equal(data['target_ids'], obs.global_target_index)
        x = np.column_stack([raw, data['phenology']])
        phase = phase_weights(data['stage_probability'][:,2])
    parts, records, probabilities, started = {}, [], [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame}, PRIMARY, out, extra_models=(*NAMES[1:],CONTROL,PHENO_CONTROL))
            if min(score_predictions(frame,n)['rmse'] for n in NAMES) >= score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='All registered first-visit ordinal/mean formulations fail temporalreference improvement'),indent=2)+'\n')
                break
        ti,vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]; assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start',split['fold'],flush=True)
        warm = joblib.load(PHENO_MODELS/f'{split["fold"]}_model.joblib')
        assert warm['train_ids'] == split['train'] and warm['test_ids'] == split['test']
        cold = fit_cold_ordinal(x[ti],train,hierarchical_weights(train),available[ti])
        fitted = dict(cold=cold,warm=warm,train_ids=split['train'],test_ids=split['test'])
        path = out/f'{split["fold"]}_model.joblib'; joblib.dump(fitted,path)

        def forecast(bundle):
            query = obs.iloc[vi].drop(columns='observed_percent')
            reference = forecast_baseline(bundle['warm']['base'],arrays,history,obs,vi)
            active = available[vi] & _category(query).isin(bundle['warm']['supported_categories']).to_numpy()
            anchored = _anchor(bundle['warm']['priors'],query,measured[vi],available[vi])
            prior = predict_phase_growth(bundle['warm']['growth'],x[vi],anchored,phase[vi],reference,active)['stage']
            first = predict_cold_ordinal(bundle['cold'],x[vi],query,reference,available[vi])
            return dict(cold_ordinal_phenology=np.where(available[vi],prior,first['ordinal']),
                cold_mean_phenology=np.where(available[vi],prior,first['mean']),
                cold_ordinal_reference=first['ordinal'],cold_mean_reference=first['mean'],
                cold_ordinal_unfiltered=reference,cold_ordinal_phenology_control=prior,
                member_count=first['member_count'],probability=first['threshold_probability'])

        values,replay = forecast(fitted),forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key],replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name in [*NAMES,CONTROL,PHENO_CONTROL]:
            part[name] = values[name]
        part['cold_ordinal_members'] = values['member_count']
        np.testing.assert_allclose(part[CONTROL],part[REFERENCE],atol=1e-11,rtol=0)
        parts.setdefault(label,[]).append(part)
        for j,upper in enumerate(cold['thresholds']):
            probabilities.append(pd.DataFrame(dict(evaluation=label,fold=split['fold'],
                global_target_index=split['test'],encoded_grid_upper_percent=upper,
                encoded_exceedance_probability=values['probability'][:,j],supported_members=values['member_count'])))
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            cold_training_rows=len(cold['cold_training_ids']),members=len(cold['members']),
            changed_first_rows=int((values['member_count']>0).sum())))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results = {label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    pd.concat(probabilities,ignore_index=True).to_parquet(out/'encoded_grid_probabilities.parquet',index=False)
    metrics,_ = assess(results,PRIMARY,out,extra_models=(*NAMES[1:],CONTROL,PHENO_CONTROL))
    groups = pd.read_csv(out/'subgroup_metrics.csv')
    decisions = {n:goal_decision(metrics,groups.loc[groups.population.eq('severe_above75')],n) for n in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,
        fitted_splits=len(records),source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path); args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
