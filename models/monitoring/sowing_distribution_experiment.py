"""Sequential disease forecasts with point or distributed RF sowing development."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .phenology_phase_growth import fit_phase_growth, predict_phase_growth, phase_weights
from .phenology_transfer_experiment import _category, _anchor
from .discrepancy_experiment import forecast_baseline, CACHE
from .weather_sequence_model import design
from .innovation_common import ROOT, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

SOWING = ROOT/'data/disease_improvement/sowing_distribution_v2_20261022'
PHENO = ROOT/'data/disease_improvement/phenology_transfer_20261022'
PRIMARY = 'sowing_distribution_phase'
MODEL_KEYS = {PRIMARY:('integrated','stage'), 'point_sowing_phase':('point','stage'),
    'sowing_distribution_shared':('integrated','shared'), 'point_sowing_shared':('point','shared')}
NAMES = list(MODEL_KEYS)
CONTROL = 'sowing_distribution_unfiltered'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/sowing_distribution_20261022').resolve()
    codes = [Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'sowing_distribution', 'sowing_distribution_data', 'sowing_rf', 'phenology_transfer',
        'phenology_transfer_data', 'phenology_phase_growth', 'phenology_transfer_experiment',
        'discrepancy_experiment', 'discrepancy_filter', 'crossfit_router', 'field_history_models',
        'models', 'robust_models', 'history_interval_hybrid', 'weather_sequence_model', 'dynamic_model']]
    for source in [SOWING, PHENO]:
        verification = json.loads((source/'verification.json').read_text())
        assert verification['passed'] and verification['source_unchanged']
    registration = register(out, PRIMARY, dict(
        hypothesis='Uncertainty in estimated sowing changes crop-development priors and phase-conditioned disease growth',
        sowing_model='Disjointpre2013RF400,priorannualmeanT+latitude,equalstationbootstrapmassinexpectation under sklearn1.9.1,19722observedsowingdates1559stations withdocumentedcoords >50kmfromeverydiseaselocation',
        sowing_prior='Alladmittedobservationssharingquerytreeleaf weightedbyoriginalstationmass withinleaf, equaltreemass;9equalquantilemidpointnodes; nottree-meanvariance orcalibratedintervals',
        controls='Pointdevelopmentatconditionalempiricalmeandate; sameconditionalprior/support/coveragemetadataasnode-integrateddevelopment',
        propagation='Meansof45developmentfeatures plus12stagepassageSDs and8sowing/coverage/geographiccovariates=65;210diseasepredictorsincludingoriginal145',
        traits='Frozenpre2013/DWD50km-disjointdevelopmentprior, correctedofficialBBCH87; no observedstage/anatomicalleafclaim',
        coverage='AtEACHquery,ifanynodeORpoint lackscompletepreviousweather,bothdesignsretainoriginalscenarioandflagcoverage;futureavailabilitycannotalterearlierqueries;all2304targetsretained',
        architecture='BothsharedgrowthandfourTPVphaseexperts;20members80%coordinateyear,200HGBiterationsdepth3/leaves15/min10/L2=3/lr.05;originalweightsbeforephasegates/.01floor',
        formulations=list(MODEL_KEYS),
        fallback='Exactoriginalreference ifnoearlierfieldhistoryorsource-nativeendpointunseeninoutertraining; measuredgrowthanchorrestoredonce',
        inputs='Dayminusoneweatherandearliernative-codeobservations; annualTonlycompletecalendarprecedingsowing; noobserveddiseaselabelsusedinsowing/phenologytransformation',
        limitations='DWDannualTtoERA5queryproducttransfer, unknownhistoricalrelocations/cultivar/actualsowing; feature/gateintegrationisnotfullBayesiandiseasedistribution',
        selection='Spatial iffanyfixedformulationbeatsoriginaltemporalreference; numerical10%/severegatesunchanged; nooutertuning',
        adaptive_extension=True), codes)
    for source in [SOWING, PHENO]:
        registration['source_sha256'].update(json.loads((source/'registration.json').read_text())['source_sha256'])
        for path in sorted(source.iterdir()):
            if path.is_file():
                registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    for path in [CACHE/'registration.json', *sorted(CACHE.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, arrays, history, frames, splits, position = load_cohort()
    raw, measured, available = design(arrays, history, obs.drop(columns='observed_percent'))
    designs, phases = {}, {}
    with np.load(SOWING/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'], obs.global_target_index)
        for kind in ['point','integrated']:
            designs[kind] = np.column_stack([raw, source[kind]])
            phases[kind] = phase_weights(source[kind+'_stage_probability'][:,2])
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv', index=False)
            assess({'temporal':frame}, PRIMARY, out, extra_models=(*NAMES[1:],CONTROL))
            if min(score_predictions(frame,n)['rmse'] for n in NAMES) >= score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Allfourpoint/distributed-sowinggrowthformulations failtemporalreferenceimprovement'), indent=2)+'\n')
                break
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start',split['fold'],flush=True)
        cached = joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert cached['train_ids'] == split['train'] and cached['test_ids'] == split['test']
        base = {k:cached['experts'][k] for k in ['process','first','event','training_ids']}
        w = hierarchical_weights(train); mass = w/w.sum(); y = train.observed_percent.to_numpy(float)/100
        priors = dict(pooled=float(np.average(y,weights=mass)),means={})
        category = _category(train)
        for key in category.unique():
            mask = category.eq(key).to_numpy()
            priors['means'][key] = float((np.sum(mass[mask]*y[mask])+.04*priors['pooled'])/(mass[mask].sum()+.04))
        anchor = _anchor(priors,train,measured[ti],available[ti])
        growth = {}
        for kind in ['point','integrated']:
            growth[kind] = fit_phase_growth(designs[kind][ti],train,anchor,w,phases[kind][ti])
            print('fitted',split['fold'],kind,'seconds',round(time.monotonic()-started,1),flush=True)
        fitted = dict(base=base,growth=growth,priors=priors,supported_categories=category.unique().tolist(),
            train_ids=split['train'],test_ids=split['test'])
        path = out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)

        def forecast(bundle):
            query = obs.iloc[vi].drop(columns='observed_percent')
            baseline = forecast_baseline(bundle['base'],arrays,history,obs,vi)
            anchor = _anchor(bundle['priors'],query,measured[vi],available[vi])
            use = available[vi] & _category(query).isin(bundle['supported_categories']).to_numpy()
            results = {kind:predict_phase_growth(model,designs[kind][vi],anchor,phases[kind][vi],baseline,use)
                for kind,model in bundle['growth'].items()}
            values = {name:results[kind][form] for name,(kind,form) in MODEL_KEYS.items()}
            values[CONTROL],values['uses_growth'] = baseline,use
            return values

        values,replay = forecast(fitted),forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key],replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name in [*NAMES,CONTROL]:
            part[name] = values[name]
        part['sowing_uses_growth'] = values['uses_growth']
        np.testing.assert_allclose(part[CONTROL],part[REFERENCE],rtol=0,atol=1e-11)
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            members_per_representation=20,models_per_member=5,corrected_rows=int(values['uses_growth'].sum())))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results = {label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_ = assess(results,PRIMARY,out,extra_models=(*NAMES[1:],CONTROL))
    groups = pd.read_csv(out/'subgroup_metrics.csv')
    decisions = {n:goal_decision(metrics,groups.loc[groups.population.eq('severe_above75')],n) for n in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,
        fitted_splits=len(records),source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
