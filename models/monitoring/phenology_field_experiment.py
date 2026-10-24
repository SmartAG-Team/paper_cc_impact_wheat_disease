"""Honest warm-history phenology growth and prequential field discrepancy."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .phenology_phase_growth import phase_weights
from .phenology_warm_oob import fit_warm_growth, predict_warm_growth
from .phenology_transfer_experiment import TRAIT_DATA
from .discrepancy_experiment import forecast_baseline, CACHE
from .discrepancy_filter import fit_discrepancy, predict_discrepancy
from .weather_sequence_model import design
from .innovation_common import ROOT, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY = 'phenology_field_logit'
NAMES = [PRIMARY, 'phenology_field_linear', 'phenology_warm_growth']
CONTROL = 'phenology_field_unfiltered'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/phenology_field_bounds_repair_20261022').resolve()
    codes = [Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'phenology_warm_oob', 'phenology_phase_growth', 'phenology_transfer_experiment',
        'phenology_transfer', 'phenology_transfer_data', 'discrepancy_experiment', 'discrepancy_filter',
        'crossfit_router', 'field_history_models', 'models', 'robust_models', 'history_interval_hybrid',
        'weather_sequence_model', 'dynamic_model']]
    registration = register(out, PRIMARY, dict(
        hypothesis='Conditionally matched growth training and persistent field errors improve the trait-supervised predictor',
        biological_inputs='Frozen stage21 development representation, 145+59=204 predictors and TPV phase gates; no stage22 selection or new disease labels',
        fitting='Only observations with strictly earlier measured field history fit growth heads; original scientific relative weights retained; no outcome-fitted cold anchor',
        architecture='20members80%warm-coordinateyear; each shared+4phase HGB200/depth3/leaves15/min10/L2=3/lr.05/floor.01, seed20261022',
        honest_growth='For each outer-training query, only members excluding its entire coordinateyear and containing its source-native category may predict; mean member-clipped phase growth',
        honest_fallback='No history, unseen member category or zero OOB members uses cached five-inner-coordinate-heldout reference, never the full-outer reference',
        inner_design='Growth OOB excludes coordinateyear; reference inner folds exclude all years at coordinate; neither is additional chronological validation',
        latent_field='Four-parameter shared-field Matern3/2+same-native-code OU with unit nugget, original weighted prequential MSE on honest training errors, L-BFGS-B60 fixed shrinkage',
        formulations='Unfiltered warm-only growth; additive percentage discrepancy; logit discrepancy with .005 link endpoint clip',
        query='Full category-supported member ensemble, causal measured anchors and dayminusone inputs; field discrepancies condition strictly earlier dates, all same-day outcomes withheld',
        fallback='If no category-supported warm growth member at query, all formulations preserve original reference exactly; no record deletion',
        interpretation='Learned field discrepancy is an error proxy, not identified cultivar or protocol; source-native codes have no asserted anatomy',
        selection='Spatial iff any registered formulation improves original temporal reference; unchanged 10% and severe gates, no outer tuning',
        adaptive_extension=True), codes)
    registration['source_sha256'].update(json.loads((TRAIT_DATA/'registration.json').read_text())['source_sha256'])
    for path in [*sorted(TRAIT_DATA.iterdir()), CACHE/'registration.json', *sorted(CACHE.glob('*_model.joblib'))]:
        if path.is_file():
            registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, arrays, history, frames, splits, position = load_cohort()
    raw, measured, available = design(arrays, history, obs.drop(columns='observed_percent'))
    with np.load(TRAIT_DATA/'predictors.npz') as data:
        np.testing.assert_array_equal(data['target_ids'], obs.global_target_index)
        x = np.column_stack([raw, data['phenology']])
        phase = phase_weights(data['stage_probability'][:, 2])
    anchor = 100*measured
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv', index=False)
            assess({'temporal':frame}, PRIMARY, out, extra_models=(*NAMES[1:], CONTROL))
            if min(score_predictions(frame,n)['rmse'] for n in NAMES) >= score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='All registered warm-growth/discrepancy formulations fail temporal reference improvement'), indent=2)+'\n')
                break
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start', split['fold'], flush=True)
        cached = joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert cached['train_ids'] == split['train'] and cached['test_ids'] == split['test']
        parent_oof = .5*(cached['inner_predictions'][:, 0]+cached['inner_predictions'][:, 1])
        # Every fallback has an explicit all-year coordinate exclusion receipt.
        held = []
        for membership in cached['inner_memberships']:
            a = position.loc[membership['train']].to_numpy(); b = position.loc[membership['test']].to_numpy()
            assert not set(zip(obs.iloc[a].latitude,obs.iloc[a].longitude)) & set(zip(obs.iloc[b].latitude,obs.iloc[b].longitude))
            held.extend(membership['test'])
        assert sorted(held) == sorted(split['train'])
        base = {k:cached['experts'][k] for k in ['process','first','event','training_ids']}
        weights = hierarchical_weights(train)
        growth = fit_warm_growth(x[ti], train, anchor[ti], weights, phase[ti], available[ti])
        honest = predict_warm_growth(growth, x[ti], train.drop(columns='observed_percent'),
            anchor[ti], phase[ti], parent_oof, available[ti], oob=True)
        for k, member in enumerate((growth['growth'] or {}).get('members', [])):
            eligible = honest['eligible_members'][:, k]
            assert not set(train.loc[eligible, 'coordinate_year'].astype(str)) & set(member['clusters'])
        corrections = {link:fit_discrepancy(train, honest['stage'], weights, link) for link in ['linear','logit']}
        for correction in corrections.values():
            correction['fitting'] = 'Weighted prequential MSE on warm-coordinateyear-OOB growth with cached inner-coordinate-heldout reference fallback; unit nugget'
        fitted = dict(base=base, growth=growth, corrections=corrections, train_ids=split['train'], test_ids=split['test'],
            inner_memberships=cached['inner_memberships'], honest_parent=parent_oof,
            honest_growth=honest, training_weight=weights)
        path = out/f'{split["fold"]}_model.joblib'; joblib.dump(fitted, path)
        ci = np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        qi = pd.Series(np.arange(len(ci)), index=ci).loc[vi].to_numpy()

        def forecast(bundle):
            parent = forecast_baseline(bundle['base'], arrays, history, obs, ci)
            prior = predict_warm_growth(bundle['growth'], x[ci], obs.iloc[ci].drop(columns='observed_percent'),
                anchor[ci], phase[ci], parent, available[ci])
            values = {link:predict_discrepancy(c['parameters'], obs.iloc[ci], prior['stage'], link)
                for link,c in bundle['corrections'].items()}
            use = prior['member_count'] > 0
            for value in values.values():
                value[~use] = parent[~use]
            values['warm'], values['unfiltered'], values['member_count'] = prior['stage'], parent, prior['member_count']
            return {key:value[qi] for key,value in values.items()}

        values, replay = forecast(fitted), forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key], replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name,key in zip([*NAMES,CONTROL], ['logit','linear','warm','unfiltered'], strict=True):
            part[name] = values[key]
        part['phenology_warm_members'] = values['member_count']
        np.testing.assert_allclose(part[CONTROL], part[REFERENCE], atol=1e-11, rtol=0)
        parts.setdefault(label, []).append(part)
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            warm_training_rows=len(growth['warm_training_ids']), honest_growth_rows=int((honest['member_count']>0).sum()),
            no_oob_growth_rows=int((honest['member_count']==0).sum()), corrected_rows=int((values['member_count']>0).sum())))
        (out/'fit_progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results = {label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label, frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'), index=False)
    metrics,_ = assess(results, PRIMARY, out, extra_models=(*NAMES[1:],CONTROL))
    groups = pd.read_csv(out/'subgroup_metrics.csv')
    decisions = {n:goal_decision(metrics,groups.loc[groups.population.eq('severe_above75')],n) for n in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,
        fitted_splits=len(records),source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
