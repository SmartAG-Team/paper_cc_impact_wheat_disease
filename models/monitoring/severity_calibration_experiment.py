"""Assess calibration without refitting weather models or reading query labels."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .autonomous_weather import OUTPUTS
from .autonomous_weather_data import AUTONOMOUS
from .autonomous_monitoring import coordinate_oob
from .severity_calibration import fit_calibration, predict_calibration
from .innovation_common import ROOT, load_cohort, register, assess, goal_decision
from .evaluation import paired_interval
from .inputs import sha

PRIOR = ROOT/'outputs/disease_improvement/autonomous_weather_20261022'
PREFIX = dict(zip(OUTPUTS, ['baseline', 'shared', 'phase']))
NAMES = [f'calibrated_{prefix}_{kind}' for prefix in PREFIX.values() for kind in ['logit', 'isotonic']]
PRIMARY = 'calibrated_phase_logit'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/severity_calibration_20261022').resolve()
    receipt = json.loads((PRIOR/'verification.json').read_text())
    assert receipt['passed'] and json.loads((PRIOR/'independent_verification.json').read_text())['passed']
    for name, digest in receipt['artifacts_sha256'].items():
        assert sha(PRIOR/name) == digest
    codes = [Path(__file__)] + [Path(__file__).with_name(name+'.py') for name in [
        'severity_calibration', 'autonomous_monitoring', 'autonomous_weather', 'phenology_phase_growth']]
    registration = register(out, PRIMARY, dict(
        recipe='Twofixedseveritymeanmaps(logitlocation/positiveslope,isotonic)foreachoffrozen3autonomouspriors; noHGBfitting',
        calibration='Onlycoordinate-year-excludedmembers; zero-countNaNneverfittedfallback; originalcovered-rowrelativeweights; entirefoldwithoutOOBsupportpreservesuncalibratedpriorviaidentitymap',
        logit='WeightedMSE+1e-4identityridge,endpointclip.005,a[-5,5],logb[log(.1),log(5)],initial0,0,L-BFGS-B100',
        isotonic='Bounded[0,100]weightedmonotoneleastsquares; nearestknotoutsideOOBsupport',
        evaluation='All20originalsplits/all2017recordsperdesign; nooutcome-dependentearlyscreen',
        information='Autonomousfourweatherinputs; noseasonaldisease/BBCH/satellite',
        goals='Originalmonitoringsixgatesunchanged; separateautonomousgatesagainstfreshclimate_reference; severeconditionscannotbeoverridden'), codes)
    paths = [*PRIOR.glob('*_model.joblib'), *[PRIOR/name for name in ['registration.json', 'verification.json',
        'independent_verification.json', 'assessment_predictions.csv', 'spatial_predictions.csv']],
        AUTONOMOUS/'predictors.npz']
    for path in paths:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, _, _, frames, splits, position = load_cohort()
    a = dict(np.load(AUTONOMOUS/'predictors.npz'))
    np.testing.assert_array_equal(a['target_ids'], obs.global_target_index)
    parent_frames = {label: pd.read_csv(PRIOR/name, float_precision='round_trip').set_index('global_target_index')
        for label, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    parts, progress, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        ti = position.loc[split['train']].to_numpy(); train = obs.iloc[ti]
        print('start', split['fold'], flush=True)
        model = joblib.load(PRIOR/f'{split["fold"]}_model.joblib')
        assert model['train_ids'] == split['train'] and model['test_ids'] == split['test']
        oob = coordinate_oob(model, a['baseline'][ti], a['augmented'][ti], a['phase'][ti], train)
        weight = hierarchical_weights(train)
        fits = {f'calibrated_{PREFIX[name]}_{kind}': fit_calibration(oob['predictions'][name],
            train.observed_percent.to_numpy(), weight, oob['count'], kind)
            for name in OUTPUTS for kind in ['logit', 'isotonic']}
        bundle = dict(calibrators=fits, oob=oob, weights=weight, train_ids=split['train'], test_ids=split['test'],
            prior_path=str(PRIOR/f'{split["fold"]}_model.joblib'), information_setting='autonomous')
        path = out/f'{split["fold"]}_model.joblib'; joblib.dump(bundle, path)
        part = frames[label].loc[split['test']].reset_index().copy()
        for source in OUTPUTS:
            raw = parent_frames[label].loc[split['test'], source].to_numpy()
            part[source] = raw
            for kind in ['logit', 'isotonic']:
                name = f'calibrated_{PREFIX[source]}_{kind}'
                predicted = predict_calibration(fits[name], raw)
                replay = predict_calibration(joblib.load(path)['calibrators'][name], raw)
                np.testing.assert_array_equal(predicted, replay)
                part[name] = predicted
        parts.setdefault(label, []).append(part)
        progress.append(dict(fold=split['fold'], seconds=time.monotonic()-started,
            covered=int((oob['count'] > 0).sum()), missing=int((oob['count'] == 0).sum()),
            converged={name: fit['converged'] for name, fit in fits.items()}))
        (out/'fit_progress.json').write_text(json.dumps(progress, indent=2)+'\n')
        print('finished', split['fold'], round(time.monotonic()-started, 1), flush=True)
    results = {label: pd.concat(part, ignore_index=True) for label, part in parts.items()}
    for label, frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
    metrics, _ = assess(results, PRIMARY, out, extra_models=[n for n in [*NAMES, *OUTPUTS] if n != PRIMARY])
    groups = pd.read_csv(out/'subgroup_metrics.csv'); severe = groups.loc[groups.population.eq('severe_above75')]
    original = {name: goal_decision(metrics, severe, name) for name in NAMES}
    autonomous = {name: goal_decision(metrics, severe, name, reference='climate_reference') for name in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(original, indent=2)+'\n')
    (out/'autonomous_goal_decisions.json').write_text(json.dumps(autonomous, indent=2)+'\n')
    intervals = []
    for label, frame in results.items():
        for source in OUTPUTS:
            for kind in ['logit', 'isotonic']:
                name = f'calibrated_{PREFIX[source]}_{kind}'
                low, high = paired_interval(frame, name, source)
                intervals.append(dict(evaluation=label, model=name, reference=source, lower=low, upper=high))
    pd.DataFrame(intervals).to_csv(out/'parent_paired_intervals.csv', index=False)
    verified = dict(passed=True, fitted_splits=20, calibrators=120, saved_model_replay=True,
        no_HGB_refitting=True, information_setting='autonomous', publication_model_promoted=False,
        source_unchanged=all(sha(ROOT/path) == digest for path, digest in registration['source_sha256'].items()),
        code_unchanged=all(sha(ROOT/path) == digest for path, digest in registration['code_sha256'].items()),
        goal_achieved=any(v['achieved'] for v in original.values()),
        artifacts_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert verified['source_unchanged'] and verified['code_unchanged']
    (out/'verification.json').write_text(json.dumps(verified, indent=2)+'\n')
    return metrics, original, autonomous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
