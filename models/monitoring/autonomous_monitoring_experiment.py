"""Optional field updating of a separately fitted autonomous weather prior."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .autonomous_weather import predict_autonomous, OUTPUTS
from .autonomous_weather_data import AUTONOMOUS
from .autonomous_monitoring import coordinate_oob, monitoring_forecast
from .discrepancy_filter import fit_discrepancy
from .innovation_common import ROOT, load_cohort, register, assess, goal_decision
from .inputs import sha

PRIOR = ROOT/'outputs/disease_improvement/autonomous_weather_20261022'
PREFIX = dict(zip(OUTPUTS, ['baseline', 'shared', 'phase']))
NAMES = [f'weather_prior_{prefix}_{link}' for prefix in PREFIX.values() for link in ['linear', 'logit']]
PRIMARY = 'weather_prior_phase_logit'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/autonomous_monitoring_20261022').resolve()
    receipt = json.loads((PRIOR/'verification.json').read_text())
    independent = json.loads((PRIOR/'independent_verification.json').read_text())
    assert receipt['passed'] and independent['passed'] and receipt['fitted_splits'] == 20
    for name, digest in receipt['artifacts_sha256'].items():
        assert sha(PRIOR/name) == digest, name
    assert not any(d['achieved'] for d in json.loads((PRIOR/'all_goal_decisions.json').read_text()).values())
    codes = [Path(__file__)] + [Path(__file__).with_name(name+'.py') for name in [
        'autonomous_monitoring', 'autonomous_weather', 'phenology_phase_growth', 'discrepancy_filter']]
    registration = register(out, PRIMARY, dict(
        hypothesis='An observation-free global weather prior followed by an optional local residual update avoids fitting weather response with observation-history shortcuts',
        priors='Threefrozenstage31autonomousHGBensembles,noadditionalHGBfitting',
        calibration='OnlymembersEXCLUDINGentirecoordinateyearproducehistoricalOOBforecasts; countzeroNaNneverfittedfallback; onlycoveredrowscalibratewithoriginalrelativeweights',
        oob_limit='Cross-sectionalhistoricalOOBisnotchronologicalvalidation; variableensemblecountnotcalibrateduncertainty',
        correction='ExistingsharedfieldMatern3/2+same-nominal-codeOU,fourpositiveparameters,unitnugget; fixedlinear/logitlinks(.005endpointclip),original60iterationL-BFGS-Bboundsandshrinkage',
        updating='StrictlyearlierOWNFIELDonly; allsame-daycodeswithheld; firstvisitexactautonomousprior; noactualBBCH',
        climate='No-observationclimaterunequalsseparatelyevaluatedstage31prior; monitoringposteriorRMSEcannotestablishclimateskill',
        selection='OnecandidatepassesALLoriginaltemporalgoalgatesbeforespatial; unchangedthresholds/weights/rows/memberships',
        satellite=False, external_validation_execution_prerequisite=False), codes)
    for path in [*PRIOR.glob('*_model.joblib'), *[PRIOR/name for name in ['registration.json', 'verification.json',
            'independent_verification.json', 'assessment_predictions.csv', 'spatial_predictions.csv']],
            AUTONOMOUS/'predictors.npz']:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, _, _, frames, splits, position = load_cohort()
    with np.load(AUTONOMOUS/'predictors.npz') as arrays:
        np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
        base, augmented, phase = [arrays[key] for key in ['baseline', 'augmented', 'phase']]
    parents = {label: pd.read_csv(PRIOR/name, float_precision='round_trip').set_index('global_target_index')
        for label, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv', index=False)
            metrics, _ = assess({'temporal': frame}, PRIMARY, out,
                extra_models=[n for n in [*NAMES, *OUTPUTS] if n != PRIMARY])
            severe = pd.read_csv(out/'subgroup_metrics.csv'); severe = severe.loc[severe.population.eq('severe_above75')]
            decisions = {n: goal_decision(metrics, severe, n) for n in NAMES}
            temporal = ['temporal_ten_percent_rmse', 'temporal_severe_rmse_improves', 'temporal_severe_absolute_bias_improves']
            if not any(all(d['gates'][gate] for gate in temporal) for d in decisions.values()):
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='No single optional-updating candidate passes all original temporal goal gates'), indent=2)+'\n')
                break
        ti, vi = [position.loc[split[key]].to_numpy() for key in ['train', 'test']]
        train = obs.iloc[ti]
        print('start', split['fold'], flush=True)
        prior = joblib.load(PRIOR/f'{split["fold"]}_model.joblib')
        assert prior['train_ids'] == split['train'] and prior['test_ids'] == split['test']
        oob = coordinate_oob(prior, base[ti], augmented[ti], phase[ti], train)
        covered = oob['count'] > 0
        assert covered.sum() > 20
        weights = hierarchical_weights(train)
        fits = {f'weather_prior_{PREFIX[name]}_{link}': fit_discrepancy(train.loc[covered],
            oob['predictions'][name][covered], weights[covered], link) for name in OUTPUTS for link in ['linear', 'logit']}
        fitted = dict(filters=fits, prior_path=str(PRIOR/f'{split["fold"]}_model.joblib'),
            prior_sha256=sha(PRIOR/f'{split["fold"]}_model.joblib'), train_ids=split['train'], test_ids=split['test'],
            calibration_ids=train.loc[covered].global_target_index.astype(int).tolist(), oob=oob,
            training_weights=weights, monitored=True, autonomous_backbone_unchanged=True)
        path = out/f'{split["fold"]}_model.joblib'; joblib.dump(fitted, path)
        context = np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        assert set(obs.iloc[context].global_target_index).isdisjoint(split['train'])
        query_position = pd.Series(np.arange(len(context)), index=context).loc[vi].to_numpy()

        def forecast(bundle):
            assert sha(Path(bundle['prior_path'])) == bundle['prior_sha256']
            backbone = joblib.load(bundle['prior_path'])
            values = predict_autonomous(backbone, base[context], augmented[context], phase[context])
            output = {}
            for prior_name in OUTPUTS:
                for link in ['linear', 'logit']:
                    name = f'weather_prior_{PREFIX[prior_name]}_{link}'
                    output[name] = monitoring_forecast(bundle['filters'][name]['parameters'], obs.iloc[context],
                        values[prior_name], link)[query_position]
                output[prior_name] = values[prior_name][query_position]
            return output

        values, replay = forecast(fitted), forecast(joblib.load(path))
        for name in values:
            np.testing.assert_array_equal(values[name], replay[name])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name in values:
            part[name] = values[name]
        for name in OUTPUTS:
            np.testing.assert_array_equal(part[name], parents[label].loc[split['test'], name])
        parts.setdefault(label, []).append(part)
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            oob_covered=int(covered.sum()), oob_missing=int((~covered).sum()),
            counts=[int(oob['count'].min()), int(oob['count'].max())],
            filters={n: dict(converged=f['converged'], iterations=f['iterations'], objective=f['objective']) for n, f in fits.items()}))
        (out/'fit_progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print('finished', split['fold'], 'seconds', round(time.monotonic()-started, 1), flush=True)
    results = {label: pd.concat(part, ignore_index=True) for label, part in parts.items()}
    for label, frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
    metrics, _ = assess(results, PRIMARY, out, extra_models=[n for n in [*NAMES, *OUTPUTS] if n != PRIMARY])
    groups = pd.read_csv(out/'subgroup_metrics.csv'); severe = groups.loc[groups.population.eq('severe_above75')]
    decisions = {name: goal_decision(metrics, severe, name) for name in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions, indent=2)+'\n')
    verification = dict(passed=True, fitted_splits=len(records), saved_model_replay=True,
        no_backbone_refitting=True, satellite_used=False, goal_achieved=any(d['achieved'] for d in decisions.values()),
        source_unchanged=all(sha(ROOT/p) == digest for p, digest in registration['source_sha256'].items()),
        code_unchanged=all(sha(ROOT/p) == digest for p, digest in registration['code_sha256'].items()),
        artifacts_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert verification['source_unchanged'] and verification['code_unchanged']
    (out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    return metrics, decisions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
