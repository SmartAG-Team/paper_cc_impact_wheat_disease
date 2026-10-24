"""Temporal gate calibration using only previously issued earlier-year forecasts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .evaluation import paired_interval, score_predictions, score_subset
from .infection_event_experiment import populations
from .inputs import sha
from .oob_history_blend import apply_weights, fit_weights

ROOT = Path(__file__).resolve().parents[2]
NAME = 'forward_history_blend'


def calibration_rows(forecasts, outer_training_ids, assessment_year):
    dates = pd.to_datetime(forecasts.date, format='mixed', errors='coerce')
    if (not forecasts.global_target_index.is_unique or dates.isna().any()
            or not dates.dt.year.eq(forecasts.season_year).all()):
        raise ValueError('Unique forecast identities and aligned assessment dates required')
    return forecasts.loc[forecasts.global_target_index.isin(outer_training_ids)
                         & forecasts.season_year.lt(assessment_year)].copy()


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    base = root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    parent = root/'outputs/disease_improvement/history_interval_20261022'
    out = Path(output or root/'outputs/disease_improvement/forward_blend_screen_20261022').resolve()
    if out.exists():
        raise FileExistsError('A new output directory is required')
    frame = pd.read_csv(parent/'assessment_predictions.csv')
    obs = pd.read_csv(base/'training_observation_ledger.csv')
    arrays = np.load(base/'predictors.npz')
    names = joblib.load(base/'fitted_candidate.joblib')['history_names']
    history = pd.DataFrame(arrays['history'],columns=names)
    position = pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    all_splits = json.loads((base/'split_membership.json').read_text())
    splits = [s for s in all_splits if s['fold'].startswith('temporal')]
    out.mkdir(parents=True)
    sources = [parent/'assessment_predictions.csv',parent/'verification.json',
               base/'training_observation_ledger.csv',base/'predictors.npz',base/'split_membership.json']
    code = [Path(__file__),Path(__file__).with_name('oob_history_blend.py'),
            Path(__file__).with_name('evaluation.py'),root/'analysis/final_study/metrics.py']
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        role='Temporal screen of one chronological calibration refinement; spatial evaluation conditional on temporal guards',
        rule='Same three history-availability weights and ten-group neutral penalty as prior OOB test; calibration labels only from earlier-year outer-training observations with saved earlier-year forecasts',
        forecast_origin='Frozen parent temporal models; every calibration forecast is itself fitted strictly before its target season',
        unavailable_calibration='The earliest evaluated year retains .5;2014 and18qualified additions lack saved forward forecasts and do not enter gate calibration',
        original_component_training_observations=2304, original_evaluation_observations=2017,
        temporal_screen_to_spatial='No worse overall RMSE, observed-window integral RMSE, severe absolute bias or cold/warm/BASF/Corteva RMSE relative to history_interval',
        component_refitting=False, posthoc_development=True, hyperparameter_search=False,
        independent_external_validation=False,publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)):sha(p) for p in code})
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for p in code:
        dest = out/'code_snapshot'/p.relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
    parts = []
    for split in splits:
        part = frame.set_index('global_target_index').loc[split['test']].reset_index()
        assert part.season_year.nunique()==1
        year = int(part.season_year.iloc[0])
        training = obs.set_index('global_target_index').loc[split['train']]
        assert training.season_year.max() < year
        calibration = calibration_rows(frame,split['train'],year)
        if len(calibration):
            ids = position.loc[calibration.global_target_index]
            h = history.iloc[ids].reset_index(drop=True)
            # Every saved forecast already averages20 complete earlier-year members.
            fitted = fit_weights(calibration.candidate_process,calibration.interval_history_component,
                                 h,calibration,np.full(len(calibration),20))
            fitted.pop('minimum_oob_members')
        else:
            fitted = dict(prior_equivalent_coordinate_years=10.,calibration_training_ids=[],
                          groups={k:dict(history_weight=.5,calibration_rows=0) for k in
                                  ['first_field','same_leaf','other_leaf_only']})
        fitted.update(calibration_scheme='strict earlier-year forecasts',component_members_per_forecast=20,
                      maximum_calibration_year=int(calibration.season_year.max()) if len(calibration) else None)
        (out/f"{split['fold']}_calibration.json").write_text(json.dumps(fitted,indent=2)+'\n')
        calibration.to_csv(out/f"{split['fold']}_calibration_rows.csv",index=False)
        h = history.iloc[position.loc[part.global_target_index]].reset_index(drop=True)
        part[NAME] = apply_weights(fitted,part.candidate_process,part.interval_history_component,h,
                                  part.drop(columns='observed_percent'))
        parts.append(part)
    result = pd.concat(parts,ignore_index=True)
    result.to_csv(out/'assessment_predictions.csv',index=False)
    metrics = pd.DataFrame([dict(model=name,**score_predictions(result,name)) for name in
                            [NAME,'history_interval','reference_replay','current']])
    metrics.to_csv(out/'metrics.csv',index=False)
    low,high = paired_interval(result,NAME,'history_interval')
    (out/'paired_interval.json').write_text(json.dumps(dict(reference='history_interval',lower=low,upper=high),indent=2)+'\n')
    h = history.iloc[position.loc[result.global_target_index]].reset_index(drop=True)
    masks = populations(result,h)
    groups = pd.DataFrame([dict(population=key,model=name,**score_subset(result,mask,name))
        for key,mask in masks.items() for name in [NAME,'history_interval']])
    groups.to_csv(out/'subgroup_metrics.csv',index=False)
    old,new = [score_predictions(result,name) for name in ['history_interval',NAME]]
    gates = dict(rmse_noninferior=new['rmse']<=old['rmse'],integral_noninferior=new['integral_rmse']<=old['integral_rmse'])
    for key in ['cold','warm','BASF','Corteva']:
        gates[key+'_noninferior'] = score_subset(result,masks[key],NAME)['rmse']<=score_subset(result,masks[key],'history_interval')['rmse']
    gates['severe_bias_noninferior'] = abs(score_subset(result,masks['severe_above75'],NAME)['bias'])<=abs(score_subset(result,masks['severe_above75'],'history_interval')['bias'])
    (out/'screen_decision.json').write_text(json.dumps(dict(proceed_to_spatial=all(gates.values()),gates=gates,
        independent_external_validation=False,publication_model_promoted=False),indent=2)+'\n')
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(run(output=args.output).to_string(index=False))
