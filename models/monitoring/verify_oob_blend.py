"""Independently replay OOB predictions, calibration arithmetic and held-out scores."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .evaluation import paired_interval, promotion_decision, score_predictions, score_subset
from .field_history_models import predict_field_history, predict_ensemble
from .infection_event_experiment import populations
from .inputs import sha
from .models import predict_candidate
from .oob_history_blend import predict
from .robust_models import predict_cluster_ensemble
from .verify_french_weather import write_verification_receipt

ROOT = Path(__file__).resolve().parents[2]
NAME = 'oob_history_blend'


def verify(root, output):
    root, out = Path(root).resolve(), Path(output).resolve()
    registration = json.loads((out/'registration.json').read_text())
    for path, digest in registration['source_sha256'].items():
        assert sha(root/path) == digest, path
    for path, digest in registration['code_sha256'].items():
        assert sha(root/path) == digest and sha(out/'code_snapshot'/path) == digest, path
    base = root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    events = root/'outputs/disease_improvement/infection_events_20261022'
    parent = root/'outputs/disease_improvement/history_interval_20261022'
    obs = pd.read_csv(base/'training_observation_ledger.csv')
    arrays, event_arrays = np.load(base/'predictors.npz'), np.load(events/'predictors.npz')
    fitted = joblib.load(out/'fitted_candidate.joblib')
    original = joblib.load(parent/'fitted_candidate.joblib')
    assert joblib.hash(fitted['components']) == joblib.hash(original)
    assert fitted['training_ids'] == original['training_ids']
    history = pd.DataFrame(arrays['history'], columns=original['base']['history_names'])
    designs = [arrays['process'], arrays['station'], event_arrays['history_events']]
    positions = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    splits = json.loads((base/'split_membership.json').read_text())
    frames = {label:pd.read_csv(out/filename) for label,filename in
              [('temporal','assessment_predictions.csv'), ('spatial_50km','spatial_predictions.csv')]}
    for label, filename in [('temporal','assessment_predictions.csv'), ('spatial_50km','spatial_predictions.csv')]:
        old = pd.read_csv(parent/filename)
        pd.testing.assert_frame_equal(frames[label][old.columns], old, check_dtype=False)
        assert len(old) == 2017 and old.global_target_index.is_unique
    replayed = 0
    for split in splits+[dict(fold='full', train=obs.global_target_index.astype(int).tolist(), test=[])]:
        fold = split['fold']
        train = positions.loc[split['train']].to_numpy()
        rows = obs.iloc[train].reset_index(drop=True)
        h = history.iloc[train].reset_index(drop=True)
        if fold == 'full':
            models = [original['base']['process'], original['base']['history'], original['events_history']]
        else:
            models = [joblib.load(base/f'{fold}_{key}.joblib') for key in ['process','history']]
            models.append(joblib.load(events/f'{fold}_history.joblib'))
            held = obs.iloc[positions.loc[split['test']]]
            assert rows.season_year.max() < held.season_year.min()
        total, counts = np.zeros((len(rows),2)), np.zeros(len(rows),int)
        assert all(len(ensemble['members']) == 20 for ensemble in models)
        for i in range(20):
            memberships = []
            for ensemble in models:
                assert set(ensemble['training_ids']) == set(rows.global_target_index)
                ids = ensemble['members'][i]['training_ids']
                assert set(ids) <= set(rows.global_target_index)
                inside = rows.global_target_index.isin(ids)
                groups = rows.loc[inside,'coordinate_year']
                np.testing.assert_array_equal(inside, rows.coordinate_year.isin(groups))
                memberships.append(~inside.to_numpy())
            assert all(np.array_equal(memberships[0], m) for m in memberships[1:])
            mask = memberships[0]
            if mask.any():
                q, hh = rows.loc[mask].drop(columns='observed_percent'), h.loc[mask]
                values = [function(ensemble['members'][i], design[train][mask], hh, q) for function,ensemble,design in
                          zip([predict_candidate,predict_field_history,predict_field_history], models, designs)]
                available = hh.prior_assessment_available.eq(1)|hh.other_leaf_available.eq(1)
                total[mask] += np.c_[values[0], np.where(available, values[2], values[1])]
                counts[mask] += 1
                replayed += 3
        mean = np.divide(total, counts[:,None], out=np.full_like(total,np.nan), where=counts[:,None]>0)
        saved = pd.read_csv(out/f'{fold}_oob.csv')
        np.testing.assert_array_equal(saved.global_target_index, rows.global_target_index)
        np.testing.assert_array_equal(saved.oob_members, counts)
        np.testing.assert_allclose(saved[['process','history_component']], mean, equal_nan=True, atol=1e-10)
        np.testing.assert_allclose(saved.observed_percent, rows.observed_percent)
        calibration = json.loads((out/f'{fold}_calibration.json').read_text())
        assert calibration['calibration_training_ids'] == rows.global_target_index.astype(int).tolist()
        w = hierarchical_weights(rows); w = w/w.sum()*rows.coordinate_year.nunique()
        group = np.where(h.prior_assessment_available.eq(1),'same_leaf',
                         np.where(h.other_leaf_available.eq(1),'other_leaf_only','first_field'))
        for key in ['first_field','same_leaf','other_leaf_only']:
            use = (counts>=2)&(group==key)
            delta = mean[use,1]-mean[use,0]
            denominator = float(np.sum(w[use]*delta**2))
            numerator = float(np.sum(w[use]*delta*(rows.observed_percent.to_numpy()[use]-mean[use,0])))
            mass = w[use].sum()
            penalty = 10.*denominator/mass if mass else 0.
            expected = np.clip((numerator+.5*penalty)/(denominator+penalty),0,1) if denominator else .5
            assert calibration['groups'][key]['calibration_rows'] == int(use.sum())
            np.testing.assert_allclose(calibration['groups'][key]['history_weight'], expected, atol=1e-12)
        if fold == 'full':
            assert fitted['calibration'] == calibration
        else:
            label = 'temporal' if fold.startswith('temporal') else 'spatial_50km'
            f = frames[label].set_index('global_target_index').loc[split['test']]
            ht = history.iloc[positions.loc[f.index]]
            names = np.where(ht.prior_assessment_available.eq(1),'same_leaf',
                             np.where(ht.other_leaf_available.eq(1),'other_leaf_only','first_field'))
            weight = np.array([calibration['groups'][name]['history_weight'] for name in names])
            expected = (1-weight)*f.candidate_process.to_numpy()+weight*f.interval_history_component.to_numpy()
            np.testing.assert_allclose(f[NAME], expected, atol=1e-10)
        print('verified', fold, flush=True)
    metrics = pd.read_csv(out/'metrics.csv')
    expected_metrics = {(label,name) for label in frames for name in
        [NAME,'history_interval','reference_replay']+(['current'] if label=='temporal' else [])}
    assert len(metrics)==len(expected_metrics) and set(zip(metrics.evaluation,metrics.model))==expected_metrics
    for row in metrics.itertuples(index=False):
        for key,value in score_predictions(frames[row.evaluation], row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row,key),value,atol=1e-9)
    intervals = pd.read_csv(out/'paired_intervals.csv')
    expected_intervals = {(label,NAME,name) for label in frames for name in
        ['history_interval','reference_replay']+(['current'] if label=='temporal' else [])}
    assert len(intervals)==len(expected_intervals) and set(zip(intervals.evaluation,intervals.model,intervals.reference))==expected_intervals
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose([row.lower,row.upper],paired_interval(frames[row.evaluation],NAME,row.reference),atol=1e-9)
    frame = frames['temporal']
    masks = populations(frame, history.iloc[positions.loc[frame.global_target_index]].reset_index(drop=True))
    subgroups = pd.read_csv(out/'subgroup_metrics.csv')
    expected_groups = {(population,model) for population in masks for model in [NAME,'history_interval','reference_replay']}
    assert len(subgroups)==len(expected_groups) and set(zip(subgroups.population,subgroups.model))==expected_groups
    for row in subgroups.itertuples(index=False):
        for key,value in score_subset(frame,masks[row.population],row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row,key),value,atol=1e-9)
    decisions = json.loads((out/'adoption_decision.json').read_text())
    assert decisions['independent_external_validation'] is False and decisions['publication_model_promoted'] is False
    assert set(decisions['references'])=={'history_interval','current','reference_replay'}
    for reference, saved in decisions['references'].items():
        deltas={key:score_subset(frame,masks[key],NAME)['rmse']-score_subset(frame,masks[key],reference)['rmse']
                for key in ['cold','warm','BASF','Corteva']}
        severe=abs(score_subset(frame,masks['severe_above75'],NAME)['bias'])-abs(score_subset(frame,masks['severe_above75'],reference)['bias'])
        ratio=score_subset(frame,masks['new_site'],NAME)['rmse']/score_subset(frame,masks['new_site'],reference)['rmse']
        expected=promotion_decision(score_predictions(frame,reference),score_predictions(frame,NAME),
            paired_interval(frame,NAME,reference),independent=False,subgroup_deltas=deltas,severe_bias_delta=severe,new_site_ratio=ratio)
        assert saved['gates']==expected['gates'] and saved['promote']==expected['promote']
        np.testing.assert_allclose(saved['severe_absolute_bias_change'],severe,atol=1e-9)
        np.testing.assert_allclose(saved['new_site_rmse_ratio'],ratio,atol=1e-9)
        for key,value in deltas.items():
            np.testing.assert_allclose(saved['subgroup_rmse_deltas'][key],value,atol=1e-9)
    contract = json.loads((out/'fitted_candidate_lock.json').read_text())
    for path,digest in contract['sha256'].items():
        assert sha(out/path) == digest, path
    assert contract['parent_contract_sha256'] == sha(parent/'fitted_candidate_lock.json')
    # A synthetic future-schema API probe checks arithmetic, not forecast accuracy.
    pick = [int(np.flatnonzero(mask)[0]) for mask in [
        history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(0),
        history.prior_assessment_available.eq(1),
        history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(1)]]
    q = obs.iloc[pick].drop(columns='observed_percent').copy()
    q['date'], q['season_year'] = '2020-06-01', 2020
    h = history.iloc[pick]
    p = predict_cluster_ensemble(original['base']['process'], designs[0][pick], h, q)
    a = predict_ensemble(original['base']['history'], designs[1][pick], h, q)
    b = predict_ensemble(original['events_history'], designs[2][pick], h, q)
    selected = np.where(h.prior_assessment_available.eq(1)|h.other_leaf_available.eq(1),b,a)
    weight = np.array([fitted['calibration']['groups'][name]['history_weight'] for name in
                       ['first_field','same_leaf','other_leaf_only']])
    np.testing.assert_allclose(predict(fitted,*[d[pick] for d in designs],h,q), (1-weight)*p+weight*selected, atol=1e-10)
    for bad in [q.assign(source='unknown'),q.assign(leaf_index=7),q.assign(date='2019-06-01',season_year=2019),q.assign(observed_percent=0.)]:
        try:
            predict(fitted,*[d[pick] for d in designs],h,bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Future API failed to reject invalid query')
    result = dict(verified=True, calibration_fits=21, component_member_predictions_replayed=replayed,
        metrics_recomputed=len(metrics), intervals_recomputed=len(intervals), subgroup_rows_recomputed=len(subgroups),
        training_records=2304,evaluation_records_per_condition=2017,original_components_identical=True,
        diagnostic_cross_products_complete=True,adoption_decisions_recomputed=3,
        held_out_labels_used_for_weight_fitting=False, future_schema_probe_is_accuracy_validation=False,
        independent_external_validation=False,publication_model_promoted=False,verifier_sha256=sha(Path(__file__)))
    write_verification_receipt(out,result,Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(ROOT,args.output),indent=2))
