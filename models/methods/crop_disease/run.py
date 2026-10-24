"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_disease_coupling_20261010')
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.methods.crop_disease import adapter, evaluation, inputs
from analysis.methods.seasonal_model.disease.run import predict_record
from analysis.methods.observation_model import observation as measurement
from calibration.seasonal_septoria.infection_events import symptom_brackets, onset_distance
RELATIVE = Path('data/final_study/evidence/phenology_disease_coupling')
HOSTS = ['retained', 'lstm_parent_q50', 'lstm_extended_q25', 'lstm_extended_q50', 'lstm_extended_q75']
LIMITATIONS = ['Retrospective reuse of previously inspected outcomes; no independent confirmatory test.', 'The original disease and field-stage calibration populations include geographic sites recurring in evaluation.', 'Calendar-scenario sowing dates and source-numbered leaves retain their original unresolved semantics.', 'The host coordinate is an empirical development index, not measured thermal accumulation or leaf area.', 'The retained leaf policy has no natural senescence; terminal carry-forward is a structural assumption.', 'One unresolved Corteva assessment extends to day 384; original observed-window endpoints remain primary.', 'Quantile timing settings are structural sensitivities, not calibrated joint trajectory uncertainty.', 'The observation mapping remains fitted to the original host and does not validate changed biological dynamics.', 'Bootstrap intervals condition on frozen fits and previously selected architectures.', 'No independent functional canopy, grain-loss or climate-transfer validation is supplied by this experiment.']

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def check_baseline(observations, raw, symptoms):
    raw, symptoms = (np.asarray(raw), np.asarray(symptoms))
    error = np.abs(raw - observations.frozen_damage_percent.to_numpy())
    if raw.shape != (len(observations),) or not np.isfinite(raw).all() or np.max(error) > 1e-10 or (not np.array_equal(symptoms, observations.frozen_symptom_day.to_numpy())):
        raise ValueError('Original host fails archived assessment or symptom-timing reproduction')
    return dict(assessment_count=len(observations), maximum_damage_difference=float(error.max()), all_symptom_days_equal=True)

def host_coordinates(bundle):
    thresholds = bundle.thresholds
    horizon = bundle.data.temperature.shape[1]
    parent_anchors = [thresholds[s] for s in [10, 31, 51, 85]]
    extended_anchors = [thresholds[s] for s in [10, 31, 37, 39, 51, 85]]
    result = {'retained': bundle.retained_coordinate}
    c = adapter.development_coordinate(bundle.parent_cdf, parent_anchors)
    result['lstm_parent_q50'] = adapter.extend_completed(c, horizon=horizon, terminal_anchor=thresholds[85])
    for q in [0.25, 0.5, 0.75]:
        c = adapter.development_coordinate(bundle.extended_cdf[:, [0, 1, 2, 3, 4, 6]], extended_anchors, quantile=q)
        result[f'lstm_extended_q{int(q * 100)}'] = adapter.extend_completed(c, horizon=horizon, terminal_anchor=thresholds[85])
    return result

def execute(output, root=inputs.ROOT):
    root, output = (Path(root), Path(output))
    protocol_path = root / RELATIVE / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    inputs.check_hashes(protocol['dependency_sha256'], root)
    if inputs.sha(root / protocol['specification']) != protocol['specification_sha256']:
        raise ValueError('Registered specification changed')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite frozen experiment results')
    bundle = inputs.load_inputs(root)
    observations = bundle.observations.copy()
    coordinates = host_coordinates(bundle)
    field, day, leaf = [observations[k].to_numpy(int) for k in ['field_index', 'day_index', 'leaf_index']]
    length = bundle.data.metadata.set_index('field_index').forcing_days
    if np.any(day < 1) or np.any(day > length.reindex(field).to_numpy(int)):
        raise ValueError('Assessment lies outside original forcing')
    intervals = symptom_brackets(observations.assign(value=observations.observed_percent), upper_three=False)
    intervals['partition'] = intervals.field_id.map(observations.groupby('field_id').partition.first())
    intervals['forcing_end_day'] = intervals.field_index.map(length)
    daily_damage, daily_area, symptom_days, diagnostics, stage_rows = ([], [], [], [], [])
    baseline = None
    for name in HOSTS:
        trajectory, host = predict_record(bundle.data, coordinates[name], bundle.thresholds, bundle.fitted)
        raw = 100 * trajectory.damage[field, day, leaf]
        symptom = trajectory.symptom_day[field, leaf]
        if name == 'retained':
            baseline = check_baseline(observations, raw, symptom)
        observations['raw_' + name] = raw
        features = observations.assign(frozen_damage_percent=raw)
        observations['mapped_' + name] = measurement.predict_model(bundle.observation_fit, features)
        observations['host_area_' + name] = host.area[field, day - 1, leaf]
        onset = trajectory.symptom_day[intervals.field_index.to_numpy(int), intervals.leaf_index.to_numpy(int)]
        intervals[name + '_symptom_day'] = onset
        intervals[name + '_distance_days'] = onset_distance(intervals, onset, intervals.forcing_end_day.to_numpy(int))
        daily_damage.append(100 * trajectory.damage)
        daily_area.append(host.area)
        symptom_days.append(trajectory.symptom_day)
        diagnostics.append(dict(host=name, max_tissue_mass_error=float(np.max(np.abs(trajectory.state.sum(axis=-1) - 1))), raw_min=float(raw.min()), raw_max=float(raw.max()), beyond_original_mapping_range=int(((raw < bundle.observation_fit['raw_training_min']) | (raw > bundle.observation_fit['raw_training_max'])).sum())))
        for stage, indices in host.stage_day_index.items():
            stage_rows.extend((dict(host=name, field_id=field_id, legacy_stage_key=int(stage), target_stage=87 if stage == 85 else int(stage), crossing_day_index=int(index)) for field_id, index in zip(bundle.registry.field_id, indices)))
        print('Completed fixed host:', name, flush=True)
    observations['leaf_mean'] = observations.leaf_mean_percent
    cols = ['leaf_mean'] + [prefix + '_' + host for prefix in ['raw', 'mapped'] for host in HOSTS]
    evaluated = observations.loc[observations.partition.isin(measurement.EVALUATION_PARTITIONS)].copy()
    windows, summary = evaluation.summarize(evaluated, cols)
    calibration_windows, calibration_summary = evaluation.summarize(observations.loc[observations.partition.eq('calibration')], cols)
    if len(evaluated) != 2033 or len(windows) != 573 or evaluated.field_id.nunique() != 188:
        raise ValueError('Original evaluation population or observed windows changed')
    old_mapping = pd.read_csv(root / inputs.OBSERVATION / 'reused_assessment_predictions.csv').set_index('global_target_index')
    old_values = old_mapping.loc[evaluated.global_target_index, 'bounded_leaf_raw'].to_numpy(float)
    if not np.allclose(old_values, evaluated.mapped_retained, rtol=0, atol=1e-10):
        raise ValueError('Frozen observation mapping fails archived reproduction')
    comparisons, draws = evaluation.all_comparisons(evaluated, windows, cols)
    decision = evaluation.decide(summary, comparisons)
    onset_rows = []
    eligible_onsets = intervals.loc[intervals.partition.isin(measurement.EVALUATION_PARTITIONS)]
    for scope, part in [('all_available', eligible_onsets), ('genuine_two_sided', eligible_onsets.loc[eligible_onsets.censoring.eq('two_sided')])]:
        for source, selected in evaluation.source_groups(part):
            weights = measurement.hierarchical_weights(selected)
            for host in HOSTS:
                onset_rows.append(dict(scope=scope, source=source, host=host, n=len(selected), sites=selected.site_id.nunique(), interval_distance_days=float(np.average(selected[host + '_distance_days'], weights=weights))))
    inputs.check_hashes(protocol['dependency_sha256'], root)
    output.mkdir(parents=True, exist_ok=True)
    tables = dict(assessment_predictions=observations, window_predictions=windows, summary=summary, calibration_window_predictions=calibration_windows, calibration_summary=calibration_summary, comparisons=comparisons, bootstrap_draws=draws, first_positive_intervals=intervals, onset_summary=pd.DataFrame(onset_rows), stage_crossings=pd.DataFrame(stage_rows), simulation_diagnostics=pd.DataFrame(diagnostics), field_registry=bundle.registry)
    for name, frame in tables.items():
        frame.to_csv(output / (name + '.csv'), index=False)
    np.savez_compressed(output / 'daily_arrays.npz', field_ids=bundle.registry.field_id.to_numpy(str), hosts=np.array(HOSTS), coordinate=np.stack([coordinates[n] for n in HOSTS]), damage_percent=np.stack(daily_damage), host_area=np.stack(daily_area), symptom_day=np.stack(symptom_days))
    write_json(output / 'decision.json', decision)
    write_json(output / 'baseline_reproduction.json', baseline)
    write_json(output / 'protocol.json', protocol)
    receipt = dict(completed_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=inputs.sha(protocol_path), evaluation_assessments=len(evaluated), evaluation_windows=len(windows), original_fields=len(bundle.registry), terminal_carry_forward_fields=bundle.registry.loc[bundle.registry.forcing_days.gt(330), 'field_id'].tolist(), result_sha256={p.name: inputs.sha(p) for p in sorted(output.iterdir()) if p.is_file()}, limitations=LIMITATIONS)
    write_json(output / 'receipt.json', receipt)
    print(json.dumps(dict(advance_host_evaluation=decision['advance_host_evaluation'], baseline=baseline, evaluation_assessments=len(evaluated), evaluation_windows=len(windows))), flush=True)
    return decision
