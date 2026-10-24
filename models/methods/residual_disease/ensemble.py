"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.methods.severity_probability.data import weights
from analysis.methods.observation_model import observation as metrics
from analysis.methods.residual_disease import study, config, evaluation
REGULARIZATION = 0.001
NAMES = list(config.FAMILY_ORDER)
ENSEMBLES = ['ensemble_composite', 'ensemble_point', 'ensemble_uniform']

def matrices(frame, names, point_only):
    weight = weights(frame)
    weight /= weight.sum()
    x = frame[names].to_numpy(float) / 100
    y = frame.observed_percent.to_numpy(float) / 100
    gram = x.T @ (weight[:, None] * x)
    linear = x.T @ (weight * y)
    constant = float(weight @ y ** 2)
    if not point_only:
        windows = metrics.windows(frame, names)
        weight = weights(windows)
        weight /= weight.sum()
        x = windows[[name + '_audpc' for name in names]].to_numpy() / 3000
        y = windows.observed_audpc.to_numpy() / 3000
        gram += x.T @ (weight[:, None] * x)
        linear += x.T @ (weight * y)
        constant += float(weight @ y ** 2)
    return (gram, linear, constant)

def objective(frame, names, value, *, prior='fold_leaf_mean', point_only=False):
    gram, linear, constant = matrices(frame, names, point_only)
    value = np.asarray(value, float)
    target = np.zeros(len(names))
    target[names.index(prior)] = 1
    delta = value - target
    return (float(value @ gram @ value - 2 * linear @ value + constant + REGULARIZATION * (delta @ delta)), 2 * (gram @ value - linear + REGULARIZATION * delta))

def fit(frame, names, *, prior='fold_leaf_mean', point_only=False):
    if frame.global_target_index.duplicated().any():
        raise ValueError('Duplicate mixture training target')
    gram, linear, constant = matrices(frame, names, point_only)
    target = np.zeros(len(names))
    target[names.index(prior)] = 1

    def loss(value):
        delta = value - target
        return (float(value @ gram @ value - 2 * linear @ value + constant + REGULARIZATION * (delta @ delta)), 2 * (gram @ value - linear + REGULARIZATION * delta))
    result = minimize(loss, target, jac=True, method='SLSQP', bounds=[(0, 1)] * len(names), constraints=[dict(type='eq', fun=lambda value: value.sum() - 1, jac=lambda value: np.ones(len(value)))], options=dict(maxiter=2000, ftol=1e-12))
    if not result.success or np.any(result.x < -1e-10) or abs(result.x.sum() - 1) > 1e-08:
        raise RuntimeError('Mixture fit failed: ' + str(result.message))
    value = np.maximum(result.x, 0)
    value /= value.sum()
    risk, gradient = loss(value)
    gap = float(gradient @ value - gradient.min())
    if gap > 1e-06:
        raise RuntimeError('Mixture simplex optimality gap too large')
    return dict(names=list(names), weights=value.tolist(), point_only=point_only, regularization=REGULARIZATION, prior=prior, objective=risk, simplex_gap=gap, convergence=str(result.message), iterations=int(result.nit), training_targets=sorted(frame.global_target_index.astype(int).tolist()), training_sites=sorted(frame.site_id.unique()), interpretation='Weights and family settings selected from training held-site predictions; no outer outcomes used')

def apply(fitted, frame):
    value = frame[fitted['names']].to_numpy(float) @ np.asarray(fitted['weights'])
    if not np.isfinite(value).all() or np.any(value < -1e-08) or np.any(value > 100 + 1e-08):
        raise ValueError('Unbounded mixture prediction')
    return np.clip(value, 0, 100)

def check_protocol():
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def decide(summary, intervals):
    s = summary.copy()
    p = intervals.copy()
    s['model'] = s.model.replace({'selected_pipeline': 'initial_selected', 'ensemble_composite': 'selected_pipeline'})
    p['candidate'] = p.candidate.replace({'selected_pipeline': 'initial_selected', 'ensemble_composite': 'selected_pipeline'})
    result = evaluation.decide(s, p)
    result.update(primary_candidate='ensemble_composite', adaptive_extension=True)
    return result

def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Empty ensemble destination required')
    protocol = check_protocol()
    output.mkdir(parents=True)
    study.write(output / 'registration.json', dict(started_before_fitting=datetime.now(timezone.utc).isoformat(), protocol_sha256=study.sha(study.HERE / 'ensemble_protocol.json')))
    initial = study.HERE / 'results'
    temporal = pd.read_csv(initial / 'assessment_predictions.csv')
    source = pd.read_csv(initial / 'source_predictions.csv')
    for name in [*[f'year_{y}' for y in study.YEARS], *[f'source_{s}' for s in sorted(source.source.unique())]]:
        root = initial / name
        inner = pd.read_csv(root / 'inner_predictions.csv')
        choices = json.loads((root / 'selected_choices.json').read_text())
        for family in NAMES:
            inner[family] = inner[f"candidate_{choices[family]['candidate_id']}"]
        fitting = pd.read_csv(root / 'training_membership.csv')
        if set(fitting.global_target_index) != set(inner.global_target_index):
            raise ValueError('Changed mixture training population')
        inner.to_csv(output / (name + '_training.csv'), index=False)
        outer = pd.read_csv(root / 'outer/validation_predictions.csv')
        predicted = outer.copy()
        fits = {}
        for candidate, point_only in [('ensemble_composite', False), ('ensemble_point', True)]:
            fits[candidate] = fit(inner, NAMES, point_only=point_only)
            predicted[candidate] = apply(fits[candidate], outer)
        fits['ensemble_uniform'] = dict(names=NAMES, weights=[1 / len(NAMES)] * len(NAMES), fitted=False)
        predicted['ensemble_uniform'] = apply(fits['ensemble_uniform'], outer)
        study.write(output / (name + '_fits.json'), fits)
        predicted.to_csv(output / (name + '_predictions.csv'), index=False)
        frame = temporal if name.startswith('year_') else source
        aligned = predicted.set_index('global_target_index').loc[frame.loc[frame.global_target_index.isin(predicted.global_target_index)].global_target_index]
        for candidate in ENSEMBLES:
            frame.loc[frame.global_target_index.isin(predicted.global_target_index), candidate] = aligned[candidate].to_numpy()
    temporal.to_csv(output / 'assessment_predictions.csv', index=False)
    source.to_csv(output / 'source_predictions.csv', index=False)
    tables = {key: [] for key in ['summary', 'windows', 'discrimination', 'paired_intervals']}
    for population, frame, refs in [('all_future_records', temporal, evaluation.REFERENCES), ('new_sites', temporal.loc[temporal.new_site], evaluation.REFERENCES), ('source_transfer', source, ['fold_leaf_mean', 'joint_base', 'trajectory_base'])]:
        for year in [0] + list(study.YEARS) if population != 'source_transfer' else [0]:
            part = frame if year == 0 else frame.loc[frame.test_year.eq(year)]
            windows, summary = evaluation.summarize(part, ENSEMBLES)
            tables['summary'].append(summary.assign(population=population, test_year=year))
            tables['discrimination'].append(evaluation.aucs(part, ENSEMBLES).assign(population=population, test_year=year))
            if year != 0:
                continue
            windows = metrics.windows(part, ENSEMBLES + refs)
            tables['windows'].append(windows.assign(population=population))
            for endpoint, data in evaluation.endpoint_frames(part, windows, ENSEMBLES + refs):
                for label, sample in evaluation.source_groups(data):
                    for candidate in ENSEMBLES:
                        for reference in refs:
                            tables['paired_intervals'].append(evaluation.paired(sample, candidate, reference, replicates=protocol['bootstrap_replicates']).assign(population=population, endpoint=endpoint, source=label, candidate=candidate, reference=reference, n=len(sample), sites=sample.site_id.nunique()))
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    result['summary'] = pd.concat([pd.read_csv(initial / 'summary.csv'), result['summary']], ignore_index=True)
    result['discrimination'] = pd.concat([pd.read_csv(initial / 'discrimination.csv'), result['discrimination']], ignore_index=True)
    for name, table in result.items():
        table.to_csv(output / (name + '.csv'), index=False)
    study.write(output / 'decision.json', decide(result['summary'], result['paired_intervals']))
    check_protocol()
    study.write(output / 'receipt.json', dict(status='completed', fitted_ensembles=14, untuned_ensembles=7, output_sha256={str(p.relative_to(output)): study.sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
