"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/severity_probability_20261011')
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from analysis.methods.severity_probability import data, models, evaluate

def choose(records):
    if not records or not all((np.isfinite(r['rps']) for r in records)):
        raise ValueError('Finite selection scores required')
    return float(min(records, key=lambda r: (r['rps'], -r['ridge']))['ridge'])

def validate_oof(prediction, training):
    if prediction.global_target_index.duplicated().any() or set(prediction.global_target_index) != set(training.global_target_index):
        raise ValueError('OOF predictions must contain every training target exactly once')

def prediction_rows(frame, model, probability, expected):
    out = frame.copy()
    out['model'] = model
    out['prediction'] = expected
    for j in range(4):
        out['p' + str(j)] = probability[:, j]
    return out

def run(output):
    data.check_lock()
    output = Path(output)
    if output.exists():
        raise FileExistsError('New result directory required')
    output.mkdir(parents=True)
    (output / 'fits').mkdir()
    (output / 'inner').mkdir()
    obs, fields, features, names, old = data.load()
    predictions = []
    scores = []
    selections = []
    members = []
    fit_count = 0
    for year in data.YEARS:
        training, test, assignment = data.partition(obs, year)
        members.append(training[['global_target_index', 'field_id', 'site_id', 'season_year']].assign(test_year=year, role='training', inner_fold=training.site_id.map(assignment)))
        members.append(test[['global_target_index', 'field_id', 'site_id', 'season_year']].assign(test_year=year, role='evaluation', inner_fold=-1))
        for family in data.FAMILIES:
            columns = data.columns(family, names)
            candidate_scores = []
            model = 'ordinal_' + family
            for ridge in models.RIDGES:
                inner = []
                for fold in range(3):
                    a = training.loc[training.site_id.map(assignment).ne(fold)]
                    b = training.loc[training.site_id.map(assignment).eq(fold)]
                    if set(a.site_id) & set(b.site_id):
                        raise ValueError('Inner site overlap')
                    fitted = models.fit(data.design(features, a, columns), a, ridge)
                    fitted.update(family=family, feature_columns=columns, test_year=year, inner_fold=fold)
                    data.write(output / 'fits' / f'{year}_{family}_r{ridge}_fold{fold}.json', fitted)
                    fit_count += 1
                    prob, mean = models.predict(fitted, data.design(features, b, columns), b.leaf_index)
                    inner.append(prediction_rows(b, model, prob, mean).assign(ridge=ridge, inner_fold=fold))
                oof = pd.concat(inner, ignore_index=True).sort_values('global_target_index')
                validate_oof(oof, training)
                loss = evaluate.losses(oof.observed_percent, oof[['p0', 'p1', 'p2', 'p3']].to_numpy(float))[0]
                score = dict(test_year=year, family=family, ridge=ridge, rps=float(np.average(loss, weights=data.weights(oof))), n=len(oof))
                scores.append(score)
                candidate_scores.append(score)
                oof.to_csv(output / 'inner' / f'{year}_{family}_r{ridge}.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
            selected = choose(candidate_scores)
            selections.append(dict(test_year=year, family=family, ridge=selected))
            fitted = models.fit(data.design(features, training, columns), training, selected)
            fitted.update(family=family, feature_columns=columns, test_year=year, inner_fold=None)
            data.write(output / 'fits' / f'{year}_{family}_final.json', fitted)
            fit_count += 1
            prob, mean = models.predict(fitted, data.design(features, test, columns), test.leaf_index)
            predictions.append(prediction_rows(test, model, prob, mean))
            print(f'{year} {family}: ridge {selected}; {fit_count} fits complete', flush=True)
        reference = models.fit_reference(training)
        data.write(output / 'fits' / f'{year}_reference.json', reference)
        prob, mean = models.predict_reference(reference, test.leaf_index)
        predictions.append(prediction_rows(test, 'leaf_frequency', prob, mean))
        prior = old.set_index('global_target_index').loc[test.global_target_index]
        for name, column in [('archived_joint', 'joint'), ('archived_leaf_mean', 'fold_leaf_mean'), ('archived_boosting', 'boosting')]:
            predictions.append(prediction_rows(test, name, np.full((len(test), 4), np.nan), prior[column].to_numpy(float)))
    if fit_count != 195:
        raise ValueError('Unexpected model count')
    prediction = pd.concat(predictions, ignore_index=True)
    prediction.to_csv(output / 'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(scores).to_csv(output / 'inner_scores.csv', index=False)
    pd.DataFrame(selections).to_csv(output / 'selections.csv', index=False)
    pd.concat(members, ignore_index=True).to_csv(output / 'membership.csv', index=False)
    tables = evaluate.evaluate(prediction)
    for name, table in tables.items():
        if name == 'paired_draws':
            table.to_csv(output / (name + '.csv.gz'), index=False, compression={'method': 'gzip', 'mtime': 0})
        else:
            table.to_csv(output / (name + '.csv'), index=False)
    decision = evaluate.decide(tables['paired_intervals'])
    data.write(output / 'decision.json', decision)
    shutil.copy2(data.ROOT / data.FILES['archived_onset'], output / 'archived_onset_summary.csv')
    receipt = dict(status='completed', ordinal_fits=fit_count, reference_fits=5, assessments=2018, new_site_assessments=1272, prediction_rows=len(prediction), inner_score_rows=len(scores), bootstrap_rows=len(tables['paired_draws']), decision=decision, protocol_sha256=data.sha(data.HERE / 'protocol.json'), source_lock_sha256=data.sha(data.HERE / 'source_lock.json'), files={str(p.relative_to(output)): data.sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    data.write(output / 'receipt.json', receipt)
    return receipt
