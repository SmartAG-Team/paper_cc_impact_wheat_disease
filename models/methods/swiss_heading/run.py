"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/fip_heading_adaptation_20261011')
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from analysis.methods.swiss_heading import inputs, models, evaluate

def write_lock(paths, target, *, root=inputs.ROOT):
    root = Path(root).resolve()
    records = {}
    for role, path in paths.items():
        path = Path(path).resolve()
        records[role] = dict(path=path.relative_to(root).as_posix(), sha256=inputs.sha(path), bytes=path.stat().st_size)
    result = dict(phase='development_complete_before_evaluation_label_scoring', created_utc=datetime.now(timezone.utc).isoformat(), files=records)
    inputs.save(target, result)
    return result

def check_lock(path, *, root=inputs.ROOT):
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def develop(output=inputs.RESULTS):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    locked = output / 'development_lock.json'
    if locked.exists():
        check_lock(locked)
        return json.loads((output / 'selection.json').read_text())
    registry, _, parent = inputs.prepare_predictors(output)
    labels, ledger = inputs.training_labels()
    if len(labels) != 1456 or len(ledger) != 2222 or ledger.event_after_horizon.any():
        raise ValueError('Changed development label eligibility')
    labels.to_csv(output / 'development_labels.csv', index=False)
    ledger.to_csv(output / 'development_eligibility.csv', index=False)
    selection, cross_validation, folds = models.cross_validate(labels, registry, parent)
    fits = {kind: models.fit(kind, labels, registry, parent) for kind in models.KINDS}
    predictions = np.stack([models.predict(fits[kind], registry, parent) for kind in models.KINDS])
    inputs.save(output / 'selection.json', selection)
    inputs.save(output / 'fits.json', fits)
    inputs.save(output / 'cross_validation_fits.json', folds)
    cross_validation.to_csv(output / 'cross_validation_scores.csv', index=False)
    np.save(output / 'prediction_cdfs.npy', predictions, allow_pickle=False)
    inputs.save(output / 'prediction_models.json', list(models.KINDS))
    names = ['registry.csv', 'weather_eligibility.csv', 'parent_cdf.npy', 'prefix_verification.json', 'development_labels.csv', 'development_eligibility.csv', 'selection.json', 'fits.json', 'cross_validation_fits.json', 'cross_validation_scores.csv', 'prediction_cdfs.npy', 'prediction_models.json']
    files = {'result/' + name: output / name for name in names}
    for name in ['protocol.json', 'source_lock.json', 'inputs.py', 'models.py', 'evaluate.py', 'run.py']:
        files['code/' + name] = inputs.HERE / name
    write_lock(files, locked)
    return selection

def score(output=inputs.RESULTS):
    output = Path(output).resolve()
    check_lock(output / 'development_lock.json')
    inputs.check_sources()
    existing = output / 'evaluation_receipt.json'
    if existing.exists():
        receipt = json.loads(existing.read_text())
        if receipt['development_lock_sha256'] != inputs.sha(output / 'development_lock.json'):
            raise ValueError('Evaluation no longer matches its development seal')
        for name, value in receipt['output_sha256'].items():
            if inputs.sha(output / name) != value:
                raise ValueError('Changed completed evaluation output: ' + name)
        return receipt
    labels, ledger = inputs.read_labels(inputs.EVALUATION)
    if len(labels) != 1349 or len(ledger) != 1836:
        raise ValueError('Changed later-year label population')
    training = pd.read_csv(output / 'development_labels.csv', dtype={'genotype_id': str})
    labels['genotype_seen_in_development'] = labels.genotype_id.isin(training.genotype_id)
    labels.to_csv(output / 'evaluation_labels.csv', index=False)
    ledger.to_csv(output / 'evaluation_eligibility.csv', index=False)
    all_ledger = pd.concat([pd.read_csv(output / 'development_eligibility.csv', dtype={'genotype_id': str}).assign(phase='development'), ledger.assign(phase='evaluation')], ignore_index=True)
    all_ledger.to_csv(output / 'all_plot_eligibility.csv', index=False)
    registry = pd.read_csv(output / 'registry.csv')
    arrays = np.load(output / 'prediction_cdfs.npy', allow_pickle=False)
    names = json.loads((output / 'prediction_models.json').read_text())
    predictions = dict(zip(names, arrays, strict=True))
    scores = pd.concat([evaluate.score_labels(training, registry, predictions, phase='development'), evaluate.score_labels(labels, registry, predictions, phase='evaluation')], ignore_index=True)
    scores.to_csv(output / 'event_predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    metrics = evaluate.summarize(scores)
    metrics.to_csv(output / 'metrics.csv', index=False)
    selection = json.loads((output / 'selection.json').read_text())
    selected = selection['selected_model']
    bootstrap, draws, accounting = evaluate.paired_bootstrap(scores, selected)
    bootstrap.to_csv(output / 'bootstrap_summary.csv', index=False)
    draws.to_csv(output / 'bootstrap_draws.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    inputs.save(output / 'bootstrap_accounting.json', accounting)
    support = labels.groupby('year').agg(heading_records=('plot_id', 'size'), genotypes=('genotype_id', 'nunique'), records_seen_genotype=('genotype_seen_in_development', 'sum')).reset_index()
    support['records_new_genotype'] = support.heading_records - support.records_seen_genotype
    support.to_csv(output / 'evaluation_support.csv', index=False)
    held = metrics[metrics.phase.eq('evaluation')]
    annual = held[held.year.ne('pooled')].pivot(index='year', columns='model', values='crps')
    comparison = bootstrap[bootstrap.reference.eq('frozen')].iloc[0]
    criterion = bool(comparison.crps_difference_upper < 0 and (annual[selected] < annual.frozen).all())
    decision = dict(selected_model=selected, pooled_crps_difference=float(comparison.crps_difference), conditional_interval_upper=float(comparison.crps_difference_upper), improvement_in_each_evaluation_year=bool((annual[selected] < annual.frozen).all()), prespecified_heading_improvement_criterion=criterion, scope='Previously unscored later source-year heading evaluation at the same station; later-developed parent and realized weather', prospective_forecast=False, independent_location=False, functional_canopy_validation=False, disease_model_changed=False, climate_adoption=False)
    inputs.save(output / 'decision.json', decision)
    check_lock(output / 'development_lock.json')
    inputs.check_sources()
    output_names = ['evaluation_labels.csv', 'evaluation_eligibility.csv', 'all_plot_eligibility.csv', 'event_predictions.csv.gz', 'metrics.csv', 'bootstrap_summary.csv', 'bootstrap_draws.csv.gz', 'bootstrap_accounting.json', 'evaluation_support.csv', 'decision.json']
    receipt = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(), development_lock_sha256=inputs.sha(output / 'development_lock.json'), protocol_sha256=inputs.PROTOCOL_SHA, development_labels=len(training), evaluation_labels=len(labels), all_source_plots=len(all_ledger), missing_heading=int((~all_ledger.eligible).sum()), event_prediction_rows=len(scores), selected_model=selected, evaluation_used_for_selection=False, output_sha256={name: inputs.sha(output / name) for name in output_names})
    inputs.save(existing, receipt)
    return receipt
