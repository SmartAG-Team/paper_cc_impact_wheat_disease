"""Retain and independently reconcile archived German station-held-out metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COHORTS = {'calibration': 'train', 'validation': 'val', 'testing': 'test'}
MODELS = ['GDD', 'T-P', 'T-P-V']
KEYS = ['PEP_ID', 'SOWING_DATE', 'BBCH']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_error_metrics(rows, reference):
    error = (rows.predicted_date - rows.true_date).dt.days.dropna().astype(float)
    assert len(error) == reference['matched']
    for name, value in [('mae_days', error.abs().mean()),
                        ('rmse_days', np.sqrt((error**2).mean())),
                        ('bias_days', error.mean())]:
        np.testing.assert_allclose(value, reference[name], rtol=0, atol=1e-12)
    return len(error)


def summarize(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    original = source / 'results/AGC-PhenFormer-v2-final-results-20260926'
    verification = subprocess.run([sys.executable, '-B',
        str(original / 'evaluation_source/verify_results.py')],
        check=True, capture_output=True, text=True, cwd=source)
    verifier_receipt = json.loads(verification.stdout)
    split = json.loads((original / 'configs/station_split.json').read_text())
    parameters = json.loads((original / 'configs/calibration.json').read_text())
    evaluation = json.loads((original / 'configs/evaluation_sites.json').read_text())
    assert set(parameters['fit_station_ids']) == set(split['train'])
    assert parameters['schema_version'] == 2 and parameters['observed_stage_conditioned'] is False
    assert not set(split['train']) & set(split['val'])
    assert not set(split['train']) & set(split['test'])
    assert not set(split['val']) & set(split['test'])
    assert set(evaluation['station_ids']) | set(evaluation['excluded_station_ids']) == set(split['test'])
    assert not set(evaluation['station_ids']) & set(evaluation['excluded_station_ids'])
    assert digest(original / 'configs/calibration.json') == digest(ROOT / 'process_model/parameters/calibration.json')
    output.mkdir(parents=True)
    evidence = output / 'source_evidence'
    evidence.mkdir()
    retained = []

    def retain(path, name):
        target = evidence / name
        shutil.copyfile(path, target)
        assert digest(path) == digest(target)
        retained.append(dict(source_path=str(path), destination_path=str(target.relative_to(output)),
            sha256=digest(path), bytes=path.stat().st_size, byte_identical=True))

    for name in ['calibration.json', 'station_split.json', 'evaluation_sites.json',
                 'calibrated_data_manifest.json', 'gdd_config.csv']:
        retain(original / 'configs' / name, name)
    for path, name in [(original / 'RESULTS_MANIFEST.json', 'RESULTS_MANIFEST.json'),
        (original / 'evaluation_source/seasonal_evaluation.py', 'seasonal_evaluation.py'),
        (original / 'FINAL_DELIVERY.md', 'FINAL_DELIVERY_original.md'),
        (source / 'process_model/calibrate.py', 'calibrate_original.py'),
        (source / 'process_model/forecast_input_contract.md', 'forecast_input_contract_original.md'),
        (source / 'dataset/sowing_records.json', 'sowing_records_original_provenance.json')]:
        retain(path, name)
    extraction = source / 'exports/.build_20260928_sowing_onward/AGC-Transformer-RTX4090-20260928-sowing-onward/dl_model/phenology/seasonal_data.py'
    retain(extraction, 'seasonal_data_original.py')
    table = []
    year_rows = []
    cohort_results = {}
    source_events = []
    cohort_years = {}
    for cohort, group in COHORTS.items():
        folder = original / 'process_models' / cohort
        metrics = json.loads((folder / 'metrics.json').read_text())
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['provenance']['model_source_sha256'] == digest(extraction)
        assert manifest['source_sha256'] == digest(original / 'evaluation_source/seasonal_evaluation.py')
        assert manifest['provenance']['data_contract']['data_sha256']['calibration.json'] == digest(original / 'configs/calibration.json')
        retain(folder / 'metrics.json', cohort + '_metrics_original.json')
        retain(folder / 'manifest.json', cohort + '_manifest_original.json')
        frame = pd.read_csv(folder / 'event_dates.csv', parse_dates=['SOWING_DATE',
            'END_DATE', 'true_date', 'predicted_date'])
        assert not frame.duplicated(KEYS + ['model']).any()
        assert set(frame.model) == set(MODELS)
        expected_sites = set(evaluation['station_ids']) if cohort == 'testing' else set(split[group])
        assert set(frame.PEP_ID) == expected_sites
        assert frame.groupby(KEYS).size().eq(3).all()
        matched = frame.true_date.notna() & frame.predicted_date.notna()
        shared = matched.groupby([frame[k] for k in KEYS]).transform('all')
        assert shared.equals(frame.common_matched)
        calculated = (frame.predicted_date - frame.true_date).dt.days
        np.testing.assert_allclose(calculated, frame.error_days, rtol=0, atol=0, equal_nan=True)
        for model in MODELS:
            selected = frame.loc[frame.model == model]
            for stage in ['overall', 10, 31, 51, 85]:
                rows = selected if stage == 'overall' else selected.loc[selected.BBCH == stage]
                own = metrics['models'][model]['overall' if stage == 'overall' else 'by_stage']
                own = own if stage == 'overall' else own[str(stage)]
                common = metrics['common_matched']['overall' if stage == 'overall' else 'by_stage']
                common = common if stage == 'overall' else common[str(stage)]
                paired = common['models'][model]
                check_error_metrics(rows, own)
                check_error_metrics(rows.loc[rows.common_matched], paired)
                assert int(rows.true_date.notna().sum()) == own['observed']
                assert int((rows.true_date.notna() & rows.predicted_date.isna()).sum()) == own['missing_predictions']
                for denominator, reference in [('own_matched', own), ('three_model_shared', paired)]:
                    table.append(dict(cohort=cohort, model=model, stage=stage,
                        denominator=denominator, matched=reference['matched'],
                        observed=own['observed'], common_matched=common['matched'],
                        coverage_fraction=reference['matched'] / own['observed'],
                        mae_days=reference['mae_days'], rmse_days=reference['rmse_days'],
                        bias_days=reference['bias_days'], missing_predictions=own['missing_predictions']))
        tpv = frame.loc[frame.model == 'T-P-V'].copy()
        frame.to_csv(output / (cohort + '_all_model_event_dates.csv.gz'), index=False,
            compression={'method': 'gzip', 'mtime': 0})
        tpv.to_csv(output / (cohort + '_tpv_event_dates.csv.gz'), index=False,
            compression={'method': 'gzip', 'mtime': 0})
        observed_year = tpv.true_date.dt.year
        cohort_years[cohort] = sorted(observed_year.dropna().astype(int).unique().tolist())
        for year in cohort_years[cohort]:
            rows = tpv.loc[observed_year == year]
            year_rows.append(dict(cohort=cohort, year=year, observed_events=len(rows),
                tpv_matched_events=int(rows.predicted_date.notna().sum()),
                common_matched_events=int(rows.common_matched.sum())))
        coverage = json.loads((folder / 'coverage.json').read_text())
        cycle_table = pd.DataFrame(coverage)
        assert int(cycle_table.eligible.sum()) == metrics['coverage']['eligible_cycles']
        cohort_results[cohort] = dict(**metrics['coverage'],
            observed_events=metrics['models']['T-P-V']['overall']['observed'],
            own_matched=metrics['models']['T-P-V']['overall']['matched'],
            common_matched=metrics['common_matched']['overall']['matched'],
            tpv_own_metrics=metrics['models']['T-P-V']['overall'],
            tpv_shared_metrics=metrics['common_matched']['overall']['models']['T-P-V'],
            observed_years=cohort_years[cohort],
            observed_min=str(tpv.true_date.min().date()), observed_max=str(tpv.true_date.max().date()),
            flagged_observation_order_conflicts=int(cycle_table.observation_order_conflict.sum()),
            ineligible_observed_events=int(tpv.loc[~tpv.eligible].true_date.notna().sum()),
            scored_sites=sorted(expected_sites))
        source_events.append(dict(cohort=cohort, source_path=str(folder / 'event_dates.csv'),
            source_sha256=digest(folder / 'event_dates.csv'), source_rows=len(frame),
            retained_all_models_file=cohort + '_all_model_event_dates.csv.gz',
            retained_tpv_rows=len(tpv), retained_file=cohort + '_tpv_event_dates.csv.gz'))
    metrics_table = pd.DataFrame(table)
    metrics_table.to_csv(output / 'metrics_long.csv', index=False)
    figure_rows = metrics_table.loc[(metrics_table.cohort == 'testing')
        & (metrics_table.denominator == 'three_model_shared')
        & (metrics_table.stage != 'overall')].copy()
    figure_rows.to_csv(output / 'heldout_test_figure_data.csv', index=False)
    pd.DataFrame(year_rows).to_csv(output / 'event_year_overlap.csv', index=False)
    sites = []
    for group, identifiers in split.items():
        for identifier in identifiers:
            sites.append(dict(PEP_ID=identifier, split=group,
                fitted=group == 'train', scored_test=identifier in evaluation['station_ids'],
                excluded_test_scoring=identifier in evaluation['excluded_station_ids']))
    pd.DataFrame(sites).sort_values('PEP_ID').to_csv(output / 'station_split_summary.csv', index=False)
    summary = dict(status='verified', outcome='German winter-wheat BBCH event dates',
        geographic_scope='German phenology network; not European Septoria disease outcomes',
        units='calendar days', BBCH85='soft dough',
        source_result_root=str(original), source_archive_updated='2026-09-28',
        original_verification=verifier_receipt,
        split_counts={key: len(values) for key, values in split.items()},
        scored_test_stations=len(evaluation['station_ids']),
        excluded_reserved_test_stations=len(evaluation['excluded_station_ids']),
        split_disjoint=True, calibration_fit_matches_training_station_ids=True,
        calibration_matches_copied_model=True, observed_stage_conditioned=False,
        heldout_dimension='station', withheld_year_validation=False,
        test_years_absent_from_training=sorted(set(cohort_years['testing']) - set(cohort_years['calibration'])),
        cohort_results=cohort_results, recomputed_numeric_metrics=len(table) * 3,
        temporal_interpretation='Known-sowing, realized-weather retrospective hindcast',
        matching_rule='Exact station/sowing-cycle/BBCH keys; first daily threshold crossing; no observed-date nearest-hit selection',
        scope_limit='German phenology evidence does not validate the European Septoria disease model, cultivar transfer, or operational issue-time forecasts.')
    (output / 'validation_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    provenance = dict(source_root=str(source), source_files_retained=retained,
        source_event_subsets=source_events,
        verification_command=[sys.executable, '-B', str(original / 'evaluation_source/verify_results.py')],
        original_verifier_stdout=verifier_receipt, original_verifier_exit_code=verification.returncode,
        derived_files={p.name: digest(p) for p in output.iterdir() if p.is_file()},
        source_mutation=False, forward_simulation=False)
    (output / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/Users/gangzhao/Documents/workspace/AGC-Transformer'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.source, args.output)
    print(json.dumps({key: report[key] for key in ['status', 'split_counts',
        'scored_test_stations', 'withheld_year_validation', 'recomputed_numeric_metrics']}, indent=2))
