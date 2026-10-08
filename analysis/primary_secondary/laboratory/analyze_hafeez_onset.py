"""Extract source-cell inoculation scores and interval-censored onset constraints.

Source workbook is immutable. All output is confined to this analysis directory.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
WORKBOOK = PROJECT / 'data/public_septoria/hafeez-2025-infection/Hafeez et al. Stb15, Nature Plants - pathology data.xlsx'
PAPER = 'https://www.nature.com/articles/s41477-025-01920-2'
DATA_DOI = 'https://doi.org/10.5281/zenodo.14515753'
SCHEDULE = [0, 7, 10, 14, 17, 20, 21, 24, 25, 27, 28, 29, 31, 32]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def scalar(value):
    if pd.isna(value):
        return None
    return str(value)


def score_value(value):
    text = scalar(value)
    if text is None or text.strip() in {'', '-', '*', 'NA'}:
        return np.nan, 'missing'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan, 'unresolved_text'
    if not 0 <= number <= 100:
        return np.nan, 'outside_0_100'
    return number, 'valid'


def extract():
    records = []
    design = []
    for sheet in pd.ExcelFile(WORKBOOK).sheet_names:
        if sheet == 'ArinaEMS_IPO88004':
            raw = pd.read_excel(WORKBOOK, sheet_name=sheet, header=None)
            for row in range(2, 7):
                for plant in range(1, 7):
                    base = dict(assay=sheet, source_row=row + 1,
                        plant_id=f'{sheet}|row{row+1}|plant{plant}', genotype=str(raw.iloc[row, 0]),
                        genotype_original=str(raw.iloc[row, 0]), mutation=scalar(raw.iloc[row, 1]),
                        isolate='IPO88004', batch=None, replicate=str(plant), scorer=None,
                        tray=None, box=None, block=None, plot=None,
                        assay_class='EMS_current_status', plant_slot_empty=False,
                        leaf='second_leaf_6cm_section', day_temperature_c=21., night_temperature_c=18.,
                        photoperiod_hours=16., inoculum_spores_ml=1e7,
                        inoculation_method='paintbrush', experimental_trial_mapping='unavailable')
                    design.append(base)
                    for offset, endpoint in [(0, 'necrosis'), (1, 'pycnidia')]:
                        column = 2 + (plant - 1) * 2 + offset
                        value, quality = score_value(raw.iloc[row, column])
                        records.append(dict(**base, source_column=f'column_{column+1}',
                            dpi=21, endpoint=endpoint, raw_value=scalar(raw.iloc[row, column]),
                            score_percent=value, quality_status=quality))
            continue
        frame = pd.read_excel(WORKBOOK, sheet_name=sheet)
        genotype_column = 'Name' if 'Name' in frame else 'Line' if 'Line' in frame else 'Variety_newname'
        score_columns = [column for column in frame if re.fullmatch('[pd][0-9]+', str(column))]
        key_columns = [c for c in ['Batch', 'Rep', 'Tray', 'Box', 'Block', 'Plot'] if c in frame]
        assert not frame.duplicated(key_columns).any(), (sheet, key_columns)
        for row, source in frame.iterrows():
            isolate = str(source.Isolate) if 'Isolate' in frame else {
                'IPO323_results_AUDPC': 'IPO323', 'IPOO88004_raw': 'IPO88004',
                'IPO90012_scores_raw': 'IPO90012'}[sheet]
            transgenic = 'transgenics' in sheet
            genotype = scalar(source[genotype_column])
            base = dict(assay=sheet, source_row=row + 2, plant_id=f'{sheet}|row{row+2}',
                genotype=genotype, genotype_original=scalar(source.get('Variety_oldname', source[genotype_column])),
                mutation=None, isolate=isolate, batch=scalar(source.get('Batch')),
                replicate=scalar(source.get('Rep')), scorer=scalar(source.get('Scorer')),
                tray=scalar(source.get('Tray')), box=scalar(source.get('Box')),
                block=scalar(source.get('Block')), plot=scalar(source.get('Plot')),
                assay_class='transgenic_longitudinal' if transgenic else 'diversity_panel_longitudinal',
                plant_slot_empty=genotype == 'Blank', leaf='primary_seedling_leaf',
                day_temperature_c=18., night_temperature_c=12., photoperiod_hours=16.,
                inoculum_spores_ml=1e7 if transgenic and isolate == 'IPO88004' else 1e6,
                inoculation_method='spray', experimental_trial_mapping=sheet)
            design.append(base)
            for column in score_columns:
                value, quality = score_value(source[column])
                records.append(dict(**base, source_column=column, dpi=int(column[1:]),
                    endpoint='pycnidia' if column[0] == 'p' else 'damage_necrosis_chlorosis',
                    raw_value=scalar(source[column]), score_percent=value, quality_status=quality))
    scores = pd.DataFrame(records)
    plants = pd.DataFrame(design)
    # Prior control roles remain distinct from observed no-onset/right-censoring states.
    for table in [scores, plants]:
        table['source_resistance_role'] = 'not_assigned_from_scores'
        known = table.genotype.isin(['Arina', 'ArinaLrFor']) & table.isolate.eq('IPO88004')
        table.loc[known, 'source_resistance_role'] = 'Arina_Stb15_resistant_background_publication'
        table.loc[table.genotype.eq('L6 (susceptible control)'), 'source_resistance_role'] = 'susceptible_control_workbook_label'
    assert len(scores) == 60828 and len(plants) == 5313
    assert not plants.plant_id.duplicated().any()
    assert not scores.duplicated(['plant_id', 'endpoint', 'dpi']).any()
    assert not scores.loc[scores.plant_slot_empty].score_percent.notna().any()
    return scores, plants


def onset_intervals(scores, threshold):
    result = []
    for (_, endpoint), frame in scores.groupby(['plant_id', 'endpoint'], sort=False):
        source = frame.iloc[0]
        valid = frame.loc[frame.quality_status == 'valid'].sort_values('dpi')
        positive = valid.score_percent.gt(0) if threshold == 0 else valid.score_percent.ge(threshold)
        first = valid.loc[positive].dpi.min()
        before = valid.loc[(valid.dpi < first) & ~positive] if pd.notna(first) else valid
        last_zero = before.dpi.max()
        reversal_count = int((~positive & valid.dpi.gt(first)).sum()) if pd.notna(first) else 0
        if not len(valid):
            status, lower, upper = 'unobserved', np.nan, np.nan
        elif pd.isna(first):
            status, lower, upper = 'right_censored_late_or_never', float(last_zero), np.nan
        elif pd.isna(last_zero):
            status, lower, upper = 'left_censored', 0., float(first)
        else:
            status, lower, upper = 'interval_censored', float(last_zero), float(first)
        record = {c: source[c] for c in ['assay', 'plant_id', 'genotype', 'genotype_original',
            'isolate', 'batch', 'replicate', 'scorer', 'tray', 'box', 'block', 'plot',
            'assay_class', 'plant_slot_empty', 'day_temperature_c', 'night_temperature_c',
            'photoperiod_hours', 'inoculum_spores_ml', 'inoculation_method', 'leaf', 'source_resistance_role']}
        record.update(endpoint=endpoint, threshold_percent=threshold,
            detection_rule='score>0' if threshold == 0 else f'score>={threshold}',
            censoring=status, lower_exclusive_dpi=lower, upper_inclusive_dpi=upper,
            first_positive_dpi=first, last_negative_before_first_positive_dpi=last_zero,
            first_valid_dpi=valid.dpi.min(), last_valid_dpi=valid.dpi.max(),
            valid_assessments=len(valid), positive_assessments=int(positive.sum()),
            post_positive_negative_assessments=reversal_count,
            reversal_flag=reversal_count > 0,
            phenotype_interpretation='not_assessed' if not len(valid) else
                'onset_observed' if pd.notna(first) else 'no_onset_detected_late_or_never_unresolved')
        result.append(record)
    return pd.DataFrame(result)


def npmle(frame):
    """Finite Turnbull-cell likelihood including an unresolved final tail cell."""
    eligible = frame.loc[frame.censoring != 'unobserved']
    lower = eligible.lower_exclusive_dpi.to_numpy(dtype=float)
    upper = eligible.upper_inclusive_dpi.fillna(np.inf).to_numpy(dtype=float)
    endpoints = sorted(set(lower) | set(upper[np.isfinite(upper)]))
    endpoints = [value for value in endpoints if value > 0]
    bounds = np.r_[0., endpoints, np.inf]
    cell_lower, cell_upper = bounds[:-1], bounds[1:]
    member = (cell_lower[None, :] >= lower[:, None]) & (cell_upper[None, :] <= upper[:, None])
    assert member.any(axis=1).all()
    patterns, counts = np.unique(member.astype(float), axis=0, return_counts=True)
    probability = np.full(member.shape[1], 1 / member.shape[1])
    previous_ll = -np.inf
    for iteration in range(1, 50001):
        denominator = patterns @ probability
        score = patterns.T @ (counts / denominator)
        updated = probability * score / counts.sum()
        likelihood = float(counts @ np.log(patterns @ updated))
        assert likelihood + 1e-9 >= previous_ll
        gap = float(np.max(patterns.T @ (counts / (patterns @ updated))) / counts.sum() - 1)
        change = float(np.max(np.abs(updated - probability)))
        probability = updated
        if change < 1e-10 and gap < 1e-7:
            break
        previous_ll = likelihood
    else:
        raise RuntimeError('NPMLE failed convergence')
    assert abs(probability.sum() - 1) < 1e-12
    median_cells = np.flatnonzero(probability.cumsum() >= .5)
    index = int(median_cells[0])
    return dict(lower=cell_lower, upper=cell_upper, probability=probability,
        log_likelihood=likelihood, iterations=iteration, kkt_gap=gap,
        n_observed=len(eligible), median_lower=float(cell_lower[index]),
        median_upper=float(cell_upper[index]), tail_mass=float(probability[-1]))


def model_summary(intervals):
    summary, cells, cdf = [], [], []
    for (assay, isolate, endpoint, threshold), original in intervals.groupby(
            ['assay', 'isolate', 'endpoint', 'threshold_percent'], sort=False):
        for sensitivity in ['all_assessed', 'exclude_presence_reversals']:
            frame = original if sensitivity == 'all_assessed' else original.loc[~original.reversal_flag]
            assessed = frame.loc[frame.censoring != 'unobserved']
            if not len(assessed):
                continue
            fit = npmle(assessed)
            info = dict(assay=assay, isolate=isolate, endpoint=endpoint,
                threshold_percent=threshold, sensitivity=sensitivity,
                assay_class=original.assay_class.iloc[0],
                day_temperature_c=original.day_temperature_c.iloc[0],
                night_temperature_c=original.night_temperature_c.iloc[0],
                inoculum_spores_ml=original.inoculum_spores_ml.iloc[0])
            counts = Counter(frame.censoring)
            horizon = float(assessed.last_valid_dpi.max())
            summary.append(dict(**info, source_plant_slots=len(original), included_slots=len(frame),
                assessed_plants=len(assessed), unobserved_slots=counts['unobserved'],
                empty_slots=int(frame.plant_slot_empty.sum()), genotypes=assessed.genotype.nunique(),
                left_censored=counts['left_censored'], interval_censored=counts['interval_censored'],
                right_censored_late_or_never=counts['right_censored_late_or_never'],
                presence_reversals=int(original.reversal_flag.sum()),
                onset_detected=int(assessed.first_positive_dpi.notna().sum()),
                maximum_followup_dpi=horizon, npmle_median_lower_dpi=fit['median_lower'],
                npmle_median_upper_dpi=fit['median_upper'],
                npmle_median_not_reached_in_followup=bool(np.isinf(fit['median_upper'])),
                npmle_tail_mass=fit['tail_mass'], log_likelihood=fit['log_likelihood'],
                em_iterations=fit['iterations'], kkt_gap=fit['kkt_gap']))
            for lower, upper, mass in zip(fit['lower'], fit['upper'], fit['probability']):
                cells.append(dict(**info, lower_exclusive_dpi=lower, upper_inclusive_dpi=upper,
                    probability=mass, cell_type='unresolved_late_or_never_tail' if np.isinf(upper) else 'finite_onset_interval'))
            for time in SCHEDULE:
                if time > horizon:
                    continue
                lo = assessed.lower_exclusive_dpi.to_numpy(float)
                up = assessed.upper_inclusive_dpi.fillna(np.inf).to_numpy(float)
                definitely = int((up <= time).sum())
                possible = int((lo < time).sum())
                # Within-cell locations remain unidentified: no artificial exact event days.
                fitted_lower = float(fit['probability'][fit['upper'] <= time].sum())
                fitted_upper = float(fit['probability'][fit['lower'] < time].sum())
                cdf.append(dict(**info, dpi=time, assessed_plants=len(assessed),
                    definitely_onset_by_time=definitely, possibly_onset_by_time=possible,
                    empirical_cdf_lower=definitely / len(assessed),
                    empirical_cdf_upper=possible / len(assessed),
                    npmle_cdf_lower=fitted_lower, npmle_cdf_upper=fitted_upper,
                    bound_type='sample_identification_interval_not_confidence_interval'))
    return pd.DataFrame(summary), pd.DataFrame(cells), pd.DataFrame(cdf)


def reconcile(scores):
    common_path = PROJECT / 'data/harmonized/observations.parquet'
    harmonized = pd.read_parquet(common_path, filters=[('dataset_id', '==', 'hafeez-2025-infection')])
    assert len(harmonized) == len(scores)
    source = scores.copy()
    source['source_row'] = source.source_row.astype(str)
    source = source.rename(columns={'assay': 'source_table'})
    joined = source.merge(harmonized, on=['source_table', 'source_row', 'source_column'],
        suffixes=('_source', '_harmonized'), validate='one_to_one')
    assert len(joined) == len(scores)
    np.testing.assert_allclose(joined.score_percent, joined.value, rtol=0, atol=0, equal_nan=True)
    assert (joined.quality_status_source == joined.quality_status_harmonized).all()
    discrepancies = []
    for row in joined.itertuples():
        for field, authoritative, archived in [('isolate', row.isolate_source, row.isolate_harmonized),
            ('organ', row.leaf, row.organ)]:
            # Generic harmonized seedling-leaf wording agrees for the standard JIC assay.
            if field == 'organ' and authoritative == 'primary_seedling_leaf' and archived == 'seedling_leaf':
                continue
            if str(authoritative) != str(archived):
                discrepancies.append(dict(source_table=row.source_table, source_row=row.source_row,
                    source_column=row.source_column, field=field, authoritative_value=authoritative,
                    harmonized_value=archived))
    return pd.DataFrame(discrepancies), dict(harmonized_path=str(common_path.relative_to(PROJECT)),
        harmonized_sha256=digest(common_path), score_cells_checked=len(joined),
        numeric_value_matches=len(joined), quality_status_matches=len(joined),
        metadata_discrepancies=len(discrepancies))


def self_check():
    # Exact binomial current-status likelihood: F(21)=3/5, unidentified within(0,21].
    frame = pd.DataFrame({'censoring': ['left_censored'] * 3 + ['right_censored_late_or_never'] * 2,
        'lower_exclusive_dpi': [0.] * 3 + [21.] * 2,
        'upper_inclusive_dpi': [21.] * 3 + [np.nan] * 2})
    fit = npmle(frame)
    np.testing.assert_allclose(fit['probability'], [.6, .4], rtol=0, atol=1e-10)
    right = frame.iloc[3:].copy()
    np.testing.assert_allclose(npmle(right)['probability'], [0., 1.], rtol=0, atol=1e-12)
    assert score_value('*')[1] == 'missing'
    assert score_value('5?')[1] == 'unresolved_text'
    assert score_value(200)[1] == 'outside_0_100'
    return dict(current_status_analytic_fit='passed', zero_only_tail_fit='passed',
        explicit_missing_uncertain_and_invalid_scores='passed')


def run(output):
    output = Path(output).resolve()
    assert output == Path(__file__).resolve().parent, 'Output must remain within assigned directory'
    output.mkdir(parents=True, exist_ok=True)
    source_hash = digest(WORKBOOK)
    manifest = json.loads((WORKBOOK.parent / 'download-manifest.json').read_text())
    assert source_hash == manifest['files'][0]['sha256']
    tests = self_check()
    scores, plants = extract()
    intervals = pd.concat([onset_intervals(scores, 0),
        onset_intervals(scores.loc[scores.endpoint == 'pycnidia'], 5)], ignore_index=True)
    summary, cells, cdf = model_summary(intervals)
    errors, reconciliation = reconcile(scores)
    scores.to_csv(output / 'scores_long.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    plants.to_csv(output / 'source_plant_design.csv', index=False)
    intervals.to_csv(output / 'plant_onset_intervals.csv', index=False)
    summary.to_csv(output / 'assay_onset_summary.csv', index=False)
    cells.to_csv(output / 'onset_npmle_cells.csv', index=False)
    cdf.to_csv(output / 'onset_cdf_constraints.csv', index=False)
    errors.to_csv(output / 'harmonization_metadata_discrepancies.csv', index=False)
    genotype = intervals.groupby(['assay', 'isolate', 'genotype', 'endpoint', 'threshold_percent']).agg(
        source_plant_slots=('plant_id', 'size'), assessed_plants=('valid_assessments', lambda v: int((v > 0).sum())),
        onset_detected=('first_positive_dpi', 'count'), presence_reversals=('reversal_flag', 'sum'),
        latest_followup_dpi=('last_valid_dpi', 'max')).reset_index()
    genotype.to_csv(output / 'genotype_onset_summary.csv', index=False)
    batch_rows = []
    primary = intervals.loc[(intervals.endpoint == 'pycnidia')
        & (intervals.threshold_percent == 0) & intervals.batch.notna()]
    for (assay, isolate, batch), frame in primary.groupby(['assay', 'isolate', 'batch']):
        assessed = frame.loc[frame.censoring != 'unobserved']
        fit = npmle(assessed)
        batch_rows.append(dict(assay=assay, isolate=isolate, batch=batch,
            assessed_plants=len(assessed), onset_detected=int(assessed.first_positive_dpi.notna().sum()),
            npmle_median_lower_dpi=fit['median_lower'], npmle_median_upper_dpi=fit['median_upper'],
            npmle_median_not_reached_in_followup=bool(np.isinf(fit['median_upper'])),
            npmle_tail_mass=fit['tail_mass'], kkt_gap=fit['kkt_gap']))
    pd.DataFrame(batch_rows).to_csv(output / 'batch_pycnidia_onset_summary.csv', index=False)
    baseline = summary.loc[(summary.endpoint == 'pycnidia') & (summary.threshold_percent == 0)
        & (summary.sensitivity == 'all_assessed')]
    parameters = dict(outcome='first_detected_visible_pycnidia_after_known_conidial_inoculation',
        time_units='days_post_inoculation', model='interval-censored onset NPMLE',
        observable_scale='plant_or_leaf_presence_probability_not_infected_area_fraction',
        experiment_source_doi=DATA_DOI, publication_url=PAPER,
        assay_parameter_summaries=json.loads(baseline.to_json(orient='records')),
        temperature_response_estimated=False, gamma_or_erlang_rate_estimated=False,
        airborne_primary_arrival_rate_estimated=False, successful_infection_probability_estimated=False,
        cure_fraction_identified=False,
        unresolved_tail_interpretation='Later onset, resistance and failed establishment are not separable',
        finite_cell_interpretation='Probability mass on intervals; positions within each cell unidentified',
        uncertainty_interpretation='CDF bounds describe sample partial identification; no confidence interval or independent-plant assumption',
        intended_use='Assay-specific timing and detection constraints; comparison against a separately specified latent-progress kernel',
        biological_limit='Inoculation time is known; successful penetration time and emitted infectious spores are unobserved')
    (output / 'latent_progress_constraints.json').write_text(json.dumps(parameters, indent=2, allow_nan=False) + '\n')
    receipt = dict(status='verified', workbook_path=str(WORKBOOK.relative_to(PROJECT)),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        workbook_sha256=source_hash, workbook_repository_checksum_matches=True,
        score_cells=len(scores), source_plant_slots=len(plants), onset_rows=len(intervals),
        model_strata=len(summary), all_em_fits_converged=True,
        maximum_kkt_gap=float(summary.kkt_gap.max()),
        source_scores_unchanged=digest(WORKBOOK) == source_hash,
        self_checks=tests, harmonized_reconciliation=reconciliation,
        result_files={name: digest(output / name) for name in [
            'scores_long.csv.gz', 'source_plant_design.csv', 'plant_onset_intervals.csv',
            'assay_onset_summary.csv', 'onset_npmle_cells.csv', 'onset_cdf_constraints.csv',
            'harmonization_metadata_discrepancies.csv', 'genotype_onset_summary.csv',
            'batch_pycnidia_onset_summary.csv', 'latent_progress_constraints.json']})
    (output / 'verification_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return baseline, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    summary, receipt = run(args.output)
    print(summary[['assay', 'isolate', 'assessed_plants', 'onset_detected',
        'right_censored_late_or_never', 'npmle_median_lower_dpi', 'npmle_median_upper_dpi', 'npmle_tail_mass']].to_string(index=False))
    print(json.dumps({k: receipt[k] for k in ['status', 'score_cells', 'source_plant_slots', 'model_strata', 'maximum_kkt_gap']}, indent=2))
