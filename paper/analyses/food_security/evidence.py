"""Claim readiness and retained forcing diagnostics without fabricated evidence."""
from pathlib import Path

import pandas as pd

from .provenance import ROOT, auth_source, output_hashes, prepare_destination, write_json


def assess_claims(evidence):
    required = {'disease', 'HAD', 'grain_yield', 'grain_loss', 'tonnes', 'adaptation'}
    if evidence.quantity.duplicated().any() or not required <= set(evidence.quantity):
        raise ValueError('Complete unique evidence endpoint registry required')
    table = evidence.set_index('quantity')
    columns = ['observed_quantity_supported', 'claim_ready', 'end_to_end_physiological_validation']
    for column in columns:
        if table[column].isna().any() or not table[column].isin([True, False]).all():
            raise ValueError('Explicit boolean evidence flags required: ' + column)
    def qualified(quantity):
        return all(bool(table.loc[quantity, column]) for column in columns)
    canopy = qualified('HAD')
    grain = canopy and qualified('grain_yield') and qualified('grain_loss')
    rows = [
        ('baseline_production_exposure', True, 'Conditional spatial exposure under original model and crop scenario',
         'Fixed production and climate response; excludes physiological loss and future production'),
        ('functional_canopy_loss', canopy, 'Independent physiological canopy validation',
         'Dated functional/reference upper-leaf area, senescence and matched disease measurements'),
        ('grain_loss', grain, 'Independent crop yield and cause-specific loss validation',
         'Comparable reference outcomes, canopy trajectories, harvest and competing stresses'),
        ('tonnes_lost', grain and qualified('tonnes'), 'Validated regional grain loss',
         'Validated yield changes, representative crop area, compatible grain moisture and uncertainty'),
        ('adaptation_benefit', grain and qualified('adaptation'), 'Validated feasible intervention outcome',
         'Independent intervention contrasts, yield, resource outcomes and costs'),
    ]
    return pd.DataFrame(rows, columns=['claim', 'supported', 'scope', 'required_evidence'])


def _forcing_changes(frame, keys, values):
    if frame.duplicated(keys + ['state']).any() or set(frame.state) != {'NEX_v2', 'secondary_aligned'}:
        raise ValueError('Unique paired forcing states required')
    raw = frame.loc[frame.state.eq('NEX_v2')].set_index(keys)
    aligned = frame.loc[frame.state.eq('secondary_aligned')].set_index(keys)
    if set(raw.index) != set(aligned.index):
        raise ValueError('Forcing diagnostics have unmatched support')
    aligned = aligned.loc[raw.index]
    result = raw[values].rename(columns={name: name + '_raw' for name in values})
    for name in values:
        result[name + '_aligned'] = aligned[name]
        result[name + '_change'] = aligned[name] - raw[name]
    return result.reset_index()


def run(output, *, root=ROOT):
    root, output = Path(root), Path(output)
    prepare_destination(output, root)
    relatives = ['data/final_study/evidence/yield_transfer_audit/evidence_gate.csv',
                 'data/final_study/evidence/research_plan/results/source_readiness.csv',
                 'data/final_study/evidence/review_response/validation/results/evidence_gaps.csv',
                 'data/final_study/evidence/review_response/forcing/daily_distribution_diagnostics.csv',
                 'data/final_study/evidence/review_response/forcing/daily_covariance_diagnostics.csv',
                 'data/final_study/evidence/winter_wheat_population/results/country_coverage.csv']
    sources = [auth_source(root, path) for path in relatives]
    registry, readiness, gaps, distributions, covariance, countries = [pd.read_csv(root / path) for path in relatives]
    claims = assess_claims(registry)
    claims.to_csv(output / 'claim_readiness.csv', index=False)
    readiness.to_csv(output / 'source_readiness.csv', index=False)
    gaps.to_csv(output / 'validation_gaps.csv', index=False)
    countries.to_csv(output / 'winter_crop_coverage.csv', index=False)
    _forcing_changes(distributions, ['model', 'period', 'variable'],
                     ['mean', 'std', 'q05', 'q50', 'q95', 'wet_day_fraction']).to_csv(output / 'forcing_distribution_changes.csv', index=False)
    _forcing_changes(covariance, ['model', 'period', 'variable_a', 'variable_b'],
                     ['within_cell_population_covariance']).to_csv(output / 'forcing_covariance_changes.csv', index=False)
    result = dict(status='assessed', sources=sources,
                  claims=claims.to_dict('records'),
                  independent_daily_weather_reference_validated=False,
                  independent_weather_response_validated=False,
                  forcing_scope='Retained calibration-period diagnostics; changes are not independent weather validation',
                  additional_climate_models_acquired=False,
                  new_physiological_observations_acquired=False,
                  outputs_sha256=output_hashes(output))
    write_json(output / 'receipt.json', result)
    return result
