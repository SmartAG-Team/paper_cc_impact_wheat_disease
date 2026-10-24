"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/winter_wheat_population_20261010')
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from analysis.methods.winter_crop_bounds.prepare import HERE, ROOT, sha, prepare
from analysis.methods.winter_crop_bounds.bounds import allocation_bounds
MODELS = ('ACCESS-CM2', 'MPI-ESM1-2-HR', 'MRI-ESM2-0')
EXPOSURE = ROOT / 'data/final_study/evidence/food_security_exposure'

def allocation_inputs(grid, fractions, margins, changes, threshold, region, scope):
    """Assemble one pathway-period/domain without losing country fractions."""
    if grid.cell_id.duplicated().any() or fractions.duplicated(['cell_id', 'country']).any():
        raise ValueError('Duplicate grid or fractional country input')
    if margins.country.duplicated().any() or margins.country_fips.duplicated().any():
        raise ValueError('Duplicate national margin')
    if not margins.supported.isin([True, False]).all():
        raise ValueError('Unknown country support state')
    mass = fractions[['production_total_tonnes', 'production_rainfed_tonnes']].to_numpy(float)
    if not np.isfinite(mass).all() or (mass < 0).any() or (mass[:, 1] > mass[:, 0] + 1e-06).any():
        raise ValueError('Invalid total or rainfed fractional mass')
    if not set(fractions.cell_id).issubset(set(grid.cell_id)):
        raise ValueError('Country fraction outside grid')
    counted = fractions.groupby('cell_id').production_rainfed_tonnes.sum().reindex(grid.cell_id, fill_value=0)
    if not np.allclose(counted.to_numpy(), grid.production_rainfed_tonnes, rtol=1e-10, atol=1e-06):
        raise ValueError('Fractional rainfed production does not conserve grid mass')
    if changes.duplicated(['cell_id', 'model']).any():
        raise ValueError('Duplicate cell-model climate response')
    if changes.scenario.nunique() != 1 or changes.period.nunique() != 1:
        raise ValueError('Expected exactly one scenario and period')
    if not set(changes.model).issubset(MODELS) or not set(changes.cell_id).issubset(grid.cell_id):
        raise ValueError('Unknown climate model or cell')
    if not changes.metric.eq('GS65_85_lost_had3').all():
        raise ValueError('Unexpected climate response metric')
    if not np.isfinite(threshold):
        raise ValueError('Invalid threshold')
    c = changes.copy()
    c['valid'] = np.isfinite(c.change) & np.isfinite(c.valid_year_pairs) & c.valid_year_pairs.gt(0)
    status = c.groupby('cell_id').agg(model_count=('model', 'size'), valid_count=('valid', 'sum'), minimum_change=('change', 'min'))
    status['available'] = status.model_count.eq(len(MODELS)) & status.valid_count.eq(len(MODELS))
    status['exposed'] = status.available & status.minimum_change.gt(threshold)
    joined = fractions.merge(grid[['cell_id', 'environment_region']], on='cell_id', validate='many_to_one')
    joined = joined.merge(status[['available', 'exposed']], left_on='cell_id', right_index=True, how='left', validate='many_to_one')
    joined[['available', 'exposed']] = joined[['available', 'exposed']].eq(True)
    joined = joined.merge(margins[['country', 'country_fips', 'supported']], on=['country', 'country_fips'], how='left', validate='many_to_one', indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Missing country or conflicting source-country code')
    if region != 'Europe':
        if region not in set(grid.environment_region):
            raise ValueError('Unknown region')
        joined = joined[joined.environment_region.eq(region)]
    if scope == 'supported_country_subset':
        joined = joined[joined.supported]
    elif scope != 'all_domain_countries':
        raise ValueError('Unknown population scope')
    joined['exposed_tonnes'] = joined.production_rainfed_tonnes * joined.exposed
    joined['unavailable_tonnes'] = joined.production_rainfed_tonnes * ~joined.available
    aggregate = joined.groupby(['country', 'country_fips'], as_index=False).agg(domain_tonnes=('production_rainfed_tonnes', 'sum'), exposed_tonnes=('exposed_tonnes', 'sum'), unavailable_tonnes=('unavailable_tonnes', 'sum'))
    aggregate = aggregate[aggregate.domain_tonnes.gt(0)]
    return aggregate.merge(margins, on=['country', 'country_fips'], validate='one_to_one').sort_values('country').reset_index(drop=True)

def verify_lock(directory=HERE):
    lock = json.loads((directory / 'analysis_lock.json').read_text())
    for name, digest in lock['sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen population input changed: ' + name)
    return lock

def run(directory=HERE):
    verify_lock(directory)
    margins = prepare(directory)
    receipt = json.loads((EXPOSURE / 'bundle_receipt.json').read_text())
    for name, digest in receipt['sha256'].items():
        if sha(EXPOSURE / name) != digest:
            raise ValueError('Original exposure input hash mismatch')
    grid = pd.read_csv(EXPOSURE / 'production_grid_input.csv')
    fractions = pd.read_csv(EXPOSURE / 'country_fraction_weights.csv')
    changes = pd.read_csv(EXPOSURE / 'paired_changes_input.csv.gz')
    protocol = json.loads((directory / 'protocol.json').read_text())
    combinations = {(s, p) for s in ['ssp126', 'ssp245', 'ssp585'] for p in ['2031-2060', '2071-2100']}
    if set(map(tuple, changes[['scenario', 'period']].drop_duplicates().values)) != combinations:
        raise ValueError('Climate scenario-period coverage changed')
    if set(protocol['domains']) != {'Europe', *grid.environment_region.unique()}:
        raise ValueError('Environmental-region coverage changed')
    country_domain = fractions.groupby(['country', 'country_fips'], as_index=False).agg(domain_all_wheat_tonnes=('production_total_tonnes', 'sum'), domain_rainfed_wheat_tonnes=('production_rainfed_tonnes', 'sum'))
    coverage = country_domain.merge(margins, on=['country', 'country_fips'], validate='one_to_one')
    coverage['share_domain_rainfed_pct'] = 100 * coverage.domain_rainfed_wheat_tonnes / coverage.domain_rainfed_wheat_tonnes.sum()
    result_rows = []
    allocation_rows = []
    witness_rows = []
    allowance_rows = []
    for (scenario, period), climate in changes.groupby(['scenario', 'period'], sort=True):
        for threshold in protocol['thresholds_days']:
            for region in protocol['domains']:
                for scope in protocol['scopes']:
                    key = dict(scenario=scenario, period=period, threshold_days=threshold, region=region, scope=scope)
                    a = allocation_inputs(grid, fractions, margins, climate, threshold, region, scope)
                    b = allocation_bounds(a)
                    no_allowance = a.copy()
                    no_allowance['total_tonnes'] = a.mapped_national_total_tonnes
                    no_allowance['winter_lower_tonnes'] = a.fraction_lower * no_allowance.total_tonnes
                    no_allowance['winter_upper_tonnes'] = a.fraction_upper * no_allowance.total_tonnes
                    nb = allocation_bounds(no_allowance)
                    allowance_rows.append({**key, **{'allowance_effect_' + name: b[name] - nb[name] for name in ['domain_winter_lower_tonnes', 'domain_winter_upper_tonnes', 'exposed_winter_lower_tonnes', 'exposed_winter_upper_tonnes', 'share_lower_pct', 'share_upper_pct']}})
                    missing = a.copy()
                    missing['exposed_tonnes'] = a.unavailable_tonnes
                    mb = allocation_bounds(missing)
                    for endpoint in ['lower', 'upper']:
                        for row in b.pop('witness_' + endpoint):
                            witness_rows.append({**key, 'endpoint': endpoint, **row})
                    result_rows.append({**key, 'country_count': len(a), 'supported_country_count': int(a.supported.sum()), 'all_rainfed_wheat_tonnes': float(a.domain_tonnes.sum()), 'supported_rainfed_wheat_tonnes': float(a.loc[a.supported, 'domain_tonnes'].sum()), 'all_rainfed_exposed_tonnes': float(a.exposed_tonnes.sum()), 'unavailable_all_rainfed_tonnes': float(a.unavailable_tonnes.sum()), 'unavailable_winter_lower_tonnes': mb['exposed_winter_lower_tonnes'], 'unavailable_winter_upper_tonnes': mb['exposed_winter_upper_tonnes'], **b})
                    allocation_rows.extend([{**key, **row} for row in a.to_dict('records')])
    out = directory / 'results'
    out.mkdir(exist_ok=True)
    coverage.to_csv(out / 'country_coverage.csv', index=False)
    pd.DataFrame(result_rows).to_csv(out / 'allocation_bounds.csv', index=False)
    pd.DataFrame(allocation_rows).to_csv(out / 'country_allocation_inputs.csv', index=False)
    pd.DataFrame(witness_rows).to_csv(out / 'fraction_witnesses.csv', index=False)
    pd.DataFrame(allowance_rows).to_csv(out / 'unassigned_allowance_sensitivity.csv', index=False)
    files = [*sorted((directory / 'inputs').glob('*')), directory / 'protocol.json', directory / 'analysis_lock.json', *sorted(directory.glob('*.py')), *sorted(out.glob('*.csv'))]
    (directory / 'result_receipt.json').write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), interpretation=protocol['uncertainty'], sha256={str(p.relative_to(ROOT)): sha(p) for p in files}), indent=2) + '\n')
    return pd.DataFrame(result_rows)
