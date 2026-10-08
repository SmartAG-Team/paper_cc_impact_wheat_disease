"""Independent archive, membership and weighted-arithmetic audit."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/paper_study/regional_parameter_uncertainty'
DEST = ROOT / 'analysis/paper_study/regional_parameter_uncertainty'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    draws = pd.read_csv(DATA / 'spatial_draws.csv')
    weights = draws.area_mean_weight.to_numpy(float)
    assert len(draws) == 64 and draws.spatial_draw_id.to_list() == list(range(64))
    assert draws.groupby('stratum').size().eq(4).all()
    np.testing.assert_allclose(weights.sum(), 1., rtol=0, atol=1e-14)
    point = ROOT / 'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json'
    parameter_paths = [point] + sorted((ROOT / 'data/paper_study/seasonal_statistics/bootstrap_fits').glob('fit_*.json'))
    expected_hashes = {str(path.relative_to(ROOT)): sha(path) for path in parameter_paths}
    assert len(expected_hashes) == 101
    for path in parameter_paths[1:]:
        fit = json.loads(path.read_text())
        assert fit['calibration_only'] and not fit['validation_or_external_targets_used']
    checks, max_error, max_mass = 0, 0., 0.
    hashes = {}
    years = list(range(1991, 2021)) + list(range(2031, 2061)) + list(range(2071, 2101))
    for model in ['ACCESS-CM2', 'MPI-ESM1-2-HR', 'MRI-ESM2-0']:
        for scenario in ['ssp126', 'ssp245', 'ssp585']:
            for year in years:
                path = DATA / 'annual' / model / scenario / f'{year}.npz'
                receipt = json.loads(path.with_suffix('.json').read_text())
                digest = sha(path)
                assert digest == receipt['output_sha256']
                assert receipt['model'] == model and receipt['scenario'] == scenario and receipt['harvest_year'] == year
                assert receipt['spatial_sample_sha256'] == sha(DATA / 'spatial_draws.csv')
                assert {entry['path']: entry['sha256'] for entry in receipt['parameter_draws']} == expected_hashes
                assert receipt['point_operator_reconciliation']
                hashes[str(path.relative_to(ROOT))] = digest
                with np.load(path) as archive:
                    values = archive['cell_damage_percent']
                    assert values.shape == (101, 64)
                    valid = np.isfinite(values)
                    expected_valid = archive['eligible'] & archive['complete_season']
                    np.testing.assert_array_equal(valid, np.broadcast_to(expected_valid, values.shape))
                    assert np.all((values[valid] >= 0) & (values[valid] <= 100 + 1e-10))
                    denominator = []
                    means = []
                    # Separate scalar loops avoid the original broadcasted sums.
                    for parameter_index in range(101):
                        keep = valid[parameter_index]
                        den = sum(float(weights[j]) for j in range(64) if keep[j])
                        total = sum(float(values[parameter_index, j]) * float(weights[j]) for j in range(64) if keep[j])
                        denominator.append(den)
                        means.append(total / den if den else np.nan)
                    error = np.max(np.abs(np.asarray(means) - archive['area_mean_percent']))
                    max_error = max(max_error, float(error))
                    np.testing.assert_allclose(means, archive['area_mean_percent'], rtol=0, atol=1e-10, equal_nan=True)
                    np.testing.assert_allclose(denominator, archive['valid_area_fraction'], rtol=0, atol=1e-14)
                    checks += values.size + 202 + 8
                max_mass = max(max_mass, receipt['maximum_mass_error'])
                assert receipt['maximum_mass_error'] < 1e-12
    result = dict(status='passed', annual_outputs=810, calibration_bootstrap_draws=100,
        scalar_and_membership_checks=checks, maximum_mean_difference_percentage_points=max_error,
        maximum_compartment_mass_error=max_mass, source_code_sha256=sha(Path(__file__)),
        frozen_parameter_hashes=expected_hashes, output_hashes=hashes,
        source_climate_bytes_independently_audited_separately=True)
    (DEST / 'independent_ensemble_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['output_hashes', 'frozen_parameter_hashes']}))


if __name__ == '__main__':
    main()
