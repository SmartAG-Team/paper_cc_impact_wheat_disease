"""Independently check retained source scores, onset intervals and NPMLE optima."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify():
    directory = Path(__file__).resolve().parent
    project = Path(__file__).resolve().parents[3]
    receipt = json.loads((directory / 'verification_receipt.json').read_text())
    book = project / receipt['workbook_path']
    assert hashlib.sha256(book.read_bytes()).hexdigest() == receipt['workbook_sha256']
    for filename, expected in receipt['result_files'].items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
    scores = pd.read_csv(directory / 'scores_long.csv.gz', low_memory=False)
    plants = pd.read_csv(directory / 'source_plant_design.csv')
    intervals = pd.read_csv(directory / 'plant_onset_intervals.csv')
    summary = pd.read_csv(directory / 'assay_onset_summary.csv')
    cells = pd.read_csv(directory / 'onset_npmle_cells.csv')
    cdf = pd.read_csv(directory / 'onset_cdf_constraints.csv')
    assert len(scores) == 60828 and len(plants) == 5313 and len(intervals) == 15939
    assert not scores.duplicated(['plant_id', 'endpoint', 'dpi']).any()
    assert not plants.plant_id.duplicated().any()
    numeric = pd.to_numeric(scores.raw_value, errors='coerce')
    valid = numeric.between(0, 100)
    np.testing.assert_allclose(scores.score_percent, numeric.where(valid), rtol=0, atol=0, equal_nan=True)
    assert scores.quality_status.eq('valid').equals(valid)
    curves = {key: frame.loc[frame.score_percent.notna()].sort_values('dpi')
        for key, frame in scores.groupby(['plant_id', 'endpoint'])}
    checked = 0
    for row in intervals.itertuples():
        curve = curves[(row.plant_id, row.endpoint)]
        positive = curve.score_percent > 0 if row.threshold_percent == 0 else curve.score_percent >= row.threshold_percent
        if not len(curve):
            assert row.censoring == 'unobserved' and pd.isna(row.lower_exclusive_dpi)
        elif not positive.any():
            assert row.censoring == 'right_censored_late_or_never'
            assert row.lower_exclusive_dpi == curve.dpi.max() and pd.isna(row.upper_inclusive_dpi)
        else:
            first = curve.loc[positive].dpi.min()
            negatives = curve.loc[(curve.dpi < first) & ~positive]
            assert row.upper_inclusive_dpi == first
            assert row.lower_exclusive_dpi == (negatives.dpi.max() if len(negatives) else 0)
            assert row.censoring == ('interval_censored' if len(negatives) else 'left_censored')
            assert row.reversal_flag == bool((~positive & (curve.dpi > first)).any())
        checked += 1
    keys = ['assay', 'isolate', 'endpoint', 'threshold_percent', 'sensitivity']
    maximum_gap = 0.
    for row in summary.itertuples():
        mask = (intervals.assay == row.assay) & (intervals.isolate == row.isolate)
        mask &= (intervals.endpoint == row.endpoint) & (intervals.threshold_percent == row.threshold_percent)
        data = intervals.loc[mask & intervals.censoring.ne('unobserved')]
        if row.sensitivity == 'exclude_presence_reversals':
            data = data.loc[~data.reversal_flag]
        selected = cells.copy()
        selected_cdf = cdf.copy()
        for key in keys:
            selected = selected.loc[selected[key] == getattr(row, key)]
            selected_cdf = selected_cdf.loc[selected_cdf[key] == getattr(row, key)]
        assert len(data) == row.assessed_plants
        lo = data.lower_exclusive_dpi.to_numpy(float)
        up = data.upper_inclusive_dpi.fillna(np.inf).to_numpy(float)
        cell_lo = selected.lower_exclusive_dpi.to_numpy(float)
        cell_up = selected.upper_inclusive_dpi.to_numpy(float)
        p = selected.probability.to_numpy(float)
        matrix = (cell_lo[None, :] >= lo[:, None]) & (cell_up[None, :] <= up[:, None])
        denominator = matrix @ p
        assert np.all(denominator > 0)
        np.testing.assert_allclose(p.sum(), 1., rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.log(denominator).sum(), row.log_likelihood, rtol=0, atol=1e-9)
        gap = float(np.max(matrix.T @ (1 / denominator)) / len(data) - 1)
        maximum_gap = max(maximum_gap, gap)
        assert gap < 1e-7
        # Indistinguishable positive-mass columns would make cell locations unidentified.
        active = np.flatnonzero(p > 1e-10)
        for a, left in enumerate(active):
            for right in active[a+1:]:
                assert not np.array_equal(matrix[:, left], matrix[:, right])
        median = np.flatnonzero(p.cumsum() >= .5)[0]
        assert cell_lo[median] == row.npmle_median_lower_dpi
        assert cell_up[median] == row.npmle_median_upper_dpi
        for point in selected_cdf.itertuples():
            np.testing.assert_allclose(point.empirical_cdf_lower, (up <= point.dpi).mean(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(point.empirical_cdf_upper, (lo < point.dpi).mean(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(point.npmle_cdf_lower, p[cell_up <= point.dpi].sum(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(point.npmle_cdf_upper, p[cell_lo < point.dpi].sum(), rtol=0, atol=1e-12)
            assert point.empirical_cdf_lower <= point.empirical_cdf_upper + 1e-12
    return dict(status='verified', score_cells_checked=len(scores),
        plant_intervals_checked=checked, npmle_fit_optima_checked=len(summary),
        maximum_kkt_gap=maximum_gap, cdf_constraint_rows_checked=len(cdf),
        positive_mass_cell_memberships_distinct=True,
        preserved_original_workbook=True, temperature_response_estimated=False,
        cure_fraction_identified=False, airborne_primary_arrival_estimated=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
