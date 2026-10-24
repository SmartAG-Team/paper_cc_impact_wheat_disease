"""Descriptive audit for the next climate-compatible experiment; no fitting."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.final_study.metrics import hierarchical_weights
from .innovation_common import ROOT
from .inputs import sha


def run():
    parent = ROOT/'outputs/disease_improvement/autonomous_weather_20261022'
    out = ROOT/'outputs/disease_improvement/next_step_diagnostic_20261022'
    if out.exists():
        raise FileExistsError('Preserve the previous diagnostic')
    rows, bins, overall = [], [], []
    sources = [parent/'assessment_predictions.csv', parent/'spatial_predictions.csv',
        ROOT/'data/final_climate/disease_parameters.json', ROOT/'analysis/final_climate/kernel.py',
        ROOT/'README.md']
    for design, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        f = pd.read_csv(parent/name, float_precision='round_trip')
        assert len(f) == 2017 and f.global_target_index.is_unique
        w = hierarchical_weights(f); w /= w.sum()
        y, p = f.observed_percent.to_numpy(), f.climate_renewal_phase.to_numpy()
        error = p-y; mse = float(np.sum(w*error**2)); bias = float(np.sum(w*error))
        centered = float(np.sum(w*(error-bias)**2))
        np.testing.assert_allclose(mse, centered+bias**2, rtol=1e-14)
        overall.append(dict(evaluation=design, rmse=np.sqrt(mse), bias=bias,
            squared_mean_error_fraction=bias**2/mse, centered_error_fraction=centered/mse))
        severity = dict(observed_0to10=y <= 10, observed_over10to75=(y > 10)&(y <= 75), observed_over75=y > 75)
        dimensions = dict(observed_severity=severity,
            source={key: f.source.eq(key).to_numpy() for key in f.source.unique()},
            year={str(key): f.season_year.eq(key).to_numpy() for key in sorted(f.season_year.unique())})
        for dimension, groups in dimensions.items():
            total = 0.
            for key, mask in groups.items():
                mass = float(w[mask].sum()); sse = float(np.sum(w[mask]*error[mask]**2))
                rows.append(dict(evaluation=design, dimension=dimension, subgroup=key, n=int(mask.sum()),
                    scientific_mass=mass, squared_error_fraction=sse/mse, rmse=np.sqrt(sse/mass),
                    bias=np.sum(w[mask]*error[mask])/mass))
                total += sse/mse
            np.testing.assert_allclose(total, 1., atol=1e-12)
        # Forecast-defined bins distinguish calibration from outcome-tail bias.
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 6)))
        membership = np.searchsorted(edges[1:-1], p, side='right')
        for group in range(len(edges)-1):
            mask = membership == group
            if not mask.any():
                continue
            bins.append(dict(evaluation=design, forecast_bin=group+1, lower=edges[group], upper=edges[group+1],
                n=int(mask.sum()), scientific_mass=w[mask].sum(),
                forecast_mean=np.average(p[mask], weights=w[mask]),
                observed_mean=np.average(y[mask], weights=w[mask]),
                bias=np.average(error[mask], weights=w[mask])))
    kernel_metadata = json.loads((ROOT/'data/final_climate/disease_parameters.json').read_text())
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        interpretation='Descriptive saved-outcome diagnostics, not causal evidence or independent confirmation; observed-severity bins cannot establish forecast miscalibration by themselves',
        metric='Original coordinate-year/field/native-endpoint/record hierarchy; percentage-point errors',
        climate_kernel_metadata=dict(disease_magnitude_fit=kernel_metadata['disease_magnitude_fit'],
            calibration_fields=kernel_metadata['calibration_fields'], validation_outcomes_used=kernel_metadata['validation_outcomes_used']),
        caveat='The climate-kernel flag establishes recorded fitting scope, not a proven cause of the separate ML errors',
        model_fitted=False, source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        code_sha256=sha(Path(__file__)))
    out.mkdir(parents=True)
    pd.DataFrame(overall).to_csv(out/'overall_error_decomposition.csv', index=False)
    pd.DataFrame(rows).to_csv(out/'stratified_error_decomposition.csv', index=False)
    pd.DataFrame(bins).to_csv(out/'forecast_bin_calibration.csv', index=False)
    receipt['artifact_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out/'diagnostic.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(pd.DataFrame(overall).to_string(index=False))
    print(pd.DataFrame(bins).to_string(index=False))
    print(json.dumps(receipt['climate_kernel_metadata'], indent=2))
    return out


if __name__ == '__main__':
    run()
