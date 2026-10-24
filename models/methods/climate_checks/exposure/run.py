"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/review_response_20261010/exposure')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
HERE = evidence_dir('analysis/paper_study/review_response_20261010/exposure')
ROOT = PROJECT_ROOT
SOURCE = ROOT / 'data/final_study/evidence/food_security_exposure'
MODELS = ('ACCESS-CM2', 'MPI-ESM1-2-HR', 'MRI-ESM2-0')
WEIGHTS = {'all_wheat': 'production_total_tonnes', 'rainfed_all_wheat': 'production_rainfed_tonnes'}
THRESHOLDS = [*np.arange(0, 5.01, 0.25), 7.5, 10.0]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def summarize(weights, pairs, thresholds=THRESHOLDS):
    """Return threshold curves and continuous means; no grain-loss conversion.

    Positive thresholds are descriptive sensitivity settings, without biological
    significance implied. Rainfed production still includes winter/spring wheat.
    """
    keys = ['cell_id', 'model', 'scenario', 'period']
    if weights.cell_id.isna().any() or weights.cell_id.duplicated().any() or pairs.duplicated(keys).any():
        raise ValueError('Null or duplicate input identities')
    if pairs[keys].isna().any().any() or set(pairs.cell_id) != set(weights.cell_id):
        raise ValueError('Incomplete input identities')
    if set(pairs.model) != set(MODELS):
        raise ValueError('Expected exactly three climate models')
    w = weights.set_index('cell_id')
    if w.environment_region.isna().any() or w.environment_region.eq('Europe').any():
        raise ValueError('Invalid environmental-region labels')
    production = w[list(WEIGHTS.values())].to_numpy(float)
    if not np.isfinite(production).all() or (production < 0).any():
        raise ValueError('Production weights must be finite and nonnegative')
    thresholds = np.asarray(thresholds, float)
    if not np.isfinite(thresholds).all() or (thresholds < 0).any() or len(set(thresholds)) != len(thresholds):
        raise ValueError('Thresholds must be unique finite nonnegative days')
    count = pairs.valid_year_pairs.to_numpy(float)
    if not np.isfinite(count).all() or (count < 0).any() or (count > 30).any() or (count != np.floor(count)).any():
        raise ValueError('Invalid paired-year count')
    if not np.array_equal(np.isfinite(pairs.change), count > 0):
        raise ValueError('Change availability and paired-year counts disagree')
    domains = [('Europe', np.ones(len(w), bool)), *[(name, w.environment_region.eq(name).to_numpy()) for name in sorted(w.environment_region.unique())]]
    curves, means = ([], [])
    for (scenario, period), group in pairs.groupby(['scenario', 'period'], sort=True):
        if len(group) != len(w) * len(MODELS) or set(group.model) != set(MODELS):
            raise ValueError('Each scenario-period requires all cells and models')
        changes = group.pivot(index='cell_id', columns='model', values='change').reindex(index=w.index, columns=MODELS).to_numpy()
        counts = group.pivot(index='cell_id', columns='model', values='valid_year_pairs').reindex(index=w.index, columns=MODELS).to_numpy()
        complete = np.isfinite(changes).all(axis=1)
        ensemble = np.mean(changes, axis=1)
        for basis, column in WEIGHTS.items():
            for domain, mask in domains:
                p = w[column].to_numpy() * mask
                baseline = float(p.sum())
                unavailable = float(p[~complete].sum())
                common = dict(scenario=scenario, period=period, domain=domain, weight_basis=basis, baseline_production_tonnes=baseline, unavailable_tonnes=unavailable)
                model_means = []
                for j in range(len(MODELS)):
                    valid = np.isfinite(changes[:, j]) & (counts[:, j] > 0)
                    ww = p[valid] * counts[valid, j]
                    model_means.append(float(ww @ changes[valid, j] / ww.sum()) if ww.sum() else np.nan)
                model_means = np.asarray(model_means)
                finite = np.isfinite(model_means).all()
                means.append(dict(**common, mean_change_days=float(model_means.mean()) if finite else np.nan, gcm_min=float(model_means.min()) if finite else np.nan, gcm_max=float(model_means.max()) if finite else np.nan, **{model: model_means[j] for j, model in enumerate(MODELS)}))
                for threshold in thresholds:
                    unanimous = complete & (changes > threshold).all(axis=1)
                    positive_mean = complete & (ensemble > threshold)
                    all_exposed, ensemble_exposed = (float(p[unanimous].sum()), float(p[positive_mean].sum()))
                    curves.append(dict(**common, threshold_days=float(threshold), all_model_exposed_tonnes=all_exposed, ensemble_exposed_tonnes=ensemble_exposed, all_model_share_pct=100 * all_exposed / baseline if baseline else np.nan, ensemble_share_pct=100 * ensemble_exposed / baseline if baseline else np.nan))
    return (pd.DataFrame(curves), pd.DataFrame(means))

def build():
    inputs = [SOURCE / 'production_grid_input.csv', SOURCE / 'paired_changes_input.csv.gz']
    weights, pairs = [pd.read_csv(p) for p in inputs]
    curves, means = summarize(weights, pairs)
    curves.to_csv(HERE / 'exposure_curves.csv', index=False)
    means.to_csv(HERE / 'weight_sensitivity.csv', index=False)
    receipt = dict(status='passed', thresholds_days=list(map(float, THRESHOLDS)), inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs}, software_sha256=sha(__file__), outputs={name: sha(HERE / name) for name in ['exposure_curves.csv', 'weight_sensitivity.csv']}, interpretation='Fixed baseline production exposed to a specified modelled damage change; not grain lost.', limitations=['Thresholds are illustrative, not validated biological or economic cutoffs.', 'Rainfed all-wheat production does not identify winter-wheat production.', 'Original rainfed and total rasters retain their independent provider values.', 'Climate-model spread is not a probability interval; no future production is estimated.'])
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return verify()

def verify():
    receipt = json.loads((HERE / 'receipt.json').read_text())
    for name, expected in receipt['inputs'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Input checksum differs: ' + name)
    if sha(__file__) != receipt['software_sha256']:
        raise ValueError('Exposure analysis code has changed; rebuild evidence')
    for name, expected in receipt['outputs'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Output checksum differs: ' + name)
    curves = pd.read_csv(HERE / 'exposure_curves.csv')
    for _, group in curves.groupby(['scenario', 'period', 'domain', 'weight_basis']):
        group = group.sort_values('threshold_days')
        if (np.diff(group.all_model_exposed_tonnes) > 1e-08).any():
            raise ValueError('Nonmonotonic exposure curve')
        if (group.all_model_exposed_tonnes > group.ensemble_exposed_tonnes + 1e-07).any():
            raise ValueError('Unanimous exposure exceeds ensemble exposure')
    primary = pd.read_csv(SOURCE / 'domain_summary.csv')
    primary = primary[primary.analysis_mode.eq('primary_any_paired_year') & primary.domain_type.isin(['Europe', 'environment_region'])]
    zero = curves[curves.weight_basis.eq('all_wheat') & curves.threshold_days.eq(0)]
    joined = zero.merge(primary, on=['scenario', 'period', 'domain'], validate='one_to_one', suffixes=('_new', '_original'))
    if len(joined) != len(zero):
        raise ValueError('Incomplete reference reconstruction')
    for left, right in [('all_model_exposed_tonnes', 'all3positive_production_tonnes'), ('ensemble_exposed_tonnes', 'ensemble_increasing_production_tonnes')]:
        np.testing.assert_allclose(joined[left], joined[right], rtol=1e-12, atol=1e-07)
    return dict(status='passed', curve_rows=len(curves), reconstructed_reference_rows=len(joined))

def read():
    verify()
    return {name: pd.read_csv(HERE / (name + '.csv')) for name in ['exposure_curves', 'weight_sensitivity']}
