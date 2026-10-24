"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_models_20261012')
import numpy as np
from analysis.methods.residual_disease.features import design as basic_design

def design(z, frame):
    x, names = basic_design(z, None, frame)
    fi = frame.field_index.to_numpy(int)
    day = frame.day_index.to_numpy(int) - 1
    daily = z['features']
    lookup = {name: i for i, name in enumerate(z['names'])}
    cdf = [daily[:, :, lookup[f'crop_cdf_{stage}']] for stage in [10, 31, 51, 87]]
    current = [value[fi, day] for value in cdf]
    extra = [1 - current[0], current[0] - current[1], current[1] - current[2], current[2] - current[3], current[3]]
    labels = ['phase_before_10', 'phase_10_31', 'phase_31_51', 'phase_51_87', 'phase_after_87']
    for stage, value in zip([51, 87], cdf[2:]):
        for lag in [7, 21]:
            extra.append(value[fi, day] - value[fi, np.maximum(day - lag, 0)])
            labels.append(f'crop_cdf_{stage}_change_{lag}d')
    for stage, value in zip([31, 51, 87], cdf[1:]):
        days = np.arange(value.shape[1])[None, :]
        passed = np.maximum.accumulate(value >= 0.5, axis=1)
        first = np.minimum.accumulate(np.where(value >= 0.5, days, value.shape[1]), axis=1)
        extra.append(np.where(passed[fi, day], day - first[fi, day], 0) / 330.0)
        labels.append(f'elapsed_since_predicted_{stage}_median_scaled')
    for stem in ['temperature', 'humidity', 'rain_log1p']:
        short = f'{stem}_mean_7d'
        long = f'{stem}_mean_42d'
        if short in lookup and long in lookup:
            extra.append(daily[fi, day, lookup[short]] - daily[fi, day, lookup[long]])
            labels.append(f'{stem}_contrast_7d_42d')
    result = np.column_stack([x, *extra])
    if not np.isfinite(result).all():
        raise ValueError('Finite prefix predictors required')
    return (result, names + labels)
