"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/yield_evidence_20261007')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from calibration.yield_transfer import fit_group_balanced_slope, leave_group_out
HERE = evidence_dir('analysis/paper_study/yield_evidence_20261007')
ROOT = PROJECT_ROOT

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def published_mean_contrasts():
    source = HERE / 'stb_independent/foulkes2006_table2_paired_predicted_means.csv'
    frame = pd.read_csv(source)
    for kind, protected, affected, absolute in [('HAD', 'healthy_HAD_GLAI_day', 'diseased_HAD_GLAI_day', 'derived_HAD_loss_GLAI_day'), ('yield', 'healthy_yield_t_ha_15pct_moisture', 'diseased_yield_t_ha_15pct_moisture', 'derived_yield_loss_t_ha')]:
        frame[absolute] = frame[protected] - frame[affected]
        frame[f'derived_{kind}_loss_fraction'] = frame[absolute] / frame[protected]
    frame['genetic_background'] = frame.genotype.str.split().str[0].replace({'M.': 'M. Huntsman'})
    assert len(frame) == 8 and frame.genetic_background.nunique() == 5
    weights = 1 / frame.groupby('genetic_background').genetic_background.transform('size')
    weights = weights / weights.sum()
    outcomes = []
    for mode, xcol, ycol, unit in [('absolute', 'derived_HAD_loss_GLAI_day', 'derived_yield_loss_t_ha', 't ha-1'), ('relative', 'derived_HAD_loss_fraction', 'derived_yield_loss_fraction', 'fraction of protected yield')]:
        x = frame[xcol].to_numpy(float)
        y = frame[ycol].to_numpy(float)
        held = leave_group_out(x, y, frame.genetic_background.to_numpy())
        frame[f'{mode}_held_out_prediction'] = held['prediction']
        frame[f'{mode}_training_only_slope'] = held['slope']
        frame[f'{mode}_training_backgrounds'] = held['training_groups']
        frame[f'{mode}_residual'] = held['prediction'] - y
        mean = np.sum(weights * y)
        sse = np.sum(weights * (held['prediction'] - y) ** 2)
        tss = np.sum(weights * (y - mean) ** 2)
        slope = fit_group_balanced_slope(x, y, frame.genetic_background.to_numpy())
        for background in frame.genetic_background.unique():
            train = frame.genetic_background.ne(background).to_numpy()
            test = ~train
            group_sizes = frame.loc[train].groupby('genetic_background').genetic_background.transform('size').to_numpy()
            w = 1 / group_sizes
            expected = np.dot(w * x[train], y[train]) / np.dot(w * x[train], x[train])
            np.testing.assert_allclose(frame.loc[test, f'{mode}_training_only_slope'], expected, rtol=0, atol=1e-15)
            np.testing.assert_allclose(frame.loc[test, f'{mode}_held_out_prediction'], expected * x[test], rtol=0, atol=1e-15)
        outcomes.append(dict(response=mode, contrasts=8, genetic_backgrounds=5, evaluation='leave-one-genetic-background-out; published aggregate treatment means', units=unit, group_balanced_MAE=float(np.sum(weights * np.abs(held['prediction'] - y))), group_balanced_RMSE=float(np.sqrt(sse)), group_balanced_bias=float(np.sum(weights * (held['prediction'] - y))), group_balanced_R2=float(1 - sse / tss), full_data_slope=slope, slope_units='t ha-1 per top-five GLAI-day' if mode == 'absolute' else 'relative yield loss per relative top-five HAD loss', raw_plot_validation=False, current_top3_yield_calibration=False))
    frame['evaluation_weight'] = weights
    return (frame, pd.DataFrame(outcomes))
