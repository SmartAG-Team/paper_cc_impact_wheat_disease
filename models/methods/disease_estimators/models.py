"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_models_20261012')
import numpy as np
from analysis.methods.severity_probability.data import weights

def metadata(frame):
    if frame.global_target_index.duplicated().any():
        raise ValueError('Duplicate fitting target')
    return dict(training_targets=sorted(frame.global_target_index.astype(int).tolist()), training_sites=sorted(frame.site_id.unique().tolist()))

def leaf_means(frame):
    w = weights(frame)
    y = frame.observed_percent.to_numpy(float) / 100
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError('Severity must be in 0–100 percent')
    pooled = float(np.average(y, weights=w))
    means = np.full(7, pooled)
    for leaf in sorted(frame.leaf_index.unique()):
        mask = frame.leaf_index.to_numpy() == leaf
        means[int(leaf)] = np.average(y[mask], weights=w[mask])
    return means
