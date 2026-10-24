"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_coupled_ml_20261012')
SEED = 20261012
FAMILIES = ['fold_leaf_mean', 'state_baseline', 'weather_rf', 'history_rf', 'history_hgb']
STRENGTHS = [0.25, 0.5, 0.75, 1.0]
RF_GRID = [dict(max_depth=4, min_samples_leaf=20), dict(max_depth=8, min_samples_leaf=10), dict(max_depth=None, min_samples_leaf=5)]
HGB_GRID = [dict(max_leaf_nodes=7, min_samples_leaf=20), dict(max_leaf_nodes=15, min_samples_leaf=10)]
MODELS = ['selected_pipeline', *FAMILIES, 'mixture_uniform', 'mixture_composite']
REFERENCES = ['joint', 'fold_leaf_mean', 'boosting', 'full_window', 'previous_residual_mixture']
