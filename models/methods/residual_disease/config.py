"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
SEED = 20261011
JOINT = dict(ridge=0.001, onset_weight=0.1)
TRAJECTORY = dict(onset_ridge=0.01, severity_ridge=0.001, integral_weight=1.0)
RF_COMMON = dict(n_estimators=200, max_features=0.8, bootstrap=True, max_samples=0.8, n_jobs=1, random_state=SEED)
RF_GRID = (dict(max_depth=4, min_samples_leaf=30), dict(max_depth=6, min_samples_leaf=15), dict(max_depth=None, min_samples_leaf=30))
HGB_COMMON = dict(max_iter=200, learning_rate=0.05, early_stopping=False, max_depth=3, random_state=SEED)
HGB_GRID = (dict(max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=0.1), dict(max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=1.0), dict(max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=1.0))
MULTIPLIERS = (0.0, 0.25, 0.5, 1.0)
FAMILY_ORDER = ('fold_leaf_mean', 'joint_base', 'trajectory_base', 'leaf_bias', 'direct_rf', 'joint_rf', 'joint_rf_geo', 'trajectory_rf', 'joint_hgb')
ML_FAMILIES = dict(joint_rf=('joint', 'rf', False), joint_rf_geo=('joint', 'rf', True), trajectory_rf=('trajectory', 'rf', False), joint_hgb=('joint', 'hgb', False), direct_rf=(None, 'rf', False))
