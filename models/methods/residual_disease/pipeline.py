"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from analysis.methods.severity_probability.data import weights
from analysis.methods.joint_disease import benchmarks
from analysis.methods.residual_disease import base, features, heads, config, study

def prepare(training, validation, train_bounds, test_bounds, fields, z, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for key in ['field_id', 'global_target_index']:
        if set(training[key]) & set(validation[key]):
            raise ValueError('Training/validation overlap in ' + key)
    training.to_csv(output / 'training_membership.csv', index=False)
    validation.to_csv(output / 'validation_membership.csv', index=False)
    oof = base.crossfit(training, train_bounds, z, output / 'crossfit')
    fitted = base.fit_pair(training, train_bounds, z, output / 'final_base')
    prediction, events = base.predict_pair(fitted, z, validation, test_bounds)
    leaf = benchmarks.fit_leaf_mean(training)
    prediction['fold_leaf_mean'] = benchmarks.predict_leaf_mean(leaf, prediction.leaf_index)
    study.write(output / 'leaf_mean.json', leaf)
    bias = heads.fit_bias(oof, oof.joint_residual)
    study.write(output / 'leaf_bias.json', bias)
    prediction['bias_residual'] = heads.predict_bias(bias, prediction.leaf_index)
    return (oof, prediction, events)

def fit_head(family, setting, oof, validation, fields, z, output):
    name, kind, geography = config.ML_FAMILIES[family]
    training_base = features.base_columns(oof, name) if name else None
    validation_base = features.base_columns(validation, name) if name else None
    x, names = features.design(z, fields, oof, training_base, geography=geography)
    v, vnames = features.design(z, fields, validation, validation_base, geography=geography)
    target = oof.observed_percent.to_numpy(float) if name is None else oof[name + '_residual'].to_numpy(float)
    fitted = heads.fit(kind, setting, x, target, weights(oof), names)
    predicted = heads.predict(fitted, v, vnames)
    path = Path(output) / 'heads'
    path.mkdir(exist_ok=True)
    filename = f'{family}_{setting}'
    joblib.dump(fitted, path / (filename + '.joblib'), compress=3)
    metadata = {k: v for k, v in fitted.items() if k != 'estimator'}
    metadata.update(family=family, base=name, geography=geography, target_definition='observed severity' if name is None else 'observed minus site-cross-fitted base severity', training_targets=sorted(oof.global_target_index.astype(int).tolist()), training_sites=sorted(oof.site_id.unique()), residual_training_sha256=study.sha(Path(output) / 'crossfit/residual_training.csv'), model_sha256=study.sha(path / (filename + '.joblib')))
    study.write(path / (filename + '_metadata.json'), metadata)
    if kind == 'rf':
        pd.DataFrame(dict(feature=names, importance=fitted['estimator'].feature_importances_)).to_csv(path / (filename + '_importances.csv'), index=False)
    return predicted

def apply_spec(spec, prediction, raw):
    family = spec['family']
    if family in ['joint_base', 'trajectory_base', 'fold_leaf_mean']:
        return prediction[family].to_numpy(float)
    if family == 'leaf_bias':
        return heads.correct(prediction.joint_base, prediction.bias_residual, spec['multiplier'])
    residual = raw[family, spec['setting']]
    if family == 'direct_rf':
        return np.clip(residual, 0, 100)
    name = config.ML_FAMILIES[family][0]
    return heads.correct(prediction[name + '_base'], residual, spec['multiplier'])

def candidates(training, validation, train_bounds, test_bounds, fields, z, output):
    if set(training.site_id) & set(validation.site_id):
        raise ValueError('Inner held-site leakage')
    output = Path(output)
    oof, prediction, events = prepare(training, validation, train_bounds, test_bounds, fields, z, output)
    raw = {}
    for family, (_, kind, _) in config.ML_FAMILIES.items():
        for setting in range(len(config.RF_GRID if kind == 'rf' else config.HGB_GRID)):
            raw[family, setting] = fit_head(family, setting, oof, prediction, fields, z, output)
            prediction[f'raw_{family}_{setting}'] = raw[family, setting]
    for spec in study.candidate_specs():
        prediction[f"candidate_{spec['candidate_id']}"] = apply_spec(spec, prediction, raw)
    prediction.to_csv(output / 'validation_predictions.csv', index=False)
    events.to_csv(output / 'base_onset_predictions.csv', index=False)
    return (prediction, events)

def selected(training, validation, train_bounds, test_bounds, fields, z, choices, output):
    output = Path(output)
    oof, prediction, events = prepare(training, validation, train_bounds, test_bounds, fields, z, output)
    keys = {(spec['family'], spec['setting']) for spec in choices.values() if spec['family'] in config.ML_FAMILIES}
    raw = {}
    for family, setting in sorted(keys):
        raw[family, setting] = fit_head(family, setting, oof, prediction, fields, z, output)
        prediction[f'raw_{family}_{setting}'] = raw[family, setting]
    for name, spec in choices.items():
        prediction[name] = apply_spec(spec, prediction, raw)
    study.write(output / 'selected_choices.json', choices)
    prediction.to_csv(output / 'validation_predictions.csv', index=False)
    events.to_csv(output / 'base_onset_predictions.csv', index=False)
    return (prediction, events)
