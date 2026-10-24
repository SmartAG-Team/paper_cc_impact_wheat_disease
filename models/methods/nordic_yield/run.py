"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/nordic_yield_validation_20261009')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import nnls
HERE = evidence_dir('analysis/paper_study/nordic_yield_validation_20261009')
ROOT = PROJECT_ROOT
SOURCE = ROOT / 'data/final_study/inputs/public_septoria/nfts-extension-20261009'
MATCHED = SOURCE / 'nfts_stb_severity_yield_matched.csv'
OTHER = {'brown': 'Brown rust,cer. % covered', 'yellow': 'Yellow rust % covered', 'mildew': 'Powdery mildew % covered', 'tan': 'Leaf spot, yellow: cereals % covered'}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def prepare():
    original = pd.read_csv(MATCHED)
    assert not original.duplicated(['registry_id', 'treatment_code', 'date_disease']).any()
    assert original.native_unit_verified.all()
    assert original.groupby(['registry_id', 'treatment_code']).yield_t_ha_15percent_moisture.nunique().eq(1).all()
    np.testing.assert_allclose(original.yield_t_ha_15percent_moisture, original.value_yield * 0.1)
    assert original.value_disease.between(0, 100).all()
    post = pd.to_datetime(original.date_disease) > pd.to_datetime(original.date_yield)
    duplicate = original.source_duplicate_control
    clean = original.loc[~post & ~duplicate].copy()
    clean['verified_stage'] = clean.growth_stage_disease.where(clean.growth_stage_disease.between(10, 99))
    clean['source_leaf_number'] = clean.registry_id.map({69307: 3, 71289: 1})
    clean['leaf_definition'] = 'Undocumented'
    clean.loc[clean.source_leaf_number.notna(), 'leaf_definition'] = 'Source-defined leaf number; final-leaf equivalence unverified'
    metadata = {m['registry_id']: m for m in json.loads((SOURCE / 'nfts_trial_metadata.json').read_text())}
    clean['latitude'] = clean.registry_id.map({k: m['latitude'] for k, m in metadata.items()})
    clean['longitude'] = clean.registry_id.map({k: m['longitude'] for k, m in metadata.items()})
    clean['harvest_year'] = pd.to_datetime(clean.date_yield).dt.year
    labels = {}
    for registry, m in metadata.items():
        frame = pd.read_csv(SOURCE / f'nfts_{registry}_treatment_rows.csv').fillna('')
        for code, part in frame.iloc[1:].groupby('col0'):
            if not str(code).isdigit():
                continue
            if m['single_variety']:
                labels[registry, int(code)] = m['single_variety'].strip()
            else:
                components = list(dict.fromkeys(((str(r.col5) + ' ' + str(r.col6)).strip() for r in part.itertuples())))
                labels[registry, int(code)] = ' / '.join(components)
    clean['cultivar_or_mixture_label'] = [labels.get((r.registry_id, r.treatment_code), 'Undocumented') for r in clean.itertuples()]
    endpoints = clean.sort_values('date_disease').groupby(['registry_id', 'treatment_code'], sort=True).tail(1).copy()
    disease = pd.read_csv(SOURCE / 'nfts_disease_observations_long.csv')
    rows = []
    controls = clean[clean.trial_type.eq('disease_control_comparison')]
    for registry, part in controls.groupby('registry_id'):
        source_control = pd.read_csv(SOURCE / f'nfts_{registry}_treatment_rows.csv').fillna('')
        no_control = source_control[source_control.col5.eq('No disease control')]
        assert set(no_control.col0.astype(int)) == {1}, registry
        reference = part[part.treatment_code.eq(1)].set_index('date_disease')
        latest = endpoints[endpoints.registry_id.eq(registry)]
        for row in latest[latest.treatment_code.ne(1)].itertuples():
            if row.date_disease not in reference.index:
                continue
            control = reference.loc[row.date_disease]
            y0 = float(control.yield_t_ha_15percent_moisture)
            yt = float(row.yield_t_ha_15percent_moisture)
            item = dict(registry_id=registry, trial_name=row.trial_name_disease, treatment_code=row.treatment_code, contrast_id=f'{registry}_{row.treatment_code}', harvest_year=row.harvest_year, cultivar_or_mixture_label=row.cultivar_or_mixture_label, assessment_date=row.date_disease, harvest_date=row.date_yield, latest_stage=row.verified_stage, source_leaf_number=row.source_leaf_number, severity_control=float(control.value_disease), severity_treated=float(row.value_disease), delta_severity=float(control.value_disease - row.value_disease) / 100, control_yield_t_ha=y0, treated_yield_t_ha=yt, response_fraction=1 - y0 / yt, observed_gain_kg_ha=1000 * (yt - y0), assessment_lead_days=(pd.Timestamp(row.date_yield) - pd.Timestamp(row.date_disease)).days)
            stem = reference[reference.verified_stage.between(31, 39)]
            item['control_stem_severity'] = np.nan if stem.empty else float(stem.sort_index().iloc[0].value_disease) / 100
            earlier = part[(part.treatment_code == row.treatment_code) & (part.date_disease < row.date_disease) & part.verified_stage.between(31, 64)]
            earlier = earlier[earlier.date_disease.isin(reference.index)]
            if earlier.empty:
                item.update(early_delta_severity=np.nan, early_stage=np.nan, early_date=None)
            else:
                first = earlier.sort_values('date_disease').iloc[-1]
                ref = reference.loc[first.date_disease]
                item.update(early_delta_severity=float(ref.value_disease - first.value_disease) / 100, early_stage=float(first.verified_stage), early_date=first.date_disease)
            for name, label in OTHER.items():
                q = disease[(disease.registry_id == registry) & (disease.measurement == label) & disease.value.notna()]
                q = q[pd.to_datetime(q.date) <= pd.Timestamp(row.date_yield)]
                dates = set(q[q.treatment_code.astype(int) == 1].date) & set(q[q.treatment_code.astype(int) == row.treatment_code].date)
                dates = [date for date in dates if abs((pd.Timestamp(date) - pd.Timestamp(row.date_disease)).days) <= 7]
                if not dates:
                    item[f'delta_{name}'] = np.nan
                else:
                    date = min(dates, key=lambda date: (abs((pd.Timestamp(date) - pd.Timestamp(row.date_disease)).days), date))
                    values = q[q.date == date].set_index(q[q.date == date].treatment_code.astype(int)).value
                    item[f'delta_{name}'] = float(values.loc[1] - values.loc[row.treatment_code]) / 100
            rows.append(item)
    contrasts = pd.DataFrame(rows).sort_values(['registry_id', 'treatment_code']).reset_index(drop=True)
    contrasts['stage_interaction'] = contrasts.delta_severity * (contrasts.latest_stage - 75) / 10
    contrasts['stem_interaction'] = contrasts.delta_severity * contrasts.control_stem_severity
    contrasts['lead_interaction'] = contrasts.delta_severity * contrasts.assessment_lead_days / 50
    quality = dict(source_records=len(original), source_trial_treatment_outcomes=len(original[['registry_id', 'treatment_code']].drop_duplicates()), source_trials=int(original.registry_id.nunique()), postharvest_records=int(post.sum()), duplicate_control_records=int(duplicate.sum()), eligible_records=len(clean), eligible_yield_outcomes=len(endpoints), eligible_trials=int(clean.registry_id.nunique()), unknown_stage_records=int(clean.verified_stage.isna().sum()), explicitly_documented_leaf_number_records=int(clean.source_leaf_number.notna().sum()), disease_control_yield_outcomes=int(endpoints.trial_type.eq('disease_control_comparison').sum()), disease_control_contrasts=len(contrasts), disease_control_trials=int(contrasts.registry_id.nunique()), source_scope='Published treatment means at 15% grain moisture; repeated dates share one yield outcome', limitations=['Disease-control contrasts do not isolate STB from other diseases or fungicide-associated physiology.', 'Disease assessment dates are not infection dates.', 'Leaf area, absorbed radiation and crop carbon balance are not measured in the matched CSV.', 'Source-defined leaf numbers cannot be assigned to the undocumented disease-control records.', 'Reported source yield intervals do not supply the covariance of contrasts sharing one control.'])
    return (clean, endpoints, contrasts, quality)

def weights(frame):
    result = 1 / frame.groupby('registry_id').registry_id.transform('size').to_numpy(dtype=float)
    return result / result.sum()

def fit_predict(train, test, model, features):
    w = weights(train)
    target = train.response_fraction.to_numpy(dtype=float)
    if model == 'training_mean':
        return (np.full(len(test), w @ target), dict(mean=float(w @ target)))
    if model == 'zero_response':
        return (np.zeros(len(test)), {})
    x = train[features].to_numpy(dtype=float)
    z = test[features].to_numpy(dtype=float)
    assert np.isfinite(x).all() and np.isfinite(z).all()
    if model in ['origin_linear', 'origin_exponential', 'two_stage_nonnegative', 'multiple_disease_nonnegative', 'stem_nonnegative', 'positive_stage_linear']:
        transformed = -np.log1p(-target) if model == 'origin_exponential' else target
        coefficients, _ = nnls(np.sqrt(w)[:, None] * x, np.sqrt(w) * transformed)
        values = z @ coefficients
        if model == 'origin_exponential':
            values = -np.expm1(-values)
        return (values, dict(coefficients=coefficients.tolist(), features=features))
    center = w @ x
    scale = np.sqrt(w @ (x - center) ** 2)
    scale = np.where(scale > 1e-12, scale, 1)
    a = (x - center) / scale
    b = (z - center) / scale
    mean = float(w @ target)
    coefficients = np.linalg.solve((a * w[:, None]).T @ a + 0.1 * np.eye(len(features)), (a * w[:, None]).T @ (target - mean))
    return (mean + b @ coefficients, dict(coefficients=coefficients.tolist(), features=features, training_center=center.tolist(), training_scale=scale.tolist(), training_mean=mean, ridge_penalty=0.1))

def leave_trial_out(frame, model, features):
    rows = []
    for trial in sorted(frame.registry_id.unique()):
        train = frame[frame.registry_id != trial]
        test = frame[frame.registry_id == trial].copy()
        assert set(train.registry_id).isdisjoint(set(test.registry_id))
        pred, parameters = fit_predict(train, test, model, features)
        test['prediction'] = pred
        test['model'] = model
        test['training_trials'] = '|'.join((str(v) for v in sorted(train.registry_id.unique())))
        test['fit_parameters'] = json.dumps(parameters, sort_keys=True)
        rows.append(test)
    return pd.concat(rows, ignore_index=True)

def score(frame):
    w = weights(frame)
    y = frame.response_fraction.to_numpy()
    p = frame.prediction.to_numpy()
    e = p - y
    var = float(w @ (y - w @ y) ** 2)
    out = dict(contrasts=len(frame), trials=int(frame.registry_id.nunique()), RMSE_pp=100 * float(np.sqrt(w @ (e * e))), MAE_pp=100 * float(w @ abs(e)), bias_pp=100 * float(w @ e), R2=1 - float(w @ (e * e)) / var if var else None, negative_observed=int((y < 0).sum()), negative_predicted=int((p < 0).sum()), predictions_at_or_above_one=int((p >= 1).sum()))
    if 'control_yield_t_ha' in frame:
        gain = np.where(p < 1, 1000 * frame.control_yield_t_ha.to_numpy() * (1 / (1 - p) - 1), np.nan)
        if np.isfinite(gain).all():
            out['conditional_gain_RMSE_kg_ha'] = float(np.sqrt(w @ (gain - frame.observed_gain_kg_ha.to_numpy()) ** 2))
    return out

def candidates(domain):
    result = {'zero_response': [], 'training_mean': [], 'origin_linear': ['delta_severity'], 'origin_exponential': ['delta_severity'], 'affine_ridge': ['delta_severity']}
    if domain == 'endpoint':
        result['stem_nonnegative'] = ['delta_severity', 'stem_interaction']
        result['lead_ridge'] = ['delta_severity', 'lead_interaction']
        result['multiple_disease_nonnegative'] = ['delta_severity', *[f'delta_{key}' for key in OTHER]]
    if domain == 'stage_known':
        result['stage_ridge'] = ['delta_severity', 'stage_interaction']
    if domain == 'two_assessments':
        result['two_stage_nonnegative'] = ['early_delta_severity', 'delta_severity']
        result['two_stage_ridge'] = ['early_delta_severity', 'delta_severity']
    return result

def choose_training_candidate(frame, choices):
    measures = []
    for model, features in choices.items():
        p = leave_trial_out(frame, model, features)
        measures.append((score(p)['RMSE_pp'], model))
    return (min(measures)[1], {name: value for value, name in measures})

def nested_predictions(frame, choices):
    output = []
    for trial in sorted(frame.registry_id.unique()):
        train = frame[frame.registry_id != trial]
        test = frame[frame.registry_id == trial].copy()
        model, inner = choose_training_candidate(train, choices)
        pred, parameters = fit_predict(train, test, model, choices[model])
        test['prediction'] = pred
        test['model'] = 'nested_selection'
        test['selected_model'] = model
        test['training_trials'] = '|'.join((str(v) for v in sorted(train.registry_id.unique())))
        test['fit_parameters'] = json.dumps(parameters, sort_keys=True)
        test['inner_scores'] = json.dumps(inner, sort_keys=True)
        output.append(test)
    return pd.concat(output, ignore_index=True)

def bootstrap_comparison(predictions, domain):
    sub = predictions[predictions.domain.eq(domain) & predictions.validation.eq('leave_trial_out')]
    trial_mse = sub.assign(sq=(sub.prediction - sub.response_fraction) ** 2).groupby(['model', 'registry_id']).sq.mean().unstack()
    generator = np.random.default_rng(20261009)
    draws = generator.integers(0, trial_mse.shape[1], size=(20000, trial_mse.shape[1]))
    baseline = np.sqrt(trial_mse.loc['training_mean'].to_numpy()[draws].mean(axis=1)) * 100
    records = []
    for model in trial_mse.index:
        rmse = np.sqrt(trial_mse.loc[model].to_numpy()[draws].mean(axis=1)) * 100
        delta = rmse - baseline
        low, high = np.quantile(delta, [0.025, 0.975])
        records.append(dict(domain=domain, model=model, RMSE_difference_lower95_pp=low, RMSE_difference_upper95_pp=high, interval='Paired trial bootstrap conditional on fixed cross-validated predictions; five or fewer trial groups'))
    return records
