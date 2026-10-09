"""Bounded exploratory Swiss yield association test; no physiological claims."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import Bundle, prediction_gate, sha256

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE/'protocol.json').read_text())


def fit_predict(train, test, features):
    allowed = set(sum(PROTOCOL['models'].values(),[]))
    if not set(features) <= allowed:
        raise ValueError('Forbidden predictor: '+str(set(features)-allowed))
    y = train.yield_t_ha.to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite training yields')
    weights = 1. / train.groupby('site_year').site_year.transform('size').to_numpy(float)
    weights *= len(train)/weights.sum()
    ymean = np.average(y,weights=weights)
    if not features:
        return np.full(len(test),ymean), {}
    train_columns, test_columns, unseen = [], [], {}
    for field in features:
        if train[field].isna().any() or test[field].isna().any():
            raise ValueError('Missing predictor: '+field)
        if field in PROTOCOL['categorical']:
            a, b = train[field].astype(str).str.strip(), test[field].astype(str).str.strip()
            levels = sorted(a.unique())
            unseen[field] = int((~b.isin(levels)).sum())
            for level in levels:
                train_columns.append(a.eq(level).to_numpy(float))
                test_columns.append(b.eq(level).to_numpy(float))
        else:
            a, b = train[field].to_numpy(float), test[field].to_numpy(float)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite predictor: '+field)
            mean = np.average(a,weights=weights)
            sd = np.sqrt(np.average((a-mean)**2,weights=weights)) or 1.
            train_columns.append((a-mean)/sd)
            test_columns.append((b-mean)/sd)
    x, xt = np.column_stack(train_columns), np.column_stack(test_columns)
    mean = np.average(x,axis=0,weights=weights)
    centered = x-mean
    # Weighted ridge with an unpenalized intercept; all preprocessing uses training data.
    lhs = centered.T @ (centered*weights[:,None]) + PROTOCOL['ridge_alpha']*np.eye(x.shape[1])
    beta = np.linalg.solve(lhs,centered.T @ (weights*(y-ymean)))
    return ymean+(xt-mean)@beta, unseen


def main():
    bundle = Bundle()
    d = bundle.csv('analysis_ready/swiss_stb_yield_724.csv')
    gate = prediction_gate(d)
    if not gate['eligible']:
        raise SystemExit('Grouped evidence gate failed; no model fitting')
    if d.record_id.duplicated().any():
        raise ValueError('Repeated plot IDs')
    all_predictions, diagnostics, unknown = [], [], []
    for split, group in [('leave_one_site_out','postcode'),('leave_one_site_year_out','site_year')]:
        for fold in sorted(d[group].unique()):
            train, test = d[d[group]!=fold].copy(), d[d[group]==fold].copy()
            diagnostic = dict(split=split,fold=str(fold),train_plots=len(train),test_plots=len(test),
                train_sites=train.postcode.nunique(),test_sites=test.postcode.nunique(),
                train_site_years=train.site_year.nunique(),test_site_years=test.site_year.nunique(),
                overlapping_split_groups=len(set(train[group])&set(test[group])),
                overlapping_plot_ids=len(set(train.record_id)&set(test.record_id)),
                overlapping_sites=len(set(train.postcode)&set(test.postcode)),
                overlapping_site_years=len(set(train.site_year)&set(test.site_year)),
                training_id_sha256=sha256('\n'.join(sorted(train.record_id)).encode()),
                testing_id_sha256=sha256('\n'.join(sorted(test.record_id)).encode()))
            if diagnostic['overlapping_split_groups'] or diagnostic['overlapping_plot_ids']:
                raise ValueError('Holdout leakage')
            diagnostics.append(diagnostic)
            for model, features in PROTOCOL['models'].items():
                prediction, unseen = fit_predict(train,test,features)
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite predictions')
                out = test[['record_id','site_year','postcode','year','yield_t_ha']].copy()
                out = out.rename(columns={'yield_t_ha':'observed_t_ha'})
                out['split'],out['fold'],out['model'] = split,str(fold),model
                out['predicted_t_ha'] = prediction
                out['residual_t_ha'] = prediction-out.observed_t_ha
                all_predictions.append(out)
                for field, count in unseen.items():
                    unknown.append(dict(split=split,fold=str(fold),model=model,feature=field,
                                        unseen_test_rows=count,test_rows=len(test)))
    pred = pd.concat(all_predictions,ignore_index=True)
    pred.to_csv(HERE/'heldout_predictions.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(HERE/'fold_diagnostics.csv',index=False)
    pd.DataFrame(unknown).to_csv(HERE/'unseen_categories.csv',index=False)
    pred['squared_error'] = pred.residual_t_ha**2
    pred['absolute_error'] = pred.residual_t_ha.abs()
    environment = pred.groupby(['split','model','postcode','site_year'],as_index=False).agg(
        n=('record_id','size'),mse=('squared_error','mean'),mae=('absolute_error','mean'),bias=('residual_t_ha','mean'))
    environment.to_csv(HERE/'environment_errors.csv',index=False)
    summary = []
    for (split,model), part in pred.groupby(['split','model']):
        env = environment[environment.split.eq(split)&environment.model.eq(model)]
        baseline = environment[environment.split.eq(split)&environment.model.eq('training_mean')].mse.mean()
        management = environment[environment.split.eq(split)&environment.model.eq('management')].mse.mean()
        mse = env.mse.mean()
        summary.append(dict(split=split,model=model,plots=len(part),site_years=len(env),sites=part.postcode.nunique(),
            rmse_environment_equal_t_ha=np.sqrt(mse),mae_environment_equal_t_ha=env.mae.mean(),
            bias_environment_equal_t_ha=env.bias.mean(),rmse_pooled_t_ha=np.sqrt(part.squared_error.mean()),
            mse_skill_vs_training_mean=1-mse/baseline,mse_skill_vs_management=1-mse/management))
    metrics = pd.DataFrame(summary)
    metrics.to_csv(HERE/'swiss_model_comparison.csv',index=False)
    site_errors = environment[environment.split.eq('leave_one_site_out')].groupby(['model','postcode']).mse.mean().unstack('model')
    comparisons, boot = [], []
    rng = np.random.default_rng(PROTOCOL['seed'])
    indices = rng.integers(0,len(site_errors),size=(PROTOCOL['bootstrap_replicates'],len(site_errors)))
    for base, augmented in [('management','management_stb'),('management_codisease','management_codisease_stb')]:
        delta = site_errors[augmented]-site_errors[base]
        for site in site_errors.index:
            comparisons.append(dict(site=site,baseline=base,augmented=augmented,
                baseline_mse=site_errors.loc[site,base],augmented_mse=site_errors.loc[site,augmented],
                delta_mse=delta.loc[site],augmented_better=bool(delta.loc[site]<0)))
        means = delta.to_numpy()[indices].mean(axis=1)
        boot.append(dict(baseline=base,augmented=augmented,sites=len(delta),
            sites_improved=int(delta.lt(0).sum()),mean_delta_mse=delta.mean(),
            descriptive_p025=float(np.quantile(means,.025)),descriptive_p975=float(np.quantile(means,.975)),
            interpretation='Five-site descriptive bootstrap stability interval conditional on fitted OOF predictions; not a confirmatory confidence interval'))
    pd.DataFrame(comparisons).to_csv(HERE/'site_paired_errors.csv',index=False)
    pd.DataFrame(boot).to_csv(HERE/'site_bootstrap_stability.csv',index=False)
    outputs = ['heldout_predictions.csv','fold_diagnostics.csv','unseen_categories.csv','environment_errors.csv',
               'swiss_model_comparison.csv','site_paired_errors.csv','site_bootstrap_stability.csv']
    receipt = dict(status='complete',analysis_status='exploratory; specified after data inspection, before model fitting; not preregistered',
        protocol_sha256=sha256((HERE/'protocol.json').read_bytes()),source_bundle_sha256=sha256(bundle.path.read_bytes()),
        code_sha256=sha256(Path(__file__).read_bytes()),
        models=list(PROTOCOL['models']),unique_plots=len(d),sites=d.postcode.nunique(),site_years=d.site_year.nunique(),
        total_fits=len(diagnostics)*len(PROTOCOL['models']),heldout_prediction_rows=len(pred),
        outcome_leakage_controls='Fold-local encoding/scaling/fitting; harvest-derived covariates excluded; whole sites or site-years excluded',
        primary_split='leave_one_site_out',secondary_split='leave_one_site_year_out',
        temporal_transfer_tested=False,causal_adaptation_tested=False,mechanistic_validation_established=False,
        ordinal_percent_conversion=False,models_promoted_to_climate_simulation=False,
        gate=gate,uncertainty=boot,software=dict(numpy=np.__version__,pandas=pd.__version__),
        output_hashes={name:sha256((HERE/name).read_bytes()) for name in outputs})
    (HERE/'prediction_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(metrics[metrics.split.eq('leave_one_site_out')].to_string(index=False))
    print(json.dumps(boot,indent=2))


if __name__ == '__main__':
    main()
