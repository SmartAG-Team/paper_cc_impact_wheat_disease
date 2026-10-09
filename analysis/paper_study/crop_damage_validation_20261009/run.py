"""Grouped field tests of observed leaf damage and harvested wheat yield.

Tunisia and Nordic observations retain their measurement definitions. Severity
associations do not identify physiological damage or cause-specific yield loss.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COLLECTION = ROOT / 'data/stb_collection_20261009'
SOURCE = COLLECTION / 'sources/public_septoria/paper-model-data-20261009/derived'
COMPOSITION = ['INRAT100', 'karim', 'salim', 'monastir']
MODELS = {
    'training_mean': [],
    'composition_ridge': COMPOSITION,
    'flag_ridge': ['severity_flag'],
    'lower_ridge': ['severity_lower'],
    'two_leaf_ridge': ['severity_flag', 'severity_lower'],
    'composition_two_leaf_ridge': COMPOSITION + ['severity_flag', 'severity_lower'],
    'nonnegative_damage': ['severity_flag', 'severity_lower'],
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare():
    combined = pd.read_csv(COLLECTION / 'analysis_ready/combined_severity_yield_assessments.csv')
    original = pd.read_csv(SOURCE / 'tunisia2020_damage_yield_plot_seasons.csv')
    assert not original.duplicated(['year', 'treatment', 'replicate']).any()
    plots = original.copy()
    plots['yield_t_ha'] = plots.kg_per_ha / 1000
    plots['severity_flag'] = plots.PLACL_weighted_flag / 100
    plots['severity_lower'] = plots.PLACL_weighted_flag_minus_1 / 100
    for column in COMPOSITION:
        plots[column] /= 100
    plots['composition_id'] = plots[COMPOSITION].astype(str).agg('|'.join, axis=1)
    assert np.allclose(plots[COMPOSITION].sum(axis=1), 1)
    # Treatment codes are year-specific; composition identifies shared genotypes.
    assert plots.groupby(['year', 'treatment']).composition_id.nunique().eq(1).all()
    references = pd.read_csv(SOURCE / 'tunisia2019_protected_reference_yield_contrasts.csv')
    contrasts = references.merge(plots[['year', 'treatment', 'replicate', 'plot_season_id',
        'severity_flag', 'severity_lower', 'composition_id']],
        on=['year', 'treatment', 'replicate'], validate='one_to_one')
    for column in COMPOSITION:
        contrasts[column] /= 100
    contrasts['response_fraction'] = contrasts.relative_yield_gap_pct / 100
    np.testing.assert_allclose(contrasts.response_fraction,
        1 - contrasts.kg_per_ha_unprotected / contrasts.kg_per_ha_protected)
    assert len(contrasts) == len(references)
    all_yields = pd.read_csv(SOURCE / 'tunisia2020_all_plot_yields.csv')
    paired = combined[combined.dataset_id.eq('tunisia2020')]
    check = paired.merge(plots[['plot_season_id', 'yield_t_ha']],
        left_on=paired.yield_unit_id.str.removeprefix('tunisia2020_'),
        right_on='plot_season_id', validate='many_to_one', suffixes=('', '_plot'))
    np.testing.assert_allclose(check.yield_t_ha, check.yield_t_ha_plot)
    nordic = pd.read_csv(COLLECTION / 'sources/public_septoria/nfts-extension-20261009/nfts_stb_severity_yield_matched.csv')
    original_nordic = ROOT / 'data/public_septoria/nfts-extension-20261009/nfts_stb_severity_yield_matched.csv'
    if original_nordic.is_file():
        assert sha(COLLECTION / 'sources/public_septoria/nfts-extension-20261009/nfts_stb_severity_yield_matched.csv') == sha(original_nordic)
    late = pd.to_datetime(nordic.date_disease) > pd.to_datetime(nordic.date_yield)
    eligible = nordic[~late & ~nordic.source_duplicate_control]
    quality = dict(combined_assessment_records=len(combined),
        combined_source_yield_units=int(combined.yield_unit_id.nunique()),
        combined_clean_assessment_records=len(paired) + len(eligible),
        combined_clean_yield_units=len(plots) + len(eligible[['registry_id','treatment_code']].drop_duplicates()),
        tunisia_plot_yields=len(plots), tunisia_leaf_assessments=len(paired),
        tunisia_years=int(plots.year.nunique()), tunisia_locations=1,
        tunisia_reference_contrasts=len(contrasts),
        tunisia_negative_reference_contrasts=int(contrasts.response_fraction.lt(0).sum()),
        protected_yield_observations=int(all_yields.inoculate_no_fungicide.eq(0).sum()),
        protected_severity_observations=0,
        nordic_postharvest_assessments_excluded=int(late.sum()),
        nordic_pooled_control_alias_assessments_excluded=int(nordic.source_duplicate_control.sum()),
        leaf_stage_scope=plots.groupby('year')[['assessment_GS_flag','assessment_GS_flag_minus_1']].first().reset_index().to_dict('records'),
        limitations=[
            'Protected-plot disease severity is unobserved; no zero severity or severity reduction is imputed.',
            'Flag and lower leaves were assessed on different dates; one observation per leaf rank cannot establish HAD or AUDPC.',
            'The lower-leaf stage differs between years (GS61 versus GS73); leaf, stage and environmental effects cannot be separately identified.',
            'Within-2019 mixture and replicate holdouts remain within one environment, rather than external environmental validation.',
            'Tunisia durum wheat and Nordic bread-wheat treatment means differ in crop type, grain and severity definition.',
            'Tunisian yield moisture is unspecified; no cross-source absolute-yield or severity coefficient is fitted.',
            'Treatment-associated yield contrasts do not identify a cause-specific STB coefficient.'])
    return plots, contrasts, quality


def weights(frame, group):
    w = 1 / frame.groupby(group)[group].transform('size').to_numpy(float)
    return w / w.sum()


def fit_predict(train, test, model, features, target, group):
    w = weights(train, group)
    y = train[target].to_numpy(float)
    mean = float(w @ y)
    if model == 'training_mean':
        return np.full(len(test), mean), dict(training_mean=mean)
    x, z = train[features].to_numpy(float), test[features].to_numpy(float)
    assert np.isfinite(x).all() and np.isfinite(z).all()
    center = w @ x
    if model == 'nonnegative_damage':
        direction = -1 if target == 'yield_t_ha' else 1
        coefficients, _ = nnls(np.sqrt(w)[:,None] * (x-center), np.sqrt(w) * direction * (y-mean))
        prediction = mean + direction * (z-center) @ coefficients
        return prediction, dict(coefficients=coefficients.tolist(), direction=direction,
            training_center=center.tolist(), training_mean=mean, features=features)
    scale = np.sqrt(w @ ((x-center)**2))
    scale = np.where(scale > 1e-12, scale, 1)
    a, b = (x-center)/scale, (z-center)/scale
    coefficient = np.linalg.solve((a*w[:,None]).T@a + .1*np.eye(len(features)), (a*w[:,None]).T@(y-mean))
    return mean + b@coefficient, dict(coefficients=coefficient.tolist(),
        training_mean=mean, training_center=center.tolist(), training_scale=scale.tolist(),
        ridge_penalty=.1, features=features)


def grouped_predictions(frame, group, model, features, target):
    rows = []
    for held in sorted(frame[group].unique()):
        train = frame[frame[group].ne(held)]
        test = frame[frame[group].eq(held)].copy()
        assert set(train[group]).isdisjoint(set(test[group]))
        prediction, parameters = fit_predict(train, test, model, features, target, group)
        test['prediction'] = prediction
        test['observed'] = test[target]
        test['model'] = model
        test['validation_group'] = group
        test['training_groups'] = '|'.join(map(str, sorted(train[group].unique())))
        test['parameters'] = json.dumps(parameters, sort_keys=True)
        rows.append(test)
    return pd.concat(rows, ignore_index=True)


def score(predictions, group, scale):
    w = weights(predictions, group)
    observed = predictions.observed.to_numpy(float)
    error = predictions.prediction.to_numpy(float) - observed
    mse = float(w @ error**2)
    variance = float(w @ (observed-w@observed)**2)
    return dict(RMSE=scale*np.sqrt(mse), MAE=scale*float(w@abs(error)),
        bias=scale*float(w@error), R2=1-mse/variance if variance>0 else None,
        n=len(predictions), groups=int(predictions[group].nunique()))


def run():
    plots, contrasts, quality = prepare()
    protocol = dict(analysis='Leaf-damage and yield associations in the expanded public field collection',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [
            COLLECTION/'analysis_ready/combined_severity_yield_assessments.csv',
            SOURCE/'tunisia2020_damage_yield_plot_seasons.csv',
            SOURCE/'tunisia2019_protected_reference_yield_contrasts.csv']},
        models=MODELS, ridge_penalty=.1,
        primary='Whole-year transfer of absolute yield; both directions, two years at one site.',
        secondary='Within-2019 protected-reference response, holding out whole mixtures or source replicate groups.',
        weights='Equal held-out groups, then equal plot outcomes; repeated leaf measurements share one outcome.',
        model_selection='All specified models reported; no performance-based promotion into climate yield conversion.',
        missing='No severity values invented for protected plots; no severity-to-HAD conversion.',
        uncertainty='Two environments cannot support a reliable environmental confidence interval.')
    path = HERE/'protocol_before_fit.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'The recorded protocol differs.'
    else:
        path.write_text(json.dumps(protocol, indent=2)+'\n')
    metrics, all_predictions = [], []
    for domain, frame, group, target, scale, units in [
        ('absolute_year_transfer', plots, 'year', 'yield_t_ha', 1, 't ha-1'),
        ('within_2019_mixture', contrasts, 'composition_id', 'response_fraction', 100, 'percentage points'),
        ('within_2019_replicate', contrasts, 'replicate', 'response_fraction', 100, 'percentage points')]:
        predictions = {}
        for model, features in MODELS.items():
            p = grouped_predictions(frame, group, model, features, target)
            p['domain'] = domain
            predictions[model] = p
            all_predictions.append(p)
        baseline = score(predictions['training_mean'], group, scale)['RMSE']
        for model, p in predictions.items():
            s = score(p, group, scale)
            metrics.append(dict(domain=domain, model=model, units=units, **s,
                baseline_RMSE=baseline, skill=1-(s['RMSE']/baseline)**2))
            if group == 'year':
                for year, part in p.groupby('year'):
                    single = score(part, group, scale)
                    b = score(predictions['training_mean'][predictions['training_mean'].year.eq(year)], group, scale)['RMSE']
                    metrics.append(dict(domain=f'{2018 if year==2019 else 2019}_to_{year}',model=model,
                        units=units,**single,baseline_RMSE=b,skill=1-(single['RMSE']/b)**2))
    plots.to_csv(HERE/'tunisia_plot_features.csv',index=False)
    contrasts.to_csv(HERE/'tunisia_reference_features.csv',index=False)
    pd.concat(all_predictions,ignore_index=True).to_csv(HERE/'held_out_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(HERE/'model_comparison_metrics.csv',index=False)
    receipt=dict(data_quality=quality,protocol_sha256=sha(path),
        analysis_inputs=protocol['source_hashes'],
        outputs={p.name:sha(p) for p in HERE.glob('*.csv')},
        climate_yield_conversion_changed=False,
        interpretation='Grouped tests of severity/yield associations; no externally validated leaf-specific physiological yield-loss function.')
    (HERE/'analysis_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(quality,indent=2))
    print(pd.DataFrame(metrics)[['domain','model','RMSE','baseline_RMSE','skill','R2']].to_string(index=False))


if __name__ == '__main__':
    run()
