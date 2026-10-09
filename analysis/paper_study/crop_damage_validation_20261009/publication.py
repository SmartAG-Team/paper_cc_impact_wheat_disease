"""Current field-yield figures, tables and source paths for the manuscript."""
from pathlib import Path
import json
import pandas as pd
from .run import HERE,ROOT,COLLECTION
from .figures import tunisia_figure,briwecs_figure,LABELS
from analysis.paper_study.nordic_yield_validation_20261009.figures import validation_figure

NORDIC=ROOT/'analysis/paper_study/nordic_yield_validation_20261009'


def supplementary_figures(destination):
    validation_figure(destination,'figS22_nordic_yield_validation')
    tunisia_figure(destination,'figS23_leaf_yield_validation')
    briwecs_figure(destination,'figS24_german_yield_validation')
    return {
      'figS22_nordic_yield_validation':
        'Figure S22 | Nordic yield-response prediction and assessment stage. (a) Same-date severity reduction and relative treated–untreated yield response, retaining negative responses. (b) Entire-trial-held-out prediction on 62 contrasts from five trials. (c) RMSE differences from the matched training mean for the 51-contrast, four-trial population with recorded growth stage. Whiskers are 95% paired trial-bootstrap intervals conditional on fixed held-out predictions. The nonnegative stage model is a secondary sign diagnostic. (d) Forward transfer from 2024 to 2025, with 38 contrasts in three evaluation trials; the training-selected model is the training mean. The data describe protection responses against multiple diseases, rather than isolated Septoria loss or measured HAD.',
      'figS23_leaf_yield_validation':
        'Figure S23 | Leaf-rank severity and yield-response transfer in Tunisia. (a) Flag-leaf incidence-weighted lesion area at GS75 and observed yield for 82 plot-season outcomes in 2018–2019. (b) Absolute-yield errors under two-way whole-year transfer, with years given equal weight. (c) Observed protected-reference yield gaps and unprotected flag-leaf severity in 40 paired 2019 outcomes. (d) Entire-mixture-held-out prediction in that single environment. The lower leaf was assessed at GS61 in 2018 and GS73 in 2019, on different dates from the flag leaf. Protected severity and functional leaf area are unobserved; the two leaf assessments do not establish a disease time integral. Yield moisture basis is unspecified.',
      'figS24_german_yield_validation':
        'Figure S24 | German multi-environment protection-response prediction. (a) Unprotected Septoria severity and observed relative protection response for 3,264 cultivar–management comparisons from 16 site-years at five locations. Negative responses remain in the data. Treatment means match cultivar, nitrogen and source-coded water regime; drought-suffixed references are excluded. (b) Whole-site-year-held-out errors, with site-years weighted equally. (c) Whole-geographic-site-held-out errors, with locations weighted equally. (d) Forward transfer from 2015–2017 to 619 comparisons in six site-years during 2018–2019. German yields share a dry-mass basis, which cancels in the relative response. Missing severity remains unknown. Protection targets multiple diseases; disease-assessment dates and final-leaf ranks are undocumented.'}


def model_label(name):
    extras={'origin_linear':'Severity, linear through origin','origin_exponential':'Severity, exponential',
        'affine_ridge':'Severity + intercept','stage_ridge':'Severity + assessment stage',
        'nested_selection':'Nested model selection','training_selected':'Training-selected model',
        'two_stage_nonnegative':'Early + late severity, nonnegative','two_stage_ridge':'Early + late severity',
        'positive_stage_linear':'Stage response, nonnegative'}
    return extras.get(name,LABELS.get(name,name.replace('_',' ')))


def supplementary_tables():
    tables=[]
    rows=[['Source','Matched yield outcomes','Predictive comparisons','Environmental coverage','Measurement support']]
    for row in [
        ['Nordic extension','181 eligible treatment means','62 treated–untreated contrasts','5 prediction trials; 11 inventory trials','Dated symptoms; leaf ranks unconfirmed in prediction trials'],
        ['Tunisia','82 unprotected plots; 40 protected references','82 cross-year yields; 40 reference gaps','2 years at one site; references only in 2019','Flag and lower leaf on different dates; protected severity missing'],
        ['Germany','Cultivar treatment means from original site files','3,264 matched protection responses','16 site-years; 5 locations','Whole-plot severity; assessment dates and leaf ranks unspecified']]:rows.append(row)
    tables.append((rows,'Table S17 | Additional field evidence for disease-related yield-response evaluation. '
        'The Tunisia/Nordic combined source inventory contains 423 assessments and 272 yield keys; '
        'excluding post-harvest measurements and one pooled-control alias leaves 412 assessments and 263 outcomes. '
        'Counts retain plot versus treatment-mean grain and do not represent independent environments. '
        'The German archive is separate from this combined inventory. None supplies compatible repeated leaf-area and harvested-yield measurements sufficient to validate the climate HAD conversion.'))
    m=pd.read_csv(NORDIC/'model_comparison_metrics.csv')
    rows=[['Population / validation','Model','n / trials','RMSE (pp)','Mean benchmark (pp)','95% ΔRMSE (pp)']]
    selections=[('endpoint','leave_trial_out',['training_mean','origin_linear','origin_exponential','affine_ridge','nested_selection']),
        ('stage_known','leave_trial_out',['training_mean','affine_ridge','stage_ridge','nested_selection']),
        ('two_assessments','leave_trial_out',['training_mean','two_stage_nonnegative','two_stage_ridge']),
        ('endpoint','2024_to_2025',['training_mean','origin_linear','affine_ridge','training_selected'])]
    for domain,validation,models in selections:
        q=m[m.domain.eq(domain)&m.validation.eq(validation)].set_index('model')
        label={'endpoint':'Endpoint','stage_known':'Stage recorded','two_assessments':'Two assessments'}[domain]+(' / trial holdout' if validation=='leave_trial_out' else ' / 2024→2025')
        for model in models:
            r=q.loc[model];interval='Not estimated' if pd.isna(r.RMSE_difference_lower95_pp) else f'{r.RMSE_difference_lower95_pp:+.2f} to {r.RMSE_difference_upper95_pp:+.2f}'
            rows.append([label,model_label(model),f'{int(r.contrasts)} / {int(r.trials)}',f'{r.RMSE_pp:.2f}',f'{r.training_mean_RMSE_pp:.2f}',interval])
    sensitivity=pd.read_csv(NORDIC/'positive_stage_comparison.csv').set_index('model').loc['positive_stage_linear']
    rows.append(['Stage recorded / secondary',model_label('positive_stage_linear'),'51 / 4',f'{sensitivity.RMSE_pp:.2f}','5.79',f'{sensitivity.RMSE_difference_lower95_pp:+.2f} to {sensitivity.RMSE_difference_upper95_pp:+.2f}'])
    tables.append((rows,'Table S18 | Nordic grouped yield-response comparisons. '
        'The target is 1 − untreated yield / treated yield; values are percentage points. '
        'All candidates use their own common-population training-mean benchmark. '
        'Whole trials retain shared controls and repeated disease assessments. '
        'Nested selection uses training trials only. Intervals resample complete trials conditional on fixed held-out predictions and exclude refitting uncertainty. '
        'The nonnegative stage row is a secondary diagnostic after inspection of the unconstrained stage-model slopes. '
        'All specified model scores remain in Source Data; fitted coefficients accompany the held-out prediction CSVs in the evidence archive.'))
    m=pd.read_csv(HERE/'model_comparison_metrics.csv')
    rows=[['Prediction task','Model','n / groups','RMSE','Mean benchmark','Units']]
    for domain,label in [('absolute_year_transfer','Between years'),('within_2019_mixture','2019 mixture holdout'),('within_2019_replicate','2019 replicate holdout')]:
        for r in m[m.domain.eq(domain)].itertuples():
            rows.append([label,model_label(r.model),f'{r.n} / {r.groups}',f'{r.RMSE:.2f}',f'{r.baseline_RMSE:.2f}','t ha⁻¹' if r.units=='t ha-1' else 'percentage points'])
    tables.append((rows,'Table S19 | Tunisian leaf-rank models under grouped yield evaluation. '
        'Absolute-yield transfer gives the two years equal weight. Within-2019 response is 1 − unprotected yield / source-matched protected yield, '
        'with complete mixture compositions or source replicate groups excluded from fitting. '
        'The latter tests remain within one environment. Flag-leaf and lower-leaf predictors are observed incidence-weighted lesion percentages; '
        'protected severity is unavailable. Cultivar-composition predictors use four source cultivar proportions. '
        'The nonnegative model constrains higher damage to lower absolute yield or a larger relative protected-reference gap. '
        'One negative measured reference gap remains in all compatible comparisons.'))
    m=pd.read_csv(HERE/'briwecs_model_comparison_metrics.csv')
    rows=[['Population / validation','Model','n / groups','RMSE (pp)','Mean benchmark (pp)']]
    for r in m.itertuples():
        domain='Unprotected endpoint' if r.domain=='unprotected_endpoint' else 'Both severities observed'
        validation={'leave_site_year_out':'site-year holdout','leave_location_out':'site holdout','2015_2017_to_2018_2019':'2018–2019 transfer'}[r.validation]
        rows.append([domain+' / '+validation,model_label(r.model),f'{r.n} / {r.groups}',f'{r.RMSE_pp:.2f}',f'{r.baseline_RMSE_pp:.2f}'])
    tables.append((rows,'Table S20 | German protection-response models under spatial and temporal transfer. '
        'All comparisons use source original site files, matching cultivar, nitrogen and source-coded water regime. '
        'The response is 1 − unprotected treatment-mean yield / protected treatment-mean yield. '
        'Drought-suffixed references and nonpositive yields are excluded; missing disease measurements are not imputed. '
        'Site-year and temporal scores weight site-years equally; geographic transfer weights locations equally. '
        'The observed-reduction subset requires recorded severity in both treatment groups, with its own benchmark. '
        'Treatment means retain their different disease and yield replicate coverage. These protection responses concern multiple diseases and are not causal estimates of STB-specific loss.'))
    return tables


def source_paths():
    sources=[]
    for directory in [HERE,NORDIC]:
        sources.extend(p for p in sorted(directory.glob('*.csv')) if 'held_out_predictions' not in p.name)
    sources.extend(COLLECTION/name for name in ['analysis_ready/combined_severity_yield_assessments.csv','analysis_ready/combined_yield_units.csv'])
    return sources
