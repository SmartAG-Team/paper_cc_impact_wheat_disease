"""Observed management-grain evidence with source-specific estimands."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import ROOT,sha
from .figures import export,clean,BLUE,GOLD,GREY

NORDIC=ROOT/'analysis/paper_study/nordic_archive_validation_20261009/outputs'
MIXTURE=ROOT/'analysis/paper_study/mixture_adaptation_20261009/results'


def read(directory,name):
    receipt=json.loads((directory/'analysis_receipt.json').read_text())
    hashes=receipt.get('output_sha256',receipt.get('results_sha256'))
    if hashes.get(name)!=sha(directory/name):
        raise ValueError('Management evidence changed: '+str(directory/name))
    return json.loads((directory/name).read_text()) if name.endswith('.json') else pd.read_csv(directory/name)


def headline():
    n=read(NORDIC,'descriptive_summary.csv').query("group_by == 'all'").iloc[0]
    f=read(MIXTURE,'french_summary.csv').set_index('endpoint')
    uncertainty=read(NORDIC,'uncertainty.json')['intervals'][0]
    return n,f,uncertainty


def empirical_management_figure(destination):
    nordic=read(NORDIC,'paired_management_grain.csv')
    france=read(MIXTURE,'french_contrasts.csv')
    n,f,uncertainty=headline()
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.7))
    fig.subplots_adjust(left=.085,right=.98,bottom=.21,top=.83,wspace=.32)
    ax=axes[0]
    ordered=nordic.sort_values('gain_t_ha')
    ax.step(ordered.gain_t_ha,100*ordered.trial_weight.cumsum()/ordered.trial_weight.sum(),where='post',color=BLUE,lw=1.8)
    ax.axvline(0,color=GREY,lw=.8,ls='--');ax.set_ylim(0,101);clean(ax)
    ax.set_xlabel('Treated − untreated grain yield (t ha⁻¹)',fontsize=9)
    ax.set_ylabel('Cumulative trial-weighted comparisons (%)',fontsize=9)
    ax.set_title('a   Nordic/Baltic management comparisons',loc='left',fontsize=10,weight='bold',pad=12)
    ax.text(.97,.12,f'307 contrasts · 263 trial identifiers\nMean {n.mean_gain_t_ha:+.3f} t ha⁻¹\n95% cluster interval {uncertainty["lower"]:.3f}–{uncertainty["upper"]:.3f}',
        transform=ax.transAxes,ha='right',va='bottom',fontsize=9,color=BLUE)
    ax=axes[1]
    for column,color,label in [('delta_t_ha',GOLD,'Raw harvests'),('blup_delta_t_ha',BLUE,'Author spatial adjustment')]:
        x=np.sort(france[column].to_numpy())
        ax.step(x,100*np.arange(1,len(x)+1)/len(x),where='post',color=color,lw=1.6,label=label)
    ax.axvline(0,color=GREY,lw=.8,ls='--');ax.set_ylim(0,101);clean(ax)
    ax.set_xlabel('Mixture − constituent grain baseline (t ha⁻¹)',fontsize=8.8)
    ax.set_ylabel('Cumulative mixture comparisons (%)',fontsize=9)
    ax.set_title('b   French mixture experiment · one site-year',loc='left',fontsize=10,weight='bold',pad=12)
    ax.legend(frameon=False,loc='upper left',fontsize=8)
    raw=f.loc['raw_primary'];adjusted=f.loc['author_BLUP_same_comparisons_sensitivity']
    ax.text(.97,.08,f'195 mixtures · shared controls\nRaw mean {raw.mean_delta_t_ha:+.3f} t ha⁻¹\n{100*raw.below_baseline_fraction:.1f}% below constituent baseline',
        transform=ax.transAxes,ha='right',va='bottom',fontsize=8.8,color=GOLD)
    stem='fig5_observed_management_grain_responses'
    export(fig,destination,stem)
    nordic.to_csv(destination/f'{stem}_nordic.csv',index=False)
    france.to_csv(destination/f'{stem}_france.csv',index=False)
    return stem,('Figure 5 | Observed management-associated grain responses. '
        '(a) Cumulative distribution of 307 treated-minus-untreated contrasts across 263 reported trial identifiers in five Nordic/Baltic countries, 2012–2016. '
        'Each trial has unit total weight, divided between contrasts sharing its control. The mean interval resamples 25 country–year groups. '
        'Yields use 15% grain moisture; disease-specific observations and plot replication are unavailable. '
        '(b) Raw mixture-minus-constituent differences for 195 complete comparisons in the randomized Mauguio durum-wheat experiment, 2018. '
        'The baseline averages both constituent pure stands. Source spatially adjusted estimates provide sensitivity on the same comparisons. '
        'Dried-grain moisture is unspecified; the two panels are separate populations and are not pooled. '
        'Mixtures occupy alternating rows and received fungicide after early disease assessment. Shared controls connect 189 French comparisons; no independent-environment confidence interval is available. '
        'These distributions describe observed management outcomes, not future failure probabilities, STB-mediated protection or validated climate-adaptation benefits.')


def supplementary_tables():
    n=read(NORDIC,'descriptive_summary.csv')
    rows=[['Grouping','Category','Contrasts','Trial identifiers','Mean gain (t ha⁻¹)','Negative share (%)']]
    for r in n.itertuples():
        rows.append([r.group_by.replace('_',' '),str(r.group),str(r.pairs),str(r.trials),f'{r.mean_gain_t_ha:.3f}',f'{100*r.negative_pair_fraction_trial_weighted:.1f}'])
    tables=[(rows,'Table S29 | Nordic/Baltic observed management-associated grain responses. '
        'Observed treated-minus-untreated yields retain 15% moisture units. Each trial has unit total weight, divided among its retained contrasts. '
        'Subgroups are descriptive and do not identify treatment effects across countries, crops or numbers of applications. '
        'The archive has no observed disease severity, leaf-specific canopy trajectory or verified healthy-yield reference.')]
    scores=read(NORDIC,'benchmark_metrics.csv')
    rows=[['Held-out group','Benchmark','Contrasts','RMSE (t ha⁻¹)','MAE (t ha⁻¹)','MSE skill']]
    for r in scores.itertuples():
        rows.append([r.scheme.replace('_',' '),r.model.replace('_',' '),str(r.pairs),f'{r.rmse_t_ha:.3f}',f'{r.mae_t_ha:.3f}',f'{r.mse_skill_vs_training_mean:.3f}'])
    tables.append((rows,'Table S30 | Transfer of fixed Nordic/Baltic management benchmarks. '
        'Country–year holdouts are primary; country, year and retrospective forward-year holdouts are secondary. '
        'Category means and fallbacks use training observations only. Common controls remain in the same evaluation group. '
        'No benchmark improves the primary training-mean RMSE; these models do not test disease severity or physiological crop response.'))
    f=read(MIXTURE,'french_summary.csv')
    rows=[['Yield endpoint','Comparisons','Mean difference (t ha⁻¹)','Mean paired relative difference (%)','Below baseline (%)','Lower decile (t ha⁻¹)']]
    for r in f.itertuples():
        rows.append([r.endpoint.replace('_',' '),str(r.n_mixtures),f'{r.mean_delta_t_ha:.3f}',f'{r.mean_relative_percent:.2f}',f'{100*r.below_baseline_fraction:.1f}',f'{r.empirical_p10_delta_t_ha:.3f}'])
    tables.append((rows,'Table S31 | French mixture performance against constituent pure stands. '
        'Both endpoints use the same 195 complete comparisons and 156 distinct constituent controls from one site-year. '
        'Mean relative differences average pair-specific ratios and differ from a ratio of overall means. '
        'Source spatial BLUPs use the full trial and are descriptive sensitivity, not independent validation. Shared controls and unreplicated genotype combinations prevent an independent-environment interval. '
        'The comparison includes mixtures without disease scores; it is not an additional independent disease cohort.'))
    rows=[['Evidence domain','Observed grain comparison','Transfer and decision boundary']]
    rows.extend([
        ['Nordic/Baltic management','307 contrasts; 263 reported trial identifiers; 25 country–year groups','Observed yield association; no cause-specific disease attribution or evaluated treatment cost'],
        ['French mixtures','195 complete constituent-baseline comparisons; one environment','Unreplicated experimental lines and alternating rows; no commercial-adoption or climate-transfer evidence'],
        ['Swiss constituent controls','0 of 637 eligible binary mixtures have complete within-trial constituent controls','National-network controls are not exchangeable with the mixture experiment; matched-source differences remain noncausal'],
        ['Climate and adaptation','Annual canopy damage and fixed-production exposure','No validated disease-specific grain-loss or future intervention-benefit conversion']])
    tables.append((rows,'Table S32 | Management comparison applicability and policy limits. '
        'Observed treatment-associated yield responses and conditional climate projections have distinct targets, populations and units. '
        'Observed mean gains do not supply regional intervention recommendations, pesticide-reduction targets, net returns or avoided tonnes lost.'))
    return tables


def source_paths():
    names={NORDIC:['descriptive_summary.csv','paired_management_grain.csv','benchmark_metrics.csv','excluded_source_records.csv'],
           MIXTURE:['french_summary.csv','french_contrasts.csv','french_dependency_groups.csv','french_measured_tradeoffs.csv',
                    'swiss_summary.csv','swiss_same_trial_constituent_gate.csv','swiss_transfer_summary.csv']}
    result=[]
    for directory,files in names.items():
        for name in files:
            read(directory,name);result.append(directory/name)
    return result
