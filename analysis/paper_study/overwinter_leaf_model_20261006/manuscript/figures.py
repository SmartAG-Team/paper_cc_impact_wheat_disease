"""Source-backed static manuscript figures; no model fitting."""
from pathlib import Path
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.paper_study.full_validation_20261007.figures import render_main
from analysis.paper_study.full_validation_20261007.workflow_figure import render_workflow
from analysis.paper_study.yield_evidence_20261007.figures import main_yield
from analysis.paper_study.map_restoration_20261007.figures import main_maps

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
COLORS = {'disease': '#235e83', 'benchmark': '#898989', 'stage37': '#235e83', 'stage39': '#52796f'}


def export(figure, destination, stem):
    for suffix in ('png', 'pdf', 'svg'):
        figure.savefig(destination/f'{stem}.{suffix}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def main(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    captions = {}
    captions['fig1_framework'] = render_workflow(destination, export)
    captions['fig2_flag_stage_validation'] = render_main(destination, export)

    detection = pd.read_csv(STUDY/'disease/assessment_detection_metrics.csv')
    onsets = pd.read_csv(STUDY/'disease/onset_bracket_metrics.csv')
    parts = ['reused_BASF2019','reused_strict_Corteva']
    fig, axes = plt.subplots(1,3,figsize=(11,4))
    for model, offset, label, color in [('overwinter_source_model',-.18,'Seasonal source model',COLORS['disease']),
        ('phenology_only',.18,'Phenology benchmark',COLORS['benchmark'])]:
        a = detection.loc[detection.model.eq(model)&detection.scenario.eq('baseline')&detection.leaf_scope.eq('top3')].set_index('partition').loc[parts]
        b = onsets.loc[onsets.model.eq(model)&onsets.scenario.eq('baseline')&onsets.leaf_scope.eq('top3')&onsets.censoring.eq('two_sided')].set_index('partition').loc[parts]
        for ax, values, title in [(axes[0],b.distance_days.to_numpy(),'First-symptom interval excess (days)'),
            (axes[1],a.sensitivity.to_numpy()*100,'Positive-assessment sensitivity (%)'),
            (axes[2],a.specificity.to_numpy()*100,'Zero-assessment specificity (%)')]:
            ax.bar(np.arange(2)+offset,values,width=.34,color=color,label=label)
            for x, value in zip(np.arange(2)+offset,values):
                ax.text(x,value+(.13 if ax is axes[0] else 1.4),f'{value:.2f}' if ax is axes[0] else f'{value:.1f}',ha='center',fontsize=8)
            ax.set_xticks(range(2),['BASF 2019','Corteva']); ax.set_ylabel(title); ax.grid(axis='y',color='#e5e5e5',linewidth=.7); ax.set_axisbelow(True)
    axes[0].set_ylim(0,6.4); axes[1].set_ylim(0,110); axes[2].set_ylim(0,110)
    axes[0].legend(frameon=False,fontsize=8,loc='upper left'); fig.tight_layout()
    export(fig,destination,'fig3_onset_and_detection')
    captions['fig3_onset_and_detection'] = ('Figure 3 | Frozen disease timing and assessment detection on reused validation records. '
        'Model selection uses only the original28 BASF2017–2018 field histories and three inner location folds. '
        'The independent phenology benchmark selects effective leaf spacing and thermal delay with the same calibration membership. '
        'The seasonal model increases sensitivity but reduces specificity and has larger first-symptom interval-excess distances. '
        'The top-three assessment counts are252 and1310; genuinely two-sided onset histories number13 and123, respectively. '
        'Height differences are point comparisons, without probabilistic confidence intervals or direct infection-date validation.')
    captions['fig4_timing_and_yield_relevance'] = main_yield(destination, export)
    # Preserve the earlier declared scenario as source evidence.
    source = HERE/'figures/fig4_specified_yield_scenarios.csv'
    target = destination/'fig4_specified_yield_scenarios.csv'
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    climate=pd.read_csv(STUDY/'climate/ensemble_paired_period_changes.csv')
    fig,axes=plt.subplots(1,3,figsize=(11,4.1))
    scenarios=['ssp126','ssp245','ssp585']
    metrics=[('F1_infection_relative_anthesis_days','Change in F1 infection relative to flowering (days)'),
        ('F1_symptom_relative_anthesis_days','Change in F1 symptoms relative to flowering (days)'),
        ('conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai','Change in conditional loss (t ha⁻¹ per unit LAI)')]
    for period,offset,marker,color in [('2031-2060',-.12,'o',COLORS['disease']),('2071-2100',.12,'s',COLORS['stage39'])]:
        for ax,(metric,ylabel) in zip(axes,metrics):
            part=climate.loc[climate.future_period.eq(period)&climate.metric.eq(metric)].set_index('scenario').loc[scenarios]
            values=part.gcm_mean_change.to_numpy();low=part.gcm_min_change.to_numpy();high=part.gcm_max_change.to_numpy()
            ax.errorbar(np.arange(3)+offset,values,yerr=[values-low,high-values],fmt=marker,color=color,capsize=4,label=period)
            ax.axhline(0,color='#555555',linewidth=.9);ax.set_xticks(range(3),['SSP1–2.6','SSP2–4.5','SSP5–8.5'])
            ax.set_ylabel(ylabel);ax.grid(axis='y',color='#e5e5e5',linewidth=.7);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=8);fig.tight_layout();export(fig,destination,'fig5_conditional_climate_changes')
    captions['fig5_conditional_climate_changes']=('Figure 5 | Conditional climate changes in upper-leaf timing and grain-fill healthy-area-duration transfer. '
        'Points average ACCESS-CM2, MPI-ESM1-2-HR and MRI-ESM2-0; whiskers span the three climate-model point estimates, not probabilistic intervals. '
        'Future periods are paired to the same model/SSP-specific1991–2020 reference on common valid draw-year pairs. '
        'The registered64 area-proportional draws retain separate identities across62 unique cells. '
        'Yield-transfer changes use nominal top-three LAI1 and b=0.018 t ha⁻¹ per GLAI-day; actual crop yield and functional green-area loss are unobserved. '
        'Spatial Monte Carlo errors and coverage accompany the source tables. Ecological-parameter, host-transfer, observation and future-management uncertainties are additional and unquantified.')
    captions['fig6_european_spatial_results'] = main_maps(destination, export)
    (destination/'captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    return captions


if __name__=='__main__':
    main(HERE/'figures')
