"""Validation distributions and stage-specific observation-window figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .evidence import HERE, PARTITIONS

GREY = '#59646e'
BLUE = '#235e83'
FIELD_STAGES = [31,32,33,37,39,51,65,85]
FIELD_SOURCES = {
    'reused_development_2019': dict(label='BASF 2019',color=GREY,marker='o',open=True),
    'reused_external_strict': dict(label='Corteva location-disjoint',color=BLUE,marker='s',open=False),
}


def field_records(partitions):
    frame = pd.read_csv(HERE/'field_stage_scores_all.csv')
    frame = frame.loc[frame.forcing_scope.eq('full_season') & frame.genuinely_bracketed
        & frame.partition.isin(partitions)].copy()
    for column in ['lower_exclusive','upper_inclusive','predicted_date']:
        frame[column] = pd.to_datetime(frame[column])
    frame['observed_first_possible_date'] = frame.lower_exclusive+pd.Timedelta(days=1)
    frame['observed_last_possible_date'] = frame.upper_inclusive
    reference = pd.to_datetime(frame.season_year.astype(str)+'-01-01')
    for column, target in [('observed_first_possible_date','observed_lower_doy'),
            ('observed_last_possible_date','observed_upper_doy'),('predicted_date','predicted_doy')]:
        frame[target] = (frame[column]-reference).dt.days+1
    frame['interval_centre_doy_for_display'] = (frame.observed_lower_doy+frame.observed_upper_doy)/2.
    return frame


def handles(sources):
    return [Line2D([],[],marker=v['marker'],linestyle='none',markersize=5,
        markerfacecolor='white' if v['open'] else v['color'],markeredgecolor=v['color'],label=v['label'])
        for v in sources.values()]


def render_main(destination, export):
    destination = Path(destination)
    frames = []
    for cohort in ['validation','testing']:
        frame = pd.read_parquet(HERE/f'german_{cohort}_tpv_events.parquet')
        frames.append(frame.loc[frame.common_matched])
    german = pd.concat(frames,ignore_index=True)
    german[['cohort','PEP_ID','SOWING_DATE','BBCH','true_date','predicted_date','error_days']].to_csv(
        destination/'fig2_german_stage_errors.csv',index=False)
    field = field_records(FIELD_SOURCES)
    field.to_csv(destination/'fig2_field_stage_intervals.csv',index=False)
    metrics = pd.read_csv(HERE/'german_stage_metrics_all.csv')
    fig, axes = plt.subplots(1,2,figsize=(9.,5.6),gridspec_kw={'width_ratios':[1,1.1]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.29,top=.81,wspace=.28)
    stages = [10,31,51,85]
    for cohort, label, color, offset in [('validation','Station validation',GREY,-.18),
            ('testing','Station test',BLUE,.18)]:
        samples = [german.loc[german.cohort.eq(cohort)&german.BBCH.eq(s),'error_days'].to_numpy() for s in stages]
        axes[0].boxplot(samples,positions=np.arange(4)+offset,widths=.29,whis=(5,95),showfliers=False,
            patch_artist=True,manage_ticks=False,
            boxprops=dict(facecolor='white' if cohort=='validation' else '#dbe7ef',edgecolor=color,linewidth=1.1),
            medianprops=dict(color=color,linewidth=1.4),whiskerprops=dict(color=color,linewidth=1.),
            capprops=dict(color=color,linewidth=1.))
    counts = [f'{s}\n'+ '\n'.join(f'{int((german.cohort.eq(c)&german.BBCH.eq(s)).sum()):,}'
        for c in ['validation','testing']) for s in stages]
    axes[0].set_xticks(range(4),counts,fontsize=8.)
    axes[0].set(xlim=(-.6,3.6),ylim=(-33,33),ylabel='Predicted − observed date (days)',xlabel='BBCH stage; n: validation, then test')
    axes[0].set_title('a   German station holdouts',loc='left',fontsize=10.4,weight='bold',pad=10)
    axes[0].legend(handles=[Line2D([],[],color=GREY,linewidth=4,label='Station validation'),
        Line2D([],[],color=BLUE,linewidth=4,label='Station test')],loc='lower left',bbox_to_anchor=(0,1.12),
        frameon=False,fontsize=8.5,ncol=2,handlelength=1.2,columnspacing=1.)
    notes = []
    for cohort,label in [('validation','Validation'),('testing','Test')]:
        row = metrics.loc[metrics.cohort.eq(cohort)&metrics.model.eq('T-P-V')
            &metrics.stage.eq('overall')&metrics.denominator.eq('three_model_shared')].iloc[0]
        notes.append(f'{label}: MAE {row.mae_days:.2f} d; RMSE {row.rmse_days:.2f} d')
    axes[0].text(0,-.40,'\n'.join(notes),transform=axes[0].transAxes,fontsize=8.6,va='top',linespacing=1.5)
    counts = []
    for index, stage in enumerate(FIELD_STAGES):
        ns = []
        for offset,(partition,style) in zip([-.19,.19],FIELD_SOURCES.items()):
            rows = field.loc[field.partition.eq(partition)&field.event.eq(stage)].sort_values(['signed_distance_days','field_id'])
            ns.append(len(rows))
            for error, repeated in rows.groupby('signed_distance_days'):
                shifts = np.linspace(-.075,.075,len(repeated)) if len(repeated)>1 else [0.]
                axes[1].scatter(index+offset+np.asarray(shifts),np.full(len(repeated),error),s=13,
                    marker=style['marker'],facecolors='white' if style['open'] else style['color'],
                    edgecolors=style['color'],linewidths=.7,alpha=.65,zorder=3)
        counts.append(f'{stage}\n{ns[0]}/{ns[1]}')
    axes[1].set_xticks(range(8),counts,fontsize=8.4)
    axes[1].set(xlim=(-.6,7.6),ylim=(-26,21),ylabel='Signed interval-excess distance (days)',
        xlabel='BBCH stage; n = BASF / Corteva')
    axes[1].set_title('b   European field validation',loc='left',fontsize=10.4,weight='bold',pad=10)
    axes[1].legend(handles=handles(FIELD_SOURCES),loc='lower left',bbox_to_anchor=(0,1.12),
        frameon=False,fontsize=8.5,ncol=2,handlelength=1.1,columnspacing=.8)
    axes[1].text(0,-.40,'Zero: predicted date within the observed window\nBBCH10: no two-sided field intervals',
        transform=axes[1].transAxes,fontsize=8.6,va='top',linespacing=1.5)
    for ax in axes:
        ax.axhline(0,color='#767676',linestyle='--',linewidth=.9,zorder=1)
        ax.grid(axis='y',color='#e8e8e8',linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y',labelsize=9)
    export(fig,destination,'fig2_flag_stage_validation')
    return ('Figure 2 | Validation across the retained crop-development stages. '
        '(a) German station-held-out T–P–V date errors at BBCH10,31,51 and85, using the common matched events '
        'of the archived GDD, T–P and T–P–V comparison. Boxes show the interquartile range, lines the median and whiskers '
        'the fifth and ninety-fifth percentiles; individual tails are retained in the source data. Counts beneath each stage give validation followed by test events. '
        'Overall error scores use all shared matched events; missing predictions remain in the coverage tables. '
        '(b) Individual field-stage distances outside genuine two-sided observation intervals at BBCH31,32,33,37,39,51,65 and85. '
        'Negative values indicate premature prediction, positive values delayed prediction and zero an admissible date. '
        'Horizontal displacement separates coincident records within each stage and dataset without changing the date distance. '
        'Counts give BASF2019/location-disjoint Corteva records; zero counts indicate unavailable interval evidence. '
        'Field dates use the frozen parameters and archived full-season weather extension. '
        'The German station holdout validates retained phenology, not the added flag stages or disease outcomes. '
        'Calibration, one-sided constraints, the overlapping full Corteva transfer and archived French transfer are reported separately '
        'in Tables S1–S6 and the Source Data. Exact-date errors and interval-excess distances have different observational resolutions; '
        'box whiskers are distribution percentiles, not confidence intervals.')


def interval_panels(destination, export, stages, stem, sources, number):
    frame = field_records(sources)
    fig,axes = plt.subplots(2,2,figsize=(7.8,8.))
    fig.subplots_adjust(left=.095,right=.975,bottom=.125,top=.91,wspace=.24,hspace=.54)
    for ax,event,letter in zip(axes.flat,stages,'abcd'):
        selected = frame.loc[frame.event.eq(event)]
        if len(selected):
            lo = 5*np.floor((min(selected.observed_lower_doy.min(),selected.predicted_doy.min())-3)/5)
            hi = 5*np.ceil((max(selected.observed_upper_doy.max(),selected.predicted_doy.max())+3)/5)
        else:lo,hi=100,180
        ax.plot([lo,hi],[lo,hi],color='#888888',linestyle='--',linewidth=.9)
        notes = []
        for partition,style in sources.items():
            rows = selected.loc[selected.partition.eq(partition)]
            for row in rows.itertuples():
                ax.hlines(row.predicted_doy,row.observed_lower_doy,row.observed_upper_doy,
                    color=style['color'],linewidth=1,alpha=.35)
                ax.scatter(row.interval_centre_doy_for_display,row.predicted_doy,s=24,
                    marker=style['marker'],facecolors='white' if style['open'] else style['color'],
                    edgecolors=style['color'],linewidths=.8,zorder=3)
            if len(rows):
                short_label = 'Corteva' if partition=='reused_external_strict' else style['label']
                notes.append(f"{short_label}: {int(rows.compatible.sum())}/{len(rows)} within; mean excess {rows.distance_days.mean():.2f} d")
            else:notes.append(f"{style['label']}: no two-sided interval")
        if selected.empty:
            ax.text(.5,.5,'No genuine two-sided intervals',ha='center',va='center',transform=ax.transAxes,fontsize=9.)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed stage window (day of year)',ylabel='Predicted stage date (day of year)')
        ax.set_aspect('equal',adjustable='box')
        ax.set_title(f'{letter}   BBCH{event}',loc='left',fontsize=10.2,weight='bold')
        ax.tick_params(labelsize=8.5)
        ax.xaxis.label.set_size(8.6);ax.yaxis.label.set_size(8.6)
        ax.grid(color='#ededed',linewidth=.6);ax.set_axisbelow(True)
        ax.text(0,-.29,'\n'.join(notes),transform=ax.transAxes,fontsize=8.,va='top',linespacing=1.5)
    legend = handles(sources)+[Line2D([],[],color='#888888',linestyle='--',label='Equal calendar date')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.53,.995),ncol=3,frameon=False,fontsize=9.)
    export(fig,Path(destination),stem)
    return (f'Figure S{number} | '+('Calibration' if 'calibration' in sources else 'Reused field validation')+
        f' at BBCH{stages[0]},{stages[1]},{stages[2]} and{stages[3]}. '
        'Horizontal lines span admissible daily dates from the last below-stage assessment plus one day to the first '
        'at-or-above-stage assessment. Symbols occupy interval centres for display rather than observed event dates. '
        'An interval crossing the equal-date diagonal contains the prediction. Empty panels identify absent two-sided evidence. '
        'All dates use unchanged parameters and the archived full-season forcing; intervals describe observations, not statistical confidence.')


def supplementary(destination, export):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    captions={}
    captions['figS1_field_early_stages']=interval_panels(destination,export,[31,32,33,37],
        'figS1_field_early_stages',FIELD_SOURCES,1)
    captions['figS2_field_late_stages']=interval_panels(destination,export,[39,51,65,85],
        'figS2_field_late_stages',FIELD_SOURCES,2)
    calibration={'calibration':dict(label='BASF 2017–18 calibration',color=GREY,marker='o',open=True)}
    captions['figS3_calibration_early_stages']=interval_panels(destination,export,[31,32,33,37],
        'figS3_calibration_early_stages',calibration,3)
    captions['figS4_calibration_late_stages']=interval_panels(destination,export,[39,51,65,85],
        'figS4_calibration_late_stages',calibration,4)
    (destination/'supplementary_captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    return captions
