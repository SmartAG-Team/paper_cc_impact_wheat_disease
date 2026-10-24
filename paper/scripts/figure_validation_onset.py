"""Complementary fixed-forecast calibration, error coverage and onset displays."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,LogNorm
from matplotlib.lines import Line2D
from .figure_validation_v2 import STYLE,COLORS,MARKERS,EVALUATIONS,_box,_label
from .validation_error_coverage import error_coverage


def validation_onset_figure(data,cohorts):
    """Show agreement, forecast calibration, full error coverage and onset timing."""
    calibration=data['calibration'].copy();calibration['panel']='b'
    coverage=error_coverage(data['points']);coverage['panel']='c'
    bracketed=cohorts['bracketed_histories'].copy();bracketed['panel']='d'
    phases=['calibration','evaluation']
    bracketed['phase_position']=bracketed.phase.map(dict(zip(phases,[0.,1.])))
    bracketed['plot_x']=np.nan
    for phase in phases:
        ix=bracketed.index[bracketed.phase.eq(phase)]
        bracketed.loc[ix,'plot_x']=bracketed.loc[ix,'phase_position']+np.linspace(-.16,.16,len(ix))
    bracketed['plot_y']=bracketed.signed_interval_excess_days
    with plt.rc_context(STYLE):
        height_cm=17.5
        fig=plt.figure(figsize=(19/2.54,height_cm/2.54))
        a=fig.add_axes([.12,.58,.34,.33])
        b=fig.add_axes([.64,.58,.33,.33])
        c=fig.add_axes([.12,.10,.34,.29])
        d=fig.add_axes([.64,.10,.33,.29])
        density=data['density'][data['density'].evaluation.eq('temporal')]
        mass=density.sort_values(['observed_lower','predicted_lower']).scientific_mass_percent.to_numpy().reshape(40,40)
        positive=mass[mass>0]
        cmap=LinearSegmentedColormap.from_list('weighted_density',['#D5E4ED','#8FBBD3','#397DA6','#073B61'])
        mesh=a.pcolormesh(np.linspace(0,100,41),np.linspace(0,100,41),np.ma.masked_equal(mass.T,0),
            cmap=cmap,norm=LogNorm(vmin=positive.min(),vmax=positive.max()),rasterized=True)
        a.plot([0,100],[0,100],ls='--',color='#555555',lw=1)
        a.set(xlim=(0,100),ylim=(0,100),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100],
            xlabel='Observed disease (%)',ylabel='Predicted disease (%)',aspect='equal')
        cax=fig.add_axes([.12,.49,.34,.018])
        bar=fig.colorbar(mesh,cax=cax,orientation='horizontal')
        ticks=[t for t in [.001,.01,.1,1,10,100] if positive.min()<=t<=positive.max()]
        bar.set_ticks(ticks,labels=[f'{t:g}' for t in ticks]);bar.minorticks_off()
        bar.set_label('Weight per cell (%)',fontsize=12,labelpad=3)
        b.plot([0,100],[0,100],ls='--',color='#666666',lw=1,zorder=0)
        for i,design in enumerate(EVALUATIONS):
            colour=COLORS[design];marker=MARKERS[design];linestyle='-' if i==0 else '--'
            q=calibration[calibration.evaluation.eq(design)].sort_values('bin_lower')
            b.plot(q.predicted_mean,q.observed_mean,color=colour,marker=marker,linestyle=linestyle,markersize=3.5,lw=1.1)
            has_ci=q.lower95.notna()&q.upper95.notna()
            b.vlines(q.loc[has_ci,'predicted_mean'],q.loc[has_ci,'lower95'],q.loc[has_ci,'upper95'],color=colour,lw=.9)
            for bound in ['lower95','upper95']:
                b.plot(q.loc[has_ci,'predicted_mean'],q.loc[has_ci,bound],linestyle='none',marker='_',color=colour,markersize=4)
            q=coverage[coverage.evaluation.eq(design)]
            c.fill_between(q.threshold_pp,q.lower95,q.upper95,step='post',color=colour,alpha=.12,lw=0,zorder=1)
            c.step(q.threshold_pp,q.covered_weight_percent,where='post',color=colour,ls=linestyle,lw=1.25,zorder=3)
            marks=q[q.threshold_pp.isin([10,20,30,40,50,60,70,80,90,100])]
            c.plot(marks.threshold_pp,marks.covered_weight_percent,linestyle='none',marker=marker,color=colour,markersize=3,zorder=4)
        b.set(xlim=(0,100),ylim=(0,100),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100],
            xlabel='Mean predicted disease (%)',ylabel='Mean observed disease (%)')
        c.set(xlim=(0,100),ylim=(0,102),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100],
            xlabel='Absolute error (pp)',ylabel='Evaluation weight (%)')
        c.grid(axis='y',color='#E5E6E7',lw=.6);c.set_axisbelow(True)
        handles=[Line2D([],[],color=COLORS[design],marker=MARKERS[design],ls='-' if i==0 else '--',lw=1.1,
            markersize=4,label='Temporal' if design=='temporal' else 'Spatial, 50 km') for i,design in enumerate(EVALUATIONS)]
        legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.995),ncol=2,frameon=False,
            fontsize=12,borderpad=0,borderaxespad=0,labelspacing=.35,handlelength=1.7,handletextpad=.5)
        for phase in phases:
            q=bracketed[bracketed.phase.eq(phase)]
            d.scatter(q.plot_x,q.plot_y,s=35,marker='o',edgecolors='#7B3F8C',lw=1.1,
                facecolors='white' if phase=='calibration' else '#7B3F8C',zorder=4)
        d.axhline(0,color='#747A7E',ls='--',lw=.9,zorder=0)
        d.set(xlim=(-.45,1.45),ylim=(-11,5),yticks=[-10,-5,0,5],xticks=[0,1],
            xticklabels=['Calibration\n(n=3)','Evaluation\n(n=4)'],ylabel='Onset deviation\n(days)')
        d.grid(axis='y',color='#E5E6E7',lw=.6);d.set_axisbelow(True);d.tick_params(axis='x',pad=2)
        for ax,letter in zip([a,b,c,d],'abcd'):_box(ax);_label(ax,letter)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for ax in [a,b,c,d]:
            for item in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]:
                box=item.get_window_extent(renderer)
                assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height,('clipped figure4 text',item.get_text())
                assert not legend.get_window_extent(renderer).overlaps(box),('legend overlaps text',item.get_text())
            assert not legend.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer)),('legend covers data',ax.get_xlabel())
        for ax in [c,d]:assert not cax.xaxis.label.get_window_extent(renderer).overlaps(ax.texts[-1].get_window_extent(renderer))
        assert not a.xaxis.label.get_window_extent(renderer).overlaps(cax.get_window_extent(renderer)), 'Density xlabel overlaps colourbar'
        assert not cax.xaxis.label.get_window_extent(renderer).overlaps(c.get_window_extent(renderer)), 'Colourbar label covers error data'
    spec=dict(width_cm=19.,height_cm=height_cm,font_pt=12,
        panels={'a':'primary temporal assessment-level severity density','b':'forecast calibration in both evaluation designs',
            'c':'full weighted absolute-error CDFs and site-bootstrap intervals','d':'signed onset errors, calibration and evaluation separate'},
        density_evaluation='temporal',density_bin_pp=2.5,calibration_bins_pp=10,
        calibration_resamples=1000,calibration_is_evaluation_forecast_diagnostic=True,
        error_coverage_full_range_pp=[0,100],error_coverage_resamples=2000,
        error_coverage_weighting='original evaluation hierarchy; not raw record fraction',
        F1_field_seasons=55,positive_field_seasons=52,calibration_field_seasons=22,evaluation_field_seasons=33,
        onset_calibration_histories=3,onset_evaluation_histories=4,calibration_is_validation=False,
        onset_operator='unchanged first daily N+I+D crossing at0.001',
        error='signed distance to daily interval: negative early, positive late, zero within observed support',
        exact_onset_MAE_identified=False,small_sample=True,onset_legend_overlaps_density_label=False,
        row_groups='calibration2017/2018 and evaluation2019; no pooled validation result',
        jitter='deterministic horizontal offsets to reveal coincident field-level errors')
    tables={'calibration':calibration,'error_coverage':coverage,'onset_field_histories':cohorts['field_histories'],
        'onset_coverage_by_year':cohorts['coverage_by_year'],'onset_bracketed_histories':bracketed,'onset_phase_metrics':cohorts['phase_metrics']}
    return fig,tables,spec


def onset_eligibility(overview,flag_records):
    """Reconcile all2019 fields using observations alone, before model scoring."""
    overview=overview[overview.season_year.eq(2019)].copy()
    if not set(flag_records.field_id).issubset(set(overview.field_id)):
        raise ValueError('Flag-leaf histories must belong to the displayed2019 cohort')
    rows=[]
    for field,all_rows in overview.groupby('field_id',sort=True):
        flag=flag_records[flag_records.field_id.eq(field)].sort_values('date')
        positive=flag[flag.observed_percent.gt(0)]
        if flag.empty:status='no_qualified_F1_observation'
        elif positive.empty:status='no_positive_F1_assessment'
        elif flag[flag.date.lt(positive.date.iloc[0])&flag.observed_percent.eq(0)].empty:
            status='F1_positive_at_first_assessment'
        else:status='two_sided_F1_onset'
        rows.append(dict(field_id=field,source=all_rows.source.iloc[0],season_year=2019,
            site_id=all_rows.site_id.iloc[0],all_leaf_assessments=len(all_rows),
            qualified_F1_assessments=len(flag),qualified_F1_dates=flag.date.nunique(),
            first_F1_date=flag.date.iloc[0] if len(flag) else None,
            last_F1_date=flag.date.iloc[-1] if len(flag) else None,
            first_F1_score=flag.observed_percent.iloc[0] if len(flag) else np.nan,
            positive_F1_assessments=int(flag.observed_percent.gt(0).sum()),
            zero_F1_assessments=int(flag.observed_percent.eq(0).sum()),onset_eligibility=status))
    return pd.DataFrame(rows)
