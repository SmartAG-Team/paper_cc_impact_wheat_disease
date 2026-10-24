"""Focused disease–heat comparison on the complete-season adaptation panel.

The full preregistered and post-audit support analyses remain in the numerical
archive and Supplementary Information. No simulation or selection is repeated.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.climate.run import digest

COLORS={'Atlantic':'#34769A','Boreal':'#8E6C8A','Continental':'#B87342',
        'Mediterranean':'#887D42','Steppic':'#438A83','Sampled design':'#353B40'}


def build_adaptation_figure(output):
    output=Path(output)
    fixed=pd.read_csv(output/'support_sensitivity_fixed_shifts.csv')
    policy=pd.read_csv(output/'support_sensitivity_same_policy.csv')
    fixed=fixed[fixed.support.eq('complete_season')&fixed.period.eq('future')&
                fixed.model.eq('ensemble')&fixed.environment_region.eq('Sampled design')].sort_values('shift_days')
    policy=policy[policy.support.eq('complete_season')&policy.model.eq('ensemble')].set_index('environment_region')
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,
                         'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12}):
        fig=plt.figure(figsize=(19/2.54,15.3/2.54),facecolor='white')
        a=fig.add_axes([.20,.66,.77,.28]);b=fig.add_axes([.20,.115,.77,.31])
        q='delta_whole_proxy_days'
        a.fill_between(fixed.shift_days,fixed[q+'_gcm_min'],fixed[q+'_gcm_max'],color='#0072B2',alpha=.16,linewidth=0)
        a.plot(fixed.shift_days,fixed[q],color='#0072B2',marker='o',markersize=4,lw=1.4)
        a.axhline(0,color='#A9ADB0',lw=.8,zorder=0)
        a.set(xticks=[-14,-7,0,7,14],xlim=(-15,15),xlabel='Sowing-date shift (days)',
              ylabel='Affected-tissue integral\nchange (days)')
        xq,yq='delta_grain_fill_tmax_above30_days','delta_whole_proxy_days'
        def point(row,label,color,marker,filled=True):
            x,y=float(row[xq]),float(row[yq])
            b.errorbar(x,y,xerr=np.maximum(0,[[x-row[xq+'_gcm_min']],[row[xq+'_gcm_max']-x]]),
                       yerr=np.maximum(0,[[y-row[yq+'_gcm_min']],[row[yq+'_gcm_max']-y]]),
                       color=color,marker=marker,markerfacecolor=color if filled else 'white',
                       ms=4.5,lw=.9,capsize=2,label=label)
        for region,color in COLORS.items():
            point(policy.loc[region],'Regional policy' if region=='Sampled design' else region,
                  color,'D' if region=='Sampled design' else 'o')
        point(fixed[fixed.shift_days.eq(14)].iloc[0],'Uniform +14 d','#353B40','^',False)
        b.axvline(0,color='#A9ADB0',lw=.8,zorder=0);b.axhline(0,color='#A9ADB0',lw=.8,zorder=0)
        b.set(xlabel='Change in GS65–85 days with Tmax >30°C',ylabel='Affected-tissue integral\nchange (days)')
        b.legend(loc='lower center',bbox_to_anchor=(.5,1.04),ncol=3,frameon=False,
                 columnspacing=.8,handlelength=1.2,handletextpad=.3,borderaxespad=0)
        for ax in [a,b]:
            for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.8)
            ax.tick_params(width=.8,length=3)
        fig.text(.025,.98,'(a)',fontsize=12,ha='left',va='top')
        fig.text(.025,.555,'(b)',fontsize=12,ha='left',va='top')
    caption=('Sowing responses under SSP5–8.5, 2071–2100. (a) Fixed sowing shifts relative to unchanged sowing; '
        'lines connect tested options and shading spans three GCMs. (b) Regional disease–heat responses of the '
        'leave-one-GCM-out policy, its sampled-design aggregate (diamond), and uniform +14-day delay (open triangle). '
        'Points are equal-GCM means; bars show model minima–maxima, not confidence intervals. Own-window integrals '
        'are normalized symptom-tissue proxies; heat exposure counts GS65–85 days with Tmax >30°C. Complete-season '
        'support retains 1,902/1,920 draw-years per GCM, 62 cells and 99.64% design weight. This support sensitivity '
        'is post-audit; original common-calendar results remain in supplementary analyses. Bounded shifts establish '
        'neither agronomic nor yield optima.')
    spec=dict(name='complete_season_disease_heat_results',width_cm=19.,height_cm=15.3,font_pt=12,
        scenario='ssp585',period='2071-2100',support='complete_season',
        source_files={name:digest(output/name) for name in ['support_sensitivity_fixed_shifts.csv','support_sensitivity_same_policy.csv']},
        panels={'a':'fixed-shift own-window disease integral, sampled-design ensemble mean and three-GCM range',
                'b':'regional policy, regional effects, and uniform plus14 disease–heat comparison on identical complete-season support'},
        caption=caption,qualification='conditional sampled-design sensitivities; all original support analyses retained')
    return fig,spec
