"""Unchanged quantitative attribution/exposure with distinct method displays."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .figure_registry import AGREEMENT_COLORS

SCENARIOS = ['ssp126','ssp245','ssp585']
COLORS = ['#0072B2','#E69F00','#D55E00']


def attribution_figure(frame):
    settings=['baseline','anthesis_early_5pct','anthesis_late_5pct','terminal_early_5pct',
        'terminal_late_5pct','reference_linear_senescence','field_ungated_damage','fixed_common_overlap']
    labels=['Whole host windows','Start threshold −5%','Start threshold +5%',
        'Terminal threshold −5%','Terminal threshold +5%','Reference senescence',
        'No symptom gate','Shared elapsed time']
    fig,ax=plt.subplots(figsize=(19/2.54,12.8/2.54),layout='constrained')
    for metric,label,color,offset in [('weather_had_days','Weather','#0072B2',-.19),
            ('host_had_days','Host','#D55E00',0),('net_had_days','Net change','#444444',.19)]:
        for i,key in enumerate(settings):
            part=frame.loc[frame.setting.eq(key)&~frame.model.eq('ensemble'),metric].to_numpy(float)
            mean=float(frame.loc[frame.setting.eq(key)&frame.model.eq('ensemble'),metric].iloc[0])
            assert len(part)==3
            ax.plot([part.min(),part.max()],[i+offset]*2,color=color,linewidth=1.25)
            ax.scatter(part,np.full(len(part),i+offset),s=14,color=color,alpha=.6)
            ax.scatter([mean],[i+offset],s=36,color=color,marker='D',
                label=label if i==0 else None,zorder=5)
    ax.axvline(0,color='#888888',linewidth=.8);ax.set_yticks(range(8),labels);ax.invert_yaxis()
    ax.set_xlabel('Contribution to normalized affected-tissue duration (days)')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=3,frameon=False,
        columnspacing=.9,handletextpad=.3)
    return fig


def exposure_figure(thresholds,regional):
    columns=['baseline_production_tonnes','unanimous_increase_tonnes',
        'unanimous_decrease_tonnes','mixed_or_zero_tonnes','unavailable_tonnes']
    order=regional.nlargest(5,'baseline_production_tonnes').domain.tolist()
    display=regional.set_index('domain').loc[order,columns].copy()
    display.loc['Other regions']=regional.loc[~regional.domain.isin(order),columns].sum()
    fig,(a,b)=plt.subplots(2,1,figsize=(19/2.54,14.7/2.54),layout='constrained',
        gridspec_kw={'height_ratios':[1.1,1.35]})
    for scenario,color,label in zip(SCENARIOS,COLORS,['SSP1–2.6','SSP2–4.5','SSP5–8.5']):
        part=thresholds.loc[thresholds.scenario.eq(scenario)].sort_values('threshold_days')
        a.plot(part.threshold_days,part.unanimous_increase_share_percent,'o-',color=color,
            label=label,markersize=4)
    a.set(xlabel='Required increase (days)',ylabel='Baseline production (%)',
        xticks=[0,1,2,3,5],ylim=(-2,101))
    a.text(0,1.05,'(a)',transform=a.transAxes,fontsize=12)
    a.legend(loc='upper right',ncol=3,frameon=False,columnspacing=.8,
        handlelength=1.3,handletextpad=.35)
    left=np.zeros(len(display))
    for column,label,color in [('unanimous_increase_tonnes','Increase',AGREEMENT_COLORS['increase']),
            ('unanimous_decrease_tonnes','Decrease',AGREEMENT_COLORS['decrease']),
            ('mixed_or_zero_tonnes','Mixed/zero',AGREEMENT_COLORS['mixed_or_zero']),
            ('unavailable_tonnes','Unavailable',AGREEMENT_COLORS['unavailable'])]:
        values=display[column].to_numpy()/1e6
        b.barh(display.index,values,left=left,color=color,label=label,height=.65);left+=values
    np.testing.assert_allclose(left,display.baseline_production_tonnes.to_numpy()/1e6,rtol=1e-12,atol=1e-10)
    b.invert_yaxis();b.set_xlim(0,float(left.max())*1.07)
    b.set_xlabel('Baseline wheat production (Mt)');b.text(0,1.12,'(b)',transform=b.transAxes,fontsize=12)
    b.legend(loc='lower center',bbox_to_anchor=(.5,1),ncol=4,frameon=False,
        columnspacing=.6,handlelength=1,handletextpad=.25)
    return fig,display.reset_index()


def support_figure(frame):
    """Retained and excluded fractions on a common100percent design denominator."""
    from matplotlib.patches import Patch
    data=frame.loc[frame.setting.eq('baseline')&frame.support.eq('across_settings')&~frame.model.eq('ensemble')].copy()
    fig,ax=plt.subplots(figsize=(19/2.54,6.2/2.54))
    fig.subplots_adjust(left=.235,right=.975,bottom=.22,top=.80)
    y=np.arange(len(data));values=100*data.common_area_time_fraction.to_numpy()
    ax.barh(y,100,color='#E7ECEF',height=.52,zorder=1)
    ax.barh(y,values,color='#34769A',height=.52,zorder=2)
    for position,value in zip(y,values):ax.text(value/2,position,f'{value:.1f}%',ha='center',va='center',color='white',fontsize=12)
    ax.set(yticks=y,yticklabels=data.model,xlim=(0,100),xticks=[0,25,50,75,100],xlabel='Design-weighted area–time support (%)')
    ax.invert_yaxis()
    fig.legend(handles=[Patch(facecolor='#34769A',label='Retained'),Patch(facecolor='#E7ECEF',label='Excluded')],
        loc='upper center',bbox_to_anchor=(.60,.985),ncol=2,frameon=False,fontsize=12)
    return fig,data


def graphical_figure(attribution):
    """Integration support and its paired host-attribution sign change."""
    values=attribution.loc[attribution.model.eq('ensemble')].set_index('setting')
    fig,ax=plt.subplots(figsize=(19/2.54,9.3/2.54))
    fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.975)
    ax.set(xlim=(0,19),ylim=(0,9.3));ax.set_axis_off()
    ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,
        edgecolor='#353B40',linewidth=.8,clip_on=False))
    ax.text(.4,8.7,'European wheat under climate change',fontsize=12,va='center')
    ax.text(.4,7.8,'Integration interval',fontsize=12,va='center')
    ax.text(15.65,7.8,'Host contribution (days)',fontsize=12,ha='center',va='center')
    for row,label,setting in [(5.9,'Whole host\nwindows','baseline'),
            (3.5,'Shared\nelapsed time','fixed_common_overlap')]:
        ax.text(.4,row,label,fontsize=12,va='center',linespacing=1.35)
        for y,start,end,color in [(row+.3,6.2,12.2,'#0072B2'),
                (row-.3,5.1,11.1,'#D55E00')]:
            if setting=='fixed_common_overlap':
                ax.plot([start,end],[y,y],color='#C9CDD0',linewidth=2.5,linestyle=(0,(2,2)))
                start,end=6.2,11.1
            ax.plot([start,end],[y,y],color=color,linewidth=5,solid_capstyle='butt')
            for endpt in [start,end]:ax.plot([endpt,endpt],[y-.15,y+.15],color=color,linewidth=.8)
        ax.annotate('',xy=(13.25,row),xytext=(12.55,row),
            arrowprops=dict(arrowstyle='->',color='#555555',linewidth=1))
        ax.add_patch(Rectangle((13.6,row-.62),4.15,1.24,facecolor='#F2F4F5',
            edgecolor='#353B40',linewidth=.8))
        value=values.loc[setting,'host_had_days']
        ax.text(15.675,row,f'{value:+.2f}'.replace('-','−'),fontsize=12,ha='center',va='center')
    ax.axhline(4.7,xmin=.025,xmax=.975,color='#D9DDE0',linewidth=.8)
    for start,text,color in [(5.1,'Historical host','#0072B2'),(10.7,'Future host','#D55E00')]:
        ax.plot([start,start+.55],[1.85,1.85],color=color,linewidth=4,solid_capstyle='butt')
        ax.text(start+.8,1.85,text,fontsize=12,va='center')
    ax.text(.4,.98,'Identical 4,117 season pairs · Disease parameters fixed',fontsize=12,va='center')
    ax.text(.4,.37,'Schematic intervals; modelled values are normalized affected-tissue days',fontsize=12,va='center')
    return fig
