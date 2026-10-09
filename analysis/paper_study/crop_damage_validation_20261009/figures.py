"""Publication figures for leaf-rank and multi-environment yield-response tests."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .run import HERE

LABELS={'training_mean':'Training mean','composition_ridge':'Cultivar composition',
    'flag_ridge':'Flag-leaf severity','lower_ridge':'Lower-leaf severity',
    'two_leaf_ridge':'Both leaf severities','composition_two_leaf_ridge':'Composition + both leaves',
    'nonnegative_damage':'Nonnegative damage','management_ridge':'Nitrogen + water regime',
    'severity_ridge':'Septoria severity','severity_management_ridge':'Severity + management',
    'reduction_ridge':'Observed severity reduction'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8)


def save(fig,destination,stem):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(destination/f'{stem}.{ext}',dpi=250,bbox_inches='tight',pad_inches=.12,facecolor='white')
    plt.close(fig)
    return destination/f'{stem}.png'


def tunisia_figure(destination=HERE,stem='Tunisia_leaf_yield_validation'):
    style();plots=pd.read_csv(HERE/'tunisia_plot_features.csv');refs=pd.read_csv(HERE/'tunisia_reference_features.csv')
    metrics=pd.read_csv(HERE/'model_comparison_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(10.3,7.5))
    fig.subplots_adjust(left=.09,right=.97,top=.94,bottom=.12,wspace=.85,hspace=.54)
    ax=axes[0,0]
    for year,color,marker in [(2018,'#275d7d','o'),(2019,'#bd862e','^')]:
        q=plots[plots.year.eq(year)]
        ax.scatter(100*q.severity_flag,q.yield_t_ha,s=25,c=color,marker=marker,edgecolor='white',lw=.35,label=str(year))
    ax.set(xlabel='Flag-leaf lesion area at GS75 (%)',ylabel='Observed grain yield (t ha⁻¹)')
    ax.legend(frameon=False,fontsize=8);ax.set_title('a   Leaf damage and annual yield differences',loc='left',fontweight='bold',pad=10);clean(ax)
    def bars(ax,domain,models,title,unit):
        q=metrics[metrics.domain.eq(domain)].set_index('model')
        vals=[q.loc[m,'RMSE'] for m in models]
        ax.barh(range(len(models)),vals,color=['#aeb4b9' if m=='training_mean' else '#275d7d' for m in models],height=.62)
        ax.set_yticks(range(len(models)),[LABELS[m] for m in models]);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.23)
        for i,v in enumerate(vals):ax.text(v+.02*max(vals),i,f'{v:.2f}',va='center',fontsize=8)
        ax.set(xlabel=f'Prediction RMSE ({unit})');ax.set_title(title,loc='left',fontweight='bold',pad=10);clean(ax)
    models=['training_mean','composition_ridge','flag_ridge','two_leaf_ridge','composition_two_leaf_ridge']
    bars(axes[0,1],'absolute_year_transfer',models,'b   Transfer between 2018 and 2019','t ha⁻¹')
    ax=axes[1,0]
    ax.scatter(100*refs.severity_flag,100*refs.response_fraction,s=28,c='#275d7d',edgecolor='white',lw=.4)
    ax.axhline(0,color='#777',lw=.7)
    ax.set(xlabel='Flag-leaf lesion area at GS75 (%)',ylabel='Protected-reference yield gap (%)')
    ax.set_title('c   Management-associated yield gaps, 2019',loc='left',fontweight='bold',pad=10);clean(ax)
    bars(axes[1,1],'within_2019_mixture',models,'d   Mixture-held-out predictions, 2019','percentage points')
    fig.text(.09,.018,'Flag and lower leaves were assessed on different dates. Protected-plot severity and healthy-area duration were unobserved.',fontsize=8,color='#555b60')
    return save(fig,destination,stem)


def briwecs_figure(destination=HERE,stem='German_yield_response_validation'):
    style();metrics=pd.read_csv(HERE/'briwecs_model_comparison_metrics.csv');features=pd.read_csv(HERE/'briwecs_treatment_response_features.csv')
    fig,axes=plt.subplots(2,2,figsize=(10.3,7.7))
    fig.subplots_adjust(left=.105,right=.98,top=.94,bottom=.17,wspace=.78,hspace=.55)
    ax=axes[0,0];palette=['#275d7d','#50776c','#bd862e','#795b87','#777d83']
    for (location,part),color,marker in zip(features.groupby('Location'),palette,['o','s','^','D','v']):
        ax.scatter(100*part.severity_unprotected,100*part.response_fraction,s=9,c=color,marker=marker,alpha=.5,label=location,rasterized=True)
    ax.axhline(0,color='#777',lw=.7);ax.set(xlabel='Source unprotected Septoria severity (%)',ylabel='Observed protection response (%)')
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.073),ncol=5,frameon=False,markerscale=1.5)
    ax.set_title('a   Responses across cultivars and environments',loc='left',fontweight='bold',pad=10);clean(ax)
    models=['training_mean','management_ridge','severity_ridge','severity_management_ridge']
    def bars(ax,domain,validation,models,title):
        q=metrics[metrics.domain.eq(domain)&metrics.validation.eq(validation)].set_index('model')
        values=[q.loc[m,'RMSE_pp'] for m in models]
        ax.barh(range(len(models)),values,color=['#aeb4b9' if m=='training_mean' else '#275d7d' for m in models],height=.62)
        ax.set_yticks(range(len(models)),[LABELS[m] for m in models]);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.24)
        for i,v in enumerate(values):ax.text(v+.3,i,f'{v:.2f}',va='center',fontsize=8)
        ax.set(xlabel='Yield-response RMSE (percentage points)');ax.set_title(title,loc='left',fontweight='bold',pad=10);clean(ax)
    bars(axes[0,1],'unprotected_endpoint','leave_site_year_out',models,'b   Whole site-year held out')
    bars(axes[1,0],'unprotected_endpoint','leave_location_out',models,'c   Whole geographic site held out')
    bars(axes[1,1],'unprotected_endpoint','2015_2017_to_2018_2019',models,'d   Transfer to 2018–2019')
    fig.text(.105,.025,'Comparisons match cultivar, nitrogen and water regime. Protection targets multiple diseases; assessment dates and leaf ranks are unspecified.',fontsize=7.7,color='#555b60')
    return save(fig,destination,stem)


if __name__=='__main__':
    tunisia_figure();briwecs_figure()
