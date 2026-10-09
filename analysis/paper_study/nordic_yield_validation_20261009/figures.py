"""Static research figures for the Nordic trial-validation comparisons."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .run import HERE

PALETTE=['#275d7d','#50776c','#bd862e','#795b87','#777d83']
LABELS={'training_mean':'Training mean','origin_linear':'Severity, linear',
    'origin_exponential':'Severity, exponential','affine_ridge':'Severity + intercept',
    'stage_ridge':'Severity + crop stage','nested_selection':'Model selection',
    'positive_stage_linear':'Nonnegative stage model','training_selected':'Training-selected model'}


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8.5)


def validation_figure(destination=HERE,stem='Nordic_yield_validation'):
    destination=Path(destination);destination.mkdir(exist_ok=True,parents=True)
    metrics=pd.read_csv(HERE/'model_comparison_metrics.csv')
    sensitivity=pd.read_csv(HERE/'positive_stage_comparison.csv')
    features=pd.read_csv(HERE/'treatment_response_features.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10,7.8))
    fig.subplots_adjust(left=.095,right=.98,bottom=.18,top=.94,wspace=.78,hspace=.52)
    ax=axes[0,0]
    for i,(trial,part) in enumerate(features.groupby('registry_id')):
        stage=part.latest_stage.iloc[0]
        label=f'{trial}: '+(f'GS{int(stage)}' if np.isfinite(stage) else 'stage unknown')
        ax.scatter(100*part.delta_severity,100*part.response_fraction,s=24,color=PALETTE[i],
                   marker=['o','s','^','D','v'][i],edgecolor='white',lw=.4,label=label)
    ax.axhline(0,color='#555b60',lw=.8);ax.axvline(0,color='#ddd',lw=.7)
    ax.set(xlabel='Reduction in Septoria severity (percentage points)',ylabel='Measured yield response (%)')
    ax.set_title('a   Severity and treatment response',loc='left',fontweight='bold',pad=12)
    handles,legend_labels=ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,frameon=False,fontsize=7.5,loc='lower center',bbox_to_anchor=(.54,.065),ncol=3)
    clean(ax)
    def bars(ax,domain,validation,models,title):
        selected=metrics[(metrics.domain==domain)&(metrics.validation==validation)].set_index('model')
        values=[float(selected.loc[model,'RMSE_pp']) for model in models]
        ax.barh(range(len(models)),values,color=['#aeb4b9' if 'mean' in model or 'selected' in model else '#275d7d' for model in models],height=.62)
        ax.set_yticks(range(len(models)),[LABELS[model] for model in models],fontsize=8.4)
        for i,value in enumerate(values):ax.text(value+.12,i,f'{value:.2f}',va='center',fontsize=8.5)
        ax.invert_yaxis();ax.set_xlim(0,max(values)*1.2)
        ax.set_xlabel('Yield-response RMSE (percentage points)')
        ax.set_title(title,loc='left',fontweight='bold',pad=12);clean(ax)
    bars(axes[0,1],'endpoint','leave_trial_out',
         ['training_mean','origin_linear','origin_exponential','affine_ridge','nested_selection'],
         'b   Trial-held-out predictions')
    ax=axes[1,0]
    models=['affine_ridge','stage_ridge','nested_selection','positive_stage_linear']
    data=sensitivity.set_index('model');baseline=float(data.loc['training_mean','RMSE_pp'])
    for i,model in enumerate(models):
        value=float(data.loc[model,'RMSE_pp'])-baseline
        lo=float(data.loc[model,'RMSE_difference_lower95_pp']);hi=float(data.loc[model,'RMSE_difference_upper95_pp'])
        ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',ms=5.5,capsize=3,color=PALETTE[0] if model=='stage_ridge' else '#777d83')
        ax.text(value,i+.23,f'{value:+.2f}',ha='center',fontsize=8)
    ax.set_yticks(range(len(models)),[LABELS[model] for model in models],fontsize=8.1)
    ax.set_ylim(len(models)-.45,-.5);ax.axvline(0,color='#555b60',lw=.8,ls='--')
    ax.set_xlabel('RMSE difference from training mean (points)')
    ax.set_title('c   Assessment-stage information',loc='left',fontweight='bold',pad=12);clean(ax)
    bars(axes[1,1],'endpoint','2024_to_2025',
         ['training_mean','origin_linear','affine_ridge','training_selected'],
         'd   Transfer from 2024 to 2025')
    fig.text(.095,.023,'Responses are treatment contrasts; infection dates, functional leaf area and leaf-specific damage are unobserved.',fontsize=8,color='#555b60')
    for ext in ['png','pdf','svg']:fig.savefig(destination/f'{stem}.{ext}',dpi=250,facecolor='white',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    return destination/f'{stem}.png'


if __name__=='__main__':validation_figure()
