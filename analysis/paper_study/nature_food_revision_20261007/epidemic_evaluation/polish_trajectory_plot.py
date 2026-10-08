"""Render already-evaluated tables; does not run or fit models."""
from pathlib import Path
import os
P=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(P/'.mplconfig'))
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.dates as md
q=pd.read_csv(P/'observed_vs_frozen_assessments.csv',parse_dates=['date']);d=pd.read_csv(P/'representative_daily_predictions.csv',parse_dates=['date']);s=pd.read_csv(P/'representative_series_selection_before_predictions.csv')
labels={'calibration':'Calibration 2017–18','reused_BASF2019':'BASF2019 (reused)','reused_strict_Corteva':'Strict Corteva (reused)'};colors={'model':'#0072B2','mean':'#D18F00','stage':'#AD5673'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':9})
f,axes=plt.subplots(3,3,figsize=(12,8.2),sharey=True)
for ax,z in zip(axes.flat,s.itertuples()):
 a=q[q.partition.eq(z.partition)&q.field_id.eq(z.field_id)&q.leaf_index.eq(z.leaf_index)].sort_values('date');b=d[d.partition.eq(z.partition)&d.field_id.eq(z.field_id)&d.leaf_index.eq(z.leaf_index)]
 ax.plot(b.date,b.model_damage_percent,c=colors['model'],lw=1.8,label='Frozen damage ×100');ax.plot(b.date,b.leaf_mean_percent,c=colors['mean'],lw=1.3,ls='--',label='Calibration leaf mean');ax.plot(a.date,a.observed_percent,c='#333333',lw=.8,marker='o',ms=4,label='Observed score');t=a[a.stage_time_available];ax.scatter(t.date,t.stage_time_percent,c=colors['stage'],marker='s',s=20,label='Calibration stage/time');ax.set_ylim(-2,102);ax.grid(axis='y',color='#e8e8e8');idx=np.unique(np.round(np.linspace(0,len(a)-1,min(4,len(a)))).astype(int));ax.set_xticks(a.date.iloc[idx]);ax.xaxis.set_major_formatter(md.DateFormatter('%d %b'));ax.tick_params(axis='x',rotation=25);ax.set_title(f'{labels[z.partition]}\nField {z.field_id.split("|")[1]} ({int(a.season_year.iloc[0])}); source leaf {z.leaf_index+1}; n={len(a)}')
f.supylabel('Severity score / modeled damage (%)',x=.01,fontsize=10);h,l=axes.flat[0].get_legend_handles_labels();f.legend(h,l,loc='upper center',ncol=4,frameon=False);f.text(.5,.01,'Selection: longest observed histories by count, span and lexical key. Corteva leaf ranks remain unverified. Observed connecting lines are guides.',ha='center',fontsize=8);f.tight_layout(rect=[.025,.04,1,.94]);f.savefig(P/'representative_trajectories.png',dpi=180);f.savefig(P/'representative_trajectories.pdf');plt.close(f)
print('Rendered representative trajectories from unchanged audited tables.')
