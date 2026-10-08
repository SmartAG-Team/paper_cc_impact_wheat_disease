"""Descriptive Septoria analysis and publication figures; no epidemic calibration."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import zipfile

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/septoria';OUT.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42,'savefig.dpi':300})
COLORS={'IPO323':'#156b85','IPO88004':'#be6d28','IPO90012':'#6457a5'}

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)

data=pd.read_parquet(ROOT/'data/harmonized/observations.parquet')
assert data.observation_id.is_unique

# Main isolate panel only; transgenics and mutant tests have separate designs.
main=data[(data.dataset_id=='hafeez-2025-infection') & data.source_table.isin(
    ['IPO323_results_AUDPC','IPOO88004_raw','IPO90012_scores_raw']) &
    (data.metric=='pycnidial_coverage_percent')].copy()
assert main.source_table.nunique()==3
intervals=[]
for (isolate,record),group in main.groupby(['isolate','record_id'],sort=False):
    scored=group[group.value.notna()].sort_values('days_post_inoculation')
    result=dict(isolate=isolate,record_id=record,cultivar=group.cultivar.iloc[0],
                scored_dates=len(scored),lower_dpi=np.nan,upper_dpi=np.nan)
    if scored.empty:result['censoring']='unobserved'
    elif not scored.value.gt(0).any():
        result['censoring']='right';result['lower_dpi']=scored.days_post_inoculation.max()
    else:
        first=scored.loc[scored.value.gt(0),'days_post_inoculation'].min()
        previous=scored[scored.days_post_inoculation<first]
        result['upper_dpi']=first
        if previous.empty:result['censoring']='left'
        else:result['censoring']='interval';result['lower_dpi']=previous.days_post_inoculation.max()
    intervals.append(result)
intervals=pd.DataFrame(intervals);intervals.to_csv(OUT/'first_pycnidia_intervals.csv',index=False)
censoring=intervals.groupby(['isolate','censoring']).size().unstack(fill_value=0)
censoring=censoring.reindex(columns=['interval','left','right','unobserved'],fill_value=0)
censoring.to_csv(OUT/'pycnidia_censoring_counts.csv')
scored=main[main.value.notna()].copy();scored['positive']=scored.value.gt(0)
progress=scored.groupby(['isolate','days_post_inoculation']).agg(scored_records=('value','size'),
    positive_fraction=('positive','mean'),median_coverage_percent=('value','median'),mean_coverage_percent=('value','mean')).reset_index()
progress.to_csv(OUT/'controlled_infection_progress.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(12.4,3.4),gridspec_kw={'width_ratios':[1.2,1.2,1.1]})
for isolate,g in progress.groupby('isolate'):
    axs[0].plot(g.days_post_inoculation,g.positive_fraction*100,'o-',color=COLORS[isolate],label=isolate)
    axs[1].plot(g.days_post_inoculation,g.median_coverage_percent,'o-',color=COLORS[isolate],label=isolate)
axs[0].set(xlabel='Days after inoculation',ylabel='Pycnidia-positive records (%)',ylim=(-2,102),title='a  Reproductive development')
axs[1].set(xlabel='Days after inoculation',ylabel='Median pycnidial coverage (%)',ylim=(-.1,max(10,progress.median_coverage_percent.max()*1.15)),title='b  Coverage among scored records')
axs[0].legend(frameon=False,fontsize=8)
bottom=np.zeros(len(censoring));censor_colors=['#156b85','#be6d28','#9eb2bc','#d5dadd']
for c,color in zip(censoring,censor_colors):
    values=censoring[c]/censoring.sum(axis=1)*100
    axs[2].bar(censoring.index,values,bottom=bottom,color=color,label=c.title());bottom+=values.to_numpy()
axs[2].set(ylabel='Source records (%)',ylim=(0,100),title='c  First-pycnidia censoring')
axs[2].tick_params(axis='x',rotation=25);axs[2].legend(frameon=False,fontsize=7,loc='lower left')
fig.text(.02,-.025,'Observed sample sizes change across dates. First visible pycnidia do not measure penetration time or spore release.',fontsize=8)
fig.tight_layout();save(fig,'controlled_infection_progress')

# Necrosis and reproduction are distinct observations. Aggregate within source
# plots and collection before calculating descriptive rank correlations.
relationships=[]
fig,axs=plt.subplots(1,2,figsize=(10.8,4.2))
for ax,dataset,title in zip(axs,['karisto-2018-field','durum-mixtures-2020'],['Swiss field, 2016','Tunisian field, 2018–2019']):
    f=data[(data.dataset_id==dataset)&data.metric.isin(['necrotic_leaf_area_percent','pycnidia_density_leaf'])]
    f=f[f.measurement_role=='observed']
    pivot=f.pivot(index=['source_file','source_table','source_row','plot_id','date','season_year'],columns='metric',values='value').reset_index()
    assert len(pivot)==(21420 if dataset=='karisto-2018-field' else 3004)
    grouped=pivot.groupby(['season_year','date','plot_id'],dropna=False).agg(
        necrosis_percent=('necrotic_leaf_area_percent','mean'),pycnidia_cm2=('pycnidia_density_leaf','mean'),leaves=('source_row','size')).reset_index()
    grouped['dataset_id']=dataset;relationships.append(grouped)
    for date,g in grouped.groupby('date'):
        ax.scatter(g.necrosis_percent,g.pycnidia_cm2,s=18,alpha=.65,label=date,edgecolor='none',rasterized=True)
    ax.set(title=title,xlabel='Mean necrotic leaf area (%)',ylabel='Mean pycnidia per cm² of leaf')
    ax.legend(frameon=False,fontsize=8)
fig.text(.015,-.01,'Each point is a plot–collection mean. Tunisian scans condition on pycnidia-positive leaves; populations are not pooled.',fontsize=8)
fig.tight_layout();save(fig,'necrosis_and_pycnidia')
relationships=pd.concat(relationships,ignore_index=True);relationships.to_csv(OUT/'plot_necrosis_pycnidia.csv',index=False)
rank_correlations=[]
for (dataset,date),g in relationships.groupby(['dataset_id','date']):
    rank_correlations.append(dict(dataset_id=dataset,date=date,plot_collections=len(g),
        spearman_r=g.necrosis_percent.rank().corr(g.pycnidia_cm2.rank())))
pd.DataFrame(rank_correlations).to_csv(OUT/'descriptive_rank_correlations.csv',index=False)

# French upper-leaf scores: plot means, then unweighted means over observed plots.
f=data[(data.dataset_id=='orellana-torrejon-2022-field') & (data.source_table=='F1_field') &
    (data.metric=='pycnidial_coverage_percent') & data.organ.isin(['F1','F2','F3']) & data.value.notna()]
plot=f.groupby(['season_year','date','cultivar','organ','plot_id'],dropna=False).agg(
    pycnidia_percent=('value','mean'),assessed_plants=('value','size')).reset_index()
plot.to_csv(OUT/'french_plot_leaf_progress.csv',index=False)
french=plot.groupby(['season_year','date','cultivar','organ']).agg(pycnidia_percent=('pycnidia_percent','mean'),
    observed_plots=('plot_id','size')).reset_index()
fig,axs=plt.subplots(1,2,figsize=(10.8,3.5),sharey=True)
for ax,year in zip(axs,[2018,2019]):
    for (cultivar,organ),g in french[french.season_year==year].groupby(['cultivar','organ']):
        ax.plot(pd.to_datetime(g.date),g.pycnidia_percent,'o-',label=f'{cultivar} {organ}',lw=1.2)
    ax.set(title=str(year),ylabel='Mean pycnidia-covered leaf area (%)',xlabel='Assessment date',ylim=(-.5,max(30,np.ceil(french.pycnidia_percent.max()/10)*10)))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='x',rotation=25);ax.legend(ncol=2,fontsize=7,frameon=False)
fig.text(.02,-.025,'APA = Apache; CEL = Cellule. Assessed plots and surviving leaf populations differ among dates; senescent leaves are excluded.',fontsize=8)
fig.tight_layout();save(fig,'french_leaf_progress')

# BASF untreated repeated-assessment coverage, at the same explicit leaf rank.
basf=pd.read_csv(ROOT/'data/basf-wheat-diseases.txt',sep='\t')
b=basf[(basf.Organism=='SEPTTR')&(basf.Treatment=='Untreated')].copy()
dates=b.groupby('TrialId').Date.nunique();rank=b[b.PlantPart.str.contains(r'LEAF, (?:FLAG|1ST|2ND|3RD|4TH|5TH|6TH)',na=False)]
same=rank.groupby(['TrialId','PlantPart']).Date.nunique();trial_repeated=same.groupby('TrialId').max().ge(3)
first_dates=b.groupby('TrialId').Date.transform('min')
first_positive=b[b.Date.eq(first_dates)].groupby('TrialId').Value.max().gt(0)
coordinates=basf[['Latitude','Longitude']].drop_duplicates().copy()
site_stats=[]
for (lat,lon),g in b.groupby(['Latitude','Longitude']):
    trials=sorted(g.TrialId.unique());eligible=int(trial_repeated.reindex(trials,fill_value=False).sum())
    site_stats.append(dict(latitude=lat,longitude=lon,trials=len(trials),same_rank_repeated_trials=eligible,
        first_year=pd.to_datetime(g.Date).dt.year.min(),last_year=pd.to_datetime(g.Date).dt.year.max(),
        coordinate_precision='rounded_0.1_degree'))
site_stats=pd.DataFrame(site_stats);site_stats.to_csv(OUT/'mapped_trial_locations.csv',index=False)

# Natural Earth public-domain national boundaries, archived with SHA256.
mapdir=ROOT/'data/geography';mapdir.mkdir(exist_ok=True)
archive=mapdir/'ne_110m_admin_0_countries.zip'
url='https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip'
if not archive.exists():
    r=requests.get(url,timeout=45);r.raise_for_status();archive.write_bytes(r.content)
(mapdir/'manifest.json').write_text(json.dumps(dict(source_url=url,
    source_page='https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/',
    license='public_domain',sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2)+'\n')
world=gpd.read_file('zip://'+str(archive));region=world.cx[-12:34,33:62].copy()
region.geometry=region.geometry.intersection(box(-12,33,34,62));region=region.to_crs(3035)
points=gpd.GeoDataFrame(site_stats,geometry=gpd.points_from_xy(site_stats.longitude,site_stats.latitude),crs=4326).to_crs(3035)
fig,axs=plt.subplots(1,2,figsize=(11.8,6.5))
for ax in axs:
    region.plot(ax=ax,color='#edf0ef',edgecolor='#aab4b7',linewidth=.45)
    ax.set_aspect('equal');ax.set_axis_off()
axs[0].scatter(points.geometry.x,points.geometry.y,s=points.trials*22+20,color='#156b85',alpha=.85,
    edgecolor='white',lw=.5,label='BASF trials, 2017–2019')
for label,lat,lon,marker,color,offset in [
    ('Swiss scans, 2016',47.449,8.682,'s','#be6d28',(12,-18)),
    ('French field, 2018–2019',48.8469444444444,1.94277777777778,'D','#6457a5',(-8,-35)),
    ('Tunisian scans, 2018–2019',36.5477472222222,9.01131388888889,'^','#8d3f54',(9,7))]:
    p=gpd.GeoSeries(gpd.points_from_xy([lon],[lat]),crs=4326).to_crs(3035).iloc[0]
    axs[0].scatter(p.x,p.y,s=85,marker=marker,facecolor='white' if label.startswith('French') else color,
        edgecolor=color,zorder=5,lw=1.7)
    axs[0].annotate(label,(p.x,p.y),xytext=offset,textcoords='offset points',fontsize=8,
        bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1),ha='left' if offset[0]>0 else 'right')
axs[0].set_title('a  Geographic coverage',loc='left');axs[0].legend(loc='lower left',frameon=False,fontsize=8)
for eligible,color,label in [(False,'#c6cdd0','No trial with ≥3 dates at one ranked leaf'),
                             (True,'#156b85','≥1 trial with ≥3 dates at one ranked leaf')]:
    p=points[points.same_rank_repeated_trials.gt(0)==eligible]
    axs[1].scatter(p.geometry.x,p.geometry.y,s=40,c=color,label=label,edgecolor='white',lw=.4)
axs[1].set_title('b  Repeated leaf-assessment coverage',loc='left');axs[1].legend(loc='lower left',frameon=False,fontsize=7)
fig.text(.02,.035,'BASF coordinates are rounded to 0.1°. French marker represents the station, not a verified plot. Marker area in panel a scales with trial count.',fontsize=8)
fig.text(.02,.012,'Projection: ETRS89 / LAEA Europe (EPSG:3035). National boundaries: Natural Earth, public domain. Laboratory assays have no field location.',fontsize=8)
fig.subplots_adjust(bottom=.1,wspace=.03);save(fig,'study_coverage_map')

phen=pd.read_csv(ROOT/'analysis/phenology/field_transfer_era5/transfer_event_dates.csv')
summary=dict(canonical_records=len(data),public_source_integrity=json.loads((ROOT/'analysis/public_septoria/summary.json').read_text()),
    basf=dict(trials=int(b.TrialId.nunique()),untreated_septoria_records=len(b),distinct_locations=len(coordinates),
        trials_three_any_dates=int(dates.ge(3).sum()),trials_three_ranked_leaf_dates=int(trial_repeated.sum()),
        initially_positive_trials=int(first_positive.sum()),untreated_crop_injury=int(b.Clarifier.eq('CROP INJURY').sum())),
    controlled_censoring=censoring.reset_index().to_dict('records'),
    descriptive_rank_correlations=rank_correlations,phenology_cases=int(phen.PEP_ID.nunique()),
    weather_validation=json.loads((ROOT/'analysis/era5/validation.json').read_text()),
    figures=['study_coverage_map','controlled_infection_progress','necrosis_and_pycnidia','french_leaf_progress'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['public_source_integrity','weather_validation']},indent=2))
