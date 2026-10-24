"""Describe the retained observation cohort, independently of evaluation scores."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pyproj import Transformer

from .figure_maps_v2 import _map_axes,_country_boundaries,STYLE

SOURCES=('BASF','Corteva')
COLORS={'BASF':'#A65893','Corteva':'#009E73'}


def prepare_overview(observations,basf_raw,corteva_raw):
    obs=observations.copy()
    assert obs.global_target_index.is_unique and len(obs)==2304
    assert obs.field_id.nunique()==218 and obs.site_id.nunique()==128
    assert set(obs.source)==set(SOURCES)
    assert not obs.assessment_after_grain_record.fillna(False).any()
    sites=obs.groupby(['source','site_id'],as_index=False).agg(
        latitude=('latitude','first'),longitude=('longitude','first'),
        assessments=('global_target_index','size'),field_seasons=('field_id','nunique'))
    assert len(sites)==128
    fields=obs.groupby(['source','field_id'],as_index=False).agg(
        visits=('date','nunique'),assessments=('global_target_index','size'),season_year=('season_year','first'))
    years=obs.groupby(['source','season_year'],as_index=False).size().rename(columns={'size':'assessments'})
    visits=fields.groupby(['source','visits'],as_index=False).size().rename(columns={'size':'field_seasons'})
    hist=[];edges=np.arange(0,101,10)
    for source in SOURCES:
        values=obs.loc[obs.source.eq(source),'observed_percent']
        counts,_=np.histogram(values,bins=edges)
        for lo,hi,n in zip(edges[:-1],edges[1:],counts):
            hist.append(dict(source=source,lower_percent=int(lo),upper_percent=int(hi),assessments=int(n),
                closure='left-closed/right-open; final bin includes 100'))
    inventory=[]
    a=basf_raw.loc[basf_raw.Organism.eq('SEPTTR')&basf_raw.Parameter.eq('INFECT')&
        basf_raw.PlantPart.str.startswith('LEAF',na=False)].copy()
    b=corteva_raw.loc[corteva_raw.PestCode.eq('SEPTTR')&corteva_raw.EvaluationType.eq('INFECT')&
        corteva_raw.SamplingUnit.str.startswith('LEAF',na=False)].copy()
    for source,frame in [('BASF',a),('Corteva',b)]:
        for (treatment,sprays),part in frame.groupby(['Treatment','NumSprays'],dropna=False):
            inventory.append(dict(source=source,treatment_label=treatment,num_sprays=sprays,
                raw_leaf_INFECT_rows=len(part),eligible_treatment_label=(treatment=='Untreated' and sprays==0),
                scope='full original source-native SEPTTR INFECT leaf rows, before cohort/leaf/chronology restrictions'))
    summary=obs.groupby('source',as_index=False).agg(assessments=('global_target_index','size'),
        field_seasons=('field_id','nunique'),sites=('site_id','nunique'),
        first_year=('season_year','min'),last_year=('season_year','max'))
    return dict(assessments=obs,sites=sites,fields=fields,years=years,
        severity=pd.DataFrame(hist),visits=visits,treatment_inventory=pd.DataFrame(inventory),source_summary=summary)


def overview_figure(data):
    width,height=19.,18.2
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(width/2.54,height/2.54))
        def axes(bounds,letter):
            ax=fig.add_axes([bounds[0]/width,bounds[1]/height,bounds[2]/width,bounds[3]/height])
            ax.text(0,1.035,f'({letter})',transform=ax.transAxes,va='bottom',fontsize=12)
            for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.8)
            return ax
        sites=data['sites'];tr=Transformer.from_crs(4326,3035,always_xy=True)
        x,y=tr.transform(sites.longitude.to_numpy(),sites.latitude.to_numpy())
        dx=np.ptp(x)+240000;dy=np.ptp(y)+240000;aspect=6.1/8.2
        dx=max(dx,dy/aspect);dy=dx*aspect;cx=(max(x)+min(x))/2;cy=(max(y)+min(y))/2
        extent=[cx-dx/2,cx+dx/2,cy-dy/2,cy+dy/2]
        a=_map_axes(fig,(.6,10.3,8.2,6.1),(width,height),extent,'a')
        _country_boundaries(a)
        for source,marker in [('BASF','o'),('Corteva','^')]:
            part=sites[sites.source.eq(source)];xx,yy=tr.transform(part.longitude.to_numpy(),part.latitude.to_numpy())
            a.scatter(xx,yy,s=17,color=COLORS[source],marker=marker,edgecolors='white',linewidths=.3,zorder=5,rasterized=True)
        b=axes((11.,10.3,7.55,6.1),'b');years=np.arange(2014,2020);bottom=np.zeros(len(years))
        for source in SOURCES:
            part=data['years'][data['years'].source.eq(source)].set_index('season_year').assessments.reindex(years,fill_value=0)
            b.bar(years,part,bottom=bottom,color=COLORS[source],width=.72);bottom+=part.to_numpy()
        b.set(xlabel='Season year',ylabel='Assessments',xticks=years,ylim=(0,float(bottom.max())*1.10))
        c=axes((2.05,1.75,6.75,6.0),'c');bottom=np.zeros(10)
        for source in SOURCES:
            part=data['severity'][data['severity'].source.eq(source)].sort_values('lower_percent')
            c.bar(part.lower_percent,part.assessments,bottom=bottom,width=10,align='edge',color=COLORS[source],edgecolor='white',linewidth=.4)
            bottom+=part.assessments.to_numpy()
        c.set(xlim=(0,100),xlabel='Recorded disease (%)',ylabel='Assessments',xticks=[0,25,50,75,100])
        d=axes((11.,1.75,7.55,6.0),'d');numbers=np.arange(1,9);bottom=np.zeros(8)
        for source in SOURCES:
            part=data['visits'][data['visits'].source.eq(source)].set_index('visits').field_seasons.reindex(numbers,fill_value=0)
            d.bar(numbers,part,bottom=bottom,color=COLORS[source],width=.72);bottom+=part.to_numpy()
        d.set(xlabel='Visits per field-season',ylabel='Field-seasons',xticks=numbers,ylim=(0,float(bottom.max())*1.12))
        handles=[Patch(facecolor=COLORS[s],edgecolor='none',label=s) for s in SOURCES]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,frameon=False,
            handlelength=1.2,columnspacing=1.5)
    geometry=dict(projection='EPSG3035',extent=extent,sites_retained=128,geographical_scale=False,
        coordinates='original retained source locations; no interpolation or jitter',
        panels=dict(a='all retained source locations',b='assessment counts by season and source',
            c='native disease-score distribution; counts',d='unique assessment dates per field-season'),
        count_grain='one field/date/source-native-leaf assessment; no duplication across evaluation designs')
    return fig,geometry
