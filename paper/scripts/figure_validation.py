"""Density, calibration and visit-context displays from fixed held-out forecasts."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

COLORS = {'temporal':'#0072B2','spatial_50km':'#D55E00'}
LABELS = {'temporal':'Temporal','spatial_50km':'Spatial, 50 km'}


def metrics(frame):
    w=frame.scientific_weight.to_numpy();y=frame.observed_percent.to_numpy();p=frame.prediction_percent.to_numpy()
    mse=np.average((p-y)**2,weights=w)
    return dict(rmse=float(np.sqrt(mse)),mae=float(np.average(abs(p-y),weights=w)),
        bias=float(np.average(p-y,weights=w)),r2=float(1-mse/np.average((y-np.average(y,weights=w))**2,weights=w)),
        observed_mean=float(np.average(y,weights=w)))


def density_figure(points):
    """Equal-scale count densities and marginals; scientific weights score forecasts."""
    edges=np.linspace(0,100,41);panels=[];density=[]
    for key in ['temporal','spatial_50km']:
        f=points[points.evaluation.eq(key)]
        counts,*_=np.histogram2d(f.observed_percent,f.prediction_percent,bins=(edges,edges))
        panels.append((key,f,counts))
        for i,j in np.argwhere(counts>0):
            density.append(dict(evaluation=key,observed_lower=edges[i],observed_upper=edges[i+1],
                predicted_lower=edges[j],predicted_upper=edges[j+1],assessments=int(counts[i,j])))
    fig=plt.figure(figsize=(19/2.54,12.8/2.54),layout='constrained')
    outer=fig.add_gridspec(3,2,height_ratios=[1,.13,.11],wspace=.12,hspace=.08)
    norm=LogNorm(vmin=1,vmax=max(a[2].max() for a in panels))
    cmap=LinearSegmentedColormap.from_list('assessment_counts',['#C6DBEF','#6BAED6','#2171B5','#08306B'])
    axes=[];scored=[]
    for column,(key,f,counts) in enumerate(panels):
        grid=outer[0,column].subgridspec(2,2,width_ratios=[1,.11],height_ratios=[.11,1],hspace=.025,wspace=.025)
        ax=fig.add_subplot(grid[1,0]);top=fig.add_subplot(grid[0,0],sharex=ax);right=fig.add_subplot(grid[1,1],sharey=ax)
        axes.append(ax)
        mesh=ax.pcolormesh(edges,edges,np.ma.masked_equal(counts.T,0),cmap=cmap,norm=norm,rasterized=True)
        ax.plot([0,100],[0,100],'--',color='#555555',linewidth=1,zorder=3)
        ax.set(xlim=(0,100),ylim=(0,100),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100],
            xlabel='Observed disease (%)',ylabel='Predicted disease (%)',aspect='equal')
        top.hist(f.observed_percent,bins=edges,color='#8397A5',edgecolor='none')
        right.hist(f.prediction_percent,bins=edges,orientation='horizontal',color='#8397A5',edgecolor='none')
        top.set_yticks([]);top.tick_params(axis='x',bottom=False,labelbottom=False)
        right.set_xticks([]);right.tick_params(axis='y',left=False,labelleft=False)
        top.text(0,1.18,f'({"ab"[column]})',transform=top.transAxes,ha='left',va='bottom',fontsize=12)
        score=metrics(f);scored.append(dict(evaluation=key,records=len(f),**score))
        strip=fig.add_subplot(outer[1,column]);strip.set_axis_off()
        strip.text(.02,.55,f"RMSE {score['rmse']:.2f} pp · MAE {score['mae']:.2f} pp\n"
            +f"$R^2$ {score['r2']:.3f} · $n$ = {len(f):,}",va='center',ha='left',fontsize=12,linespacing=1.45)
    cax=fig.add_subplot(outer[2,:]);cax.set_axis_off()
    bar_ax=cax.inset_axes([.18,.58,.64,.36])
    bar=fig.colorbar(mesh,cax=bar_ax,orientation='horizontal')
    bar.set_label('Assessments per 2.5 × 2.5 percentage-point cell',fontsize=12,labelpad=4)
    # Restrict count ticks to actual integer counts; log colour is purely visual.
    ticks=[v for v in [1,2,5,10,20,50,100,200,500] if v<=norm.vmax]
    bar.set_ticks(ticks,labels=[str(v) for v in ticks]);bar.minorticks_off()
    return fig,pd.DataFrame(density),pd.DataFrame(scored)


def diagnostics(points, chronology):
    """Retain original hierarchy mass inside all descriptive strata."""
    chrono=chronology.assign(date=pd.to_datetime(chronology.date))
    starts=chrono.groupby('field_id').date.min();ends=chrono.groupby('field_id').date.max()
    p=points.copy();dates=pd.to_datetime(p.date)
    first=dates.eq(p.field_id.map(starts));last=dates.eq(p.field_id.map(ends))
    p['first_visit']=first;p['last_visit']=last;p['only_visit']=first&last
    p['visit_context']=np.select([first&last,first,last],['Only visit','First visit','Last visit'],default='Intermediate')
    calibration=[];bias=[];contexts=[]
    edges=np.arange(0,101,10);bounds=[0,10,25,50,75,100]
    rng=np.random.default_rng(20261023)
    for key,f in p.groupby('evaluation',sort=False):
        site_ids=sorted(f.site_id.unique());site_index=pd.Series(np.arange(len(site_ids)),index=site_ids)
        multiplicity=rng.multinomial(len(site_ids),np.full(len(site_ids),1/len(site_ids)),size=1000)
        forecast_bin=np.minimum((f.prediction_percent/10).astype(int),9)
        for i in range(10):
            part=f[forecast_bin.eq(i)]
            if part.empty:continue
            w=part.scientific_weight.to_numpy();y=part.observed_percent.to_numpy()
            pos=site_index.loc[part.site_id].to_numpy()
            site_w=np.bincount(pos,weights=w,minlength=len(site_ids));site_y=np.bincount(pos,weights=w*y,minlength=len(site_ids))
            den=multiplicity@site_w;num=multiplicity@site_y;reps=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
            lo,hi=np.nanquantile(reps,[.025,.975]) if part.site_id.nunique()>=4 else (np.nan,np.nan)
            calibration.append(dict(evaluation=key,bin_lower=edges[i],bin_upper=edges[i+1],n=len(part),sites=part.site_id.nunique(),
                predicted_mean=np.average(part.prediction_percent,weights=w),observed_mean=np.average(y,weights=w),lower95=lo,upper95=hi))
        outcome_bin=pd.cut(f.observed_percent,bounds,include_lowest=True)
        for i,interval in enumerate(outcome_bin.cat.categories):
            part=f[outcome_bin.eq(interval)]
            if len(part):bias.append(dict(evaluation=key,bin_lower=bounds[i],bin_upper=bounds[i+1],n=len(part),**metrics(part)))
        for context,mask in [('First visit',f.first_visit),('Intermediate',~f.first_visit&~f.last_visit),
                             ('Last visit',f.last_visit),('Only visit',f.only_visit)]:
            part=f[mask]
            if len(part):contexts.append(dict(evaluation=key,visit_context=context,n=len(part),**metrics(part)))
    return p,pd.DataFrame(calibration),pd.DataFrame(bias),pd.DataFrame(contexts)


def diagnostic_figure(calibration,bias,contexts):
    fig=plt.figure(figsize=(19/2.54,15.3/2.54),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[1.2,1],hspace=.12,wspace=.13)
    a=fig.add_subplot(grid[0,:]);b=fig.add_subplot(grid[1,0]);c=fig.add_subplot(grid[1,1])
    a.plot([0,100],[0,100],ls='--',color='#666666',linewidth=1)
    order=['First visit','Intermediate','Last visit','Only visit']
    for j,key in enumerate(['temporal','spatial_50km']):
        color=COLORS[key];marker='o' if j==0 else 's'
        part=calibration[calibration.evaluation.eq(key)].sort_values('bin_lower')
        a.errorbar(part.predicted_mean,part.observed_mean,
            yerr=np.array([part.observed_mean-part.lower95,part.upper95-part.observed_mean]),
            fmt=marker+'-',color=color,markersize=4,capsize=2,linewidth=1.2,label=LABELS[key])
        part=bias[bias.evaluation.eq(key)].sort_values('bin_lower')
        b.plot(part.observed_mean,part.bias,marker+'-',color=color,markersize=4,linewidth=1.2)
        part=contexts[contexts.evaluation.eq(key)].set_index('visit_context').reindex(order).dropna()
        y=np.array([order.index(v) for v in part.index])+(-.12 if j==0 else .12)
        c.scatter(part.rmse,y,marker=marker,color=color,s=30)
    a.set(xlim=(0,100),ylim=(-3,103),xlabel='Mean predicted disease (%)',ylabel='Mean observed disease (%)',
        xticks=[0,25,50,75,100],yticks=[0,25,50,75,100])
    a.legend(loc='upper left',frameon=False,ncol=2)
    b.axhline(0,color='#888888',linewidth=.8)
    b.set(xlim=(0,100),xticks=[0,25,50,75,100],
        xlabel='Mean observed disease (%)',ylabel='Prediction bias (pp)')
    c.set(yticks=range(4),yticklabels=order,xlabel='RMSE (pp)',xlim=(0,contexts.rmse.max()*1.15),ylim=(3.5,-.5))
    for ax,label in zip([a,b,c],'abc'):ax.text(0,1.04,f'({label})',transform=ax.transAxes,va='bottom',ha='left',fontsize=12)
    return fig
