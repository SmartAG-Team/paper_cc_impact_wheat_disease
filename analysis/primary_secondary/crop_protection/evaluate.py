"""Crop-protection diagnostics from frozen v2 predictions; no model refitting."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibration.primary_secondary.calibrate import score
from calibration.primary_secondary.crop_protection_metrics import attach_forecast_context, threshold_diagnostics
from calibration.primary_secondary.uncertainty import paired_location_bootstrap

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASF=ROOT/'analysis/primary_secondary/calibration_v2_age'
FRENCH=ROOT/'analysis/primary_secondary/external_french'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def slices(frame):
    for name,mask in [('all_eligible',np.ones(len(frame),bool)),
            ('upper_three',frame.leaf_rank.le(3)),('flag_leaf',frame.leaf_rank.eq(1))]:
        subset=frame.loc[mask]
        if subset.empty:continue
        yield name,'all','all',subset
        for horizon,group in subset.groupby('horizon_group',sort=True):
            yield name,horizon,'all',group
        for stage,group in subset.groupby('stage_group',sort=True):
            yield name,'all',stage,group


def load():
    p=pd.read_csv(BASF/'predictions.csv')
    p=attach_forecast_context(p,pd.read_csv(BASF/'episodes.csv'))
    p['dataset']='BASF';p['endpoint']='recorded_percent_infection'
    p['evaluation_status']='development-era grouped transfer'
    french=pd.read_csv(FRENCH/'transfer_v2/frozen_transfer_predictions.csv')
    french=attach_forecast_context(french,pd.read_csv(FRENCH/'episodes.csv'))
    french['dataset']='French field'
    french['fold']='external_'+french.year.astype(str)
    french['fold_type']='external_station_year'
    french['endpoint']='adult_leaf_pycnidial_area_percent'
    french['evaluation_status']='BASF-frozen cross-endpoint transfer; one independent station'
    return pd.concat([p,french],ignore_index=True)


def figure(metrics):
    selected=metrics.loc[metrics.dataset.eq('BASF')&metrics.fold.str.startswith('forward')
        &metrics.leaf_scope.eq('upper_three')&metrics.stage_scope.eq('all')]
    models=['primary_secondary_hidden','primary_secondary_no_weather','global_logit_trend','persistence']
    labels=['Weather SEIR','No-weather SEIR','Logit trend','Persistence']
    colors=['#187c83','#bf7823','#6e529a','#677785']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),sharey=True)
    horizons=['01-07','08-14','15-28','29+']
    for ax,fold in zip(axes,['forward_2018','forward_2019']):
        for model,label,color in zip(models,labels,colors):
            f=selected.loc[selected.fold.eq(fold)&selected.model.eq(model)].set_index('horizon_scope')
            values=[f.loc[h,'rmse_pp'] if h in f.index else np.nan for h in horizons]
            ax.plot(range(4),values,'o-',lw=1.5,ms=5,label=label,color=color)
        reference=selected.loc[selected.fold.eq(fold)&selected.model.eq(models[0])].set_index('horizon_scope')
        ticks=[label+'\n'+(f"n={int(reference.loc[h,'n_targets'])}" if h in reference.index else 'n=0')
               for h,label in zip(horizons,['1–7','8–14','15–28','≥29'])]
        ax.set_xticks(range(4),ticks);ax.set_xlabel('Days since first disease assessment',labelpad=10)
        ax.set_title(f"{fold[-4:]} — upper three leaves")
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('RMSE (percentage points)');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Conditional Septoria prediction with realized future weather',fontsize=12)
    fig.tight_layout(rect=[0,.025,1,.94]);fig.savefig(OUT/'upper_leaf_forecast_horizons.png',dpi=220)
    fig.savefig(OUT/'upper_leaf_forecast_horizons.pdf');plt.close(fig)


def map_errors(predictions):
    f=predictions.loc[predictions.dataset.eq('BASF')&predictions.fold.str.startswith('forward')
        &predictions.model.eq('primary_secondary_hidden')&predictions.leaf_rank.le(3)].copy()
    f['error']=f.predicted_percent-f.observed_percent
    f['square']=f.error**2
    grouped=f.groupby(['location_id','year','series_id']).agg(square=('square','mean'),bias=('error','mean'),
        Latitude=('Latitude','first'),Longitude=('Longitude','first')).reset_index()
    points=grouped.groupby(['location_id','year']).agg(square=('square','mean'),bias=('bias','mean'),
        latitude=('Latitude','first'),longitude=('Longitude','first'),series=('series_id','nunique')).reset_index()
    points['rmse_pp']=np.sqrt(points.square)
    points.to_csv(OUT/'upper_leaf_error_map_data.csv',index=False)
    geography=json.loads((ROOT/'data/era5/europe/europe_region_countries.geojson').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9.5,5),sharex=True,sharey=True)
    for ax,year in zip(axes,[2018,2019]):
        for feature in geography['features']:
            geometry=feature['geometry']
            polygons=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
            if geometry['type'] not in ['Polygon','MultiPolygon']:continue
            for polygon in polygons:
                ring=np.array(polygon[0]);ax.fill(ring[:,0],ring[:,1],facecolor='#f0f1f2',edgecolor='#bdc5ca',lw=.35,zorder=0)
        p=points.loc[points.year.eq(year)]
        scatter=ax.scatter(p.longitude,p.latitude,c=p.rmse_pp,cmap='YlOrRd',vmin=0,vmax=55,
            s=50,edgecolors='#293847',linewidths=.6,zorder=2)
        ax.set_title(f'{year}: {len(p)} coordinate-years');ax.set_xlim(-12,30);ax.set_ylim(38,61)
        ax.set_aspect(1.4);ax.set_xlabel('Longitude');ax.grid(alpha=.15)
    axes[0].set_ylabel('Latitude')
    fig.suptitle('Field error locations — observed trials only',fontsize=12)
    fig.subplots_adjust(left=.07,right=.84,bottom=.12,top=.88,wspace=.15)
    colour_axis=fig.add_axes([.88,.24,.022,.49])
    fig.colorbar(scatter,cax=colour_axis,label='Upper-leaf RMSE (percentage points)')
    fig.savefig(OUT/'upper_leaf_field_error_map.png',dpi=220)
    fig.savefig(OUT/'upper_leaf_field_error_map.pdf');plt.close(fig)


def main():
    inputs=[BASF/'predictions.csv',BASF/'episodes.csv',FRENCH/'episodes.csv',
        FRENCH/'transfer_v2/frozen_transfer_predictions.csv',OUT/'evaluation_contract.json',
        ROOT/'calibration/primary_secondary/crop_protection_metrics.py',Path(__file__),
        ROOT/'calibration/primary_secondary/uncertainty.py',
        ROOT/'data/era5/europe/europe_region_countries.geojson']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    predictions=load()
    if predictions.duplicated(['dataset','fold','model','series_id','Date']).any():
        raise ValueError('Duplicate forecast target identities.')
    predictions.to_csv(OUT/'forecast_context.csv.gz',index=False,compression='gzip')
    severity=[];diagnostics=[]
    for (dataset,fold,model),f in predictions.groupby(['dataset','fold','model'],sort=True):
        for leaf,horizon,stage,subset in slices(f):
            identity={'dataset':dataset,'fold':fold,'model':model,'leaf_scope':leaf,
                'horizon_scope':horizon,'stage_scope':stage,'endpoint':subset.endpoint.iloc[0]}
            severity.append({**identity,**score(subset),
                'n_locations':subset.location_id.nunique(),
                'n_trials_or_plots':subset.TrialId.nunique()})
            for cutoff in [1,5,10,25]:
                for eligibility,selected in [('all',subset),('initial_below_cutoff',subset.loc[subset.initial_percent.lt(cutoff)])]:
                    if not selected.empty:
                        diagnostics.append({**identity,'eligibility':eligibility,
                            **threshold_diagnostics(selected,cutoff),
                            'n_locations':selected.location_id.nunique()})
    severity=pd.DataFrame(severity);severity.to_csv(OUT/'severity_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'threshold_diagnostics.csv',index=False)
    comparisons=[]
    for fold,f in predictions.loc[predictions.dataset.eq('BASF')&predictions.fold.str.startswith('forward')
            &predictions.leaf_rank.le(3)].groupby('fold'):
        for baseline in ['global_logit_trend','canopy_logit_trend','primary_secondary_no_weather','persistence']:
            comparisons.append({'fold':fold,'leaf_scope':'upper_three',
                **paired_location_bootstrap(f,'primary_secondary_hidden',baseline)})
    pd.DataFrame(comparisons).to_csv(OUT/'upper_leaf_paired_comparisons.csv',index=False)
    figure(severity);map_errors(predictions)
    unique=predictions.drop_duplicates(['dataset','fold','series_id','Date'])
    unique.groupby(['dataset','fold','horizon_group','stage_group','leaf_rank'],dropna=False).agg(
        targets=('Date','size'),series=('series_id','nunique'),locations=('location_id','nunique')).reset_index().to_csv(
        OUT/'assessment_coverage.csv',index=False)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
        'inputs_immutable':all(digest(ROOT/p)==h for p,h in hashes.items()),
        'forecast_rows':len(predictions),'severity_metric_rows':len(severity),
        'threshold_diagnostic_rows':len(diagnostics),'new_parameters_fitted':False,
        'initial_assessments_scored':False,'future_observed_growth_stage_used':False,
        'deterministic_severity_treated_as_probability':False,
        'economic_or_spray_threshold_validated':False,'operational_weather_forecast_validated':False,
        'continental_disease_prediction_validated':False}
    (OUT/'verification_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
