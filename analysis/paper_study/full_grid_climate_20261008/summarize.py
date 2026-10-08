"""Exact land-use-area summaries and raster maps after all annual jobs finish."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MODELS=['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']
SCENARIOS=['ssp126','ssp245','ssp585']
PERIODS={'1991-2020':range(1991,2021),'2031-2060':range(2031,2061),'2071-2100':range(2071,2101)}
METRICS=['GS65_85_lost_had3','F1_symptom_relative_anthesis_days','grain_fill_days',
         'conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai','GS31_85_lost_had3',
         'mean_temperature_grain_fill_c','rainfall_grain_fill_mm',
         'F1_symptom_before85','all_top3_symptom_before85',
         'F1_symptomatic_grain_fill_days','F1_symptomatic_grain_fill_fraction',
         'GS65_85_functional_lost_fraction',
         'conditional_yield_loss_GS65_85_b0141_t_ha_per_unit_lai',
         'conditional_yield_loss_GS65_85_b0207_t_ha_per_unit_lai',
         'conditional_yield_loss_GS31_85_b0180_t_ha_per_unit_lai',
         'F1_symptom_day_after_sowing']
OUTPUT=ROOT/'analysis/paper_study/nature_food_revision_20261007/continental_replay/annual_outputs/nasa'


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def summarize(*,periods=None,destination=None,include_figures=True):
    periods=PERIODS if periods is None else periods
    destination=HERE/'results' if destination is None else Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    future_periods=[p for p in periods if p!='1991-2020']
    expected_jobs=len(MODELS)*len(SCENARIOS)*sum(len(years) for years in periods.values())
    cells=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet')
    regions=pd.read_csv(ROOT/'analysis/paper_study/nature_food_submission_20261008/spatial_sources/wheat_cells_eea_regions.csv')
    cells=cells.merge(regions[['cell_id','environment_region','calendar_valid']],on='cell_id',validate='one_to_one')
    assert len(cells)==14941 and cells.calendar_valid.sum()==14932
    domains=[('environment_region',name,
              np.ones(len(cells),bool) if name=='Europe' else cells.environment_region.eq(name).to_numpy())
             for name in ['Europe']+sorted(cells.environment_region.unique())]
    domains += [('country',name,cells.dominant_source_country.eq(name).to_numpy())
                for name in sorted(cells.dominant_source_country.dropna().unique())]
    weights=cells.harvested_total_ha.to_numpy(float)
    total_area=float(weights.sum())
    regional=[]; grid_records=[]; annual=[]; coverage=[]
    source_records=[]
    for model in MODELS:
        for scenario in SCENARIOS:
            arrays={}
            for period,years in periods.items():
                values=np.empty((len(cells),30,len(METRICS)))
                for index,year in enumerate(years):
                    path=OUTPUT/model/scenario/f'{year}.parquet'
                    receipt=json.loads(path.with_suffix('.json').read_text())
                    if receipt['output_sha256']!=_sha(path):raise ValueError('Full-grid output checksum changed: '+str(path))
                    frame=pd.read_parquet(path,columns=['cell_id','status','valid_complete_season',*METRICS])
                    if not np.array_equal(frame.cell_id,cells.cell_id):raise ValueError('Full-grid cell order changed.')
                    values[:,index,:]=frame[METRICS].to_numpy(float)
                    source_records.append(dict(path=str(path.relative_to(ROOT)),sha256=receipt['output_sha256']))
                    for status,part in frame.groupby('status'):
                        mask=frame.status.eq(status).to_numpy()
                        coverage.append(dict(model=model,scenario=scenario,period=period,year=year,status=status,
                            cells=int(mask.sum()),harvested_area_ha=float(weights[mask].sum())))
                    for j,metric in enumerate(METRICS):
                        finite=np.isfinite(values[:,index,j]);den=float(weights[finite].sum())
                        mean=float(weights[finite]@values[finite,index,j]/den) if den else np.nan
                        annual.append(dict(model=model,scenario=scenario,year=year,period=period,metric=metric,
                            area_weighted_mean=mean,valid_cells=int(finite.sum()),valid_landuse_area_fraction=den/total_area))
                arrays[period]=values
            for period in future_periods:
                history=arrays['1991-2020'];future=arrays[period]
                for j,metric in enumerate(METRICS):
                    a=history[:,:,j];b=future[:,:,j]
                    common=np.isfinite(a)&np.isfinite(b)
                    counts=common.sum(axis=1);z=counts/30
                    x=np.where(common,b-a,0).mean(axis=1)
                    rx=np.where(common,a,0).mean(axis=1);fx=np.where(common,b,0).mean(axis=1)
                    value=np.divide(x,z,out=np.full(len(cells),np.nan),where=z>0)
                    grid_records.append(pd.DataFrame(dict(cell_id=cells.cell_id,model=model,scenario=scenario,period=period,
                        metric=metric,change=value,reference_on_common=np.divide(rx,z,out=np.full(len(cells),np.nan),where=z>0),
                        future_on_common=np.divide(fx,z,out=np.full(len(cells),np.nan),where=z>0),valid_year_pairs=counts)))
                    for domain_type,region,indicator in domains:
                        w=weights*indicator;den=float(w@z)
                        mu=float(w@x/den) if den else np.nan
                        reference=float(w@rx/den) if den else np.nan;projected=float(w@fx/den) if den else np.nan
                        if den:assert np.isclose(projected-reference,mu,atol=1e-11)
                        regional.append(dict(domain_type=domain_type,environment_region=region,model=model,scenario=scenario,period=period,metric=metric,
                            change=mu,reference_on_common=reference,future_on_common=projected,
                            landuse_cells=int(indicator.sum()),calendar_eligible_cells=int((indicator&cells.calendar_valid.to_numpy()).sum()),
                            valid_year_pairs=int(counts[indicator].sum()),
                            reference_area_ha=float(w.sum()),valid_landuse_area_time_fraction=den/w.sum() if w.sum() else np.nan,
                            population='Exact fixed-landuse-area census; no spatial sampling'))
    assert len(source_records)==expected_jobs
    grid=pd.concat(grid_records,ignore_index=True)
    grid.to_parquet(destination/'full_grid_paired_changes_by_gcm.parquet',index=False,compression='zstd')
    regional=pd.DataFrame(regional)
    regional[regional.domain_type.eq('environment_region')].to_csv(destination/'environmental_region_changes_by_gcm.csv',index=False)
    regional[regional.domain_type.eq('country')].rename(columns={'environment_region':'country'}).to_csv(destination/'country_changes_by_gcm.csv',index=False)
    grouping=['domain_type','environment_region','scenario','period','metric']
    rows=[]
    for key,part in regional.groupby(grouping,sort=True):
        assert len(part)==3
        complete=part.change.notna().all()
        rows.append(dict(zip(grouping,key),mean_change=float(part.change.mean()) if complete else np.nan,
            gcm_min=float(part.change.min()) if complete else np.nan,gcm_max=float(part.change.max()) if complete else np.nan,
            reference_on_common=float(part.reference_on_common.mean()) if complete else np.nan,
            future_on_common=float(part.future_on_common.mean()) if complete else np.nan,
            landuse_cells=int(part.landuse_cells.iloc[0]),calendar_eligible_cells=int(part.calendar_eligible_cells.iloc[0]),
            reference_area_ha=float(part.reference_area_ha.iloc[0]),minimum_valid_area_time_fraction=float(part.valid_landuse_area_time_fraction.min()),
            maximum_valid_area_time_fraction=float(part.valid_landuse_area_time_fraction.max()),spatial_sampling_error_applicable=False))
    ensemble_regions=pd.DataFrame(rows)
    ensemble_regions[ensemble_regions.domain_type.eq('environment_region')].to_csv(destination/'environmental_region_changes.csv',index=False)
    ensemble_regions[ensemble_regions.domain_type.eq('country')].rename(columns={'environment_region':'country'}).to_csv(destination/'country_changes.csv',index=False)
    key=['cell_id','scenario','period','metric']
    grid['positive']=grid.change.gt(0).astype('int8')
    grid['negative']=grid.change.lt(0).astype('int8')
    grouped=grid.groupby(key,sort=False)
    ensemble=grouped.agg(mean_change=('change','mean'),gcm_min=('change','min'),gcm_max=('change','max'),
        reference_on_common=('reference_on_common','mean'),future_on_common=('future_on_common','mean'),
        available_gcms=('change','count'),minimum_valid_year_pairs=('valid_year_pairs','min'),maximum_valid_year_pairs=('valid_year_pairs','max'),
        positive_gcm_count=('positive','sum'),negative_gcm_count=('negative','sum')).reset_index()
    for c in ['mean_change','gcm_min','gcm_max','reference_on_common','future_on_common']:ensemble.loc[ensemble.available_gcms.ne(3),c]=np.nan
    ensemble=ensemble.merge(cells[['cell_id','row','col','latitude','longitude','harvested_total_ha','environment_region','calendar_valid','dominant_source_country']],on='cell_id',validate='many_to_one')
    ensemble.to_parquet(destination/'full_grid_ensemble_paired_changes.parquet',index=False,compression='zstd')
    for metric,label in [(METRICS[0],'canopy'),(METRICS[1],'symptoms')]:
        ensemble[ensemble.metric.eq(metric)].to_csv(destination/f'full_grid_{label}_scenario_changes.csv.gz',index=False)
    shares=[]
    for (scenario,period),part in ensemble[ensemble.metric.eq(METRICS[0])].groupby(['scenario','period']):
        complete=part.available_gcms.eq(3)
        area=float(part.loc[complete,'harvested_total_ha'].sum())
        for label,mask in [('positive_ensemble',part.mean_change.gt(0)),('positive_all_three_gcm',part.positive_gcm_count.eq(3)),
                           ('negative_all_three_gcm',part.negative_gcm_count.eq(3)),
                           ('mixed_gcm_sign',part.positive_gcm_count.gt(0)&part.negative_gcm_count.gt(0))]:
            selected=complete&mask
            shares.append(dict(scenario=scenario,period=period,classification=label,
                landuse_area_share_percent=100*float(part.loc[selected,'harvested_total_ha'].sum())/area,
                cells=int(selected.sum()),complete_gcm_landuse_area_fraction=area/total_area,spatial_sampling_error_applicable=False))
    pd.DataFrame(shares).to_csv(destination/'canopy_change_sign_area_shares.csv',index=False)
    pd.DataFrame(annual).to_csv(destination/'annual_area_weighted_means.csv',index=False)
    pd.DataFrame(coverage).to_csv(destination/'annual_status_and_area_coverage.csv',index=False)
    cells.to_csv(destination/'full_landuse_cell_registry.csv',index=False)
    if include_figures:
        _figures(destination,ensemble,cells)
        _results_text(destination,ensemble_regions[ensemble_regions.domain_type.eq('environment_region')],pd.DataFrame(shares))
    receipt=dict(status='complete',annual_jobs=expected_jobs,landuse_cells=14941,calendar_eligible_cells=14932,
        model_grid_seasons=14932*expected_jobs,full_grid_census=True,all_planned_periods_complete=expected_jobs==810,
        periods=list(periods),model_parameters_changed=False,
        summary_code_sha256=_sha(Path(__file__)),
        source_outputs=source_records,
        outputs_sha256={p.name:_sha(p) for p in destination.iterdir() if p.is_file() and p.name!='completion_receipt.json'},
        inference_scope='Conditional frozen-model canopy response; no newly validated absolute-yield forecast.')
    (destination/'completion_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def _figures(destination,ensemble,cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from analysis.paper_study.map_restoration_20261007.figures import geography,raster,symmetric_norm,DIVERGING
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    labels={'ssp126':'SSP1–2.6','ssp245':'SSP2–4.5','ssp585':'SSP5–8.5'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    for metric,stem,unit in [(METRICS[0],'full_grid_canopy_scenario_maps','HAD deficit change (days / reference upper-three-leaf LAI)'),
                             (METRICS[1],'full_grid_symptom_scenario_maps','Symptom timing change relative to flowering (days)')]:
        data=ensemble[ensemble.metric.eq(metric)];norm=symmetric_norm(data.mean_change.to_numpy(),2)
        fig,axes=plt.subplots(2,3,figsize=(10.5,6.3))
        fig.subplots_adjust(left=.055,right=.985,top=.94,bottom=.15,wspace=.12,hspace=.25)
        for r,period in enumerate(['2031-2060','2071-2100']):
            for c,scenario in enumerate(SCENARIOS):
                ax=axes[r,c];geography(ax,countries)
                part=data[data.scenario.eq(scenario)&data.period.eq(period)]
                assert len(part)==14941
                im=raster(ax,part,'mean_change',norm=norm,cmap=DIVERGING)
                ax.set_title(f'{chr(97+r*3+c)}   {labels[scenario]}; {period}',loc='left',fontsize=9.2,fontweight='bold')
                ax.set_xlabel('');ax.set_ylabel('Latitude (°N)' if c==0 else '')
                if c:ax.set_yticklabels([])
                if not r:ax.set_xticklabels([])
                else:ax.set_xticklabels(['0°','20°E','40°E','60°E'])
        barax=fig.add_axes([.24,.065,.54,.023]);bar=fig.colorbar(im,cax=barax,orientation='horizontal');bar.set_label(unit,fontsize=9)
        for suffix in ['png','pdf','svg']:fig.savefig(destination/f'{stem}.{suffix}',dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    (destination/'figure_captions.txt').write_text(
        'Full-grid canopy response across three emissions pathways. Panels show arithmetic three-GCM means of paired period changes at every modeled land-use wheat cell. '
        'Each future harvest year is paired with the corresponding 1991–2020 reference-year index within the same GCM and SSP. '
        'Gray or uncolored wheat cells have unavailable three-GCM response estimates; invalid seasons are not assigned zero damage. '
        'All 14,941 land-use wheat cells are retained, with 14,932 eligible crop calendars. The imposed winter/rainfed management scenario and fixed SPAM2020 area define the population. '
        'GCM means and ranges describe frozen-model response rather than observed disease prevalence or validated absolute yield loss.\n\n'
        'Full-grid symptom timing response. Panel arrangement, paired populations and fixed land-use mask follow the canopy map. '
        'Timing is conditional on symptoms being detected in both paired periods; valid-year counts and GCM completeness accompany each cell.\n')


def _results_text(destination,regions,shares):
    data=regions[regions.period.eq('2071-2100')&regions.metric.eq(METRICS[0])].set_index(['environment_region','scenario'])
    total=[data.loc['Europe',scenario].mean_change for scenario in SCENARIOS]
    p=['Spatial climate responses across the wheat land-use domain',
       'The fixed SPAM2020 wheat mask contains 14,941 quarter-degree cells, of which 14,932 have an eligible imposed winter/rainfed calendar. '
       'The full-domain comparison spans three climate models, three emissions pathways and three 30-year harvest periods, giving 810 annual model combinations. '
       'Grid-level responses are computed independently from daily weather; crop area supplies exact aggregation weights.',
       f'Late-century European upper-three-leaf HAD-deficit changes are {total[0]:+.2f}, {total[1]:+.2f} and {total[2]:+.2f} days per nominal reference LAI under SSP1–2.6, SSP2–4.5 and SSP5–8.5, respectively. '
       'These means use paired valid area-time support and describe the fixed-model management scenario. Full-grid estimation removes the spatial sampling component of uncertainty; climate-model spread and biological-transfer uncertainty remain distinct.']
    named=['Atlantic','Continental','Boreal','Mediterranean','Steppic']
    p.append('Under late-century SSP5–8.5, environmental-region HAD changes are '+', '.join(
        f'{region} {data.loc[region,"ssp585"].mean_change:+.2f}' for region in named)+
        ' days per nominal reference LAI. Regional model ranges, calendar coverage and valid-year support accompany these estimates. The EEA boundaries are fixed across periods and are independent of the modeled response.')
    s=shares[shares.scenario.eq('ssp585')&shares.period.eq('2071-2100')].set_index('classification')
    p.append(f'A positive three-GCM ensemble mean covers {s.loc["positive_ensemble","landuse_area_share_percent"]:.1f}% of the available reference wheat area. '
        f'All three GCMs retain positive changes on {s.loc["positive_all_three_gcm","landuse_area_share_percent"]:.1f}% and negative changes on {s.loc["negative_all_three_gcm","landuse_area_share_percent"]:.1f}%. '
        'Agreement across these deterministic scenarios does not identify a probability of future disease or a tested adaptation benefit.')
    (destination/'spatial_results.txt').write_text('\n\n'.join(p)+'\n')


if __name__=='__main__':
    receipt=summarize()
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_outputs'},indent=2))
