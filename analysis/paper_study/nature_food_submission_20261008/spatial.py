"""Spatial and environmental-domain summaries of frozen climate simulations."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/climate'
DECOMPOSITION = ROOT/'analysis/paper_study/nature_food_fix_20261007/climate/decomposition'
DEST = HERE/'derived/spatial'
MODELS = ['ACCESS-CM2', 'MPI-ESM1-2-HR', 'MRI-ESM2-0']
SCENARIOS = ['ssp126', 'ssp245', 'ssp585']
PERIODS = ['2031-2060', '2071-2100']
METRICS = ['GS65_85_lost_had3', 'F1_symptom_relative_anthesis_days', 'grain_fill_days']
MAJOR_REGIONS = ['Atlantic', 'Continental', 'Boreal', 'Mediterranean', 'Steppic']


def _variance(vector, draws):
    return sum(float(g.area_mean_weight.sum())**2 *
               float(np.var(vector[g.index.to_numpy()], ddof=1))/len(g)
               for _, g in draws.groupby('stratum', sort=False))


def _ratio(x, z, draws):
    weights = draws.area_mean_weight.to_numpy()
    denominator = float(weights @ z)
    if denominator <= 0:
        return np.nan, np.full(len(draws), np.nan), 0.
    mean = float(weights @ x)/denominator
    return mean, (x - mean*z)/denominator, denominator


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assign_regions(raw_geojson, destination=None):
    """Recreate frozen centroid assignments from the checksum-identified EEA source."""
    import geopandas as gpd
    registry=pd.read_csv(ROOT/'analysis/paper_study/map_restoration_20261007/map_wheat_domain.csv')
    points=gpd.GeoDataFrame(registry,geometry=gpd.points_from_xy(registry.longitude,registry.latitude),crs='EPSG:4326')
    regions=gpd.read_file(raw_geojson)
    joined=gpd.sjoin(points,regions[['code','geometry']],how='left',predicate='intersects')
    if len(joined)!=len(registry):raise ValueError('Overlapping region assignments require inspection.')
    joined['environment_region']=joined.code.replace({'BlackSea':'Black Sea','Outside':'Outside EEA regions'}).fillna('Unassigned')
    result=joined.drop(columns=['geometry','index_right','code'])
    calendars=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_calendar_scenarios.parquet')
    calendars=calendars[calendars.crop_season.eq('winter_wheat')&calendars.water_system.eq('rainfed')][['cell_id','calendar_valid']]
    result=result.merge(calendars,on='cell_id',validate='one_to_one')
    if destination is not None:result.to_csv(destination,index=False)
    return result


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    registry_path = HERE/'spatial_sources/wheat_cells_eea_regions.csv'
    registry = pd.read_csv(registry_path)
    draw_path = ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv'
    draws = pd.read_csv(draw_path).sort_values('spatial_draw_id').reset_index(drop=True)
    draws = draws.merge(registry[['cell_id', 'environment_region']], on='cell_id', validate='many_to_one')
    assert len(draws) == 64 and draws.cell_id.nunique() == 62
    assert draws.groupby('stratum').size().eq(4).all()
    assert np.isclose(draws.area_mean_weight.sum(), 1.)
    assert draws.environment_region.notna().all()
    draws.to_csv(DEST/'sample_region_membership.csv', index=False)
    eligible = registry[registry.calendar_valid].copy()
    total_area = eligible.harvested_total_ha.sum()
    sampling = json.loads((draw_path.parent/'sampling_receipt.json').read_text())
    assert np.isclose(total_area, sampling['eligible_harvested_ha'], rtol=0, atol=.01)
    region_names = sorted(set(registry.environment_region))
    domains = ['Europe'] + region_names
    context = []
    for region in domains:
        cells = eligible if region == 'Europe' else eligible[eligible.environment_region.eq(region)]
        sample = draws if region == 'Europe' else draws[draws.environment_region.eq(region)]
        context.append(dict(environment_region=region, eligible_wheat_cells=len(cells),
            reference_harvested_area_ha=float(cells.harvested_total_ha.sum()),
            reference_area_share_percent=100*float(cells.harvested_total_ha.sum())/total_area,
            sample_draws=len(sample), unique_sampled_cells=sample.cell_id.nunique(),
            sample_design_area_share_percent=100*float(sample.area_mean_weight.sum()),
            display_group=('Main regional comparison' if region in MAJOR_REGIONS else 'All-domain or sparse/unrepresented')))
    pd.DataFrame(context).to_csv(DEST/'region_sampling_context.csv', index=False)
    paired_path = STUDY/'paired_draw_period_moments.csv'
    paired = pd.read_csv(paired_path)
    paired = paired[paired.metric.isin(METRICS)].copy()
    paired = paired.merge(draws[['spatial_draw_id', 'cell_id', 'latitude', 'longitude', 'environment_region']],
                          on='spatial_draw_id', validate='many_to_one')
    rows = []; gcm_rows = []; cell_rows = []; shares = []
    for (scenario, period, metric), group in paired.groupby(['scenario', 'future_period', 'metric'], sort=True):
        for region in domains:
            indicator = np.ones(len(draws)) if region == 'Europe' else draws.environment_region.eq(region).to_numpy(float)
            count = int(indicator.sum()); means = []; influences = []; coverage = []; refs = []; futures = []
            for model in MODELS:
                part = group[group.model.eq(model)].sort_values('spatial_draw_id')
                assert len(part) == len(draws)
                x = part.numerator.to_numpy()*indicator
                z = part.denominator.to_numpy()*indicator
                mean, influence, denominator = _ratio(x, z, draws)
                ref, _, _ = _ratio(part.reference_numerator.to_numpy()*indicator, z, draws)
                future, _, _ = _ratio(part.future_numerator.to_numpy()*indicator, z, draws)
                region_weight = float(draws.area_mean_weight.to_numpy() @ indicator)
                gcm_rows.append(dict(environment_region=region, scenario=scenario, period=period, metric=metric,
                    climate_model=model, mean_change=mean, reference_on_common=ref, future_on_common=future,
                    spatial_mcse=(float(np.sqrt(_variance(influence, draws))) if count >= 4 else np.nan),
                    valid_area_time_coverage=denominator/region_weight if region_weight else np.nan,
                    valid_year_pairs=int((part.valid_year_pairs.to_numpy()*indicator).sum())))
                means.append(mean); influences.append(influence); coverage.append(denominator/region_weight if region_weight else np.nan)
                refs.append(ref); futures.append(future)
            complete = all(np.isfinite(means))
            rows.append(dict(environment_region=region, scenario=scenario, period=period, metric=metric,
                mean_change=float(np.mean(means)) if complete else np.nan,
                gcm_min=float(np.min(means)) if complete else np.nan,
                gcm_max=float(np.max(means)) if complete else np.nan,
                spatial_mcse=float(np.sqrt(_variance(np.mean(influences, axis=0), draws))) if complete and count >= 4 else np.nan,
                reference_on_common=float(np.mean(refs)) if complete else np.nan,
                future_on_common=float(np.mean(futures)) if complete else np.nan,
                sample_draws=count, unique_sampled_cells=(draws.cell_id.nunique() if region == 'Europe' else draws.loc[indicator.astype(bool), 'cell_id'].nunique()),
                minimum_gcm_coverage=float(np.min(coverage)) if complete else np.nan,
                maximum_gcm_coverage=float(np.max(coverage)) if complete else np.nan,
                resolved_all_gcm_sign=('positive' if complete and min(means) > 0 else 'negative' if complete and max(means) < 0 else 'mixed' if complete else 'unavailable')))
        for cell, part in group.groupby('cell_id', sort=True):
            unique = part.drop_duplicates('model')
            assert set(unique.model) == set(MODELS)
            for model, duplicates in part.groupby('model'):
                assert np.allclose(duplicates.numerator, duplicates.numerator.iloc[0], equal_nan=True)
                assert np.allclose(duplicates.denominator, duplicates.denominator.iloc[0], equal_nan=True)
            values = unique.numerator.to_numpy()/np.where(unique.denominator.to_numpy() > 0, unique.denominator.to_numpy(), np.nan)
            complete = np.isfinite(values).all()
            cell_rows.append(dict(scenario=scenario, period=period, metric=metric, cell_id=cell,
                longitude=part.longitude.iloc[0], latitude=part.latitude.iloc[0], environment_region=part.environment_region.iloc[0],
                mean_change=float(np.mean(values)) if complete else np.nan,
                gcm_min=float(np.min(values)) if complete else np.nan, gcm_max=float(np.max(values)) if complete else np.nan,
                positive_gcm_count=int((values > 0).sum()) if complete else 0,
                negative_gcm_count=int((values < 0).sum()) if complete else 0,
                complete_gcm_count=int(np.isfinite(values).sum()),
                minimum_valid_year_pairs=int(unique.valid_year_pairs.min()), maximum_valid_year_pairs=int(unique.valid_year_pairs.max())))
    regional = pd.DataFrame(rows); gcm = pd.DataFrame(gcm_rows); cells = pd.DataFrame(cell_rows)
    regional.to_csv(DEST/'region_paired_changes.csv', index=False)
    gcm.to_csv(DEST/'region_paired_changes_by_gcm.csv', index=False)
    cells.to_csv(DEST/'sampled_cell_scenario_changes.csv', index=False)
    for (scenario, period), part in cells[cells.metric.eq(METRICS[0])].groupby(['scenario', 'period']):
        part = draws[['spatial_draw_id','cell_id','area_mean_weight','stratum']].merge(part, on='cell_id', validate='many_to_one')
        assert len(part) == 64
        complete = part.complete_gcm_count.eq(3).to_numpy(float)
        for label, indicator in [('positive_ensemble', part.mean_change.gt(0).to_numpy(float)),
                                 ('positive_all_three_gcm', part.positive_gcm_count.eq(3).to_numpy(float)),
                                 ('negative_all_three_gcm', part.negative_gcm_count.eq(3).to_numpy(float)),
                                 ('mixed_gcm_sign', (part.positive_gcm_count.gt(0)&part.negative_gcm_count.gt(0)).to_numpy(float))]:
            mean, influence, coverage = _ratio(indicator*complete, complete, draws)
            shares.append(dict(scenario=scenario, period=period, classification=label,
                estimated_reference_area_share_percent=100*mean,
                spatial_mcse_percentage_points=100*float(np.sqrt(_variance(influence, draws))),
                represented_reference_area_fraction=coverage,
                sample_draws=int((indicator*complete).sum()),
                unique_sampled_cells=int(part.loc[(indicator*complete).astype(bool), 'cell_id'].nunique())))
    pd.DataFrame(shares).to_csv(DEST/'canopy_change_sign_area_shares.csv', index=False)
    _decompose_regions(draws, domains)
    # Regional ratios partition the original design-weighted numerator and denominator.
    old = pd.read_csv(STUDY/'ensemble_paired_period_changes.csv')
    for row in regional[regional.environment_region.eq('Europe')].itertuples():
        match = old[old.scenario.eq(row.scenario)&old.future_period.eq(row.period)&old.metric.eq(row.metric)].iloc[0]
        assert np.isclose(row.mean_change, match.gcm_mean_change, atol=1e-12, rtol=0)
        assert np.isclose(row.spatial_mcse, match.spatial_mcse_gcm_mean_change, atol=1e-12, rtol=0)
    provenance = dict(status='derived from frozen current-model results',
        spatial_cells=62, spatial_draws=64, full_period_grid_census=False,
        environmental_source='EEA Biogeographical regions, Europe 2016, version 1',
        assignments='0.25-degree cell-centroid intersections; unmatched and outside-region cells retained explicitly',
        regional_statistics='Exploratory domain ratios under the original stratified area-proportional design; shared-draw climate-model covariance retained',
        uncertainty='Spatial MCSE only; deterministic GCM ranges are separate. MCSE withheld below four regional draws; unsampled domains have no response estimate.',
        major_region_rule='All named regions represented by at least four registered spatial draws',
        source_sha256={str(p.relative_to(ROOT)):_sha(p) for p in [registry_path, draw_path, paired_path,
            DECOMPOSITION/'four_corner_draw_year_outputs.parquet', DECOMPOSITION/'supported_forcing_draw_pair_flags.csv']})
    (DEST/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


def _decompose_regions(draws, domains):
    corners = pd.read_parquet(DECOMPOSITION/'four_corner_draw_year_outputs.parquet')
    flags = pd.read_csv(DECOMPOSITION/'supported_forcing_draw_pair_flags.csv')
    by_model = []; moments = {}
    for model in MODELS:
        arrays = {name:g.sort_values(['spatial_draw_id','reference_year'])[METRICS[0]].to_numpy().reshape(64,30)
                  for name,g in corners[corners.model.eq(model)].groupby('corner')}
        a=arrays['Wreference_Hreference']; b=arrays['Wfuture_Hreference']
        c=arrays['Wreference_Hfuture']; d=arrays['Wfuture_Hfuture']
        bad=flags[flags.model.eq(model)].sort_values(['spatial_draw_id','reference_year']).excluded_unsupported_hybrid_forcing.to_numpy().reshape(64,30)
        common=np.isfinite(a)&np.isfinite(b)&np.isfinite(c)&np.isfinite(d)&~bad
        interaction=(d-b)-(c-a)
        values={'weather':b-a+interaction/2,'host':c-a+interaction/2,'net':d-a}
        for region in domains:
            indicator=np.ones(64) if region=='Europe' else draws.environment_region.eq(region).to_numpy(float)
            z=common.mean(axis=1)*indicator; record=dict(environment_region=region,climate_model=model)
            for name, array in values.items():
                mean, influence, coverage=_ratio(np.where(common,array,0).mean(axis=1)*indicator,z,draws)
                record[name]=mean; moments[region,model,name]=influence
            record['host_offset_percent']=-100*record['host']/record['weather'] if abs(record['weather'])>1e-12 else np.nan
            record['valid_year_pairs']=int((common*indicator[:,None]).sum())
            by_model.append(record)
    by_model=pd.DataFrame(by_model); records=[]
    for region, part in by_model.groupby('environment_region',sort=True):
        complete=part[['weather','host','net']].notna().all().all()
        means=part[['weather','host','net']].mean() if complete else pd.Series({'weather':np.nan,'host':np.nan,'net':np.nan})
        count=len(draws) if region=='Europe' else int(draws.environment_region.eq(region).sum())
        record=dict(environment_region=region,**means.to_dict(),
            host_offset_percent=-100*means['host']/means['weather'] if abs(means['weather'])>1e-12 else np.nan,
            offset_gcm_min=float(part.host_offset_percent.min()) if complete else np.nan,
            offset_gcm_max=float(part.host_offset_percent.max()) if complete else np.nan,
            sample_draws=count)
        for name in ['weather','host','net']:
            vector=np.mean([moments[region,model,name] for model in MODELS],axis=0)
            record[name+'_spatial_mcse']=float(np.sqrt(_variance(vector,draws))) if complete and count>=4 else np.nan
        records.append(record)
    result=pd.DataFrame(records)
    assert np.allclose(result.weather+result.host,result.net,atol=1e-12,equal_nan=True)
    europe=result[result.environment_region.eq('Europe')].iloc[0]
    assert np.isclose(europe.host_offset_percent,70.63467958402005,atol=1e-10)
    by_model.to_csv(DEST/'region_supported_decomposition_by_gcm.csv',index=False)
    result.to_csv(DEST/'region_supported_decomposition.csv',index=False)


def supplementary_tables():
    context=pd.read_csv(DEST/'region_sampling_context.csv')
    changes=pd.read_csv(DEST/'region_paired_changes.csv')
    shares=pd.read_csv(DEST/'canopy_change_sign_area_shares.csv')
    decomposed=pd.read_csv(DEST/'region_supported_decomposition.csv')
    tables=[]
    rows=[['EEA region','Eligible cells','Reference area (million ha)','Area (%)','Draws / cells','Estimated sample area (%)']]
    for r in context.itertuples():
        rows.append([r.environment_region,str(r.eligible_wheat_cells),f'{r.reference_harvested_area_ha/1e6:.3f}',
                     f'{r.reference_area_share_percent:.2f}',f'{r.sample_draws} / {r.unique_sampled_cells}',f'{r.sample_design_area_share_percent:.2f}'])
    tables.append((rows,'Table S14a | Environmental-region coverage of the fixed wheat-management scenario. EEA 2016 regions are assigned by wheat-cell centroid. Reference area is the sum over eligible calendar cells; estimated sample area uses the original stratified design weights. Zero sampled draws indicate unavailable regional response estimates. Unassigned centroids and areas outside EEA regions retain separate labels.'))
    labels={'ssp126':'SSP1–2.6','ssp245':'SSP2–4.5','ssp585':'SSP5–8.5'}
    for letter,metric,name in [('b',METRICS[0],'upper-three-leaf HAD deficit'),('c',METRICS[1],'flag-leaf symptoms relative to flowering')]:
        rows=[['Region','SSP','Draws / cells','Mean change (days)','GCM range (days)','Spatial MCSE','Coverage (%)']]
        for region in ['Europe']+MAJOR_REGIONS+['Pannonian','Unassigned']:
            part=changes[changes.environment_region.eq(region)&changes.metric.eq(metric)&changes.period.eq('2071-2100')]
            for r in part.itertuples():
                rows.append([region,labels[r.scenario],f'{r.sample_draws} / {r.unique_sampled_cells}',f'{r.mean_change:+.2f}',
                    f'{r.gcm_min:+.2f} to {r.gcm_max:+.2f}',f'{r.spatial_mcse:.2f}' if np.isfinite(r.spatial_mcse) else '—',
                    f'{100*r.minimum_gcm_coverage:.1f}–{100*r.maximum_gcm_coverage:.1f}'])
        units=' HAD units are days per nominal upper-three-leaf reference LAI.' if metric==METRICS[0] else ''
        tables.append((rows,f'Table S14{letter} | Late-century environmental-region changes in modeled {name}, relative to each SSP-specific 1991–2020 reference. Means average three design-weighted climate-model domain ratios. Climate-model ranges and spatial Monte Carlo standard errors quantify different sources of spread; neither covers model-fitting or biological-transfer uncertainty. Sparse domains retain point estimates but have no reported MCSE. All periods and unrepresented regions are retained in source CSVs; unsampled responses remain unavailable.'+units))
    rows=[['Period','SSP','Positive ensemble (%)','Positive in all GCMs (%)','Negative in all GCMs (%)','Mixed signs (%)']]
    for period in PERIODS:
        for scenario in SCENARIOS:
            part=shares[shares.period.eq(period)&shares.scenario.eq(scenario)].set_index('classification')
            rows.append([period,labels[scenario]]+[f'{part.loc[key,"estimated_reference_area_share_percent"]:.1f}' for key in
                ['positive_ensemble','positive_all_three_gcm','negative_all_three_gcm','mixed_gcm_sign']])
    tables.append((rows,'Table S14d | Estimated shares of reference wheat area represented by sampled-cell canopy-response signs. Original draw identities and design weights are retained, including duplicate sampled cells. Classification uses each cell’s three-model paired period means on valid years. Agreement across three GCMs describes deterministic scenario consistency, not a probability of future damage. Positive, negative and mixed categories partition available three-model sign classifications; exact zero is retained in source data.'))
    rows=[['Region','Draws','Weather (days)','Host (days)','Net (days)','Host offset (%)','GCM offset range (%)']]
    for region in ['Europe']+MAJOR_REGIONS+['Pannonian','Unassigned']:
        r=decomposed[decomposed.environment_region.eq(region)].iloc[0]
        rows.append([region,str(r.sample_draws),f'{r.weather:+.2f}',f'{r.host:+.2f}',f'{r.net:+.2f}',f'{r.host_offset_percent:.1f}',f'{r.offset_gcm_min:.1f}–{r.offset_gcm_max:.1f}'])
    tables.append((rows,'Table S14e | Environmental-region weather–host decomposition of late-century SSP5–8.5 HAD deficit. All three components use the same supported four-corner population, excluding unsupported hybrid forcing. Host offset is the ratio of ensemble component means; GCM ranges are deterministic spread. Estimates for one-draw domains are descriptive and do not establish regional precision or adaptation efficacy. Reference area is the nominal upper-three-leaf LAI of one.'))
    return tables


if __name__ == '__main__':
    print(json.dumps(prepare(),indent=2))
