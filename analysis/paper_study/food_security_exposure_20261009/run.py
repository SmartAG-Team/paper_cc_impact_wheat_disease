"""Run original-source or compact-bundle production-exposure analysis."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from .exposure import (METRIC, GROUP, KEY, PRODUCTION, AREAS, classify_cells, validate_changes,
                       reconcile_production, aggregate_domains, reference_summary, require, _same, unique)
from .provenance import HERE, ROOT, digest, load_verified_inputs

SUMMARY_COLUMNS = GROUP+['analysis_mode','metric','baseline_production_tonnes','production_weighted_change',
    'gcm_min','gcm_max','all3positive_production_tonnes','all3positive_share_baseline_pct',
    'ensemble_increasing_production_tonnes','ensemble_increasing_share_baseline_pct',
    'all3negative_production_tonnes','all3negative_share_baseline_pct','mixed_sign_production_tonnes',
    'mixed_sign_share_baseline_pct','nonunanimous_production_tonnes','nonunanimous_share_baseline_pct',
    'unavailable_production_tonnes','unavailable_share_baseline_pct','complete3_production_tonnes',
    'minimum_valid_production_time_fraction','maximum_valid_production_time_fraction']
NAMED_REGIONS = ['Alpine','Atlantic','Black Sea','Boreal','Continental','Mediterranean','Pannonian','Steppic']


def build_tables(paired, weights):
    summaries,details,categories,classifications = [],[],[],[]
    for threshold in [1,27]:
        c = classify_cells(paired,min_pairs=threshold)
        s,d,l = aggregate_domains(paired,c,weights,min_pairs=threshold)
        c['analysis_mode'] = s.analysis_mode.iloc[0]
        summaries.append(s);details.append(d);categories.append(l);classifications.append(c)
    summary = pd.concat(summaries,ignore_index=True)
    categories = pd.concat(categories,ignore_index=True)
    # Each classification axis is exhaustive; each geographical partition conserves Europe mass.
    for _,part in categories.groupby(GROUP+['analysis_mode','classification_axis']):
        _same(part.production_tonnes.sum(),part.baseline_production_tonnes.iloc[0],'Exposure partition does not conserve baseline')
    for _,part in summary.groupby(['scenario','period','analysis_mode']):
        europe = part[part.domain_type.eq('Europe')].iloc[0]
        for level in ['country','environment_region']:
            p = part[part.domain_type.eq(level)]
            for col in ['baseline_production_tonnes','all3positive_production_tonnes','all3negative_production_tonnes',
                        'nonunanimous_production_tonnes','unavailable_production_tonnes','ensemble_increasing_production_tonnes']:
                _same(p[col].sum(),europe[col],f'{level} partition mismatch: {col}')
    summary['partition_residual_tonnes'] = summary.baseline_production_tonnes-summary[[
        'all3positive_production_tonnes','all3negative_production_tonnes','nonunanimous_production_tonnes','unavailable_production_tonnes']].sum(axis=1)
    _same(summary.partition_residual_tonnes,0,'Stack partition residual')
    compact = summary[SUMMARY_COLUMNS]
    region = compact[compact.domain_type.eq('environment_region')].copy()
    stacks = []
    for category,column in [('all3increase','all3positive_production_tonnes'),('mixed_or_zero','nonunanimous_production_tonnes'),
                             ('all3decrease','all3negative_production_tonnes'),('unavailable','unavailable_production_tonnes')]:
        s = compact[GROUP+['analysis_mode','baseline_production_tonnes',column]].rename(columns={column:'production_tonnes'})
        s['category'] = category
        s['share_baseline_pct'] = 100*s.production_tonnes/s.baseline_production_tonnes.replace(0,np.nan)
        stacks.append(s)
    stacks = pd.concat(stacks,ignore_index=True)
    table1 = compact[(compact.domain_type.eq('Europe')|compact.domain.isin(NAMED_REGIONS))&
                     compact.period.eq('2071-2100')].copy()
    table1['baseline_production_mt'] = table1.baseline_production_tonnes/1e6
    table1['all3increase_mt'] = table1.all3positive_production_tonnes/1e6
    table1['all3increase_share_pct'] = table1.all3positive_share_baseline_pct
    table1 = table1[GROUP+['analysis_mode','baseline_production_mt','all3increase_mt','all3increase_share_pct',
                          'production_weighted_change','gcm_min','gcm_max']]
    priorities = region.sort_values(['analysis_mode','scenario','period','all3positive_production_tonnes'],
                                    ascending=[True,True,True,False]).copy()
    priorities['ordering_measure'] = 'all3positive_production_tonnes; descending within mode/scenario/period'
    priorities['decision_use'] = 'Exposure supports surveillance and field-validation prioritization; mixed signs support scenario-contingent planning.'
    priorities['evidence_limit'] = 'No causal yield loss, adaptation efficacy, economic vulnerability or composite priority score.'
    readiness = summary[GROUP+['analysis_mode','baseline_production_tonnes','harvested_area_ha','physical_area_ha',
        'grid_cells','calendar_eligible_cells','calendar_eligible_production_tonnes','complete3_production_tonnes',
        'unavailable_production_tonnes','minimum_valid_production_time_fraction','maximum_valid_production_time_fraction']]
    return dict(domain_summary=summary,region_summary=region,
                country_summary=compact[compact.domain_type.eq('country')].copy(),
                europe_summary=compact[compact.domain_type.eq('Europe')].copy(),
                changes_by_gcm=pd.concat(details,ignore_index=True),exposure_categories=categories,
                region_exposure_stacked=stacks[stacks.domain_type.eq('environment_region')].copy(),
                domain_exposure_stacked=stacks,
                late_century_exposure_by_ssp=compact[compact.domain_type.eq('Europe')&compact.period.eq('2071-2100')].copy(),
                table1=table1,region_priorities=priorities,reference_production_area=reference_summary(weights),
                metric_readiness=readiness,cell_classifications=pd.concat(classifications,ignore_index=True))


def _write_csv(frame,path):
    frame.to_csv(path,index=False,float_format='%.17g')


def write_bundle(cells,weights,paired,source,destination):
    grid = cells[['cell_id','row','col','latitude','longitude','environment_region','calendar_valid',
                  *PRODUCTION,*AREAS,'production_reference_year']]
    country = weights[weights.domain_type.eq('country')][['cell_id','domain','FIPS0',*PRODUCTION,*AREAS]].rename(
        columns={'domain':'country','FIPS0':'country_fips'})
    country = country.merge(grid[['cell_id','production_total_tonnes']].rename(columns={'production_total_tonnes':'cell_production_tonnes'}),
                             on='cell_id',validate='many_to_one')
    country['production_fraction_of_cell'] = country.production_total_tonnes/country.cell_production_tonnes.replace(0,np.nan)
    country['positive_cell_production'] = country.cell_production_tonnes.gt(0)
    paths = ['production_grid_input.csv','country_fraction_weights.csv','paired_changes_input.csv.gz','native_source_fixture.csv']
    _write_csv(grid,destination/paths[0]);_write_csv(country,destination/paths[1])
    paired.sort_values(KEY+['model']).to_csv(destination/paths[2],index=False,float_format='%.17g',
                                            compression={'method':'gzip','mtime':0})
    # Whole selected coarse cells retain independently checkable fractional native sums.
    src = source.copy()
    src['cell_id'] = [f'g025_r{int(r):03d}_c{int(c):04d}' for r,c in zip(src.row,src.col)]
    border = country.groupby('cell_id').size().loc[lambda s:s>1].index
    partial = src.loc[src.europe_fraction.lt(1),'cell_id'].unique()
    zero = grid.loc[grid.production_total_tonnes.eq(0),'cell_id'].tolist()
    ids = sorted(set(grid.cell_id.iloc[:2])|set(border[:3])|set(partial[:3])|set(zero[:2]))
    fixture = src[src.cell_id.isin(ids)][['cell_id','source_row','source_col','row','col','ADM0_NAME','FIPS0',
        'europe_fraction','mask_rule',*PRODUCTION,*AREAS,*[c+'_source_raster_valid' for c in PRODUCTION]]]
    _write_csv(fixture,destination/paths[3])
    bundle = dict(schema_version=1,rows={paths[0]:len(grid),paths[1]:len(country),paths[2]:len(paired),paths[3]:len(fixture)},
                  sha256={name:digest(destination/name) for name in paths},
                  paired_metric=METRIC,verification='Native values and masks verified against provider-hashed TIFFs at original build; bundle reproduces aggregates without TIFFs.',
                  fixture_scope='Small complete-cell subset for fractional source reconstruction; not a continental production estimate.')
    (destination/'bundle_receipt.json').write_text(json.dumps(bundle,indent=2)+'\n')


def load_bundle(source_dir=HERE):
    source_dir = Path(source_dir)
    receipt = json.loads((source_dir/'receipt.json').read_text())
    bundle_path = source_dir/'bundle_receipt.json'
    require(digest(bundle_path)==receipt['output_sha256']['bundle_receipt.json'],'Bundle receipt checksum mismatch')
    bundle = json.loads(bundle_path.read_text())
    for name,expected in bundle['sha256'].items():
        require(digest(source_dir/name)==expected,f'Bundle checksum mismatch: {name}')
    grid = pd.read_csv(source_dir/'production_grid_input.csv',float_precision='round_trip')
    country = pd.read_csv(source_dir/'country_fraction_weights.csv',float_precision='round_trip')
    paired = pd.read_csv(source_dir/'paired_changes_input.csv.gz',float_precision='round_trip')
    unique(grid,['cell_id'],'bundled grid');unique(country,['cell_id','country_fips'],'bundled countries')
    for column in PRODUCTION+AREAS:
        c = country.groupby('cell_id')[column].sum().reindex(grid.cell_id)
        _same(c,grid[column],f'Bundled country-to-grid conservation: {column}')
    europe = grid[['cell_id',*PRODUCTION,*AREAS,'calendar_valid']].assign(domain_type='Europe',domain='Europe')
    regions = grid[['cell_id',*PRODUCTION,*AREAS,'calendar_valid','environment_region']].rename(columns={'environment_region':'domain'}).assign(domain_type='environment_region')
    countries = country.rename(columns={'country':'domain','country_fips':'FIPS0'}).merge(grid[['cell_id','calendar_valid']],on='cell_id',validate='many_to_one').assign(domain_type='country')
    weights = pd.concat([europe,regions,countries],ignore_index=True)
    ensemble = classify_cells(paired).rename(columns={'ensemble_mean_change':'mean_change',
        'ensemble_reference_on_common':'reference_on_common','ensemble_future_on_common':'future_on_common'})
    validate_changes(paired,ensemble,grid)
    return paired,weights,bundle


def run(destination=HERE,bundled=False,source_dir=HERE):
    destination = Path(destination).resolve()
    if destination != HERE.resolve():
        require(not destination.exists() or not any(destination.iterdir()),'Reproduction destination must be empty')
    destination.mkdir(parents=True,exist_ok=True)
    if bundled:
        require(destination!=Path(source_dir).resolve(),'Bundled reproduction requires a separate output subdirectory')
        paired,weights,bundle = load_bundle(source_dir)
        tables = build_tables(paired,weights)
        provenance = dict(mode='verified_compact_bundle',bundle=bundle,
                          source_receipt_sha256=digest(Path(source_dir)/'receipt.json'))
    else:
        inputs,provenance = load_verified_inputs()
        cells,weights,audit = reconcile_production(inputs['registry'],inputs['production'],inputs['source'],inputs['country'])
        validate_changes(inputs['paired'],inputs['ensemble'],cells)
        paired = inputs['paired']
        tables = build_tables(paired,weights)
        tables['mass_conservation'] = audit
        write_bundle(cells,weights,paired,inputs['source'],destination)
        provenance['technology_total_minus_components_tonnes'] = float((cells.production_total_tonnes-
            cells.production_irrigated_tonnes-cells.production_rainfed_tonnes).sum())
        provenance['technology_components_note'] = 'Provider component rasters have small rounding residuals; total-production raster remains the baseline.'
        provenance['source_country_border_cells'] = int(inputs['country'].groupby('cell_id').size().gt(1).sum())
    files = {}
    for name,frame in tables.items():
        if name=='cell_classifications':
            path = destination/(name+'.parquet')
            frame.to_parquet(path,index=False,compression='zstd')
        else:
            path = destination/(name+'.csv')
            _write_csv(frame,path)
        files[name] = path.name
    extra = ['bundle_receipt.json','production_grid_input.csv','country_fraction_weights.csv',
             'paired_changes_input.csv.gz','native_source_fixture.csv'] if not bundled else []
    receipt = dict(status='passed',schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,metric=METRIC,
        canopy_unit='normalized top-three-leaf healthy-area duration lost, in normalized canopy-area days',
        production_unit='metric tonnes of fixed SPAM2020 baseline production',
        climate_reference_period='1991-2020',future_periods=['2031-2060','2071-2100'],
        interpretation='Production exposed to simulated canopy-damage changes; not tonnes of yield lost or saved.',
        paired_year_rule='Within each GCM, weight cell paired-mean change by P_i*n_i/30. Average the three resulting domain means equally; min/max are the three GCM estimates, not confidence limits.',
        classification_rule='Equal-three-GCM cell means; all three required. Exposed tonnes use static P_i without year weighting. Ensemble-positive and all3positive are overlapping but distinct measures.',
        sensitivity_rule='Require >=27/30 metric-valid pairs in every GCM within a cell; use that shared cell support in all three continuous estimates, retaining n_i/30 weights. Excluded cells remain unavailable in fixed-baseline shares.',
        baseline_denominator_rule='Every share uses the entire fixed baseline of its domain, including unavailable production; not a valid-cell-only denominator.',
        zero_rule='Strict >0/<0 signs without tolerance. Mixed_or_zero combines mixed signs, all-three-zero, and one-sided changes containing zero.',
        table1_scope='Late century; Europe plus eight named regions. Europe includes Unassigned and Outside EEA regions; regional rows alone do not sum to Europe.',
        inference_limits=['Frozen all-wheat 2020 production and spatial support; no forecast of future production or harvested area.',
            'Normalized canopy HAD does not identify actual yield losses, adaptation efficacy, food prices, trade, nutrition or welfare.',
            'Primary continuous model means use model-specific valid-year support, while classifications require all three models.',
            'Countries use fractional native source membership; environmental regions inherit the current coarse-cell assignment.',
            'Sensitivity pair counts cannot prove the same 27 calendar-pair identities across models; it is a count-based coverage filter.',
            'GCM range excludes structural, production-map, disease-parameter and management uncertainty.'],
        provenance=provenance,tables=files,table_rows={k:len(v) for k,v in tables.items()},
        output_sha256={name:digest(destination/name) for name in [*files.values(),*extra]},
        code_sha256={p.name:digest(p) for p in sorted(HERE.glob('*.py'))})
    (destination/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundled',action='store_true')
    parser.add_argument('--destination',type=Path,default=HERE)
    parser.add_argument('--source-dir',type=Path,default=HERE)
    args = parser.parse_args()
    result = run(args.destination,args.bundled,args.source_dir)
    print(json.dumps({'status':result['status'],'rows':result['table_rows']}))
