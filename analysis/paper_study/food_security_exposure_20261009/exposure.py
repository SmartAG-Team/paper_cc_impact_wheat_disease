"""Fixed-reference production exposure; no conversion of canopy damage to yield."""
import numpy as np
import pandas as pd

METRIC = 'GS65_85_lost_had3'
MODELS = ('ACCESS-CM2', 'MPI-ESM1-2-HR', 'MRI-ESM2-0')
SCENARIOS = ('ssp126', 'ssp245', 'ssp585')
PERIODS = ('2031-2060', '2071-2100')
KEY = ['cell_id', 'scenario', 'period']
GROUP = ['domain_type', 'domain', 'scenario', 'period']
PRODUCTION = ['production_total_tonnes', 'production_irrigated_tonnes', 'production_rainfed_tonnes']
AREAS = ['harvested_total_ha', 'physical_total_ha']
AGREEMENT = ['all3positive', 'all3negative', 'mixed_sign', 'all3zero',
             'nonnegative_with_zero', 'nonpositive_with_zero', 'unavailable']
DIRECTION = ['increasing', 'decreasing', 'unchanged', 'unavailable']


def read(table=None, verify=True, directory=None):
    """Integration entry point; also available from the package itself."""
    from analysis.paper_study.food_security_exposure_20261009 import read as package_read
    kwargs = {} if directory is None else {'directory': directory}
    return package_read(table=table, verify=verify, **kwargs)


def source_paths(directory=None):
    """Return receipt-verified CSV source paths for source-workbook integration."""
    from analysis.paper_study.food_security_exposure_20261009 import source_paths as package_paths
    return package_paths() if directory is None else package_paths(directory=directory)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def unique(frame, keys, label):
    require(not frame[keys].isna().any().any(), f'{label}: null keys')
    require(not frame.duplicated(keys).any(), f'{label}: duplicate keys')


def finite_nonnegative(frame, columns, label):
    x = frame[columns].to_numpy(float)
    require(np.isfinite(x).all() and (x >= 0).all(), f'{label}: invalid or negative values')


def _same(a, b, label, atol=1e-6):
    require(np.allclose(a, b, rtol=1e-11, atol=atol, equal_nan=True), label)


def reconcile_production(registry, production, source, country):
    """Reconstruct fractional source-country and grid totals before any climate join."""
    for frame, key, name in [(registry, ['cell_id'], 'registry'),
                              (production, ['cell_id'], 'production'),
                              (source, ['source_row', 'source_col'], 'native source'),
                              (country, ['cell_id', 'FIPS0'], 'country')]:
        unique(frame, key, name)
    for frame in [registry, production, country]:
        expected = [f'g025_r{int(r):03d}_c{int(c):04d}' for r, c in zip(frame.row, frame.col)]
        require(frame.cell_id.tolist() == expected, 'cell_id disagrees with grid row/column')
    require(set(registry.cell_id) == set(production.cell_id) == set(country.cell_id), 'Current grid IDs differ')
    require(production.production_reference_year.eq(2020).all(), 'Expected SPAM2020 reference')
    for frame, cols, label in [(production, PRODUCTION+AREAS, 'production'),
                               (source, PRODUCTION+AREAS, 'source'),
                               (country, PRODUCTION, 'country'), (registry, AREAS, 'registry')]:
        finite_nonnegative(frame, cols, label)
    require(source.europe_fraction.between(0, 1).all(), 'Invalid Europe fraction')
    require((source.source_row // 3).eq(source.row).all() and
            (source.source_col // 3).eq(source.col).all(), 'Wrong native-to-current-grid mapping')
    p = production.set_index('cell_id').loc[registry.cell_id]
    for col in ['row', 'col', 'latitude', 'longitude', *AREAS]:
        _same(registry[col], p[col], f'Registry mismatch: {col}', atol=1e-8)
    _same(registry.latitude, 90-(registry.row+.5)*.25, 'Latitude inconsistent with grid', atol=1e-9)
    _same(registry.longitude, -180+(registry.col+.5)*.25, 'Longitude inconsistent with grid', atol=1e-9)
    require(registry.environment_region.notna().all(), 'Missing environmental region')
    require(registry.calendar_valid.isin([True, False]).all(), 'Invalid calendar eligibility')
    native = source.copy()
    native['cell_id'] = [f'g025_r{int(r):03d}_c{int(c):04d}' for r,c in zip(native.row,native.col)]
    for col in PRODUCTION+AREAS:
        native[col] = native[col]*native.europe_fraction
    native_country = native.groupby(['cell_id','ADM0_NAME','FIPS0'], sort=True)[PRODUCTION+AREAS].sum()
    country_keys = ['cell_id','ADM0_NAME','FIPS0']
    original_country = country.set_index(country_keys).sort_index()
    require(native_country.index.equals(original_country.index), 'Native country keys disagree')
    native_grid = native.groupby('cell_id')[PRODUCTION+AREAS].sum().reindex(p.index)
    audits = []
    for col in PRODUCTION+AREAS:
        _same(native_grid[col],p[col],f'Native-to-grid conservation failed: {col}')
        if col in PRODUCTION:
            _same(native_country[col],original_country[col],f'Country conservation failed: {col}')
        audits.append(dict(check=f'native_fraction_to_grid_{col}', passed=True,
                           source_total=float(native[col].sum()), grid_total=float(p[col].sum()),
                           max_abs_cell_residual=float(np.abs(native_grid[col]-p[col]).max())))
    cells = registry.merge(production[['cell_id',*PRODUCTION,'production_reference_year']],
                           on='cell_id', validate='one_to_one')
    europe = cells[['cell_id',*PRODUCTION,*AREAS,'calendar_valid']].assign(domain_type='Europe',domain='Europe')
    regions = cells[['cell_id',*PRODUCTION,*AREAS,'calendar_valid','environment_region']].rename(
        columns={'environment_region':'domain'}).assign(domain_type='environment_region')
    countries = native_country.reset_index().merge(cells[['cell_id','calendar_valid']],on='cell_id',validate='many_to_one')
    countries = countries.rename(columns={'ADM0_NAME':'domain'}).assign(domain_type='country')
    weights = pd.concat([europe,regions,countries],ignore_index=True)
    unique(weights,['domain_type','domain','cell_id'],'domain weights')
    return cells, weights, pd.DataFrame(audits)


def classify_cells(paired, min_pairs=1):
    """Equal-three-model cell means and a disjoint sign-agreement partition."""
    require(isinstance(min_pairs,int) and 1 <= min_pairs <= 30, 'min_pairs must be 1..30')
    unique(paired,KEY+['model'],'paired changes')
    x = paired.copy()
    x['eligible'] = np.isfinite(x.change) & x.valid_year_pairs.ge(min_pairs)
    x['positive'] = x.eligible & x.change.gt(0)
    x['negative'] = x.eligible & x.change.lt(0)
    out = x.groupby(KEY,sort=True).agg(
        ensemble_mean_change=('change','mean'),gcm_min=('change','min'),gcm_max=('change','max'),
        ensemble_reference_on_common=('reference_on_common','mean'),
        ensemble_future_on_common=('future_on_common','mean'),
        available_gcms=('eligible','sum'),positive_gcm_count=('positive','sum'),negative_gcm_count=('negative','sum'),
        minimum_valid_year_pairs=('valid_year_pairs','min'),maximum_valid_year_pairs=('valid_year_pairs','max')).reset_index()
    complete = out.available_gcms.eq(3)
    for col in ['ensemble_mean_change','gcm_min','gcm_max','ensemble_reference_on_common','ensemble_future_on_common']:
        out.loc[~complete,col] = np.nan
    pos, neg = out.positive_gcm_count, out.negative_gcm_count
    out['ensemble_direction'] = np.select([~complete,out.ensemble_mean_change.gt(0),out.ensemble_mean_change.lt(0)],
                                          ['unavailable','increasing','decreasing'],default='unchanged')
    out['gcm_agreement'] = np.select([~complete,pos.eq(3),neg.eq(3),pos.gt(0)&neg.gt(0),pos.eq(0)&neg.eq(0),pos.gt(0)],
                                     ['unavailable','all3positive','all3negative','mixed_sign','all3zero','nonnegative_with_zero'],
                                     default='nonpositive_with_zero')
    out['metric'] = METRIC
    return out


def validate_changes(paired, ensemble, cells, scenarios=SCENARIOS, periods=PERIODS):
    unique(paired,KEY+['model'],'paired changes')
    unique(ensemble,KEY,'ensemble changes')
    require(paired.metric.eq(METRIC).all() and ensemble.metric.eq(METRIC).all(),'Unexpected metric')
    require(set(paired.model) == set(MODELS),'Unexpected/missing GCM')
    for col, levels in [('scenario',scenarios),('period',periods)]:
        require(set(paired[col]) == set(levels),f'Unexpected {col}')
    expected = pd.MultiIndex.from_product([cells.cell_id,scenarios,periods,MODELS],names=KEY+['model'])
    actual = pd.MultiIndex.from_frame(paired[KEY+['model']])
    require(len(actual)==len(expected) and actual.difference(expected).empty,'Incomplete or stale paired grid')
    counts = paired.valid_year_pairs.to_numpy(float)
    require(np.isfinite(counts).all() and ((counts>=0)&(counts<=30)&(counts==np.floor(counts))).all(),
            'Invalid paired-year counts')
    for col in ['change','reference_on_common','future_on_common']:
        v = paired[col].to_numpy(float)
        require(not np.isinf(v).any() and np.array_equal(np.isfinite(v),counts>0),
                f'{col}: finite values must correspond exactly to positive pair counts')
    _same(paired.future_on_common-paired.reference_on_common,paired.change,'Future-reference disagrees with change',atol=1e-10)
    got = classify_cells(paired).set_index(KEY).sort_index()
    supplied = ensemble.set_index(KEY).sort_index()
    require(got.index.equals(supplied.index),'Ensemble grid differs')
    for col, supplied_col in [('ensemble_mean_change','mean_change'),('ensemble_reference_on_common','reference_on_common'),
                               ('ensemble_future_on_common','future_on_common'),('gcm_min','gcm_min'),('gcm_max','gcm_max'),
                               *[(c,c) for c in ['available_gcms','positive_gcm_count','negative_gcm_count',
                                                'minimum_valid_year_pairs','maximum_valid_year_pairs']]]:
        _same(got[col],supplied[supplied_col],f'Ensemble mismatch: {supplied_col}',atol=1e-10)
    return got.reset_index()


def reference_summary(weights):
    x = weights.copy()
    x['calendar_eligible_production_tonnes'] = x.production_total_tonnes*x.calendar_valid
    x['calendar_eligible_harvested_area_ha'] = x.harvested_total_ha*x.calendar_valid
    out = x.groupby(['domain_type','domain'],sort=True).agg(
        baseline_production_tonnes=('production_total_tonnes','sum'),harvested_area_ha=('harvested_total_ha','sum'),
        physical_area_ha=('physical_total_ha','sum'),grid_cells=('cell_id','nunique'),
        calendar_eligible_cells=('calendar_valid','sum'),calendar_eligible_production_tonnes=('calendar_eligible_production_tonnes','sum'),
        calendar_eligible_harvested_area_ha=('calendar_eligible_harvested_area_ha','sum')).reset_index()
    out['production_reference_year'] = 2020
    return out


def aggregate_domains(paired, classified, weights, min_pairs=1):
    """Return summary, GCM means, and disjoint exposure categories for one threshold.

    Primary: each GCM uses its own finite metric years. Sensitivity: retain only
    cells meeting the threshold in all three models, then retain n/30 weighting.
    """
    mode = 'primary_any_paired_year' if min_pairs==1 else f'robust_all3_ge{min_pairs}'
    finite_nonnegative(weights,['production_total_tonnes',*AREAS],'weights')
    unique(weights,['domain_type','domain','cell_id'],'weights')
    x = weights.merge(paired,on='cell_id',validate='many_to_many')
    x = x.merge(classified[KEY+['available_gcms']],on=KEY,validate='many_to_one')
    eligible = np.isfinite(x.change) & x.valid_year_pairs.ge(min_pairs)
    if min_pairs>1:
        eligible &= x.available_gcms.eq(3)
    x['effective_production_tonnes'] = np.where(eligible,x.production_total_tonnes*x.valid_year_pairs/30,0.)
    x['eligible_production_tonnes'] = np.where(eligible,x.production_total_tonnes,0.)
    x['eligible_cell'] = eligible.astype(int)
    for col in ['change','reference_on_common','future_on_common']:
        x['numerator_'+col] = x.effective_production_tonnes*x[col].fillna(0)
    detail = x.groupby(GROUP+['model'],sort=True)[['effective_production_tonnes','eligible_production_tonnes',
                                                   'eligible_cell','numerator_change','numerator_reference_on_common',
                                                   'numerator_future_on_common']].sum().reset_index()
    den = detail.effective_production_tonnes.replace(0,np.nan)
    for col in ['change','reference_on_common','future_on_common']:
        detail['production_weighted_'+col] = detail.pop('numerator_'+col)/den
    means = detail.groupby(GROUP,sort=True).agg(
        production_weighted_change=('production_weighted_change','mean'),gcm_min=('production_weighted_change','min'),
        gcm_max=('production_weighted_change','max'),aggregate_available_gcms=('production_weighted_change','count'),
        production_weighted_reference=('production_weighted_reference_on_common','mean'),
        production_weighted_future=('production_weighted_future_on_common','mean'),
        min_effective_production_tonnes=('effective_production_tonnes','min'),
        max_effective_production_tonnes=('effective_production_tonnes','max')).reset_index()
    means.loc[means.aggregate_available_gcms.ne(3),['production_weighted_change','gcm_min','gcm_max',
                                                  'production_weighted_reference','production_weighted_future']] = np.nan
    out = means.merge(reference_summary(weights),on=['domain_type','domain'],validate='many_to_one')
    y = weights.merge(classified,on='cell_id',validate='many_to_many')
    long = []
    for axis, levels in [('ensemble_direction',DIRECTION),('gcm_agreement',AGREEMENT)]:
        for category in levels:
            subset = y[y[axis].eq(category)]
            vals = subset.groupby(GROUP).agg(production_tonnes=('production_total_tonnes','sum'),cells=('cell_id','nunique'))
            base = out[GROUP+['baseline_production_tonnes']].merge(vals,on=GROUP,how='left').fillna({'production_tonnes':0.,'cells':0})
            base['classification_axis'],base['category'] = axis,category
            base['share_baseline_pct'] = 100*base.production_tonnes/base.baseline_production_tonnes.replace(0,np.nan)
            prefix = ('ensemble_'+category) if axis=='ensemble_direction' else category
            out[prefix+'_production_tonnes'] = base.production_tonnes.to_numpy()
            out[prefix+'_share_baseline_pct'] = base.share_baseline_pct.to_numpy()
            long.append(base)
    out['nonunanimous_production_tonnes'] = out[[c+'_production_tonnes' for c in
        ['mixed_sign','all3zero','nonnegative_with_zero','nonpositive_with_zero']]].sum(axis=1)
    out['nonunanimous_share_baseline_pct'] = 100*out.nonunanimous_production_tonnes/out.baseline_production_tonnes.replace(0,np.nan)
    out['complete3_production_tonnes'] = out.baseline_production_tonnes-out.unavailable_production_tonnes
    out['minimum_valid_production_time_fraction'] = out.min_effective_production_tonnes/out.baseline_production_tonnes.replace(0,np.nan)
    out['maximum_valid_production_time_fraction'] = out.max_effective_production_tonnes/out.baseline_production_tonnes.replace(0,np.nan)
    for table in [out,detail]:
        table['analysis_mode'],table['metric'] = mode,METRIC
    categories = pd.concat(long,ignore_index=True)
    categories['analysis_mode'],categories['metric'] = mode,METRIC
    return out,detail,categories
