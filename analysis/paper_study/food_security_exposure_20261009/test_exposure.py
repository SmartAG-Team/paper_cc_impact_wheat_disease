"""Small numerical counterexamples to common exposure-analysis errors."""
import numpy as np
import pandas as pd
import pytest

from analysis.paper_study.food_security_exposure_20261009.exposure import (
    MODELS, METRIC, classify_cells, aggregate_domains, reconcile_production,
    validate_changes,
)


def paired_fixture(values, counts=None):
    if counts is None:
        counts = [[30] * 3 for _ in values]
    rows = []
    for i, (changes, ns) in enumerate(zip(values, counts)):
        for model, change, n in zip(MODELS, changes, ns):
            rows.append(dict(cell_id=f'c{i}', model=model, scenario='ssp126',
                             period='2031-2060', metric=METRIC, change=change,
                             reference_on_common=10. if n else np.nan,
                             future_on_common=10. + change if n else np.nan,
                             valid_year_pairs=n))
    return pd.DataFrame(rows)


def weights_fixture(production):
    return pd.DataFrame(dict(cell_id=[f'c{i}' for i in range(len(production))],
                             domain_type='Europe', domain='Europe',
                             production_total_tonnes=production,
                             harvested_total_ha=np.ones(len(production)),
                             physical_total_ha=np.ones(len(production)),
                             calendar_valid=True))


def test_sign_agreement_is_not_ensemble_direction_and_zeros_are_explicit():
    p = paired_fixture([[3, 3, -1], [-2, -2, -2], [0, 0, 0],
                        [1, 0, 0], [-1, 0, 0], [1, np.nan, 1], [1, -1, 0]],
                       [[30]*3]*5 + [[30, 0, 30], [30]*3])
    c = classify_cells(p)
    assert c.ensemble_direction.tolist() == ['increasing', 'decreasing', 'unchanged',
                                             'increasing', 'decreasing', 'unavailable', 'unchanged']
    assert c.gcm_agreement.tolist() == ['mixed_sign', 'all3negative', 'all3zero',
                                       'nonnegative_with_zero', 'nonpositive_with_zero',
                                       'unavailable', 'mixed_sign']
    assert c.loc[0, 'ensemble_mean_change'] == pytest.approx(5/3)
    assert np.isnan(c.loc[5, 'ensemble_mean_change'])


def test_metric_year_weighting_and_equal_model_means_are_distinct():
    p = paired_fixture([[9, -3, 3], [0, 3, 0]], [[1, 30, 30], [30, 30, 30]])
    c = classify_cells(p)
    out, per_gcm, categories = aggregate_domains(p, c, weights_fixture([90., 10.]))
    r = out.iloc[0]
    expected_gcm = np.array([27/13, -2.4, 2.7])
    np.testing.assert_allclose(per_gcm.production_weighted_change, expected_gcm)
    assert r.production_weighted_change == pytest.approx(expected_gcm.mean())
    assert r.gcm_min == pytest.approx(-2.4)
    assert r.gcm_max == pytest.approx(2.7)
    assert r.ensemble_increasing_production_tonnes == 100
    assert r.all3positive_production_tonnes == 0
    assert r.mixed_sign_production_tonnes == 90
    assert r.production_weighted_change != pytest.approx(2.8)  # static cell-mean weight
    assert r.production_weighted_change != pytest.approx(57/213)  # pooling models
    assert categories.groupby('classification_axis').production_tonnes.sum().eq(100).all()


def test_strict_sensitivity_uses_common_support_and_total_baseline_denominator():
    p = paired_fixture([[1, 2, 3], [8, 8, 8], [-1, -1, -1]],
                       [[27, 30, 28], [26, 30, 30], [0, 0, 0]])
    p.loc[p.valid_year_pairs.eq(0), ['change', 'reference_on_common', 'future_on_common']] = np.nan
    c = classify_cells(p, min_pairs=27)
    out, detail, _ = aggregate_domains(p, c, weights_fixture([20., 30., 50.]), min_pairs=27)
    r = out.iloc[0]
    assert c.ensemble_direction.tolist() == ['increasing', 'unavailable', 'unavailable']
    assert r.all3positive_production_tonnes == 20
    assert r.all3positive_share_baseline_pct == 20
    assert r.unavailable_production_tonnes == 80
    assert r.production_weighted_change == 2
    assert detail.effective_production_tonnes.tolist() == pytest.approx([18, 20, 20*28/30])


@pytest.mark.parametrize('production', [[0., 0.], [10., 20.]])
def test_missing_metric_and_zero_production_never_become_zero_damage(production):
    p = paired_fixture([[np.nan]*3]*2, [[0]*3]*2)
    c = classify_cells(p)
    out, detail, _ = aggregate_domains(p, c, weights_fixture(production))
    assert np.isnan(out.iloc[0].production_weighted_change)
    assert detail.production_weighted_change.isna().all()
    assert out.iloc[0].unavailable_production_tonnes == sum(production)
    if sum(production) == 0:
        assert np.isnan(out.iloc[0].unavailable_share_baseline_pct)


def production_fixture():
    registry = pd.DataFrame(dict(cell_id=['g025_r100_c0800'], row=[100], col=[800],
                                latitude=[64.875], longitude=[20.125],
                                harvested_total_ha=[4.], physical_total_ha=[3.],
                                environment_region=['Boreal'], calendar_valid=[True]))
    prod = registry.drop(columns=['environment_region','calendar_valid']).copy()
    prod['production_total_tonnes'] = 10.
    prod['production_irrigated_tonnes'] = 0.
    prod['production_rainfed_tonnes'] = 10.
    prod['production_reference_year'] = 2020
    source = pd.DataFrame(dict(row=[100,100],col=[800,800],source_row=[300,301],
                               source_col=[2400,2400],ADM0_NAME=['A','B'],FIPS0=['AA','BB'],
                               europe_fraction=[.5,1.],production_total_tonnes=[4.,8.],
                               production_irrigated_tonnes=[0.,0.],production_rainfed_tonnes=[4.,8.],
                               harvested_total_ha=[2.,3.],physical_total_ha=[2.,2.]))
    country = pd.DataFrame(dict(row=[100,100],col=[800,800],
                                cell_id=['g025_r100_c0800']*2,ADM0_NAME=['A','B'],FIPS0=['AA','BB'],
                                production_total_tonnes=[2.,8.],production_irrigated_tonnes=[0.,0.],
                                production_rainfed_tonnes=[2.,8.]))
    return registry, prod, source, country


def test_fractional_country_mass_and_area_are_reconstructed_once():
    cells, weights, audit = reconcile_production(*production_fixture())
    countries = weights[weights.domain_type.eq('country')].set_index('domain')
    assert countries.production_total_tonnes.to_dict() == {'A':2., 'B':8.}
    assert countries.harvested_total_ha.to_dict() == {'A':1., 'B':3.}
    assert cells.production_total_tonnes.sum() == 10
    assert audit.passed.all()


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_id', 'country_mass', 'negative', 'nan',
                                       'bad_fraction', 'wrong_coordinates', 'wrong_source_grid'])
def test_bad_production_joins_and_invalid_values_fail(mutation):
    registry, prod, source, country = production_fixture()
    if mutation == 'duplicate': prod = pd.concat([prod,prod],ignore_index=True)
    if mutation == 'missing_id': prod.loc[0,'cell_id'] = 'stale_id'
    if mutation == 'country_mass': country.loc[0,'production_total_tonnes'] += 1
    if mutation == 'negative': prod.loc[0,'production_total_tonnes'] = -1
    if mutation == 'nan': prod.loc[0,'production_total_tonnes'] = np.nan
    if mutation == 'bad_fraction': source.loc[0,'europe_fraction'] = 1.1
    if mutation == 'wrong_coordinates': prod.loc[0,'latitude'] += .25
    if mutation == 'wrong_source_grid': source.loc[0,'source_row'] += 10
    with pytest.raises(ValueError): reconcile_production(registry,prod,source,country)


def ensemble_fixture(p):
    c = classify_cells(p)
    return c.rename(columns={'ensemble_mean_change':'mean_change',
                             'ensemble_reference_on_common':'reference_on_common',
                             'ensemble_future_on_common':'future_on_common'})


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_model', 'stale_ensemble', 'bad_count',
                                       'finite_zero_pairs', 'missing_value_with_pairs', 'difference'])
def test_current_change_contract_rejects_inconsistency(mutation):
    p = paired_fixture([[1.,2.,3.]])
    e = ensemble_fixture(p)
    ids = pd.DataFrame({'cell_id':['c0']})
    if mutation == 'duplicate': p = pd.concat([p,p.iloc[:1]],ignore_index=True)
    if mutation == 'missing_model': p = p.iloc[1:]
    if mutation == 'stale_ensemble': e.loc[0,'mean_change'] = 12.
    if mutation == 'bad_count': p.loc[0,'valid_year_pairs'] = 31
    if mutation == 'finite_zero_pairs': p.loc[0,'valid_year_pairs'] = 0
    if mutation == 'missing_value_with_pairs': p.loc[0,'change'] = np.nan
    if mutation == 'difference': p.loc[0,'future_on_common'] += 3
    with pytest.raises(ValueError): validate_changes(p,e,ids,scenarios=('ssp126',),periods=('2031-2060',))


def test_verified_integration_contract_and_stack_conservation():
    from analysis.paper_study.food_security_exposure_20261009 import read, source_paths
    tables = read()
    summary = tables['domain_summary']
    required = ['domain_type','domain','scenario','period','baseline_production_tonnes',
                'production_weighted_change','gcm_min','gcm_max','all3positive_production_tonnes',
                'all3positive_share_baseline_pct','ensemble_increasing_production_tonnes',
                'ensemble_increasing_share_baseline_pct','all3negative_production_tonnes',
                'mixed_sign_production_tonnes','unavailable_production_tonnes','nonunanimous_production_tonnes']
    assert set(required).issubset(summary.columns)
    assert len(summary) == 624
    assert len(tables['table1']) == 54
    assert len(source_paths()) >= 18
    total = summary[['all3positive_production_tonnes','all3negative_production_tonnes',
                     'nonunanimous_production_tonnes','unavailable_production_tonnes']].sum(axis=1)
    np.testing.assert_allclose(total,summary.baseline_production_tonnes,rtol=1e-12,atol=1e-6)
    assert (summary.all3positive_production_tonnes <= summary.ensemble_increasing_production_tonnes+1e-6).all()


def test_reader_rejects_hash_mismatch(monkeypatch):
    import analysis.paper_study.food_security_exposure_20261009 as api
    monkeypatch.setattr(api,'digest',lambda path:'deliberate-corruption')
    with pytest.raises(ValueError,match='checksum mismatch'):
        api.read('domain_summary')


def test_exposure_module_exports_integration_api():
    from analysis.paper_study.food_security_exposure_20261009 import exposure
    summary = exposure.read()['domain_summary']
    primary_regions = summary[(summary.analysis_mode=='primary_any_paired_year') &
                              (summary.domain_type=='environment_region')]
    assert len(primary_regions) == 60
    assert len(exposure.source_paths()) >= 18


def test_native_fixture_has_complete_cells_and_reproduces_country_fractions():
    from pathlib import Path
    here = Path(__file__).resolve().parent
    s = pd.read_csv(here/'native_source_fixture.csv',float_precision='round_trip')
    country = pd.read_csv(here/'country_fraction_weights.csv',float_precision='round_trip')
    grid = pd.read_csv(here/'production_grid_input.csv',float_precision='round_trip').set_index('cell_id')
    assert s.europe_fraction.lt(1).any()
    assert s.groupby('cell_id').ADM0_NAME.nunique().gt(1).any()
    for col in ['production_total_tonnes','harvested_total_ha','physical_total_ha']:
        x = s.assign(weighted=s[col]*s.europe_fraction).groupby('cell_id').weighted.sum()
        np.testing.assert_allclose(x,grid.loc[x.index,col],rtol=1e-12,atol=1e-6)
    present = country.positive_cell_production
    np.testing.assert_allclose(country.loc[present,'production_fraction_of_cell']*country.loc[present,'cell_production_tonnes'],
                               country.loc[present,'production_total_tonnes'],rtol=1e-12,atol=1e-6)
    assert country.loc[~present,'production_fraction_of_cell'].isna().all()
