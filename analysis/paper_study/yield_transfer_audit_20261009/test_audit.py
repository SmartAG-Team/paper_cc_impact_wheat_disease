"""Scientific invariants; synthetic values below are test fixtures only."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def implementation():
    assert (HERE / 'audit.py').exists(), 'The reproducible audit is not implemented'
    import audit
    return audit


def test_repeated_assessments_count_one_harvest_and_conflicts_fail():
    a = implementation()
    d = pd.DataFrame({'yield_unit_id': ['p1', 'p1', 'p2'], 'yield_t_ha': [4., 4., 5.]})
    assert len(a.harvest_units(d)) == 2
    d.loc[1, 'yield_t_ha'] = 4.2
    with pytest.raises(ValueError, match='Conflicting'):
        a.harvest_units(d)


def test_four_subrows_require_complete_yield_and_unique_keys():
    a = implementation()
    d = pd.DataFrame({'row': [1]*8, 'column': [1]*4+[2]*4,
                      'sub_column': list(range(8)), 'GY': [100,100,200,200,300,300,400,np.nan]})
    q = a.raw_plot_yields(d).set_index('column')
    assert q.loc[1, 'yield_raw_t_ha'] == 1.5
    assert pd.isna(q.loc[2, 'yield_raw_t_ha'])
    with pytest.raises(ValueError, match='Duplicate'):
        a.raw_plot_yields(pd.concat([d, d.iloc[[0]]]))


def test_source_identity_uses_id_not_coincident_equal_yields():
    a = implementation()
    d = pd.DataFrame({'yield_unit_id': ['p1', 'p2'], 'yield_t_ha': [4., 4.]})
    assert len(a.harvest_units(d)) == 2


def test_group_gate_rejects_one_environment_and_sparse_environment():
    a = implementation()
    d = pd.DataFrame({'postcode': [1]*25, 'site_year': ['1_2020']*25})
    assert not a.prediction_gate(d)['eligible']
    d = pd.DataFrame({'postcode': np.repeat(range(5), 40),
                      'site_year': np.repeat([f'{s}_{y}' for s in range(5) for y in (2019,2020)],20)})
    assert a.prediction_gate(d)['eligible']
    assert not a.prediction_gate(d.iloc[1:])['eligible']


def test_future_contract_rejects_missing_tables():
    implementation()
    import validate_contract
    empty = validate_contract.validate_tables({})
    assert not empty['input_ready']
    assert not empty['mechanistic_validation_established']


def test_heldout_outcomes_do_not_affect_predictions():
    assert (HERE/'prediction.py').exists(), 'Bounded prediction not implemented'
    import prediction
    a = implementation()
    d = a.Bundle().csv('analysis_ready/swiss_stb_yield_724.csv')
    train, test = d[d.postcode!=1260].copy(), d[d.postcode==1260].copy()
    features = ['name_var','mono_mix','density','stb_score_ordinal']
    pred, _ = prediction.fit_predict(train,test,features)
    test['yield_t_ha'] += 1000
    test['yield_dtha'] *= -100
    test['prot'] = 999
    changed, _ = prediction.fit_predict(train,test,features)
    np.testing.assert_array_equal(pred,changed)
    with pytest.raises(ValueError,match='Forbidden'):
        prediction.fit_predict(train,test,['yield_dtha'])


def test_unseen_ordinal_category_is_reported_and_not_converted():
    assert (HERE/'prediction.py').exists(), 'Bounded prediction not implemented'
    import prediction
    train = pd.DataFrame({'site_year':['a','a','b','b'],'yield_t_ha':[2.,3.,4.,5.],
                          'stb_score_ordinal':[0.,2.,0.,2.]})
    test = pd.DataFrame({'stb_score_ordinal':[9.]})
    predicted,unknown = prediction.fit_predict(train,test,['stb_score_ordinal'])
    assert np.isfinite(predicted).all()
    assert unknown['stb_score_ordinal'] == 1


def test_generated_evidence_preserves_grain_scope_and_split_isolation():
    implementation()
    import json
    receipt = json.loads((HERE / 'audit_receipt.json').read_text())
    assert receipt['checks_failed'] == 0
    assert receipt['new_unique_plot_yields'] == 948
    assert receipt['french_trait_linked_pairs'] == 166
    assert receipt['model_fits_in_readiness_audit'] == 0
    # audit.py itself never fits; the separate supplementary comparison does.
    pred = pd.read_csv(HERE/'heldout_predictions.csv')
    assert not pred.duplicated(['split','model','record_id']).any()
    assert pred.groupby(['split','model']).size().eq(724).all()
    folds = pd.read_csv(HERE/'fold_diagnostics.csv')
    assert folds.overlapping_split_groups.eq(0).all()
    assert folds.overlapping_plot_ids.eq(0).all()
    assert folds[folds.split.eq('leave_one_site_out')].overlapping_sites.eq(0).all()
    gate = pd.read_csv(HERE / 'evidence_gate.csv').set_index('quantity')
    assert not gate.loc[['HAD','tonnes','adaptation'], 'claim_ready'].any()


def contract_fixture():
    """Small artificial schema fixture, never used as scientific observations."""
    p = pd.DataFrame([dict(plot_id=f'p{i}',environment_id=f'e{i}',site_id=f's{i}',harvest_year=2020,
        replicate=1,cultivar='test_fixture',treatment_id='fixture',split='calibration' if i==0 else 'validation',
        sowing_date='2020-06-01',anthesis_date='2020-06-02',maturity_date='2020-06-04',harvest_date='2020-06-05',
        yield_t_ha=5.,grain_moisture_percent=15.,soil_profile_ref='fixture',cultivar_parameter_ref='fixture',
        source_ref='test_only',source_plot_id=f'p{i}',measurement_origin='observed') for i in range(3)])
    canopy = pd.DataFrame([dict(plot_id=plot,date=date,leaf_rank=rank,reference_lai=1.,green_lai=.8,
        functional_loss_fraction=.2,stb_severity_percent=10.,lai_unit='m2/m2',severity_unit='percent_leaf_area',
        reference_method='test_fixture',functional_method='test_fixture',measurement_origin='observed',source_ref='test_only')
        for plot in p.plot_id for rank in [1,2,3] for date in ['2020-06-02','2020-06-03','2020-06-04']])
    weather = pd.DataFrame([dict(environment_id=env,date=date,tmin_c=10.,tmax_c=20.,precip_mm=1.,
        solar_mj_m2=15.,relative_humidity_percent=80.,source_ref='test_only')
        for env in p.environment_id for date in ['2020-06-01','2020-06-02','2020-06-03','2020-06-04']])
    management = pd.DataFrame([dict(plot_id=plot,date='2020-06-01',action='sowing',amount=350.,unit='seeds/m2',source_ref='test_only')
                              for plot in p.plot_id])
    return dict(plots=p,canopy=canopy,weather=weather,management=management)


def test_valid_structural_fixture_does_not_certify_physiology():
    import validate_contract
    result = validate_contract.validate_tables(contract_fixture())
    assert result['errors'] == []
    assert result['input_ready']
    assert not result['mechanistic_validation_established']


@pytest.mark.parametrize('failure',['single_snapshot','ordinal','nonfinite','source_alias','weather_gap','split_leakage','environment_alias'])
def test_contract_rejects_scientifically_material_failures(failure):
    import validate_contract
    tables = contract_fixture()
    if failure=='single_snapshot':
        tables['canopy'] = tables['canopy'][tables['canopy'].date.eq('2020-06-02')]
    elif failure=='ordinal':
        tables['canopy']['severity_unit'] = 'ordinal_0_9'
    elif failure=='nonfinite':
        tables['canopy'].loc[0,'functional_loss_fraction'] = np.inf
    elif failure=='source_alias':
        tables['plots'].loc[1,'source_plot_id'] = 'p0'
    elif failure=='weather_gap':
        tables['weather'] = tables['weather'].iloc[1:]
    elif failure=='split_leakage':
        tables['plots'].loc[1,'environment_id'] = 'e0'
    elif failure=='environment_alias':
        tables['plots'].loc[1,'site_id'] = 's0'
    assert not validate_contract.validate_tables(tables)['input_ready']
