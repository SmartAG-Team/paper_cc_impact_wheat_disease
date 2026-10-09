"""Behavioral tests with hand-calculated fixtures; source integration is also checked."""
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nordic_archive_analysis_under_test',HERE/'run_analysis.py')
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def records():
    # A single real-shaped trial summary, not invented plot replication.
    return pd.DataFrame([
        dict(year=2012, country='SE', trial_id='2012SE01', crop='WW',
             treatment_code=t, treated=int(t != 'A'), yield_kg_ha=y,
             station_raw='Uppsala', region='SE_Uppsala', source_row=i,
             source_file='Yield.xlsx', reported_gain_kg_ha=g)
        for t, y, g, i in [('A', 8000., np.nan, 5), ('B', 9000., 1000., 6),
                            ('C', 10000., 2000., 7)]
    ])


def test_shared_control_not_duplicated_as_independent_yield():
    p, excluded = a.build_pairs(records())
    assert len(p) == 2 and excluded.empty
    assert p.control_id.nunique() == 1
    assert p.environment_group.nunique() == 1
    assert p.trial_weight.tolist() == [.5, .5]
    assert p.control_yield_t_ha.tolist() == [8., 8.]


def test_denominators_have_distinct_hand_calculated_meanings():
    p, _ = a.build_pairs(records())
    assert p.gain_t_ha.tolist() == [1., 2.]
    assert p.gain_pct_untreated.tolist() == [12.5, 25.]
    assert p.contrast_pct_treated.tolist() == pytest.approx([100 / 9, 20.])


def test_duplicate_treatment_key_excludes_whole_trial_without_averaging():
    d = pd.concat([records(), records().iloc[[1]]], ignore_index=True)
    d.loc[3, 'yield_kg_ha'] = 9999
    p, ex = a.build_pairs(d)
    assert p.empty
    assert set(ex.reason) == {'ambiguous_treatment_key'}
    assert len(ex) == 4


def test_no_control_is_not_reconstructed_from_reported_gain():
    p, ex = a.build_pairs(records().iloc[1:])
    assert p.empty and set(ex.reason) == {'missing_control'}


@pytest.mark.parametrize('bad', [0, -1, np.nan, np.inf])
def test_invalid_control_denominator_fails_closed(bad):
    d = records(); d.loc[0, 'yield_kg_ha'] = bad
    p, ex = a.build_pairs(d)
    assert p.empty and set(ex.reason) == {'invalid_control_yield'}


def test_missing_treatment_yield_does_not_discard_valid_other_arm():
    d = records(); d.loc[1, 'yield_kg_ha'] = np.nan
    p, ex = a.build_pairs(d)
    assert p.treatment_code.tolist() == ['C']
    assert p.trial_weight.tolist() == [1.]
    assert set(ex.reason) == {'invalid_treated_yield'}


def test_negative_gain_preserved_and_source_error_not_used_as_response():
    d = records(); d.loc[1, 'yield_kg_ha'] = 7000
    p, _ = a.build_pairs(d)
    assert p.iloc[0].gain_t_ha == -1
    assert p.iloc[0].gain_inconsistent
    assert p.iloc[0].gain_audit_difference_kg_ha == -2000


def test_mismatched_crop_never_pairs():
    d = records(); d.loc[1, 'crop'] = 'SW'
    p, ex = a.build_pairs(d)
    assert p.empty and set(ex.reason) == {'inconsistent_trial_metadata'}


def test_post_2016_records_preserved_only_in_exclusion_audit():
    d = records(); d['year'] = 2017; d['trial_id'] = '2017SE01'
    p, ex = a.build_pairs(d)
    assert p.empty and set(ex.reason) == {'outside_2012_2016'}


def test_year_country_trial_all_part_of_pair_identity():
    d = records(); e = records(); e['year'] = 2013
    e['yield_kg_ha'] += 100
    p, _ = a.build_pairs(pd.concat([d, e], ignore_index=True))
    assert len(p) == 4
    assert p.control_id.nunique() == 2
    assert p.pair_id.nunique() == 4


def test_ambiguous_sas_join_cannot_expand_primary_rows():
    y = records(); sas = pd.concat([y, y.iloc[[1]]], ignore_index=True)
    joined = a.audit_sas_join(y, sas)
    assert len(joined) == len(y)
    assert joined.loc[joined.treatment_code.eq('B'), 'join_status'].iloc[0] == 'ambiguous_sas_key'
    assert pd.isna(joined.loc[joined.treatment_code.eq('B'), 'sas_yield_kg_ha'].iloc[0])


def test_unmatched_sas_row_is_not_guessed_from_similar_trial_name():
    y = records(); sas = records(); sas['trial_id'] = '2012SE02'
    joined = a.audit_sas_join(y, sas)
    assert set(joined.join_status) == {'no_sas_match'}
    assert joined.sas_yield_kg_ha.isna().all()


def test_sas_rounding_is_distinguished_from_disagreement():
    y = records(); sas = records(); sas.loc[0, 'yield_kg_ha'] += .5
    sas.loc[1, 'yield_kg_ha'] += 5
    joined = a.audit_sas_join(y, sas)
    assert joined.iloc[0].numeric_agreement
    assert not joined.iloc[1].numeric_agreement


def train_fixture():
    return pd.DataFrame(dict(
        gain_t_ha=[1., 3., 10.], trial_weight=[.5, .5, 1.],
        crop=['WW', 'WW', 'SW'], treatment_code=['B', 'C', 'B']))


def test_mean_training_weights_do_not_overweight_shared_controls():
    pred, levels = a.benchmark_predict(train_fixture(), train_fixture(), 'training_mean')
    assert pred.tolist() == [6., 6., 6.]


def test_management_and_unseen_crop_fallback_use_training_only():
    test = pd.DataFrame(dict(crop=['SW', 'XX'], treatment_code=['C', 'B'], gain_t_ha=[999., 999.]))
    pred, levels = a.benchmark_predict(train_fixture(), test, 'crop_management_mean')
    assert pred.tolist() == [3., 7.]
    assert levels.tolist() == ['management', 'management']
    test['gain_t_ha'] = -5000
    pred2, _ = a.benchmark_predict(train_fixture(), test, 'crop_management_mean')
    assert np.array_equal(pred, pred2)


def test_empty_training_fails_closed():
    with pytest.raises(ValueError):
        a.benchmark_predict(train_fixture().iloc[:0], train_fixture(), 'training_mean')


def split_fixture():
    rows = []
    for country in ['SE', 'DK']:
        for year in range(2012, 2017):
            for trial in [1, 2]:
                for arm in ['B', 'C']:
                    unit = f'{country}:{year}:{trial}'
                    rows.append(dict(country=country, year=year, trial_key=unit,
                                     control_id=unit + ':A', environment_group=f'{country}:{year}',
                                     pair_id=unit + ':' + arm))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('scheme', ['leave_country_year_out', 'leave_country_out', 'leave_year_out', 'forward_year'])
def test_all_environment_splits_keep_controls_and_trials_together(scheme):
    d = split_fixture(); seen = []
    for label, tr, te in a.make_splits(d, scheme):
        a.assert_no_leakage(d.loc[tr], d.loc[te], forward=scheme == 'forward_year')
        seen.extend(te)
        assert not set(d.loc[tr, 'control_id']) & set(d.loc[te, 'control_id'])
        assert not set(d.loc[tr, 'environment_group']) & set(d.loc[te, 'environment_group'])
        if scheme == 'forward_year':
            assert d.loc[tr, 'year'].max() < d.loc[te, 'year'].min()
    expected = d.index[d.year.ge(2014)] if scheme == 'forward_year' else d.index
    assert sorted(seen) == sorted(expected)


def test_explicit_shared_environment_leakage_is_rejected():
    d = split_fixture()
    with pytest.raises(ValueError):
        a.assert_no_leakage(d.iloc[[0]], d.iloc[[1]])


def test_forward_time_leakage_is_rejected():
    d = split_fixture()
    with pytest.raises(ValueError):
        a.assert_no_leakage(d[d.year.eq(2016)], d[d.year.eq(2015)], forward=True)


def test_recommendations_do_not_unlock_observed_disease_validation():
    gate = a.compatibility_gate(['year', 'yield_kg_ha', 'r85h14', 'day6', 'average_stage_doy'])
    assert not gate['observed_severity_yield_pairs']['available']
    assert not gate['leaf_stage_repeated_comparison']['available']
    assert not gate['functional_canopy']['available']


def test_source_bundle_verifies_and_tampering_fails_closed(tmp_path):
    bundle = HERE / 'source_bundle/nordic_public_sources.zip'
    manifest = HERE / 'source_bundle/manifest.json'
    data = a.verify_bundle(bundle, manifest)
    assert 'Yield.xlsx' in data
    corrupt = tmp_path / 'bad.zip'; corrupt.write_bytes(bundle.read_bytes() + b'x')
    with pytest.raises(ValueError):
        a.verify_bundle(corrupt, manifest)


def test_archived_primary_measurement_unit_is_unambiguous():
    data = a.verify_bundle(HERE / 'source_bundle/nordic_public_sources.zip', HERE / 'source_bundle/manifest.json')
    import io
    df = pd.read_excel(io.BytesIO(data['Yield.xlsx']), header=3)
    assert 'YieldKgHa15%' in df.columns
    assert 'Treatment_code**' in df.columns


def test_regional_date_cells_never_become_trial_observations():
    data = a.verify_bundle(HERE / 'source_bundle/nordic_public_sources.zip', HERE / 'source_bundle/manifest.json')
    ref = a.harmonize_stage_reference(data)
    first = ref.iloc[0]
    assert first.published_average_day_number == 134
    assert first.reference_date_cell == '2022-05-15'
    assert first.reference_date_day_of_year == 135
    assert first.day_number_discrepancy == 1
    assert not ref.actual_trial_observation.any()


def test_real_source_rows_reconcile_without_losing_untreated_records():
    data = a.verify_bundle(HERE / 'source_bundle/nordic_public_sources.zip', HERE / 'source_bundle/manifest.json')
    y, sas = a.read_sources(data)
    p, ex = a.build_pairs(y)
    used = set(p.control_source_row) | set(p.treated_source_row)
    assert not used & set(ex.source_row)
    assert used | set(ex.source_row) == set(y.source_row)
    # No generated recommendations enter the observed yield table or paired predictors.
    assert 'r85h14' in sas and 'r85h14' not in p
    assert not y.duplicated(['source_file', 'source_row']).any()
    assert p.groupby('trial_key').trial_weight.sum().eq(1).all()


def test_source_group_id_reuse_cannot_collapse_different_years():
    data = a.verify_bundle(HERE / 'source_bundle/nordic_public_sources.zip', HERE / 'source_bundle/manifest.json')
    y, sas = a.read_sources(data)
    audit, reused = a.audit_source_groups(sas, y)
    assert not audit.empty
    assert audit[['source_file', 'country', 'year', 'Group']].duplicated().sum() == 0
    assert audit.trial_count.eq(1).all()
    assert not reused.empty
    assert reused.groupby('possible_control_signature').environment_group.nunique().eq(1).all()


def test_conflicting_source_group_to_trial_mapping_fails_closed():
    y = records(); s = y.copy(); s['Group'] = 1
    other = s.copy(); other['trial_id'] = '2012SE02'
    with pytest.raises(ValueError):
        a.audit_source_groups(pd.concat([s, other], ignore_index=True), y)


def test_group_metric_denominator_is_trial_weighted_not_raw_row_count():
    d = pd.DataFrame(dict(scheme=['example']*3, model=['training_mean']*3,
        pair_id=['a', 'b', 'c'], trial_key=['x', 'x', 'z'],
        environment_group=['SE:2012', 'SE:2012', 'DK:2012'],
        trial_weight=[.5, .5, 1.], error_t_ha=[1., 3., 5.]))
    score = a.score_predictions(d).iloc[0]
    assert score.rmse_t_ha == pytest.approx(np.sqrt(15.))
    assert score.mae_t_ha == 3.5


def test_publication_uncertainty_and_compatibility_are_hash_verified():
    receipt=json.loads((HERE/'outputs/analysis_receipt.json').read_text())
    for name in ['uncertainty.json','compatibility.json']:
        assert name in receipt['output_sha256']
        assert receipt['output_sha256'][name]==a.sha256((HERE/'outputs'/name).read_bytes())
