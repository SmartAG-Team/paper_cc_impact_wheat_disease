"""Calibration must be invariant to all held-out outcomes and weather."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from process_model._support.station_split import load_station_split, assert_training_sites


def fixture():
    rows = []
    for site, gdd in [(1, 10), (2, 20), (3, 100), (4, 200)]:
        for day, code in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8]):
            rows.append(dict(PEP_ID=site, DATE=pd.Timestamp('2001-10-01') + pd.Timedelta(days=day),
                CODE=code, BBCH=[0, 0, 10, 10, 31, 31, 51, 51, 85][day],
                GDD=gdd, LAT=50, t_mean=5, t_max=10,
                SOWING_DATE='2001-10-01', SOWING_KNOWN_AT='2001-10-01'))
    return pd.DataFrame(rows)


def module():
    path = ROOT / 'process_model' / 'calibrate.py'
    assert path.exists(), 'Executable training-only calibration pipeline is missing'
    spec = importlib.util.spec_from_file_location('calibrate', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_calibration_ignores_heldout_outcomes_and_covariates(tmp_path):
    m = module()
    path = tmp_path / 'station_split.json'
    path.write_text(json.dumps({'train': [1, 2], 'val': [3], 'test': [4]}))
    split = load_station_split(path)
    raw = fixture()
    original, train_features = m.fit_calibration(raw, split)
    changed = raw.copy()
    heldout = changed.PEP_ID.isin([3, 4])
    changed.loc[heldout, ['CODE', 'BBCH', 'GDD', 't_mean']] = [8, 85, 99999, 99]
    perturbed, other_features = m.fit_calibration(changed, split)
    assert original == perturbed
    pd.testing.assert_frame_equal(train_features, other_features)
    assert original['photoperiod_onset_gdd'] == 45.0
    assert original['thresholds'][0]['Cumulative_GDD'] == 15.0
    assert original['thresholds'][-1]['Cumulative_GDD'] == 135.0
    assert original['fit_station_ids'] == [1, 2]


def test_recomputation_discards_archived_derived_columns(tmp_path):
    m = module()
    path = tmp_path / 'station_split.json'
    path.write_text(json.dumps({'train': [1, 2], 'val': [3], 'test': [4]}))
    raw = fixture()
    clean, _ = m.fit_calibration(raw, load_station_split(path))
    raw['Cumulative_GDD'] = -999
    raw['Cumulative_t_pp_GDD'] = 99999
    raw['Cumulative_t_pp_v_GDD'] = np.nan
    raw['daylength'] = 0
    other, _ = m.fit_calibration(raw, load_station_split(path))
    assert clean == other


def test_missing_training_stage_fails(tmp_path):
    m = module()
    path = tmp_path / 'station_split.json'
    path.write_text(json.dumps({'train': [1, 2], 'val': [3], 'test': [4]}))
    raw = fixture()
    raw.loc[(raw.PEP_ID <= 2) & (raw.CODE == 8), 'CODE'] = 7
    with pytest.raises(ValueError, match='stage|CODE'):
        m.fit_calibration(raw, load_station_split(path))


@pytest.mark.parametrize('groups', [
    {'train': [1], 'val': [2], 'test': [1, 3]},
    {'train': [1], 'val': [], 'test': [2, 3]},
])
def test_invalid_split_fails(groups, tmp_path):
    path = tmp_path / 'split.json'; path.write_text(json.dumps(groups))
    with pytest.raises(ValueError):
        load_station_split(path)


def test_fit_guard_rejects_test_and_validation_sites(tmp_path):
    path = tmp_path / 'split.json'
    path.write_text(json.dumps({'train': [1], 'val': [2], 'test': [3]}))
    split = load_station_split(path)
    for ids in [[1, 2], [1, 3]]:
        with pytest.raises(ValueError, match='non-training'):
            assert_training_sites(ids, split)


def test_adjustment_starts_after_onset_and_resets_at_sowing():
    m = module()
    raw = pd.DataFrame(dict(PEP_ID=[1]*5, DATE=pd.date_range('2001-01-01', periods=5),
        CODE=[0, 1, 2, 4, 0], BBCH=[0, 0, 10, 31, 0], GDD=[10.]*5,
        LAT=[0.]*5, t_mean=[5.]*5, t_max=[10.]*5,
        SOWING_DATE=['2001-01-01']*4+['2001-01-05'],
        SOWING_KNOWN_AT=['2001-01-01']*4+['2001-01-05']))
    result = m.transform(raw, dict(photoperiod_onset_gdd=20, vernalization_onset_tpp=20,
        photoperiod_stop_gdd=30, vernalization_stop_tpp=25))
    np.testing.assert_allclose(result.Cumulative_GDD, [10, 20, 30, 40, 10])
    np.testing.assert_allclose(result.ppfun, [1, 1, .64, 1, 1])
    np.testing.assert_allclose(result.verfun, [1, 1, .3175, 1, 1])
    np.testing.assert_allclose(result.Cumulative_t_pp_v_GDD, [10, 20, 22.032, 32.032, 10])


def test_pipeline_exports_only_frozen_test_sites(tmp_path):
    m = module()
    raw = fixture()
    raw['CODE_new'] = raw.BBCH.map({0: 0, 10: 1, 31: 2, 51: 3, 85: 4})
    source = tmp_path/'raw.csv'; raw.to_csv(source, index=False)
    path = tmp_path/'split.json'
    path.write_text(json.dumps({'train': [1, 2], 'val': [3], 'test': [4]}))
    m.run(source, path, tmp_path/'out')
    evaluated = pd.read_csv(tmp_path/'out/simulation_test.csv')
    assert set(evaluated.PEP_ID) == {4}
    report = json.loads((tmp_path/'out/process_test_metrics.json').read_text())
    assert report['test_station_ids'] == [4]
    assert all(v['rows'] == 9 for v in report['metrics'].values())
    assert json.loads((tmp_path/'out/manifest.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        m.run(source, path, tmp_path/'out')






def test_common_process_evaluation_filters_exclusions_without_refitting(tmp_path):
    import importlib
    from process_model._support.station_split import split_identity
    from process_model._support.data_contract import file_hash
    path = ROOT/'process_model/evaluate_common_sites.py'
    assert path.exists(), 'Common-site process evaluator is missing'
    evaluate = importlib.import_module('process_model.evaluate_common_sites').evaluate
    m = module()
    raw = fixture()
    raw['CODE_new'] = raw.BBCH.map({0: 0, 10: 1, 31: 2, 51: 3, 85: 4})
    # Add a second reserved test site; exclude it from scoring, never reassign it.
    extra = raw.loc[raw.PEP_ID == 4].assign(PEP_ID=5, CODE_new=999)
    raw = pd.concat([raw, extra], ignore_index=True)
    source = tmp_path/'raw.csv'; raw.to_csv(source, index=False)
    split_path = tmp_path/'split.json'
    split_path.write_text(json.dumps({'train': [1, 2], 'val': [3], 'test': [4, 5]}))
    m.run(source, split_path, tmp_path/'data')
    calibration_before = (tmp_path/'data/calibration.json').read_bytes()
    protocol = dict(schema_version=1, split_sha256=split_identity(load_station_split(split_path)),
        seq_len=3, pred_len=2, station_ids=[4], excluded_station_ids=[5],
        exclusion_reason='No valid 5-day window',
        data_manifest_sha256=file_hash(tmp_path/'data/manifest.json'))
    selection = tmp_path/'evaluation.json'; selection.write_text(json.dumps(protocol))
    report = evaluate(tmp_path/'data', selection, tmp_path/'scored')
    assert report['test_station_ids'] == [4]
    assert report['excluded_test_station_ids'] == [5]
    assert all(v['rows'] == 9 for v in report['metrics'].values())
    assert set(pd.read_csv(tmp_path/'scored/simulation_test.csv').PEP_ID) == {4}
    assert (tmp_path/'data/calibration.json').read_bytes() == calibration_before
    assert set(load_station_split(tmp_path/'data/station_split.json').query("split == 'test'").PEP_ID) == {4, 5}


def test_process_inputs_and_predictions_ignore_observed_stage_labels(tmp_path):
    m = module()
    path = tmp_path/'split.json'
    path.write_text(json.dumps({'train': [1, 2], 'val': [3], 'test': [4]}))
    raw = fixture()
    calibration, _ = m.fit_calibration(raw, load_station_split(path))
    baseline = m.transform(raw, calibration)
    changed = raw.assign(CODE=0, BBCH=85, CODE_new=999)
    changed = m.transform(changed, calibration)
    columns = ['Cumulative_GDD', 'ppfun', 'Cumulative_t_pp_GDD', 'VERDAY',
               'CUMVER', 'verfun', 'Cumulative_t_pp_v_GDD']
    pd.testing.assert_frame_equal(baseline[columns], changed[columns])
    no_labels = m.transform(raw.drop(columns=['CODE', 'BBCH']), calibration)
    pd.testing.assert_frame_equal(baseline[columns], no_labels[columns])
    predictions = m.simulate_frame(no_labels, calibration)
    assert {'gdd_simulate', 'tpp_simulate', 'tppv_simulate'} <= set(predictions)


def test_process_prefix_is_invariant_to_future_weather_and_later_sowing(tmp_path):
    m = module()
    path = tmp_path/'split.json'
    path.write_text(json.dumps({'train': [1, 2], 'val': [3], 'test': [4]}))
    raw = fixture()
    calibration, _ = m.fit_calibration(raw, load_station_split(path))
    baseline = m.transform(raw, calibration)
    changed = raw.copy()
    future = changed.DATE >= pd.Timestamp('2001-10-06')
    changed.loc[future, ['GDD', 't_mean', 't_max', 'BBCH', 'CODE']] = [500, 30, 40, 85, 0]
    changed.loc[future, ['SOWING_DATE', 'SOWING_KNOWN_AT']] = '2001-10-06'
    result = m.transform(changed, calibration)
    prefix = baseline.DATE < pd.Timestamp('2001-10-06')
    columns = ['Cumulative_GDD', 'ppfun', 'Cumulative_t_pp_GDD', 'verfun', 'Cumulative_t_pp_v_GDD']
    pd.testing.assert_frame_equal(baseline.loc[prefix, columns], result.loc[prefix, columns])


def test_future_or_unavailable_sowing_records_are_rejected():
    m = module()
    raw = fixture()
    for column in ['SOWING_DATE', 'SOWING_KNOWN_AT']:
        bad = raw.copy(); bad[column] = '2002-10-01'
        with pytest.raises(ValueError, match='sowing|Sowing'):
            m.prepare(bad)


def test_sowing_assignment_is_as_of_occurrence_even_when_future_plan_is_known():
    from process_model.sowing import attach_sowing
    weather = pd.DataFrame(dict(PEP_ID=[1]*6, DATE=pd.date_range('2001-01-01', periods=6)))
    first = pd.DataFrame(dict(PEP_ID=[1], SOWING_DATE=['2001-01-02'], SOWING_KNOWN_AT=['2001-01-02']))
    later = pd.concat([first, pd.DataFrame(dict(PEP_ID=[1], SOWING_DATE=['2001-01-05'], SOWING_KNOWN_AT=['2001-01-01']))])
    baseline = attach_sowing(weather, first)
    result = attach_sowing(weather, later)
    pd.testing.assert_frame_equal(baseline.iloc[:4], result.iloc[:4])
    assert pd.isna(result.SOWING_DATE.iloc[0])
    assert result.SOWING_DATE.iloc[4] == pd.Timestamp('2001-01-05')


def test_transform_requires_explicit_sowing_instead_of_deriving_from_labels():
    m = module()
    raw = fixture().drop(columns=['SOWING_DATE', 'SOWING_KNOWN_AT'])
    with pytest.raises(ValueError, match='SOWING'):
        m.prepare(raw)
