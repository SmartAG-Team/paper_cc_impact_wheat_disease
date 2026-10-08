"""Event-date scoring must not choose predictions using observed event dates."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def evaluator():
    assert (ROOT/'process_model/evaluate_event_dates.py').exists(), 'Event-date evaluator is missing'
    return importlib.import_module('process_model.evaluate_event_dates')


def fixture():
    frame = pd.DataFrame({
        'PEP_ID': [1]*6, 'DATE': pd.date_range('2000-12-29', periods=6),
        'SOWING_DATE': ['2000-12-29']*6, 'CODE': [0, 2, 1, 4, 6, 8],
        'Cumulative_GDD': [0, 12, 22, 32, 42, 52],
        'Cumulative_t_pp_GDD': [0, 6, 12, 22, 32, 42],
        'Cumulative_t_pp_v_GDD': [0, 2, 6, 12, 22, 32],
    })
    thresholds = [dict(BBCH=stage, **{c: value for c in frame if c.startswith('Cumulative')})
                  for stage, value in [(0, 0), (10, 10), (31, 20), (51, 30), (85, 40)]]
    return frame, {'thresholds': thresholds}


def test_first_crossing_matches_across_years_and_does_not_follow_truth():
    m = evaluator(); frame, parameters = fixture()
    rows = m.event_rows(frame, parameters).set_index('BBCH')
    assert rows.loc[31, 'GDD_date'] == pd.Timestamp('2000-12-31')
    assert rows.loc[31, 'true_date'] == pd.Timestamp('2001-01-01')
    changed = frame.copy(); changed['CODE'] = [0, 1, 2, 3, 4, 8]
    other = m.event_rows(changed, parameters).set_index('BBCH')
    pd.testing.assert_frame_equal(rows[['GDD_date', 'T-P_date', 'T-P-V_date']],
                                  other[['GDD_date', 'T-P_date', 'T-P-V_date']])


def test_shared_event_denominator_and_missing_crossings_are_explicit():
    m = evaluator(); frame, parameters = fixture()
    metrics = m.summarize(m.event_rows(frame, parameters), include_sowing=False)
    assert metrics['observed_events'] == 4
    assert metrics['common_matched_events'] == 3
    assert metrics['models']['T-P-V']['missing_predictions'] == 1
    assert metrics['models']['GDD']['MAE_days'] == pytest.approx(2/3)
    assert metrics['models']['GDD']['RMSE_days'] == pytest.approx(np.sqrt(2/3))
    assert metrics['models']['T-P']['MAE_days'] == pytest.approx(1/3)
    assert metrics['models']['T-P-V']['MAE_days'] == pytest.approx(4/3)


def test_gaps_do_not_produce_exact_event_dates_and_duplicates_are_rejected():
    m = evaluator(); frame, parameters = fixture()
    rows = m.event_rows(frame.drop(index=2), parameters)
    assert not rows.cycle_complete.any()
    assert rows.loc[rows.BBCH != 0, ['GDD_date', 'T-P_date', 'T-P-V_date']].isna().all().all()
    with pytest.raises(ValueError, match='Duplicate'):
        m.event_rows(pd.concat([frame, frame.iloc[[1]]]), parameters)


def test_cycle_resets_and_stage_jumps_keep_distinct_event_keys():
    m = evaluator(); frame, parameters = fixture()
    second = frame.copy()
    second['DATE'] += pd.Timedelta(days=365)
    second['SOWING_DATE'] = '2001-12-29'
    second.loc[1:, 'Cumulative_GDD'] = 50
    rows = m.event_rows(pd.concat([frame, second]), parameters)
    second_rows = rows.loc[rows.SOWING_DATE == pd.Timestamp('2001-12-29')]
    assert len(second_rows) == 5
    assert (second_rows.loc[second_rows.BBCH != 0, 'GDD_date'] == pd.Timestamp('2001-12-30')).all()


def test_known_sowing_is_excluded_from_primary_errors_unless_requested():
    m = evaluator(); frame, parameters = fixture()
    rows = m.event_rows(frame, parameters)
    metrics = m.summarize(rows, include_sowing=True)
    assert metrics['common_matched_events'] == 4
    assert metrics['models']['GDD']['MAE_days'] == .5


def test_event_evaluation_preserves_reserved_sites_and_frozen_parameters(tmp_path):
    import json
    from process_model.calibrate import run
    from process_model._support.data_contract import file_hash
    from process_model._support.station_split import load_station_split, split_identity
    m = evaluator(); frame, _ = fixture()
    frame['GDD'] = [0, 12, 10, 10, 10, 10]
    frame['LAT'] = 50
    frame['t_mean'] = 5
    frame['t_max'] = 10
    frame['SOWING_KNOWN_AT'] = frame.SOWING_DATE
    frame['CODE_new'] = [0, 1, 1, 2, 3, 4]
    tail = frame.iloc[[-1]].copy()
    tail['DATE'] += pd.Timedelta(days=1)
    tail['CODE'] = 9
    frame = pd.concat([frame, tail], ignore_index=True)
    raw = pd.concat([frame.assign(PEP_ID=site) for site in [1, 2, 3, 4]], ignore_index=True)
    source = tmp_path/'source.csv'; raw.to_csv(source, index=False)
    split = tmp_path/'split.json'
    split.write_text(json.dumps({'train': [1], 'val': [2], 'test': [3, 4]}))
    data = tmp_path/'data'
    run(source, split, data)
    before = (data/'calibration.json').read_bytes()
    protocol = dict(schema_version=1, split_sha256=split_identity(load_station_split(split)),
        seq_len=3, pred_len=2, station_ids=[3], excluded_station_ids=[4],
        exclusion_reason='Explicit reserved-site exclusion for this fixture',
        data_manifest_sha256=file_hash(data/'manifest.json'))
    selection = tmp_path/'selection.json'; selection.write_text(json.dumps(protocol))
    report = m.evaluate(data, selection, tmp_path/'events')
    assert report['test_station_ids'] == [3]
    assert report['excluded_test_station_ids'] == [4]
    assert report['summary']['observed_events'] == 4
    assert report['summary']['common_matched_events'] == 4
    assert report['summary']['models']['GDD']['MAE_days'] == 0
    assert set(pd.read_csv(tmp_path/'events/event_dates.csv').PEP_ID) == {3}
    assert (data/'calibration.json').read_bytes() == before
    assert set(load_station_split(data/'station_split.json').query("split == 'test'").PEP_ID) == {3, 4}
