"""Field transfer contracts; no candidate fitting or checkpoint selection."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.paper_study.wheat_dl_phenology_20261009 import field_transfer as ft

HERE = Path(__file__).resolve().parent
PREPARED = HERE / 'field_evidence/prepared'


def test_new_output_accepts_caller_selected_relative_and_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = Path('portable release') / 'new results'
    expected = tmp_path / relative
    assert ft.new_output(relative) == expected.resolve()
    assert ft.new_output(str(expected)) == expected.resolve()
    assert not expected.exists()


@pytest.mark.parametrize('kind', ['file', 'directory', 'dangling_symlink'])
def test_new_output_refuses_existing_paths_without_modifying_them(tmp_path, kind):
    output = tmp_path / 'existing'
    if kind == 'directory':
        output.mkdir()
        marker = output / 'preserve.txt'
        marker.write_text('original result')
    elif kind == 'file':
        output.write_text('original result')
        marker = output
    else:
        target = tmp_path / 'missing_target'
        output.symlink_to(target)
    with pytest.raises(FileExistsError):
        ft.new_output(output)
    if kind == 'dangling_symlink':
        assert output.is_symlink() and output.readlink() == target
        assert not target.exists()
    else:
        assert marker.read_text() == 'original result'


def test_open_lower_closed_upper_and_one_sided_bounds():
    lower, upper = pd.Timestamp('2020-06-10'), pd.Timestamp('2020-06-20')
    for date, error in [('2020-06-09', -2), ('2020-06-10', -1),
                        ('2020-06-11', 0), ('2020-06-20', 0), ('2020-06-22', 2)]:
        result = ft.interval_score(pd.Timestamp(date), lower, upper, upper)
        assert result['signed_distance_days'] == error
        assert result['distance_days'] == abs(error)
        assert not result['distance_is_lower_bound']
    assert ft.interval_score(lower, pd.NaT, upper, upper)['distance_days'] == 0
    assert ft.interval_score(lower, lower, pd.NaT, upper)['distance_days'] == 1


def test_missing_crossing_is_only_a_bound_and_not_an_imputed_event():
    result = ft.interval_score(pd.NaT, pd.Timestamp('2020-06-10'),
                               pd.Timestamp('2020-06-20'), pd.Timestamp('2020-06-25'))
    assert result['distance_days'] == 6
    assert result['distance_is_lower_bound']
    assert pd.isna(result['compatible'])
    result = ft.interval_score(pd.NaT, pd.Timestamp('2020-06-10'), pd.NaT,
                               pd.Timestamp('2020-06-25'))
    assert result['distance_days'] == 0
    assert result['compatible_or_not_excluded']
    assert pd.isna(result['compatible'])
    with pytest.raises(ValueError, match='bound'):
        ft.interval_score(pd.NaT, pd.NaT, pd.NaT, pd.Timestamp('2020-06-25'))


def source_frames():
    members = pd.DataFrame(dict(field_id=['a', 'b'], partition=['calibration'] * 2,
        sowing_date=['2020-01-01'] * 2, last_forcing_date=['2020-06-30'] * 2))
    bounds = pd.DataFrame(dict(field_id=['a'], partition=['calibration'], event=[31],
        lower_exclusive=['2020-06-10'], upper_inclusive=['2020-06-20'],
        eligible_constraint=[True], genuinely_bracketed=[True], censoring=['interval'],
        bracket_width_days=[10.], reason=[None]))
    baseline = pd.DataFrame([dict(field_id=f, partition='calibration', event=s,
        predicted_date='2020-06-15', forcing_end='2020-06-30')
        for f in ['a', 'b'] for s in ft.STAGES])
    return members, bounds, baseline


def test_alignment_retains_unobserved_fields_without_manufacturing_constraints():
    members, bounds, baseline = source_frames()
    aligned = ft.align_sources(members, bounds, baseline)
    assert len(aligned) == 8
    assert aligned.eligible_constraint.sum() == 1
    assert aligned.loc[aligned.field_id.eq('b'), 'lower_exclusive'].isna().all()
    assert aligned.loc[aligned.field_id.eq('b'), 'reason'].eq('no_archived_stage_constraint').all()
    assert aligned.loc[aligned.eligible_constraint, 'bracket_width_days'].tolist() == [10.]


def test_alignment_rejects_wrong_partition_horizon_duplicate_and_missing_prediction_row():
    for mutation in ['partition', 'horizon', 'duplicate', 'missing']:
        members, bounds, baseline = source_frames()
        if mutation == 'partition': baseline.loc[0, 'partition'] = 'reused_external_strict'
        if mutation == 'horizon': baseline.loc[0, 'forcing_end'] = '2020-07-01'
        if mutation == 'duplicate': baseline = pd.concat([baseline, baseline.iloc[:1]])
        if mutation == 'missing': baseline = baseline.iloc[1:]
        with pytest.raises(ValueError): ft.align_sources(members, bounds, baseline)


def test_ensemble_averages_cdfs_before_crossing_and_masks_padding():
    cdfs = np.array([[[[.9, .9, 1.]]], [[[.1, .2, 1.]]], [[[.1, .45, 1.]]]])
    assert ft.ensemble_median_days(cdfs, np.array([3])).item() == 2
    assert np.isnan(ft.ensemble_median_days(cdfs, np.array([1])).item())
    with pytest.raises(ValueError, match='three'):
        ft.ensemble_median_days(cdfs[:2], np.array([3]))


def test_weather_uses_recorded_tmean_and_frozen_transform():
    weather = pd.DataFrame(dict(date=pd.date_range('2020-10-01', periods=3),
        tmean_c=[6., 7., 8.], tmin_c=[1., 1., 1.], tmax_c=[19., 19., 19.]))
    member = dict(sowing_date='2020-10-01', latitude=50., field_id='a')
    params = json.loads((ft.ROOT / 'process_model/parameters/calibration.json').read_text())
    frame = ft.transform_weather(weather, member, params)
    np.testing.assert_array_equal(frame.t_mean, [6., 7., 8.])
    np.testing.assert_array_equal(frame.Cumulative_GDD, [6., 13., 21.])
    assert frame.DATE.iloc[0] == pd.Timestamp(member['sowing_date'])
    with pytest.raises(ValueError, match='temperature'):
        ft.transform_weather(weather.assign(tmean_c=41.), member, params)


def test_summary_keeps_missingness_widths_and_paired_support_separate():
    members, bounds, baseline = source_frames()
    aligned = ft.align_sources(members, bounds, baseline)
    predicted = aligned[['field_id', 'partition', 'event']].copy()
    predicted['predicted_date'] = pd.NaT
    scored = ft.compare_predictions(aligned, predicted)
    summary = ft.summarize(scored)
    row = summary.loc[summary.partition.eq('calibration') & summary.event.eq(31)
                      & summary.scope.eq('genuine_two_sided')].iloc[0]
    assert row.constraints == 1 and row.paired_reached == 0
    assert row.neural_missing == 1 and row.tpv_missing == 0
    assert pd.isna(row.neural_mean_distance_days)
    assert row.neural_mean_distance_lower_bound_days == 11
    assert row.median_interval_width_days == 10
    all_rows = summary.loc[summary.event.eq(31) & summary.scope.eq('all_constraints')].iloc[0]
    assert all_rows.registered_fields == 2 and all_rows.unavailable_constraints == 1


def test_transfer_cdf_matches_native_model_at_shared_thresholds():
    torch = pytest.importorskip('torch')
    from model.wheat_phenology_dl.network import WheatDevelopmentModel
    model = WheatDevelopmentModel(np.array([142.01, 542.3, 986.65, 1760.39]) / 1760.39, 8)
    state, native = model(torch.zeros(2, 5, 8), torch.ones(2, 5) * 600, 1760.39)
    extra = ft.threshold_cdf(state, model.raw_sigma,
                             np.array([542.3, 807.799125, 1165.925558, 1760.39]) / 1760.39)
    torch.testing.assert_close(extra[:, 0], native[:, 1])
    torch.testing.assert_close(extra[:, 3], native[:, 3])
    assert torch.all(extra[:, 0] >= extra[:, 1])
    assert torch.all(extra[:, 1] >= extra[:, 2])
    assert torch.all(extra[:, 2] >= extra[:, 3])


def test_checkpoint_metadata_rejects_feature_scale_and_normalizer_drift():
    good = dict(features=ft.FEATURES, stages=[10, 31, 51, 85], horizon=366,
        thermal_scale=1760.39, mean=[0.] * 8, std=[1.] * 8,
        thresholds=(np.array([142.01, 542.3, 986.65, 1760.39]) / 1760.39).tolist())
    ft.validate_checkpoint_metadata(good)
    for patch in [dict(features=list(reversed(ft.FEATURES))), dict(std=[0.] * 8),
                  dict(mean=[float('nan')] * 8), dict(thermal_scale=1700), dict(stages=[10, 31, 65, 85])]:
        with pytest.raises(ValueError): ft.validate_checkpoint_metadata({**good, **patch})


def test_prepared_bundle_reconciles_all_archived_sources():
    """Check the bundled release inputs under both Python environments."""
    arrays, fields, records, metadata = ft.load_prepared(PREPARED)
    assert metadata['field_counts'] == ft.FIELD_COUNTS
    assert len(fields) == 216 and len(records) == 864
    assert records.eligible_constraint.sum() == 749
    assert arrays['lengths'].sum() == 54110
    assert metadata['archived_tpv_reconciliation']['different'] == 0
    assert metadata['archived_score_reconciliation']['rows'] == 749
    assert metadata['horizon'] >= 384
    assert arrays['X'].shape[-1] == len(ft.FEATURES)


def test_synthetic_checkpoint_round_trip_uses_each_normalizer_and_keeps_missing(tmp_path, monkeypatch):
    """Untrained zero-head fixtures test I/O; these are not candidate results."""
    torch = pytest.importorskip('torch')
    from model.wheat_phenology_dl.network import WheatDevelopmentModel
    arrays, fields, records, _ = ft.load_prepared(PREPARED)
    native = (np.array([142.01, 542.3, 986.65, 1760.39]) / 1760.39).tolist()
    checkpoints, normalizers = [], []
    for seed in range(3):
        model = WheatDevelopmentModel(native, len(ft.FEATURES), encoder='tcn', width=2)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        mean = (np.arange(8, dtype=np.float32) + seed).tolist()
        std = (np.arange(8, dtype=np.float32) + seed + 1).tolist()
        metadata = dict(features=ft.FEATURES, stages=[10, 31, 51, 85], horizon=366,
                        thermal_scale=1760.39, thresholds=native, mean=mean, std=std)
        checkpoint = tmp_path / f'synthetic_untrained_{seed}.pt'
        torch.save(dict(metadata=metadata, config=dict(encoder='tcn', width=2, seed=seed),
                        state_dict=model.state_dict(), best_epoch=0, validation_crps=0.,
                        input_arrays_sha256='synthetic_test_only', protocol_sha256='synthetic_test_only',
                        network_sha256=ft.sha(ft.ROOT / ft.SOURCE_PATHS['network'])), checkpoint)
        checkpoints.append(checkpoint)
        normalizers.append((np.asarray(mean, np.float32), np.asarray(std, np.float32)))
    observed_inputs = []
    original_forward = WheatDevelopmentModel.forward

    def record_inputs(self, x, thermal, scale):
        observed_inputs.append((x.cpu().numpy().copy(), thermal.cpu().numpy().copy()))
        return original_forward(self, x, thermal, scale)

    monkeypatch.setattr(WheatDevelopmentModel, 'forward', record_inputs)
    output = tmp_path / 'synthetic_evaluation'
    report = ft.evaluate(PREPARED, checkpoints, output, batch_size=256)
    assert report['field_counts'] == ft.FIELD_COUNTS and report['comparison_rows'] == 864
    assert report['eligible_constraints'] == 749
    assert not report['selection_independence_verified']
    assert len(observed_inputs) == 3
    for (x, thermal), (mean, std) in zip(observed_inputs, normalizers):
        np.testing.assert_array_equal(x, (arrays['X'] - mean) / std)
        np.testing.assert_array_equal(thermal, arrays['G'])
    predictions = pd.read_csv(output / 'neural_predictions.csv', parse_dates=['predicted_date'])
    paired = records.merge(predictions, on=ft.KEYS, validate='one_to_one')
    assert (paired.tpv_date.eq(paired.predicted_date)
            | (paired.tpv_date.isna() & paired.predicted_date.isna())).all()
    assert predictions.predicted_date.isna().sum() > 200
    summary = pd.read_csv(output / 'interval_metrics.csv')
    assert len(summary) == 48 and set(summary.partition) == set(ft.FIELD_COUNTS)
    assert summary.paired_neural_minus_tpv_distance_days.dropna().eq(0).all()


def test_selected_checkpoints_reproduce_bundled_scientific_results(tmp_path):
    pytest.importorskip('torch')
    checkpoints = [HERE / f'fits/tcn_s{seed}.pt' for seed in [17, 29, 43]]
    reference = HERE / 'field_evidence/evaluation'
    expected = json.loads((reference / 'evaluation.json').read_text())
    for path, checkpoint in zip(checkpoints, expected['checkpoints']):
        assert ft.sha(path) == checkpoint['sha256']
    output = tmp_path / 'selected_checkpoint_reproduction'
    result = ft.evaluate(PREPARED, checkpoints, output)
    assert result['seeds'] == [17, 29, 43]
    assert result['field_counts'] == expected['field_counts']
    assert result['eligible_constraints'] == expected['eligible_constraints']
    for name in ['neural_predictions.csv', 'field_stage_comparison.csv', 'interval_metrics.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(output / name), pd.read_csv(reference / name),
                                      check_exact=True)


def test_evaluation_rejects_checkpoint_objects_outside_weights_only_contract(tmp_path):
    torch = pytest.importorskip('torch')
    checkpoint = tmp_path / 'unsupported_checkpoint_object.pt'
    torch.save({'unsupported_object': Path('not_a_tensor_or_primitive')}, checkpoint)
    output = tmp_path / 'rejected_checkpoint_evaluation'
    with pytest.raises(pickle.UnpicklingError, match='Weights only load failed'):
        ft.evaluate(PREPARED, [checkpoint] * 3, output)
    assert not output.exists()
