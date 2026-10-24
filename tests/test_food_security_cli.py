import pytest

from paper.analyses.food_security.__main__ import parse_args


def test_full_period_selects_all_thirty_reference_years():
    args = parse_args(['--output', 'outputs/new', '--full-period'])
    assert args.reference_years == list(range(1991, 2021))
    assert args.stages == ['benchmark', 'attribution', 'exposure', 'evidence']


def test_duplicate_models_and_outside_registered_years_are_rejected():
    with pytest.raises(SystemExit):
        parse_args(['--output', 'outputs/new', '--models', 'ACCESS-CM2', 'ACCESS-CM2'])
    with pytest.raises(SystemExit):
        parse_args(['--output', 'outputs/new', '--reference-years', '2030'])


def test_explicit_short_run_cannot_be_labelled_full_period():
    args = parse_args(['--output', 'outputs/new', '--reference-years', '1991', '2020'])
    assert args.reference_years == [1991, 2020]
    assert not args.full_period
