import hashlib
import json

import pytest

from paper.analyses.food_security.provenance import auth_source, prepare_destination, write_json


def test_authenticated_input_rejects_changed_bytes(tmp_path):
    p = tmp_path / 'data/final_study'
    p.mkdir(parents=True)
    target = p / 'source.csv'
    target.write_text('a,b\n1,2\n')
    record = dict(path='data/final_study/source.csv', bytes=target.stat().st_size,
                  sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (p / 'path_manifest.json').write_text(json.dumps({'files': {'original': record}}))
    assert auth_source(tmp_path, record['path'])['sha256'] == record['sha256']
    target.write_text('a,b\n1,3\n')
    with pytest.raises(ValueError, match='checksum'):
        auth_source(tmp_path, record['path'])


def test_unregistered_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='registered'):
        auth_source(tmp_path, 'unknown.csv')


def test_output_cannot_overwrite_frozen_directories_or_existing_run(tmp_path):
    with pytest.raises(ValueError, match='frozen'):
        prepare_destination(tmp_path / 'data/new', tmp_path)
    p = prepare_destination(tmp_path / 'outputs/new', tmp_path)
    assert p.is_dir()
    with pytest.raises(FileExistsError):
        prepare_destination(p, tmp_path)


def test_output_must_stay_in_repositories_outputs_directory(tmp_path):
    with pytest.raises(ValueError, match='outputs'):
        prepare_destination(tmp_path / 'docs/new', tmp_path)
    with pytest.raises(ValueError, match='outputs'):
        prepare_destination(tmp_path.parent / (tmp_path.name + '_external'), tmp_path)


def test_json_never_serializes_nonfinite_as_measurements(tmp_path):
    p = tmp_path / 'receipt.json'
    write_json(p, {'undefined': float('nan'), 'infinite': float('inf'), 'value': 2.})
    assert json.loads(p.read_text()) == {'undefined': None, 'infinite': None, 'value': 2.}

