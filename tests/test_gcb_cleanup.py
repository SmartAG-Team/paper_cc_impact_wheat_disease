from pathlib import Path
import pytest


def test_externalized_cache_preserves_original_path_bytes_and_inode(tmp_path):
    from paper.scripts.cleanup import externalize
    root = tmp_path/'repo'; root.mkdir()
    source = root/'outputs/unused'; source.mkdir(parents=True)
    data = source/'fit.bin'; data.write_bytes(b'unchanged evidence')
    inode = data.stat().st_ino
    result = externalize(root, 'outputs/unused', tmp_path/'archive')
    assert source.is_symlink() and data.read_bytes() == b'unchanged evidence'
    assert data.stat().st_ino == inode
    assert result['bytes'] == len(b'unchanged evidence')
    assert externalize(root, 'outputs/unused', tmp_path/'archive')['status'] == 'already_external'


def test_cleanup_refuses_protected_paths_and_archive_overwrite(tmp_path):
    from paper.scripts.cleanup import externalize
    root = tmp_path/'repo'; root.mkdir()
    for name in ['.git', 'data/final_climate', 'data/final_models', 'data/final_study']:
        p = root/name; p.mkdir(parents=True,exist_ok=True)
        with pytest.raises(ValueError): externalize(root, name, tmp_path/'archive')
    p = root/'outputs/unused'; p.mkdir(parents=True)
    target = tmp_path/'archive/outputs/unused'; target.mkdir(parents=True)
    with pytest.raises(FileExistsError): externalize(root, 'outputs/unused', tmp_path/'archive')
    assert not p.is_symlink() and target.is_dir()
