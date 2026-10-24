"""Integrity checks restricted to the portable model's actual local dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/final_models"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest():
    return json.loads((DATA / "manifest.json").read_text())


def verify_record(record):
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or digest(path) != record["sha256"]:
        raise ValueError("Portable model artifact changed: " + str(path))


def verify(component=None):
    contract = manifest()
    for record in contract["files"]:
        if component is None or component in record["components"]:
            verify_record(record)
    return contract


def canonical_identity():
    return manifest()["original_canonical_source"]
