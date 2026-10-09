"""Copy public source bytes into a self-contained, deterministic compact bundle."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / "yield_transfer_audit_20261009/public_source_subset.zip"
MEMBERS = [
    "sources/swiss-mixtures/blendit_full_pub.csv", "sources/swiss-mixtures/README.txt",
    "sources/swiss-mixtures-v1/blendit_full.csv", "sources/zenodo_17432866.json",
    "sources/zenodo_10209176.json", "sources/zenodo_5393959.json",
    "sources/montazeaud2022/raw_yield_variables.csv", "sources/montazeaud2022/GY_RAW_RYT.csv",
    "sources/montazeaud2022/STB_RAW_RYT.csv", "sources/montazeaud2022/STB_symptoms.csv",
    "source_code/Allelic_richness_phenotypic_file_prep.R", "source_code/Spatial_analyses_yield_variables.R",
    "context/START_HERE.txt", "context/french_archive_verification.json", "ATTRIBUTION.txt",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source=DEFAULT_SOURCE):
    lock = json.loads((HERE / "protocol_lock.json").read_text())
    assert sha(source.read_bytes()) == lock["prior_bundle_sha256"], "Prior bundle changed"
    provenance = []
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(HERE / "public_inputs.zip", "w", compression=zipfile.ZIP_DEFLATED) as out:
        for member in sorted(MEMBERS):
            data = original.read(member)
            info = zipfile.ZipInfo(member, date_time=(2026, 10, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            out.writestr(info, data)
            doi = ("10.5281/zenodo.10209176" if "swiss-mixtures-v1" in member or "10209176" in member else
                   "10.5281/zenodo.17432866" if "swiss-mixtures" in member or "17432866" in member else
                   "10.5281/zenodo.5393959" if "montazeaud2022" in member or "5393959" in member or member.startswith("source_code/") else None)
            provenance.append({"member": member, "bytes": len(data), "sha256": sha(data), "dataset_doi": doi,
                               "license": "CC-BY-4.0" if doi else "original attribution/local archived provenance retained",
                               "role": "source measurements or metadata" if doi else "archived provenance; not independent evidence",
                               "origin": "yield_transfer_audit_20261009/public_source_subset.zip::" + member})
    manifest = {"schema_version": "1.0", "bundle": "public_inputs.zip", "bundle_sha256": sha((HERE / "public_inputs.zip").read_bytes()),
                "parent_bundle_sha256": lock["prior_bundle_sha256"], "members": provenance,
                "datasets": [
                    {"doi": "10.5281/zenodo.17432866", "creators": "Laura Stefan; Dario Fossati; Karl-Heinz Camp; Didier Pellet; Flavio Foiada; Lilia Levy Häner", "role": "Swiss current public observations"},
                    {"doi": "10.5281/zenodo.10209176", "creators": "Laura Stefan", "role": "Swiss identifiers; same observations, not replication"},
                    {"doi": "10.5281/zenodo.5393959", "creators": "Germain Montazeaud; Timothée Flutre; Elsa Ballini; Jean-Benoit Morel; Jacques David; Johanna Girodolle; Aline Rocher; Aurélie Ducasse; Cyrille Violle; Florian Fort; Hélène Fréville", "role": "French raw and adjusted phenotypes and original processing code"}],
                "methods_verification": {"doi": "10.1111/nph.17915", "url": "https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.17915", "access_date": "2026-10-09", "sections": ["Experimental design", "Productivity measurements", "Symptom measurements", "Relative yield total computation"], "journal": "New Phytologist", "journal_scope": "recognized leading specialist plant-science journal", "stored_article_copy": False},
                "literature_policy": "No MDPI/Frontiers sources; Swiss associated article is archived provenance only. Dataset records are primary sources."}
    (HERE / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared {len(provenance)} public members, {(HERE / 'public_inputs.zip').stat().st_size} bytes")


if __name__ == "__main__":
    prepare()
