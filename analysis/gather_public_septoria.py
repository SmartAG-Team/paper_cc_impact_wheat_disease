"""Download public Septoria data with source metadata and checksum validation.

Original files are preserved. Run from the project root with Python and requests.
Public mirrors are catalogued as copies of the original dataset, not new studies.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "public_septoria"
SOURCES = [
    ("karisto-2018-field", 4933099, "10.5061/dryad.171q4"),
    ("hafeez-2025-infection", 14515753, "10.5281/zenodo.14515753"),
    ("durum-mixtures-2020", 3979476, "10.5281/zenodo.3979476"),
    ("karisto-2022-dispersal-code", 5713639, "10.5061/dryad.kkwh70s5r"),
]
FIGSHARE = [
    ("chaloner-2019-weather-response", 7937516, None),
    ("chaloner-2019-weather-response", 7937525, None),
    ("nordic-baltic-2012-2016", 19203377, 4),
    ("nordic-baltic-2012-2016", 19203398, 3),
]


def get_json(url):
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    return response.json()


def save_file(folder, filename, url, expected_md5=None, expected_size=None):
    if Path(filename).name != filename:
        raise ValueError(f"Unsafe filename: {filename}")
    path = folder / filename
    if path.exists():
        payload = path.read_bytes()
    else:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.content
    actual_md5 = hashlib.md5(payload).hexdigest()
    if expected_md5 and expected_md5 != actual_md5:
        raise ValueError(f"Checksum failure: {filename}")
    if expected_size is not None and expected_size != len(payload):
        raise ValueError(f"Size failure: {filename}")
    if not path.exists():
        path.write_bytes(payload)
    return {
        "filename": filename,
        "project_relative_path": str(path.relative_to(ROOT)),
        "source_url": url,
        "bytes": len(payload),
        "md5": actual_md5,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "repository_checksum_verified": bool(expected_md5),
    }


def fetch_figshare(spec):
    slug, article_id, version = spec
    folder = DEST / slug
    folder.mkdir(parents=True, exist_ok=True)
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    if version is not None:
        url += f"/versions/{version}"
    metadata = get_json(url)
    (folder / f"figshare-{article_id}-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset_id": slug,
        "title": metadata["title"],
        "original_dataset_doi": metadata.get("doi"),
        "source_page": metadata["url_public_html"],
        "metadata_url": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "license": metadata.get("license"),
        "version": metadata.get("version", version),
        "files": [],
    }
    for item in metadata["files"]:
        manifest["files"].append(save_file(
            folder, item["name"], item["download_url"],
            item.get("computed_md5") or item.get("supplied_md5"), item["size"]
        ))
    return manifest


def fetch_inrae():
    slug = "orellana-torrejon-2022-field"
    folder = DEST / slug
    folder.mkdir(parents=True, exist_ok=True)
    url = "https://data.inrae.fr/api/datasets/:persistentId/?persistentId=doi:10.15454/4MAAI0"
    metadata = get_json(url)
    (folder / "inrae-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    version = metadata["data"]["latestVersion"]
    fields = version["metadataBlocks"]["citation"]["fields"]
    manifest = {
        "dataset_id": slug,
        "title": next(f["value"] for f in fields if f["typeName"] == "title"),
        "original_dataset_doi": "10.15454/4MAAI0",
        "source_page": "https://doi.org/10.15454/4MAAI0",
        "metadata_url": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "license": version.get("license"),
        "version": f"{version['versionNumber']}.{version['versionMinorNumber']}",
        "files": [],
    }
    for item in version["files"]:
        f = item["dataFile"]
        if item.get("restricted"):
            continue
        # Dataverse tabular ingestion preserves the original spreadsheet's MD5;
        # fetch its original bytes rather than compare transformed TSV bytes.
        original = f["filename"].endswith(".tab") and f.get("originalFileName")
        filename = f["originalFileName"] if original else f["filename"]
        size = f["originalFileSize"] if original else f["filesize"]
        access_url = f"https://data.inrae.fr/api/access/datafile/{f['id']}"
        if original:
            access_url += "?format=original"
        checksum = f.get("checksum", {})
        md5 = checksum.get("value") if checksum.get("type") == "MD5" else None
        manifest["files"].append(save_file(
            folder, filename, access_url, md5, size
        ))
    return manifest


def fetch_source(spec):
    slug, record_id, original_doi = spec
    folder = DEST / slug
    folder.mkdir(parents=True, exist_ok=True)
    metadata_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(metadata_url, timeout=45)
    response.raise_for_status()
    record = response.json()
    (folder / "zenodo-metadata.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset_id": slug,
        "title": record["metadata"]["title"],
        "original_dataset_doi": original_doi,
        "download_record_doi": record["metadata"]["doi"],
        "source_page": f"https://zenodo.org/records/{record_id}",
        "metadata_url": metadata_url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "license": record["metadata"].get("license"),
        "publication_date": record["metadata"].get("publication_date"),
        "version": record["metadata"].get("version"),
        "files": [],
    }
    for source_file in record.get("files", []):
        filename = source_file["key"]
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe filename: {filename}")
        expected_md5 = source_file["checksum"].removeprefix("md5:")
        path = folder / filename
        if path.exists():
            payload = path.read_bytes()
        else:
            download = requests.get(source_file["links"]["self"], timeout=60)
            download.raise_for_status()
            payload = download.content
        actual_md5 = hashlib.md5(payload).hexdigest()
        if actual_md5 != expected_md5 or len(payload) != source_file["size"]:
            raise ValueError(f"Integrity failure: {slug}/{filename}")
        if not path.exists():
            path.write_bytes(payload)
        manifest["files"].append({
            "filename": filename,
            "project_relative_path": str(path.relative_to(ROOT)),
            "source_url": source_file["links"]["self"],
            "bytes": len(payload),
            "md5": actual_md5,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "repository_checksum_verified": True,
        })
    (folder / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifests = list(pool.map(fetch_source, SOURCES))
    # Files in a shared dataset folder are fetched sequentially.
    manifests.extend(fetch_figshare(item) for item in FIGSHARE)
    manifests.append(fetch_inrae())
    (DEST / "download-manifest.json").write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for item in manifests:
        print(item["dataset_id"], len(item["files"]),
              sum(f["bytes"] for f in item["files"]), "bytes; checksums verified")


if __name__ == "__main__":
    main()
