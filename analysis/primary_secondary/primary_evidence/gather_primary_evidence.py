"""Retrieve public process observations without changing source bytes.

Existing files are verified against repository checksums and never overwritten.
No authentication, challenge bypass, or retry against denied endpoints is used.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "data/public_septoria/primary_evidence"
REPORT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(spec: dict) -> dict:
    path = DEST / spec["relative_path"]
    record = {**spec, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            record["status"] = "existing_verified"
        else:
            response = requests.get(spec["url"], timeout=(20, 60), stream=True)
            record["http_status"] = response.status_code
            final_url = urlsplit(response.url)
            # Repository redirects may carry temporary public storage signatures.
            # Retain the stable target without its temporary query credentials.
            record["final_url"] = final_url._replace(query="", fragment="").geturl() if "x-amz-" in final_url.query.lower() else response.url
            record["content_type"] = response.headers.get("Content-Type", "")
            if not response.ok:
                record["status"] = "unavailable"
                return record
            temporary = path.with_name(path.name + ".partial")
            with temporary.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
            temporary.rename(path)
            record["status"] = "downloaded"
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
        if spec.get("expected_digest"):
            algorithm = spec["digest_type"].replace("-", "")
            h = hashlib.new(algorithm)
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            record["repository_checksum_verified"] = h.hexdigest() == spec["expected_digest"]
            if not record["repository_checksum_verified"]:
                record["status"] = "checksum_mismatch"
        if spec.get("expected_bytes") is not None:
            record["repository_size_verified"] = record["bytes"] == spec["expected_bytes"]
            if not record["repository_size_verified"]:
                record["status"] = "size_mismatch"
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def main() -> None:
    specifications = []
    for name, subdirectory, doi in [
        ("efficiency", "karisto-2019-infection-efficiency", "10.5061/dryad.k0n0676"),
        ("splash", "karisto-2021-splash", "10.5061/dryad.kkwh70s5r"),
    ]:
        metadata = json.loads((DEST / f"dryad_{name}_files.json").read_text())
        dataset = json.loads((DEST / f"dryad_{name}_api.json").read_text())
        landing = BeautifulSoup((DEST / f"dryad_{name}_landing.html").read_text(), "html.parser")
        public_links = {a.get_text(strip=True): a["href"] for a in landing.select("a.js-individual-dl")}
        for file in metadata["_embedded"]["stash:files"]:
            specifications.append({
                "dataset_doi": doi,
                "relative_path": f"{subdirectory}/{file['path']}",
                "url": "https://datadryad.org" + public_links[file["path"]],
                "license_url": dataset["license"],
                "expected_digest": file["digest"],
                "digest_type": file["digestType"],
                "expected_bytes": file["size"],
            })
    metadata = json.loads((DEST / "ascospore_dataverse.json").read_text())
    version = metadata["data"]["latestVersion"]
    for file in version["files"]:
        source = file["dataFile"]
        if file.get("restricted"):
            continue
        specifications.append({
            "dataset_doi": "10.15454/JGWC5A",
            "relative_path": "orellana-2022-interseason/" + source.get("originalFileName", source["filename"]),
            "url": f"https://entrepot.recherche.data.gouv.fr/api/access/datafile/{source['id']}?format=original",
            "license_url": version["license"]["uri"],
            "expected_digest": source["checksum"]["value"],
            "digest_type": source["checksum"]["type"],
            "expected_bytes": source.get("originalFileSize", source["filesize"]),
        })
    previous_path = DEST / "download_manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    denied = {r["relative_path"]: r for r in previous.get("files", []) if r.get("http_status") in (401, 403, 429)}
    # A normal rerun never retries an endpoint that was denied in this archive.
    pending = [s for s in specifications if s["relative_path"] not in denied]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(denied.values()) + list(pool.map(download, pending))
    manifest = {"checked_utc": datetime.now(timezone.utc).isoformat(), "files": records}
    (DEST / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for record in records:
        print(json.dumps({k: record.get(k) for k in ["relative_path", "status", "bytes", "http_status", "repository_checksum_verified"]}), flush=True)


if __name__ == "__main__":
    main()
