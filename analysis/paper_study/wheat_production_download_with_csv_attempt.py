"""Acquire the matching SPAM2020v2r2 production sources with repository hashes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/paper_study/wheat_production'
OUT.mkdir(parents=True, exist_ok=True)
METADATA = ROOT / 'data/paper_study/wheat_area/spam2020_dataverse.json'
FILES = json.loads(METADATA.read_text())['data']['latestVersion']['files']
URLS = {
    'spam2020V2r2_global_production.geotiff.zip':
    'https://www.dropbox.com/scl/fi/euolvnpdooxhd5ljskl4q/spam2020V2r2_global_production.geotiff.zip?dl=1&rlkey=mttc09xhfr4m1zsd4ik40wvz1&st=q36ax4cq',
    'spam2020V2r2_global_production.csv.zip':
    'https://www.dropbox.com/scl/fi/asxrhdtpvu2kbymii5a6z/spam2020V2r2_global_production.csv.zip?dl=1&rlkey=lhoh7dpeskozqhuh7lli20udu&st=zn9mh4wl',
}


def download(item):
    name, url = item
    source = next(f['dataFile'] for f in FILES if f['dataFile']['filename'] == name)
    path = OUT / name
    if path.exists() and not zipfile.is_zipfile(path):
        rejected=path.with_suffix('.rejected_html')
        path.replace(rejected)
    transport_url=url.replace('www.dropbox.com','dl.dropboxusercontent.com')
    if not path.exists():
        r = requests.get(transport_url, stream=True, timeout=(30, 180))
        r.raise_for_status()
        temporary = path.with_suffix('.download')
        with temporary.open('wb') as stream:
            for block in r.iter_content(1024 * 1024):
                stream.write(block)
        temporary.replace(path)
    blob = path.read_bytes()
    assert len(blob) == source['filesize']
    assert hashlib.md5(blob).hexdigest() == source['md5']
    members = []
    with zipfile.ZipFile(path) as archive:
        if 'geotiff' in name:
            for member in archive.infolist():
                if '_WHEA_' in member.filename and member.filename.endswith('.tif'):
                    target = OUT / Path(member.filename).name
                    data = archive.read(member)
                    if target.exists():
                        assert target.read_bytes() == data
                    else:
                        target.write_bytes(data)
                    members.append(dict(name=member.filename, bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest(), zip_crc32=member.CRC))
        else:
            members = [dict(name=m.filename, bytes=m.file_size, zip_crc32=m.CRC)
                for m in archive.infolist() if m.filename.endswith('.csv')]
    return dict(file=name, public_url=url, transport_url=transport_url, repository_file_id=source['id'],
        bytes=len(blob), md5=source['md5'], sha256=hashlib.sha256(blob).hexdigest(),
        native_members=members)


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(download, URLS.items()))
    receipt = dict(status='complete_and_hash_verified', acquired_utc=datetime.now(timezone.utc).isoformat(),
        provider='IFPRI SPAM2020 version2 release2; Harvard Dataverse version6', reference_year=2020,
        metadata_sha256=hashlib.sha256(METADATA.read_bytes()).hexdigest(),
        source_archives=receipts, production_is_disease_loss=False)
    (OUT / 'download_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
