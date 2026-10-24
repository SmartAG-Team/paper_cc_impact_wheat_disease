"""Export the authenticated paper package, including linked plotting data."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

PACKAGE=Path(__file__).resolve().parent


def export(output=None):
    manifest=json.loads((PACKAGE/'publication_manifest.json').read_text())
    output=Path(output) if output else PACKAGE/'exports/GCB_Submission_20261024.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.tmp')
    prefix='GCB_Submission_20261024/'
    with ZipFile(temporary,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for name,record in manifest['files'].items():
            path=PACKAGE/name
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=record['sha256']:raise ValueError('Changed package file: '+name)
            archive.write(path,prefix+name)
        archive.write(PACKAGE/'publication_manifest.json',prefix+'publication_manifest.json')
    with ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for name,record in manifest['files'].items():
            assert hashlib.sha256(archive.read(prefix+name)).hexdigest()==record['sha256'],name
    temporary.replace(output)
    return dict(status='passed',files=len(manifest['files'])+1,bytes=output.stat().st_size,output=str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    print(json.dumps(export(parser.parse_args().output),indent=2))
