"""Bounded public-source retrieval; writes only beside this script. No credentials."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parent
LIMIT = 12 * 1024 * 1024


def fetch(url, relative_path):
    target = (ROOT / relative_path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError('Destination outside owned directory')
    if target.exists():
        raise FileExistsError(target)
    record = {'retrieved_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'requested_url': url, 'local_path': relative_path,
              'role': 'unclassified_source_response'}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'WheatPhysiologyEvidenceAudit/1.0',
                                                   'Accept': '*/*'})
        with urllib.request.urlopen(req, timeout=35) as response:
            body = response.read(LIMIT + 1)
            if len(body) > LIMIT:
                raise ValueError('Response exceeds 12 MiB; payload not retained')
            final_url = response.url
            parsed = urllib.parse.urlsplit(final_url)
            if 'x-amz-' in parsed.query.lower() or 'signature=' in parsed.query.lower():
                final_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))
                record['temporary_signed_query_omitted'] = True
            record.update(http_status=response.status, final_url=final_url,
                          content_type=response.headers.get('Content-Type'), bytes=len(body),
                          sha256=hashlib.sha256(body).hexdigest(), state='retrieved')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
    except Exception as error:
        record.update(state='failed', error=str(error), http_status=getattr(error, 'code', None))
    with (ROOT / 'retrieval_log.jsonl').open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('path')
    args = parser.parse_args()
    fetch(args.url, args.path)
