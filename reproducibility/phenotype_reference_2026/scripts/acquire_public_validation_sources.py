#!/usr/bin/env python3
"""Acquire a bounded list of public research URLs, retaining receipts and raw bodies.

No automatic retry, credentials, browser challenges, or access-control workarounds.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request


def acquire(job, root, timeout, max_bytes):
    path = root / job['file']
    if path.is_absolute() and not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Output outside acquisition directory')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe output path')
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(job)
    receipt['started_utc'] = datetime.now(timezone.utc).isoformat()
    try:
        request = urllib.request.Request(job['url'], headers={
            'User-Agent': 'AMRReferenceValidation/1.0 (public research data audit)',
            'Accept': job.get('accept', '*/*'),
        })
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError('Response exceeds configured maximum size')
            path.write_bytes(data)
            receipt.update(http_status=response.status, final_url=response.url,
                           content_type=response.headers.get('Content-Type'),
                           content_range=response.headers.get('Content-Range'),
                           bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                           signature_hex=data[:12].hex())
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
    path.with_name(path.name + '.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=25)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-mib', type=int, default=40)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    if len({j['file'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate output filenames')
    args.out.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        receipts = list(executor.map(lambda job: acquire(job, args.out, args.timeout, args.max_mib * 1024**2), jobs))
    (args.out / 'manifest.json').write_text(json.dumps(receipts, indent=2) + '\n')
    for receipt in receipts:
        print(json.dumps({key:receipt[key] for key in ['file','http_status','bytes','error'] if key in receipt}), flush=True)


if __name__ == '__main__':
    main()
