from pathlib import Path
import argparse, csv, hashlib, requests

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--name', action='append', default=[])
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    rows=list(csv.DictReader((root/'RAW_METADATA_SOURCES.csv').open()))
    if not args.all and not args.name:
        for row in rows: print(row['file'], row['bytes'], row['url'])
        return
    for row in rows:
        if not args.all and row['file'] not in args.name: continue
        target=root/row['archive_relative_path']
        if target.exists():
            if sha(target)!=row['sha256']: raise RuntimeError('Existing file differs: '+str(target))
            print('Already verified:', target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary=target.with_name(target.name+'.part')
        h=hashlib.sha256()
        with requests.get(row['url'], stream=True, timeout=(10,60)) as response:
            response.raise_for_status()
            with temporary.open('wb') as handle:
                for block in response.iter_content(1024*1024):
                    handle.write(block);h.update(block)
        if h.hexdigest()!=row['sha256']:
            raise RuntimeError('Download hash differs from archived snapshot: '+row['file'])
        temporary.replace(target)
        print('Verified:', target)

if __name__=='__main__': main()
