#!/usr/bin/env python3
"""Download official recognition models and verify every file, or check installed files."""
import argparse,hashlib,io,json,subprocess,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--proxy');args=p.parse_args()
    archives={}
    for item in json.loads((ROOT/'models/manifest.json').read_text()):
        path=ROOT/item['path']
        if path.is_file() and digest(path.read_bytes())==item['sha256']:
            print('verified',item['path']);continue
        if args.check:raise SystemExit('Missing or hash mismatch: '+item['path'])
        url=item['url']
        def download(url):
            with tempfile.TemporaryDirectory() as directory:
                target=Path(directory)/'download'
                cmd=['curl','-fL','--connect-timeout','10','--max-time','240','--speed-limit','10000','--speed-time','30','-o',str(target)]
                cmd+=['-x',args.proxy] if args.proxy else ['--noproxy','*']
                if 'accept' in item:cmd+=['-H','Accept: '+item['accept']]
                subprocess.run([*cmd,url],check=True)
                return target.read_bytes()
        if 'archive_sha256' in item:
            if url not in archives:
                blob=download(url)
                if digest(blob)!=item['archive_sha256']:raise SystemExit('Archive SHA-256 mismatch: '+url)
                archives[url]=blob
            with tarfile.open(fileobj=io.BytesIO(archives[url])) as archive:
                members=[m for m in archive.getmembers() if m.isfile() and Path(m.name).name==path.name]
                if len(members)!=1:raise SystemExit('Ambiguous model archive: '+path.name)
                data=archive.extractfile(members[0]).read()
        else:
            try:data=download(url)
            except subprocess.CalledProcessError:
                if not item.get('mirror'):raise
                data=download(item['mirror'])
        if digest(data)!=item['sha256']:raise SystemExit('Model SHA-256 mismatch: '+item['path'])
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.partial');temporary.write_bytes(data);temporary.replace(path)
        print('installed',item['path'])


if __name__=='__main__':main()
