#!/usr/bin/env python3
"""Fetch the fixed Grove sources/binaries; preserve existing working sources."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    cache = ROOT / '.runtime/downloads'
    cache.mkdir(parents=True, exist_ok=True)
    for entry in json.loads((ROOT / 'g1/sources.lock.json').read_text()):
        archive = cache / (entry['name'] + '.tar.gz')
        if not archive.exists():
            if args.verify_only:
                raise RuntimeError(f'Missing cached archive: {archive}')
            partial = archive.with_suffix('.part')
            command = ['curl', '-fL', '--connect-timeout', '10', '--max-time', '240',
                       '--speed-limit', '50000', '--speed-time', '30',
                       '-H', 'Accept: ' + entry.get('accept', 'application/vnd.github+json'),
                       entry['url'], '-o', str(partial)]
            result = subprocess.run(command + ['-x', 'http://127.0.0.1:17890'])
            if result.returncode:
                # These pinned Grove forks / release assets have no published trusted
                # mainland mirror in the upstream source/dependency configuration.
                print('Official DIRECT failed; no applicable upstream domestic mirror. Trying AUTO.', flush=True)
                subprocess.run(command + ['-x', 'http://127.0.0.1:17892'], check=True)
            partial.replace(archive)
        with archive.open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != entry['sha256']:
            raise RuntimeError(f'SHA-256 mismatch; preserved for inspection: {archive}')
        print(entry['name'] + ': SHA-256 verified', flush=True)
        if args.verify_only:
            continue
        destination = ROOT / entry['destination']
        if destination.exists() and any(destination.iterdir()):
            print('  preserving existing directory: ' + str(destination), flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.fetch-', dir=destination.parent) as temp:
            subprocess.run(['tar', '--no-same-owner', '--strip-components=1', '-xzf', str(archive),
                            '-C', temp], check=True)
            if destination.exists():
                destination.rmdir()  # Only an empty submodule placeholder can reach here.
            Path(temp).rename(destination)
            # TemporaryDirectory tolerates its path having been renamed.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
