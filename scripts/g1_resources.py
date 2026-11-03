#!/usr/bin/env python3
"""Observe RTAB RSS and stop the launch before exhausting host memory."""
import argparse
import json
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch-pid', type=int, required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--rss-limit-mib', type=float, default=12288)
    parser.add_argument('--min-available-mib', type=float, default=3072)
    args = parser.parse_args()
    stopped = False
    report = {'passed': False, 'launch_pid': args.launch_pid, 'rss_limit_mib': args.rss_limit_mib,
              'minimum_host_available_mib': args.min_available_mib,
              'max_rtabmap_rss_mib': 0., 'samples': []}

    def stop(*_):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    def persist():
        path = Path(args.report)
        partial = path.with_suffix('.partial')
        partial.write_text(json.dumps(report, indent=2)+'\n')
        partial.replace(path)

    began, seen = time.monotonic(), False
    last_persist = 0.
    missing_since = None
    try:
        while not stopped:
            pid = None
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
                    if argv[0] != b'/opt/ros/jazzy/lib/rtabmap_slam/rtabmap':
                        continue
                    parent = int(proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[1])
                    if parent == args.launch_pid:
                        pid = int(proc.name)
                        break
                except (OSError, ValueError):
                    continue
            elapsed = time.monotonic()-began
            status = None
            if pid is not None:
                try:
                    status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
                    if 'VmRSS' not in status:
                        pid = None
                except OSError:
                    pid = None
            if pid is None:
                missing_since = missing_since or time.monotonic()
                # During normal shutdown RTAB and this observer receive SIGINT
                # separately. Allow teardown signal delivery before rejecting a
                # vanished process; the launch's exit callback still stops it.
                if (seen and time.monotonic()-missing_since > 2) or elapsed > 240:
                    raise RuntimeError('RTAB-Map process disappeared or did not start')
            else:
                missing_since = None
                memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
                rss = int(status['VmRSS'].split()[0])/1024
                available = int(memory['MemAvailable'].split()[0])/1024
                seen = True
                report['rtabmap_pid'] = pid
                report['max_rtabmap_rss_mib'] = max(report['max_rtabmap_rss_mib'], rss)
                report['samples'].append([elapsed, rss, available])
                if rss > args.rss_limit_mib or available < args.min_available_mib:
                    raise RuntimeError(f'Memory budget exceeded: RTAB RSS={rss:.1f} MiB, host available={available:.1f} MiB')
            if time.monotonic()-last_persist > 2:
                report.update(status='observing', wall_seconds=elapsed)
                persist()
                last_persist = time.monotonic()
            time.sleep(.5)
        report.update(status='stopped', passed=seen and len(report['samples']) >= 2)
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
    finally:
        report['wall_seconds'] = time.monotonic()-began
        persist()
    print(json.dumps({k:v for k,v in report.items() if k != 'samples'}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
