#!/usr/bin/env python3
"""One shutdown signal to ROS launch; let launch own its children's shutdown.

Shell Ctrl-C and test harnesses target this supervisor's process group. ROS
launch lives in a separate session, so its children do not get a second SIGINT.
"""
import os
import signal
import subprocess
import sys


def main():
    child=None
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        if not stopping and child is not None and child.poll() is None:
            stopping=True
            os.kill(child.pid,signal.SIGINT)
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    child=subprocess.Popen(sys.argv[1:],start_new_session=True)
    return child.wait()


if __name__=='__main__':raise SystemExit(main())
