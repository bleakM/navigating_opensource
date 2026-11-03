#!/usr/bin/env python3
"""Keep the sandbox supervisor alive while the ROS child handles Ctrl-C itself."""
import os
import signal
import sys

signal.signal(signal.SIGINT, signal.SIG_IGN if sys.argv[1] == 'ignore' else signal.SIG_DFL)
os.execvp(sys.argv[2], sys.argv[2:])
