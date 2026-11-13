#!/bin/sh
# Block until no other benchmark python process is running (for latency-sensitive jobs).
while pgrep -f "^\.venv/bin/python -u (tracks|-m core)" >/dev/null; do sleep 60; done
