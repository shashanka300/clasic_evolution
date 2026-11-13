#!/bin/sh
# Sequential job queue: one command per line in results/queue.txt; done lines move to results/queue_done.txt.
# Run:  nohup sh scripts/queue.sh > results/queue.log 2>&1 &
cd "$(dirname "$0")/.."
while true; do
  line=$(head -n 1 results/queue.txt 2>/dev/null)
  if [ -z "$line" ]; then sleep 60; continue; fi
  sed -i '' '1d' results/queue.txt
  echo "$(date '+%F %T') START $line"
  sh -c "$line"
  echo "$(date '+%F %T') END rc=$? $line"
  echo "$line" >> results/queue_done.txt
done
