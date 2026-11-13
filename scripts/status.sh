#!/bin/sh
# Per-track status from the auto-generated VERDICT.md files + job queue state.
cd "$(dirname "$0")/.."
for v in results/*/VERDICT.md; do
  t=$(basename "$(dirname "$v")")
  printf "%-24s %s\n" "$t" "$(grep -m1 'Outcome' "$v" | sed 's/^- //')"
done
echo; echo "running:"; pgrep -fl "^\.venv/bin/python -u" | cut -c1-100
echo; echo "queued: $(grep -c . results/queue.txt 2>/dev/null) jobs"; tail -2 results/queue.log 2>/dev/null
