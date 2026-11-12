#!/usr/bin/env bash
# One-line status per worktree: branch, last commit, latest LOG.md line, number of result rows.
# Usage: ./scripts/status.sh [WORKTREE_DIR]
WT="${1:-./classic-algos-wt}"
printf "%-22s %-10s %-6s %s\n" TRACK COMMIT ROWS "LATEST LOG"
for dir in "$WT"/*/; do
  slug="$(basename "$dir")"
  commit="$(git -C "$dir" log -1 --format=%h 2>/dev/null)"
  rows="$(cat "$dir"/results/"$slug"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')"
  log="$(tail -1 "$dir/results/$slug/LOG.md" 2>/dev/null)"
  [ -f "$dir/results/$slug/NEGATIVE.md" ] && log="[KILLED] $log"
  printf "%-22s %-10s %-6s %s\n" "$slug" "$commit" "$rows" "$log"
done
