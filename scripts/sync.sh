#!/usr/bin/env bash
# Rebase every track branch onto the latest core (or a given ref).
# Usage: ./scripts/sync.sh [WORKTREE_DIR] [BASE_REF]   (default base: core)
# Stops on the first conflict so you can resolve it in that worktree.
set -euo pipefail
WT="${1:-./classic-algos-wt}"; BASE="${2:-core}"
for dir in "$WT"/*/; do
  slug="$(basename "$dir")"; [ "$slug" = core ] && continue
  if [ -n "$(git -C "$dir" status --porcelain)" ]; then
    echo "skip $slug: uncommitted changes"; continue
  fi
  echo "rebase $slug onto $BASE"
  git -C "$dir" rebase -q "$BASE" || { echo "CONFLICT in $dir — resolve, then re-run"; exit 1; }
done
echo "All tracks on $BASE."
