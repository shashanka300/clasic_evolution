#!/usr/bin/env bash
# Build a clean, single-commit release tree of the paper's code and results.
#   bash scripts/make_release.sh [OUT_DIR]       (default ../sa-em-release; run from the repo root)
# Keeps: core, the five paper families + law fits, scripts, tests, results, paper, PAPER_LOG, RESEARCH_PLAN.
# Drops: companion-paper tracks (IVF / OPQ / HNSW / drift / seeding / adaptive k), internal machine notes,
#        literature-review notes. Scrubs the local home path from every text file.
set -euo pipefail
OUT=${1:-../sa-em-release}
URL=$(git remote get-url origin); REPO=$(basename -s .git "$URL"); OWNER=$(basename "$(dirname "$URL")")
rm -rf "$OUT"; mkdir -p "$OUT"
git archive HEAD | tar -x -C "$OUT"
cd "$OUT"

COMPANION="adaptive-k balanced-ivf cheap-seeding drift-balanced-ivf drift-rewarm hnsw-scaling-laws opq-joint-rotation"
for t in $COMPANION; do rm -rf "tracks/$t" "results/$t"; done
rm -f tests/test_adaptive.py tests/test_balanced_ivf.py tests/test_drift.py tests/test_drift_balanced_ivf.py \
      tests/test_hnsw.py tests/test_opq.py tests/test_seeding.py
rm -rf MIGRATION_5090.md MAC_AND_DATA_PLAN.md research_notes reports \
       results/jobs_5090.txt results/queue.txt results/queue.log
# job lists and queue logs: drop lines for companion tracks
pat=$(echo $COMPANION | sed 's/ /\\|/g')
rm -f results/figures/ivf_recall_latency.* results/figures/drift_window_mse.*
for f in results/jobs_*.txt results/pqueue_done*.txt results/pqueue*.log results/queue*.txt; do
  if [ -f "$f" ]; then sed -i "/$pat/d" "$f"; fi
done

mv paper/README_release.md README.md
# scrub local paths and the private repo name from every text file
grep -rIl -e "$HOME" -e "$REPO" -e "$OWNER" . | xargs -r sed -i -e "s#$HOME#\$HOME#g" \
  -e "s#$OWNER/$REPO#<repo>#g" -e "s#$REPO#<repo>#g" -e "s#$OWNER#<user>#g"

left=$(grep -rl --exclude=make_release.sh -e "$HOME" -e "$(whoami)" -e "$OWNER" -e "$REPO" -e "@gmail" . || true)
if [ -n "$left" ]; then echo "personal strings remain in:"; echo "$left"; exit 1; fi
echo "release tree: $OUT ($(find . -type f | wc -l) files, $(du -sh . | cut -f1))"
