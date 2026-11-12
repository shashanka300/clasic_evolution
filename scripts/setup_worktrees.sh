#!/usr/bin/env bash
# Create the research repo and one git worktree per track defined in RESEARCH_PLAN.md.
#
# Usage:  ./scripts/setup_worktrees.sh [REPO_DIR] [WORKTREE_DIR]
#   REPO_DIR      main repo (default: ./classic-algos)
#   WORKTREE_DIR  where worktrees go (default: ./classic-algos-wt)
#
# Tracks are read from the "<!-- track:ID slug:NAME phase:P deps:... -->" markers in the plan,
# so the plan is the single source of truth. Re-running is safe: existing branches/worktrees are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PLAN="$HERE/RESEARCH_PLAN.md"
REPO="$(mkdir -p "${1:-./classic-algos}" && cd "${1:-./classic-algos}" && pwd)"
WT="$(mkdir -p "${2:-./classic-algos-wt}" && cd "${2:-./classic-algos-wt}" && pwd)"

[ -f "$PLAN" ] || { echo "Plan not found: $PLAN" >&2; exit 1; }

mkdir -p "$HERE/data/raw" "$HERE/data/processed" "$HERE/data/cache"

link_shared() {
  local target_dir="$1"
  [ -d "$target_dir" ] || return 0
  if [ -d "$HERE/.venv" ]; then
    local rel_venv
    rel_venv="$(python3 -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$HERE/.venv" "$target_dir")"
    if [ ! -e "$target_dir/.venv" ] || [ -L "$target_dir/.venv" ]; then
      ln -sfn "$rel_venv" "$target_dir/.venv"
    fi
  fi
  if [ -d "$HERE/data" ]; then
    local rel_data
    rel_data="$(python3 -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$HERE/data" "$target_dir")"
    if [ ! -e "$target_dir/data" ] || [ -L "$target_dir/data" ]; then
      ln -sfn "$rel_data" "$target_dir/data"
    fi
  fi
}

# ---------- 1. main repo skeleton ----------
cd "$REPO"
if [ ! -d .git ]; then
  git init -q -b main
  mkdir -p core tracks results scripts
  cp "$PLAN" RESEARCH_PLAN.md
  cp "$HERE"/scripts/*.sh scripts/ 2>/dev/null || true
  [ -f "$HERE/MAC_AND_DATA_PLAN.md" ] && cp "$HERE/MAC_AND_DATA_PLAN.md" MAC_AND_DATA_PLAN.md
  [ -f "$HERE/pyproject.toml" ] && cp "$HERE/pyproject.toml" pyproject.toml
  [ -f "$HERE/uv.lock" ] && cp "$HERE/uv.lock" uv.lock
  [ -f "$HERE/.python-version" ] && cp "$HERE/.python-version" .python-version
  if [ -d "$HERE/tests" ]; then
    mkdir -p tests
    cp -R "$HERE"/tests/. tests/
  fi
  touch core/__init__.py tracks/.keep results/.keep
  cat > README.md <<'EOF'
# classic-algos
Parallel research on modernizing classic algorithms with the e-means recipe.
See RESEARCH_PLAN.md and MAC_AND_DATA_PLAN.md. One git worktree per track. `core/` is owned by the `core` branch only.
EOF
  cat > .gitignore <<'EOF'
# Note: no trailing slash on data/.venv so symlinks to shared bench dirs are ignored too
data
*.npy
*.npz
*.fvecs
*.bvecs
*.ivecs
*.fbin
*.ibin
*.u8bin
*.hdf5
*.h5
*.parquet
*.pt
*.tar.gz
*.gz
*.zip
*.bz2
__pycache__/
.pytest_cache/
.venv
.DS_Store
# TASK.md is generated per worktree from RESEARCH_PLAN.md; kept untracked so rebases never conflict on it
TASK.md
EOF
  git add -A && git commit -qm "Scaffold research repo"
  echo "Created repo at $REPO"
fi
link_shared "$REPO"

# ---------- 2. extract a track's TASK.md from the plan ----------
extract_task() {   # $1 = track id, e.g. T03
  awk -v id="$1" '
    /<!-- track:/ { on = ($0 ~ "track:" id " ") ; next }
    on && /^## /   { exit }
    on             { print }
  ' "$PLAN"
}

# ---------- 3. create branches + worktrees ----------
BASE="core"
git show-ref -q --verify refs/heads/core || git branch core main
git show-ref -q --verify refs/tags/core-v1 && BASE="core-v1"

if [ ! -d "$WT/core" ]; then
  git worktree add -q "$WT/core" core
  { echo "# TASK — T00 core"; echo; extract_task T00; } > "$WT/core/TASK.md"
  echo "  + core  -> $WT/core"
fi
link_shared "$WT/core"

grep -o '<!-- track:T[0-9]* slug:[a-z0-9-]* phase:[0-9] deps:[^ ]*' "$PLAN" |
while read -r _ id slug phase deps; do
  id="${id#track:}"; slug="${slug#slug:}"; phase="${phase#phase:}"; deps="${deps#deps:}"
  [ "$id" = "T00" ] && continue
  branch="track/$slug"; dir="$WT/$slug"
  if [ -d "$dir" ]; then
    link_shared "$dir"
    echo "  = $slug (exists)"
    continue
  fi

  git show-ref -q --verify "refs/heads/$branch" || git branch "$branch" "$BASE"
  git worktree add -q "$dir" "$branch"
  mkdir -p "$dir/tracks/$slug" "$dir/results/$slug"
  {
    echo "# TASK — $id $slug   (phase $phase, depends on: $deps)"
    [ "$phase" = "2" ] && echo && echo "> **BLOCKED until $deps have results in results/.** Merge their branches before starting."
    echo
    echo "Rules: edit only tracks/$slug/ and results/$slug/. Import core/, never edit it."
    echo "Log one line per day to results/$slug/LOG.md. Results go to results/$slug/*.jsonl (schema in RESEARCH_PLAN.md section 1)."
    echo
    extract_task "$id"
  } > "$dir/TASK.md"
  echo "# $id $slug log" > "$dir/results/$slug/LOG.md"
  touch "$dir/tracks/$slug/__init__.py"
  (cd "$dir" && git add -A && git commit -qm "$id: scaffold $slug")
  link_shared "$dir"
  echo "  + $slug (phase $phase) -> $dir"
done

echo
echo "Done. Branch base: $BASE"
[ "$BASE" = "core" ] && echo "Note: core-v1 is not tagged yet. When T00 is done, tag it and run scripts/sync.sh."
git worktree list
