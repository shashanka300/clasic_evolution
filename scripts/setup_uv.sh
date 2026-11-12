#!/usr/bin/env bash
# Sync the shared uv Python 3.12 environment, apply macOS Apple Silicon fixes
# (SDKROOT for hnswlib, unified libomp.dylib across torch/faiss/sklearn, and
# CWD-aware core import hook), and link .venv + data/ into repo and worktrees.
#
# Usage: ./scripts/setup_uv.sh [--verify]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
VERIFY=0
[ "${1:-}" = "--verify" ] && VERIFY=1

# Resolve shared .venv and bench root even if invoked from inside a worktree.
if [ -L "$HERE/.venv" ]; then
  TARGET="$(readlink "$HERE/.venv")"
  case "$TARGET" in
    /*) VENV_DIR="$TARGET" ;;
    *)  VENV_DIR="$(cd "$HERE/$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")" ;;
  esac
  BENCH_ROOT="$(cd "$(dirname "$VENV_DIR")" && pwd)"
else
  BENCH_ROOT="$HERE"
  VENV_DIR="$HERE/.venv"
fi

SDK_FLAGS=()
if [ "$(uname -s)" = "Darwin" ]; then
  SDK_PATH="$(xcrun --show-sdk-path 2>/dev/null || echo /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)"
  export SDKROOT="$SDK_PATH"
fi

echo "Syncing uv environment at: $VENV_DIR (project: $HERE)"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --project "$HERE" --python-preference only-managed

# Install CWD-aware core/ path hook so `uv run python tracks/<slug>/script.py`
# resolves `import core` from the active worktree without PYTHONPATH=.
for sp in "$VENV_DIR"/lib/python3.*/site-packages; do
  [ -d "$sp" ] || continue
  cat > "$sp/_cwd_core.pth" <<'EOF'
import os, sys; p = os.getcwd(); sys.path.insert(0, p) if os.path.isdir(os.path.join(p, "core")) and p not in sys.path else None
EOF

  # Deduplicate OpenMP runtimes on macOS so torch + faiss + sklearn never abort
  # with OMP Error #15 or segfault when loaded in the same process.
  if [ "$(uname -s)" = "Darwin" ] && [ -f "$sp/torch/lib/libomp.dylib" ]; then
    for dup in "$sp/faiss/.dylibs/libomp.dylib" "$sp/sklearn/.dylibs/libomp.dylib"; do
      if [ -e "$dup" ] && [ ! -L "$dup" ]; then
        ln -sf ../../torch/lib/libomp.dylib "$dup"
        echo "Unified OpenMP runtime: $dup -> ../../torch/lib/libomp.dylib"
      fi
    done
  fi
done

mkdir -p "$BENCH_ROOT/data/raw" "$BENCH_ROOT/data/processed" "$BENCH_ROOT/data/cache"

link_shared() {
  local target_dir="$1"
  [ -d "$target_dir" ] || return 0
  local rel_venv rel_data
  rel_venv="$(python3 -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$VENV_DIR" "$target_dir")"
  rel_data="$(python3 -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$BENCH_ROOT/data" "$target_dir")"
  if [ ! -e "$target_dir/.venv" ] || [ -L "$target_dir/.venv" ]; then
    ln -sfn "$rel_venv" "$target_dir/.venv"
  fi
  if [ ! -e "$target_dir/data" ] || [ -L "$target_dir/data" ]; then
    ln -sfn "$rel_data" "$target_dir/data"
  fi
}

link_shared "$BENCH_ROOT/classic-algos"
if [ -d "$BENCH_ROOT/classic-algos-wt" ]; then
  for wt in "$BENCH_ROOT/classic-algos-wt"/*/; do
    [ -d "$wt" ] && link_shared "${wt%/}"
  done
fi

if [ "$VERIFY" = "1" ]; then
  echo "Running environment verification tests..."
  UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv run --project "$HERE" --no-sync pytest
fi
