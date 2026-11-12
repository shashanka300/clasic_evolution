#!/usr/bin/env bash
# Open one tmux window per worktree and (optionally) start an agent in each.
#
# Usage:  ./scripts/launch.sh [WORKTREE_DIR] [phase]
#   phase: 0, 1, 2 or all (default 1)
#   AGENT_CMD           command to run in each window (default: none, just a shell).
#                       Example: AGENT_CMD='claude "Read TASK.md and RESEARCH_PLAN.md, then execute the task."'
#   GPUS                comma list of GPU ids to round-robin as CUDA_VISIBLE_DEVICES (optional)
#   THREADS_PER_WORKER  cap OpenMP/Accelerate/BLAS threads per window (recommended on Mac: 4)
set -euo pipefail
WT="${1:-./classic-algos-wt}"; PHASE="${2:-1}"; SESSION="classic-algos"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-}"

tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n status
i=0
for dir in "$WT"/*/; do
  [ -f "$dir/TASK.md" ] || continue
  slug="$(basename "$dir")"
  p="$(head -1 "$dir/TASK.md" | sed -n 's/.*(phase \([0-9]\).*/\1/p')"; [ "$slug" = core ] && p=0
  [ "$PHASE" != all ] && [ "$p" != "$PHASE" ] && continue
  tmux new-window -t "$SESSION" -n "$slug" -c "$dir"
  if [ ${#GPU_LIST[@]} -gt 0 ]; then
    tmux send-keys -t "$SESSION:$slug" "export CUDA_VISIBLE_DEVICES=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" C-m
  fi
  if [ -n "${THREADS_PER_WORKER:-}" ]; then
    tmux send-keys -t "$SESSION:$slug" "export OMP_NUM_THREADS=$THREADS_PER_WORKER OPENBLAS_NUM_THREADS=$THREADS_PER_WORKER MKL_NUM_THREADS=$THREADS_PER_WORKER VECLIB_MAXIMUM_THREADS=$THREADS_PER_WORKER NUMEXPR_NUM_THREADS=$THREADS_PER_WORKER" C-m
  fi
  [ -n "${AGENT_CMD:-}" ] && tmux send-keys -t "$SESSION:$slug" "$AGENT_CMD" C-m
  i=$((i+1))
done
echo "Launched $i windows in tmux session '$SESSION'. Attach: tmux attach -t $SESSION"
