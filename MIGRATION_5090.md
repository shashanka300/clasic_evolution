# Migration to the RTX 5090 desktop

Branch: `rework` (worktree `classic-algos-wt/core`). All code needed is committed; results JSONL is committed
as it lands. Data (`data/`, ~3.8 GB) is NOT in git.

## 0. Status at hand-off (written 2026-11-13 18:45 IST)

Done on the Mac, verdicts in `results/*/VERDICT.md`: T00 core, T01 GMM (diag), T06 ALS, T07 HMM, T05 LDA phase 1.
Mac overnight queue (`results/queue.txt`, expected finished ~08:30): T04 dictlearn (running), T02 OPQ, T08 drift,
T10 adaptive-k, T09 seeding, T13 live IVF. Every job resumes from its JSONL, so an unfinished one just continues
on the desktop.

Desktop queue (`results/queue_desktop.txt`): T03 IVF + T11 HNSW (latency/QPS: measured on ONE machine only),
GMM full covariance, law sweeps (GMM, HMM, ALS, LDA, dictlearn), LDA cosine law, T12 fit, final evaluate + figures.

## 1. Morning, on the Mac

```sh
cd classic-algos-wt/core
sh scripts/status.sh                         # what finished
pkill -f "sh scripts/queue.sh"               # stop the queue (the running job can be left to finish or killed;
pkill -f "^\.venv/bin/python -u tracks"      #  it resumes on the desktop either way)
git add results && git commit -m "Mac overnight results"
git push -u origin rework                    # github.com/shashanka300/clasic_evolution (ask before pushing)
# data: copy the directory the worktree's data/ symlink points to (research-classic-algos/data, ~3.8 GB)
rsync -a --progress ../../data/ <desktop>:<repo>/data/      # or USB drive
```
Copy `data/` rather than re-downloading: AG News / DINOv2 / BGE embeddings recomputed on another GPU differ in the
last bits, which would change dataset hashes vs. the Mac runs.

## 2. Desktop setup (Linux or WSL2 assumed)

```sh
git clone -b rework https://github.com/shashanka300/clasic_evolution.git && cd clasic_evolution
ln -s /path/to/data data                     # or put data/ in the repo root
uv sync                                      # Python 3.12; Linux torch wheels ship CUDA 12.x
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- RTX 5090 (Blackwell, sm_120) needs a torch build with CUDA >= 12.8. If `cuda.is_available()` is False or kernels
  fail: `uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`.
- Keep `faiss-cpu`: IVF/HNSW latency tracks are CPU-latency benchmarks by design. faiss-gpu is optional (only for
  training-time speedups later) and its Blackwell support must be checked before use.
- Device selection is automatic (`core/device.py`: cuda > mps > cpu; override `CLASSIC_ALGOS_DEVICE=cpu`).

Verify before launching anything:
```sh
.venv/bin/python -m pytest -q tests                        # expect 131 passed
.venv/bin/python -m core.acceptance --quick --seeds 1       # T00 on GPU; writes results/core (hw-tagged)
.venv/bin/python scripts/evaluate.py core                   # must still SURVIVE
```

## 3. Run the desktop queue

```sh
export NT=$(nproc)                            # thread count used by every queued job
cp results/queue_desktop.txt results/queue.txt
nohup sh scripts/queue.sh > results/queue.log 2>&1 &
sh scripts/status.sh                          # any time
```
Also re-run anything the Mac did not finish (check `results/queue.txt` from the Mac commit; prepend those lines).
Estimated desktop time: CPU-bound numba/numpy tracks scale with core count; with a 16-24 core CPU expect
roughly 12-20 h for the queue as-is, before any GPU porting.

## 4. Rules that must hold across machines

- Every record carries `extra.hw` (host, GPU, CPU count, torch). `VERDICT.md` warns when a comparison group
  mixes hosts: **wall-clock / speedup columns are only valid within one host.** Epoch-based metrics are portable.
- Latency (T03) and QPS (T11) are measured only on the desktop, alone (`scripts/wait_idle.sh`).
- Kill criteria stay frozen (`scripts/kill_criteria.py`); changes only as logged amendments.

## 5. Phase 2 on the 5090 (after the queue)

1. Port hot loops to torch CUDA (est. 1-1.5 days): GMM E-step (`tracks/gmm-ema/gmm.py`), FISTA
   (`tracks/dictlearn-scaling/dictlearn.py`), LDA E-step (`tracks/lda-scaling/lda_v2.py`), HMM forward-backward
   (`tracks/hmm-ema/hmm_v2.py`), batched ALS solves (`tracks/als-ema/als_v2.py`). Each port must reproduce the
   CPU version's numbers on a fixed seed (add a test per port) before it is used for results.
2. Scale-up runs at the plan's scale, 5 seeds for headline groups: Deep10M (ann-benchmarks deep-image-96, ~3.6 GB),
   SIFT10M/BIGANN, IVF nlist up to 65k, MovieLens-25M, full NYTimes (300k docs).
3. Re-run `scripts/evaluate.py` and `scripts/make_figures.py`; figures read only JSONL.
