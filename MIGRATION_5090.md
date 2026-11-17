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

## 3b. As actually run on the desktop (Windows 11 + WSL2 Ubuntu 24.04, Ryzen 9 9900X 24T, RTX 5090) — replaces §3

- Runs from a WSL ext4 clone, `~/clasic_evolution` (origin = GitHub), not from `F:\clasic_evolution`: Linux paths
  as the scripts expect, fast IO, and atomic line appends when two stages of one track write the same JSONL
  (not guaranteed on `/mnt/f`). torch 2.15+cu132 sees the 5090 (sm_120); `pytest` 131 passed.
- `scripts/pqueue.py` + `results/jobs_5090.txt` replace the one-at-a-time `queue.sh`: jobs are packed onto the
  24-thread budget (each gets its own OMP/NUMBA/BLAS thread count), GPU e-means tracks run beside CPU tracks,
  `needs=data` jobs wait for `data/.ready`, T03/T11 run `exclusive` (alone, for latency/QPS), T12 fit and
  evaluate run behind a `barrier`. Restart-safe: finished ids are kept in `results/pqueue_done.txt`.
  ```sh
  cd ~/clasic_evolution && python3 scripts/pqueue.py 2>&1 | tee -a results/pqueue.log   # start / resume (adopts running jobs)
  python3 scripts/pqueue.py --status                                                     # progress
  ```
- Data: downloaded fresh with `.venv/bin/python scripts/fetch_data.py` (parallel downloads -> DINOv2 / MiniLM / BGE
  embeddings on the 5090 -> every track's processed cache built once, so concurrent jobs never race on a `.npz`);
  it touches `data/.ready` on success and the waiting jobs start on the next poll. Embedding caches are built with
  `val_size=0` first, as on the Mac (the cache path ignores `val_size`; DINOv2 train must be 50k).
  Re-embedded data differs from the Mac's in the last bits, but every group on embedding-derived data that the
  queue still runs is computed entirely here (Mac results on those datasets are complete groups), so no group mixes
  the two; raw downloads (SIFT, GloVe, MovieLens, UCI, 20NG, CIFAR pixels, Brown) are byte-identical.
- `wall_s` of jobs that ran side by side includes contention. Verdicts are epoch/compute-based (T09 charges
  seeding in distance evaluations), so no kill criterion depends on it; T03 latency and T11 QPS run alone.
- Utilisation: jobs are split one process per dataset / scenario and the budget is ~1.7x the thread count
  (ported jobs wait on the GPU), giving CPU 100% / GPU ~67% with 17 concurrent jobs. Beyond that the remaining
  limit is the 12-core CPU, not the GPU: the experiments' mini-batches (B = 64-2048) are a studied parameter.
- **Wall-clock for the paper comes only from the isolated timing pass** (phase C2 of `results/jobs_5090.txt`):
  per track, the full-batch reference@25, the untuned law method (`*_kmlaw`) and the val-selected swept method
  (`scripts/timing_swept.py`) re-run alone, CPU only (`CLASSIC_ALGOS_DEVICE=cpu`, no GPU ports), 12 threads, 3 seeds,
  after an untimed warm-up (numba compile). Records go to `results/_timing/<track>/`; `evaluate.py` adds an
  "Isolated timing" section to each VERDICT and labels the main table's column `wall× (queue)`.
- T00 re-run here (`core_t00`) so all tracks share one embedding version; the Mac's T00 records are in
  `results/core/mac_mps_embeddings/`.
- T03 recall@10 is computed with all threads (exact, thread-independent; tested), the timed per-query loops of T03
  and T11 run pinned to one CPU (`CLASSIC_ALGOS_PIN_CPU`, default 2). For the systems paper report scanned codes per
  query (`ndis`) beside p99 latency; T03's frozen kill criterion stays on p99. Close heavy apps before phase C.
- LDA Enron / NYTimes and dictlearn CIFAR-12 are split by stage and, for the big sweep / online stages, by seed
  (`--seed-list`, added to both runners; LDA also gained a standalone `kmlaw` stage). Online workers run with
  `CLASSIC_ALGOS_STEPWISE_CELLS_ONLY=1` (cells only); one final full-seed `online` job then picks (tau0, kappa)
  from all seeds' val loss, exactly as the unsplit stage does. `emit` runs after all sweep seeds.
- `queue.sh` now pops lines portably (BSD `sed -i ''` would re-run line 1 forever on GNU sed).

## 4. Rules that must hold across machines

- Every record carries `extra.hw` (host, GPU, CPU count, torch). `VERDICT.md` warns when a comparison group
  mixes hosts: **wall-clock / speedup columns are only valid within one host.** Epoch-based metrics are portable.
- Latency (T03) and QPS (T11) are measured only on the desktop, alone (`scripts/wait_idle.sh`).
- Kill criteria stay frozen (`scripts/kill_criteria.py`); changes only as logged amendments.

## 5. Phase 2 on the 5090

1. DONE (2026-11-14): hot loops ported to torch CUDA, float64, behind `CLASSIC_ALGOS_GPU_PORTS=gmm,dictlearn,lda,hmm,als`
   (or `all`; off by default; recorded in `extra.hw.gpu_ports`, and `evaluate.py` warns when a group mixes port
   settings). Parity vs the CPU path on fixed seeds: `tests/test_gpu_ports.py` (15 tests). Each call routes by size
   (same results either way) from `scripts/bench_gpu_ports.py`, 5090 vs 4 CPU threads:
   | port | wins | stays on CPU |
   |---|---|---|
   | GMM E-step + stats | diag B>=1024 x24-48, full x6-13 | diag B<1024 |
   | FISTA | B>=4096 x3.9-5 (evaluation chunks) | training mini-batches |
   | LDA E-step | tokens x K >= 2e6: NYTimes x4-5, 20NG 4096 docs x2.9 | small 20NG batches |
   | HMM forward-backward | >=64 seqs x1.3, V=5000 x6-14 | log-likelihood (numba faster) |
   | ALS | row solves x2.3, 2048-user batch x3.4 | item solve from EMA stats |
   Turned on for queue jobs that had not started (gmm_full, als_law, lda_law, lda_coslaw, lda_all).
   **Bug found while porting:** the numba LDA E-step aliased `g`/`newg` under `parallel=True` and stopped every doc
   after 2 iterations; fixed, regression test added, old T05 results moved to
   `results/lda-scaling/superseded_estep_alias_bug/`, all LDA stages rerun (`results/jobs_lda_rerun.txt`).
2. Scale-up runs at the plan's scale, 5 seeds for headline groups: Deep10M (ann-benchmarks deep-image-96, ~3.6 GB),
   SIFT10M/BIGANN, IVF nlist up to 65k, MovieLens-25M, full NYTimes (300k docs).
3. Re-run `scripts/evaluate.py` and `scripts/make_figures.py`; figures read only JSONL.
