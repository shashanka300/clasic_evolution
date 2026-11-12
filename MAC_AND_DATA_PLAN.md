# Data & Mac Execution Plan (`classic-algos`)

This document specifies (1) the verified `uv` benchmark setup, (2) the full dataset acquisition, sizing, and caching plan across `T00`–`T13`, and (3) the execution plan tailored to this Apple Silicon Mac.

---

## 1. Verified Hardware & UV Environment Profile

| Resource | Observed Local Spec | Operational Constraint |
|---|---|---|
| Architecture / Chip | `arm64` — Apple M1 Pro | No CUDA; use Apple Accelerate/NEON + PyTorch MPS |
| CPU Cores | 10 total (8 Performance + 2 Efficiency) | Cap benchmark OpenMP threads to `8` (or `1` for p50/p99 latency) |
| Unified Memory | 32 GB shared between CPU & GPU | Keep peak RSS `<= 22 GB` to avoid NVMe swap |
| Storage | ~385 GiB free on APFS SSD | Tiered dataset budget `~16–24 GB` total; shared `data/` across all worktrees |
| Python / `uv` | `uv 0.8.14`, managed CPython `3.12.11` | System Python is `3.15.0`; always use `.python-version` (`3.12`) via `uv` |
| Shared `.venv` | `./.venv` (~1.5 GB), symlinked into all 14 worktrees | Saves ~19.5 GB vs 14 copies; unified `libomp.dylib` required on macOS |

### Key Mac Compatibility Findings & Mitigations
1. **OpenMP `libomp.dylib` conflict (`torch` + `faiss-cpu` + `scikit-learn`)**:
   - `torch 2.15.0`, `faiss-cpu 1.16.0`, and `scikit-learn 1.9.1` each bundle separate `libomp.dylib` binaries. Importing `torch` and `faiss` together aborts (`OMP: Error #15`), and `KMP_DUPLICATE_LIB_OK=TRUE` segfaults (`exit 139`).
   - **Fix baked into `scripts/setup_uv.sh`:** symlink `faiss/.dylibs/libomp.dylib` and `sklearn/.dylibs/libomp.dylib` to `../../torch/lib/libomp.dylib` so `dyld` loads one OpenMP runtime per process.
2. **`hnswlib==0.8.0` build SDK**:
   - `hnswlib` compiles from source on Python 3.12 arm64. `scripts/setup_uv.sh` exports `SDKROOT=$(xcrun --show-sdk-path)` and selects `uv`'s managed Python 3.12 so `clang++` links cleanly.
3. **Worktree `core/` imports**:
   - Running `uv run python tracks/<slug>/script.py` normally puts `tracks/<slug>/` on `sys.path[0]`. `scripts/setup_uv.sh` installs `_cwd_core.pth` in `.venv` so `import core` automatically resolves to the active worktree's `core/` directory.
4. **Upstream library checks**:
   - `faiss-cpu 1.16.0` does **not** ship built-in `e_means` (`ClusteringParameters` includes `afkmc2_chain_length` and `use_super_kmeans`), so T00 tests against our reference `core/emeans.py` implementation of arXiv 2611.09501.
   - `pyspark` / JDK is **not** installed locally (`java` absent); T06 uses `implicit` ALS + exact NumPy/SciPy ridge-ALS parity + SGD MF.

---

## 2. Data Plan (`data/` shared across all worktrees)

All datasets live in `research-classic-algos/data/` (`raw/`, `processed/`, `cache/`) and are symlinked into `classic-algos/data` and `classic-algos-wt/*/data` so no dataset is duplicated across branches.

### Tier Summary & Disk Footprint

| Tier | Purpose | Estimated Disk | When to Download |
|---|---|---|---|
| **Tier 1 (Core & Fast Dev)** | Phase 0 (`core` acceptance) + fast sweeps across all 11 Phase 1 tracks | **~2.6 GB** | Immediately when implementing `core/bench.py` |
| **Tier 2 (Full Mac Suite)** | Headline 3–5 seed benchmarks + multi-domain scaling laws (`T01`–`T13`) | **~14.5 GB** | After `core-v1` passes acceptance tests |
| **Tier 3 (Memmap / Optional)** | 10M-scale IVF/OPQ stress tests & optional external datasets | **~4.5 GB** | Only for T02/T03/T05 final scale checks |
| **Total (Tiers 1 + 2 + 3)** | Complete program | **~21.6 GB** | Fits easily inside ~385 GiB free SSD |

### Dataset Inventory by Track

#### A. Dense Vector & Embedding Datasets (`T00`, `T01`, `T02`, `T03`, `T08`, `T09`, `T10`, `T11`, `T13`)
| Dataset | Tracks | Shape / Dims | Tier | Disk | Source & Mac Handling |
|---|---|---|---|---|---|
| **SIFT1M** | T00, T01, T02, T03, T08, T09, T10, T11 | `1M × 128` train, `10k` query, `100-NN` GT (`float32`) | Tier 1 | ~500 MB | `ann-benchmarks` (`sift-128-euclidean.hdf5`) or IRISA `sift.tar.gz` (use HTTP HDF5 primary; hold out 100k from train or load `sift_learn` for val). |
| **ann-benchmarks Core Pack** (`glove-100-angular`, `mnist-784-euclidean`, `fashion-mnist-784-euclidean`, `nytimes-256-angular`, `lastfm-64-dot`) | T11 (fit set) | `60k–1.18M × 64–784` (`float32`) | Tier 1 | ~1.3 GB | Direct HTTP HDF5s from `ann-benchmarks.com`. |
| **DINOv2 Embeddings** (`ViT-S/14` 384-d, `ViT-B/14` 768-d) | T00, T01, T02, T08, T09, T10 | `250k × 384` (+ `50k × 768` val/upgrade) (`float32`) | Tier 2 | ~550 MB | Extract locally on MPS (~5 min) from CIFAR-100 / Imagenette / HF `imagenet-1k` val+subset via `torchvision`/`transformers`, cache as `.npy` in `data/cache/`. |
| **MS MARCO Passage Embeddings** (`all-MiniLM-L6-v2` & `BAAI/bge-small-en-v1.5`, 384-d) | T00, T01, T02, T03, T08, T10, T13 | `500k × 384` (both encoders for `100k` slice) (`float32`) | Tier 2 | ~1.1 GB | Embed `BeIR/msmarco` / `sentence-transformers/msmarco` passages on MPS; paired MiniLM → BGE embeddings power T08 scenario (c) encoder upgrade. |
| **GIST1M** | T00, T02, T11 (held-out) | `1M × 960` train, `1k` query (`float32`) | Tier 2 | ~3.6 GB | `gist-960-euclidean.hdf5` (`ann-benchmarks`). Load via `h5py` slicing / `np.memmap`; run 1 job at a time (~8–12 GB peak RAM). |
| **Deep10M (+ Deep1M slice)** | T00, T02, T03, T11 (held-out) | `10M × 96` (`float32`, 3.84 GB) + `1M` dev slice (384 MB) | Tier 2 | ~4.0 GB | `deep-image-96-angular.hdf5` or Yandex `big-ann` `base.10M.fbin`. **Must use `np.memmap` streaming batches** for 10M runs on 32 GB RAM; use `Deep1M` slice for sweeps. |
| **SIFT10M (BigANN prefix)** | T03 (`nlist=65k`) | `10M × 128` (`uint8`) | Tier 3 | ~1.32 GB | **Do NOT download 128 GB BigANN.** Fetch exact first `10,000,000 × 132` bytes of `bigann_base.bvecs` via HTTP `Range` or `big-ann-benchmarks` 10M split; decode batches to `float32` on the fly. |
| **T01 PCA-64/128 Cache + Synthetic GMMs** | T01, T10 | `1M × 64/128` + synthetic | Tier 1/2 | ~500 MB | Derived in `data/cache/` from SIFT1M / DINOv2 / MS MARCO via randomized PCA; synthetic GMMs generated on the fly. |

#### B. Dictionary Learning, Audio & Image Patches (`T04`)
| Dataset | Shape / Size | Tier | Disk | Source & Mac Handling |
|---|---|---|---|---|
| **CIFAR-10 / CIFAR-100 & MNIST** | 60k–120k images → `500k` 8×8 / 16×16 patches | Tier 1 | ~220 MB | `torchvision.datasets`; extract & ZCA/mean-center patches into cached `.npy` (`~250 MB`). |
| **Audio Spectrograms (ESC-50 / Speech Commands v0.02 subset)** | ~2k–10k clips → `200k × 64` log-mel frames | Tier 2 | ~450 MB | `torchaudio` / `librosa` mel-spectrogram frames cached as `float32` `.npy`. |
| **BoW for NMF** | Shares 20NG & NYTimes BoW | Tier 1/2 | 0 extra | Reuse T05 sparse `.npz` matrices. |

#### C. Topic Modeling Corpora (`T05`)
| Dataset | Size / Vocab | Tier | Disk | Source & Mac Handling |
|---|---|---|---|---|
| **20 Newsgroups** | 18,846 docs, ~30k vocab | Tier 1 | ~15 MB | `sklearn.datasets.fetch_20newsgroups` (built-in). |
| **UCI NYTimes BoW** | 300k docs, 102k vocab, ~100M tokens | Tier 2 | ~350 MB | UCI ML Repository `docword.nytimes.txt.gz`; convert once to CSR `.npz` (~400 MB). |
| **Wikipedia English Subset** | `300k` articles, 50k vocab | Tier 2 | ~450 MB | HuggingFace `wikimedia/wikipedia` (`20231101.en`, `streaming=True` first 300k docs) → CSR `.npz`. Avoids downloading 20+ GB full dump. |
| **UCI PubMed Subset** | `500k–1M` doc stratified sample (full = 8.2M) | Tier 3 | ~1.2 GB | Stream `docword.pubmed.txt.gz` and subsample 500k–1M docs for batch-variational baseline parity; full 8.2M optional for online-only scaling verification. |

#### D. Recommender Matrices (`T06`)
| Dataset | Interactions | Tier | Disk | Source & Mac Handling |
|---|---|---|---|---|
| **MovieLens-25M** | 25M ratings, 162k users, 62k items | Tier 1 | ~250 MB zip (~1.2 GB CSV → ~300 MB CSR) | GroupLens `ml-25m.zip`; convert once to `scipy.sparse.csr_matrix` `.npz` for instant loading. |
| **Amazon Reviews 2023 (Electronics / Books 5-core)** | ~10M–30M interactions | Tier 2 | ~1.5 GB | `McAuley-Lab/Amazon-Reviews-2023` on HuggingFace (`parquet`/streaming); keep 5-core dense user/item IDs. |
| **Netflix Prize (Substitution Note)** | 100M ratings | Tier 3 (Optional) | ~700 MB | **Officially withdrawn by Netflix (2010).** Use MovieLens-25M + Amazon Reviews 2023 (+ optional **LastFM-360K / MSD Taste Profile** ~300 MB) as clean public benchmarks; support user-supplied Kaggle `netflix-prize-data` path if present. |

#### E. Sequences & Hidden Markov Models (`T07`)
| Dataset | Sequences / Tokens | Tier | Disk | Source & Mac Handling |
|---|---|---|---|---|
| **Synthetic HMMs** | Known transition/emission params | Tier 1 | 0 MB | Generated in RAM (discrete & Gaussian emissions). |
| **POS Induction (Penn Treebank Substitution Note)** | 3.9k–57k sentences | Tier 1 | ~30 MB | **Full Penn Treebank WSJ requires paid LDC99T42 license.** Primary free benchmarks: **Universal Dependencies `en_ewt`** (~16k sents, ~250k tokens) + **NLTK `brown`** (~57k sents, 1.16M tokens) + **NLTK `treebank` 10% WSJ sample** (3,914 sents). Add optional loader if user supplies local LDC PTB path. |
| **Activity Recognition (UCI HAR + PAMAP2)** | 10k windows (UCI HAR) + 3.8M continuous samples (PAMAP2) | Tier 1/2 | ~60 MB + ~690 MB | UCI ML Repository zips; PAMAP2 provides long continuous multi-subject IMU sequences ideal for Baum–Welch. |

---

## 3. Plan for Running on Mac (Apple M1 Pro, 32 GB)

### Rule 1 — Sequential Timing, Limited Development Concurrency
- **Timed experiments (`wall_s`, `p50`/`p99` latency, `epochs-to-match`) MUST run 1 benchmark process at a time.** CPU and MPS share the same 32 GB unified memory and ~200 GB/s memory controller; concurrent jobs corrupt `wall_s` and risk NVMe swapping.
- **Parallel worktrees are for parallel coding / light unit tests only.** If using `./scripts/launch.sh`, keep `<= 2` active heavy shells and set `THREADS_PER_WORKER=4`:
  ```bash
  THREADS_PER_WORKER=4 ./scripts/launch.sh ./classic-algos-wt 1
  ```

### Rule 2 — Hardware Dispatch & Precision Policy (MPS vs Apple Accelerate CPU)
| Workload | Target Backend | Dtype & Threading Guidance |
|---|---|---|
| Dense mini-batch distance / GEMM (`B >= 1024, d >= 128, k >= 256`), Sinkhorn (T03), embedding extraction (T00/T08) | **PyTorch MPS** | **`float32` only** (MPS raises `TypeError` on `float64`). Always call `torch.mps.synchronize()` around timers! |
| Sufficient-statistic accumulation requiring high dynamic range (`N=1e-10`, T01 full covariance $\Sigma \gamma xx^\top$, T04 Cholesky/normal eqs) | **CPU Accelerate (`float64`)** or centered `float32` residuals | Verify MPS `float32` against CPU `float64` in unit tests to prevent variance/covariance drift. |
| `faiss-cpu` k-means/PQ/OPQ/IVF (T00, T02, T03, T09, T10), `hnswlib` (T11) | **CPU NEON + Accelerate** | Pin `faiss.omp_set_num_threads(8)` (8 P-cores only; mixing 2 slower E-cores hurts barrier-synced loops). |
| Single-query latency benchmarks (`p50`/`p99` in T03, T11) | **CPU single-thread** | Set `faiss.omp_set_num_threads(1)` / `index.set_num_threads(1)` for clean tail-latency curves. |
| Sparse CSR/CSC (`implicit` ALS in T06, LDA BoW in T05), sequential HMM DP (T07) | **CPU (`scipy.sparse`, `implicit`, `numba`)** | MPS has no advantage on irregular sparse/DP loops; Numba JIT (`@numba.njit`) + Accelerate BLAS is fastest. |

### Rule 3 — JSONL Logging Adaptation for Mac (`RESEARCH_PLAN.md` §1)
Preserve the frozen JSONL schema exactly:
```json
{"track": "...", "method": "...", "dataset": "...", "n": 1000000, "d": 128, "k_or_units": 1024, "epochs": 2, "batch": 4096, "alpha0": 0.1, "seed": 0, "train_loss": 0.0, "test_loss": 0.0, "wall_s": 0.0, "gpu_s": 0.0, "extra": {"device": "cpu|mps", "mps_s": 0.0, "cpu_threads": 8, "hw": "m1_pro_32gb"}}
```
- Set `gpu_s: 0.0` (no CUDA) and record MPS device time in `extra.mps_s` so T12 can separate Apple MPS vs CPU timings cleanly without breaking schema parsing.

### Rule 4 — Memory Guardrails for 32 GB Unified Memory
- **10M-vector datasets (`Deep10M`, `SIFT10M`)**: Never materialize full float32 copies + index structures simultaneously (`10M × 128 × 4B = 5.12 GB` raw, `2–3×` with temp buffers). Store as `float32` `.npy` / `uint8` `.bvecs` and stream mini-batches via `np.load(..., mmap_mode="r")` or `np.memmap`.
- **Large sparse matrices (NYTimes, PubMed subset, Amazon 2023)**: Store and slice in `scipy.sparse.csr_matrix` (`int32` indices, `float32` data) and clear intermediate arrays (`del ...; gc.collect()`) between seeds.
- **Thermal consistency**: Insert a brief cooldown (or run 2s warmup before timing each method) during multi-hour sweeps, and run heavy 10M / 5-seed headline suites sequentially in an overnight queue.

### Rule 5 — Recommended Mac Execution Sequence
1. **Phase 0 (`core`, ~1–2 days):** Implement `core/{stats,schedule,revival,finalize,scaling,emeans,bench,plugin}.py`. Download **Tier 1** data (~2.6 GB). Pass SIFT1M + DINOv2/MS MARCO subset acceptance test (`<= ~2.6 epochs` to Lloyd@25) and tag `core-v1`. Sync all tracks via `./scripts/sync.sh`.
2. **Phase 1 — Wave A (Fast CPU/MPS tracks, 1–2 hrs/track):** `T09 cheap-seeding`, `T10 adaptive-k`, `T08 drift-rewarm`, `T01 gmm-ema`, `T07 hmm-ema`. Run 3-seed sweeps on Tier 1 first; promote winners to 5 seeds.
3. **Phase 1 — Wave B (Medium-compute tracks, 2–5 hrs/track):** `T04 dictlearn-scaling`, `T05 lda-scaling`, `T06 als-ema`, `T11 hnsw-scaling-laws`. Download Tier 2 datasets per track just-in-time.
4. **Phase 1 — Wave C (10M memmap / overnight tracks):** `T02 opq-joint-rotation` (`SIFT1M` + `Deep1M` dev, then `Deep10M` memmap), `T03 balanced-ivf` (`SIFT1M` + `Deep10M` + `SIFT10M` prefix).
5. **Week-2 Kill Review:** Enforce `RESEARCH_PLAN.md` kill criteria; write `NEGATIVE.md` for any killed track (still valuable input for `T12`!).
6. **Phase 2 (`T12`, `T13`):** Merge `results/*/*.jsonl` into `T12` (`statsmodels`/`scipy`/`matplotlib` universal scaling law + failure taxonomy) and merge `T03` + `T08` into `T13`.
