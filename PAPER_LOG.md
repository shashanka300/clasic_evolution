# Paper log

Running record of every decision, deviation, bug, result and analysis that the paper(s) will need, so that the
methods, limitations and appendix sections can be written from this file alone. **Append-only**: add dated entries
at the end of each section; never rewrite an earlier entry (mark it *superseded* and point to the new one).
Numbers here are copied from auto-generated files (`results/*/VERDICT.md`, `results/summary.json`,
`results/universal-scaling-law/*.json`); when they disagree, the generated files win.

Related documents: `RESEARCH_PLAN.md` (frozen hypotheses, kill criteria, rules), `MIGRATION_5090.md` (how the
desktop run was set up), `reports/Novel directions for EMA statistics.md` (literature review, 2026-11-14),
`scripts/kill_criteria.py` (frozen criteria + logged amendments).

---

## 1. Experimental setup (methods section)

**2026-11-14 — hardware and software.**
- Desktop: AMD Ryzen 9 9900X (12 cores / 24 threads), NVIDIA RTX 5090 (Blackwell, sm_120), Windows 11 + WSL2
  Ubuntu 24.04, Python 3.12, torch 2.15 + CUDA 13.2, faiss-cpu. Host tag in records: `Dummi2`.
- Earlier results (T00, T01 diag, T06, T07 and the first T05/T04 runs) were produced on an Apple-silicon Mac (MPS);
  those records carry `untagged` / pre-hw-tag hardware. Every record has `extra.hw`; `evaluate.py` warns when a
  comparison group mixes hosts, and **wall-clock comparisons are only made within one host**.
- Data was downloaded fresh on the desktop (`scripts/fetch_data.py`), not copied: SIFT1M / GloVe (ann-benchmarks),
  MovieLens 1M/10M, UCI bag-of-words Enron / NYTimes, UCI HAR, NLTK Brown + universal tagset, 20 Newsgroups,
  CIFAR-100, AG News. Raw downloads are byte-identical to the Mac's; embeddings (DINOv2, MiniLM, BGE) were
  recomputed on the 5090 and differ in the last bits, so T00 was re-run on the desktop and the Mac T00 records are
  archived in `results/core/mac_mps_embeddings/`. No comparison group mixes the two embedding versions.

**2026-11-14 — GPU ports (implementation detail, not a method change).**
- Hot loops of GMM (E-step + statistics), FISTA (dictionary learning), the LDA E-step, HMM forward–backward and ALS
  row solves were ported to torch CUDA in float64, behind `CLASSIC_ALGOS_GPU_PORTS`. Parity with the CPU path on
  fixed seeds is tested (`tests/test_gpu_ports.py`). Calls are routed by size (small batches stay on CPU). The port
  set is recorded in `extra.hw.gpu_ports`.
- Ports change wall-clock only, never the algorithm; epoch-based metrics are unaffected.

**2026-11-14 — how wall-clock numbers for the paper are measured (isolated timing pass).**
- The main queue ran up to ~37 jobs concurrently, so its `wall_s` includes contention; it is labelled
  `wall× (queue)` in the verdicts and is **not** a headline number.
- Headline wall-clock comes only from `results/_timing/<track>/`: the full-batch reference@25, the untuned law
  method (`*_kmlaw`) and the val-selected swept method (`scripts/timing_swept.py`), each re-run alone, CPU only
  (`CLASSIC_ALGOS_DEVICE=cpu`, no GPU ports), 12 threads, 3 seeds, after an untimed warm-up (numba compilation).
- T03 recall is computed with all threads (exact, thread-independent, tested); the timed per-query loops of T03 and
  T11 run alone, pinned to one CPU. Report scanned codes per query (`ndis`) beside p99 latency.

**2026-11-14 — evaluation protocol (as implemented).**
- Epochs-to-match = first epoch at which the seed-mean test curve reaches the seed-mean full-batch reference@25,
  linearly interpolated; gain = 25 / epochs-to-match. 95% CI by seed bootstrap; paired t-test by seed.
- alpha0 swept on the grid of 8 log-spaced values in [0.01, 1] (`core.sweep.ALPHA_GRID`), selected on val per T.
- Baselines selected on val the same way: stepwise / online EM (tau0, kappa) grid.

## 2. Deviations from RESEARCH_PLAN.md (limitations section — every one must be fixed or disclosed)

**2026-11-14 — found while reading the interim verdicts.**
1. **Budgets stop at 5 epochs** (`core.trackkit.BUDGETS = (1, 2, 3, 5)`). Rule 4 asks for loss at 1, 2, 3, 5,
   10, 25 epochs. Consequence: any method needing more than 5 epochs is scored gain 0, so the smallest measurable
   gain is 5x while the kill criteria test 2x (12.5 epochs). **T06 ALS "KILLED, 0.00x" is this artefact**: on
   MovieLens-10M the swept method's 5-epoch RMSE is 0.7925 vs ALS@25 0.7924 (p = 0.67), and its 3-epoch RMSE
   already beats ALS@10. Same truncation affects GMM full covariance and LDA Enron/NYTimes. Planned fix: add
   budgets 10 and 25 (and sub-epoch points) for our methods and the tuned baselines.
2. **"25x" is a ceiling**, reported whenever epoch 1 already beats the reference. The full-batch reference is the
   weak comparator: tuned stepwise / online EM also beats it in 1–4 epochs. The defensible headline is ours vs tuned
   prior art; sub-epoch budgets are needed to separate them when both need 1 epoch.
3. **3 seeds everywhere**; rule 5 asks for 5 seeds for headline numbers.
4. **Datasets narrower than planned**: ALS on MovieLens 1M/10M (plan: 25M, Netflix); HMM on Brown (plan: Penn
   Treebank); LDA without PubMed / Wikipedia; dictionary learning without audio spectrograms.
5. **alpha0 grid floor 0.01 censors LDA**: 20 of 60 LDA sweep cells have their optimum at 0.01 (see §5).
6. Suspected weak baselines to check before publication: tuned online EM with full covariance (GMM) is far behind
   (e.g. AG News 79.6 vs ours 70.7 nats); T11's grid-search "oracle" is worse than hnswlib defaults with tuned
   efSearch at some recall targets, so the grid misses good settings (and T11 has 1 seed).

**2026-11-14 — protocol v3 (fixes deviations 1–3; `core/budgets.py`, `scripts/make_jobs_v3.py`).**
- Budgets become (0.25, 0.5, 1, 2, 3, 5, 10, 25) epochs for every curve-producing method: our swept and untuned-law
  methods, the tuned stepwise / online-EM baselines, LDA's Hoffman default and cosine law, sklearn mini-batch DL
  (integer budgets only: its `max_iter` counts whole epochs). Full-batch references already have 25 iterations.
- Seeds 0–4 for all 19 headline groups (GMM 8, dictionary learning 3, LDA 3, ALS 2, HMM 3); new references and
  library sanity checks for seeds 3–4.
- Search at the long budgets is restricted, frozen before any long run in `results/<track>/long_budget_plan.json`
  (computed from seeds 0–2 at T=5): alpha0 = the four extended-grid values around the T=5 optimum (two below, one
  above; the fitted laws say alpha0 falls with T, ~T^-0.27), and the three best (tau0, kappa) at T=5 for the tuned
  baselines. Short budgets keep the full alpha0 grid (8 values) and full (tau0, kappa) grids (18 or 24).
  Disclose in the paper: at T=10/25 both our method and the baseline get a 3–4-point local search, not the full
  grid; the law fits treat an optimum at either edge of a restricted grid as censored (amendment to the
  pre-registered T12 censoring rule, which only censored the lower edge because the full grid's top is alpha0 = 1).
- All v2 derived records are archived to `results/<track>/superseded_budget_v2/` (`scripts/archive_v3.py`) and
  regenerated; sweep rows and references are kept and extended. The isolated timing pass is re-run on the v3
  budgets (law and swept methods; timed references kept).
- Epochs-to-match can now be below 1 (first point 0.25), so gains above 25x are possible; ties at "1 epoch" in v2
  become distinguishable.
- 2026-11-14: GMM `sift1m_pca64 | full` never had a swept / online-EM search in v2 (only reference and law curves),
  so it has no T=5 anchor. Its whole short-budget search runs in v3, and its long-budget plan is frozen lazily by the
  first long-budget worker after seeds 0–2's short budgets finish (the seeds used are recorded in the plan file's
  `anchor_seeds`). The other 18 groups were frozen up front from seeds 0–2.
- 2026-11-14 (operational, no effect on numbers): the v3 job `fin_kmlaw_gmm_agnews_pca64_diag` wrote seeds 0–3
  (~2.5 min each) and then stalled on seed 4 for 7.9 h (process running, no progress, under ~2x CPU
  oversubscription). Run alone, the same seed/config finishes in 35 s with the same loss range as seeds 0–3, so it was
  a stall, not a property of the config. The process was killed and seed 4 was rerun alone, same code and settings
  (note in `results/pqueue_v3_notes.log`). Its test-NLL curve (79.86 → 78.74) matches seeds 0–3. Wall times inside
  `kmlaw` rows are from a contended machine anyway; the paper's timing numbers come only from the isolated timing pass.
- 2026-11-15 (operational): from ~00:10 UTC no job finished for 6 h. Windows had 0.3 of 61.6 GB free: WSL was capped
  at 56 GB and full GPU memory (31/32.6 GB, 13 concurrent ALS ml10m jobs) spills into Windows RAM. WSL was shut down
  (the 27 running jobs were killed; finished cells are kept, run_sweep skips recorded configs), the WSL cap lowered to
  44 GB (`~/.wslconfig`, backup `.wslconfig.bak-20261115`), and the queue relaunched at budget 32 / max 16 jobs
  (was 56 / 28). No results files were corrupted (every jsonl line parses).
- 2026-11-15: **LDA seeds 3–4 on grid-extension alphas.** The alpha0 < 0.01 values from `extend_grid.py` (§5) were run
  for seeds 0–2 only; `select_alpha` keeps only seeds that have every alpha in a cell, so 7 enron/nytimes cells
  (T ≤ 5) dropped seeds 3–4. `tracks/lda-scaling/fill_seeds.py` runs the missing (alpha, seed) pairs (job
  `fill_lda_<corpus>`); all swept records are then regenerated from the complete cells by `scripts/reemit_v3.py` (job
  `v3_reemit`, before the timing pass, which reads its alpha selection from the same cells).
- 2026-11-15 (user decision): **the v3 isolated timing pass times the short budgets only (T ≤ 5).** It sets
  `CLASSIC_ALGOS_BUDGETS=0.25,…,5` for the law-curve and swept timing runs (`timing_swept.py`, `stage_kmlaw` honour
  `core.budgets.active()`). Reason: the queue was running a day late; timing T = 10 and 25 would have added several
  hours of serial runs. Consequence for the paper: wall-clock speed-ups exist only for methods that match the reference
  within 5 epochs; matches at 10–25 epochs are reported in epochs only (the reference's own 25-epoch wall time comes
  from the kept timing references). Accuracy results (main runs) still cover all budgets.
- 2026-11-15: a stale WSL VM process from before the restart (`vmmemWSL`, 7 GB RAM / 41 GB commit) could not be
  stopped without admin rights; it keeps Windows memory tight until the next reboot. No effect on results.
- 2026-11-15: LDA `_budget_curve` (hoffman default, cosine law) now resumes per seed; it used to skip a whole group as
  soon as one seed was recorded (hoffman nytimes had seeds 0–1 when the WSL restart killed it).
- 2026-11-16: **17 HMM jobs crashed with `CUDA_ERROR_UNKNOWN`** (2026-11-15 ~18:00 UTC, while Windows had < 1 GB RAM
  free; GPU forward-backward `_fb_torch`): sweep / stepwise cells for HMM synthetic (seeds 0–4) and brown_pos
  (seeds 0–2), plus `fin_online_hmm_synthetic`. The v3 queue still ran its finals, so the first v3 evaluation (00:49
  UTC) had `hmm_ema_swept` with 2 seeds on synthetic and brown_pos, no `stepwise_em_tuned` for synthetic, and a brown_pos
  tuned baseline selected from incomplete cells. Fix (`results/jobs_hmm_fix_v3.txt`, all rc=0, CUDA healthy after the
  load dropped): re-ran the 16 cell jobs (recorded configs skipped), moved the brown_pos tuned rows to
  `results/hmm-ema/superseded_cuda_crash.jsonl` (`scripts/drop_rows.py`) and re-selected, re-ran the synthetic tuned
  selection, re-emitted HMM swept records (`reemit_v3.py hmm-ema`), re-timed the HMM swept method, re-evaluated. Every
  HMM method now has 5 seeds. No other track had a failed job.
- 2026-11-16: **ALS ml10m, T = 10, alpha0 = 0.518, seed 2 was missing** (the only incomplete sweep cell in any track;
  killed in the 2026-11-15 WSL restart and not re-run because the job's other configs were recorded). It made
  `als_ema_swept` ml10m a 4-seed record. Filled by `results/jobs_als_fix_v3.txt`, then ALS re-emit, re-time, re-evaluate.

## 3. Bugs found and their effect on results

**2026-11-14 — LDA E-step aliasing (numba `parallel=True`).** The numba LDA E-step hoisted an allocation so that
`g` and `newg` aliased; every document stopped after 2 E-step iterations instead of iterating to tolerance. Found
because the new GPU port disagreed with the CPU path (serial vs parallel iteration counts 2 vs 11–50). Fixed,
regression test added; all LDA results before the fix moved to `results/lda-scaling/superseded_estep_alias_bug/`
and every LDA stage was re-run. Any LDA number from before 2026-11-14 is invalid.

**2026-11-14 — make_figures.py**: matplotlib renamed `boxplot(labels=)` to `tick_labels=`; the last figure
(`law_penalties`) failed in the final queue step. Fixed and regenerated; no data affected.

**2026-11-15 — scheduler adoption (scripts/pqueue.py).** On restart the queue adopts still-running jobs by matching
the process argv. v3 jobs of one (group, seed) differ only in the env var `CLASSIC_ALGOS_BUDGETS` (short / 10 / 25), so
one process was adopted under 2–3 job ids, and all of them were recorded as done when it exited. On the 2026-11-14
restarts this hit 73 jobs (GMM sift1m diag+full, dinov2 full, synthetic full; dictlearn cifar_p8 seed 0). Effect: the
tuned online-EM / stepwise baselines and swept records of those 5 groups were computed from incomplete cells (e.g.
gmm_ema_swept had 4 seeds for three groups). Fix: adoption also matches the job's env and never reuses a pid. The 73
jobs and the 10 dependent finals were re-queued (cells already recorded are skipped); the stale tuned rows were
moved to `results/<track>/superseded_adopt_bug.jsonl`, and every swept record is regenerated by `v3_reemit`.

## 4. Results snapshot

**2026-11-16 — protocol v3 final verdicts** (budgets 0.25–25 epochs, 5 seeds; `results/evaluate_all.log`). Supersedes the
2026-11-14 v2 snapshot below. Gain = 25 / epochs-to-match the full-batch reference@25 (seed means).

| track | outcome | headline (v3) |
|---|---|---|
| T00 core | SURVIVES | 4 datasets reach Lloyd@25 in <= 3 epochs |
| T01 gmm-ema | SURVIVES | 0/4 datasets need > 8 epochs (diag); full covariance needs 2–9.6 epochs (swept) / 2.9–23 (law) |
| T02 opq-joint-rotation | SURVIVES | 6/6 groups |
| T03 balanced-ivf | SURVIVES (marginal) | median p99@R0.9 +10.1% vs required +10% |
| T04 dictlearn-scaling | SURVIVES | median gain 30.1x (cifar_p12 30.1x, cifar_p8 100x = matched at the first budget 0.25 ep, nmf_20ng 10.8x) |
| T05 lda-scaling | SURVIVES | fitted cosine law worse than default online LDA on 0/3 corpora |
| T06 als-ema | SURVIVES (marginal, see below) | median gain 2.14x vs 2.0x required: ml10m 4.28x (swept), ml1m 0x |
| T07 hmm-ema | SURVIVES | median gain 38.0x (brown_pos 39.0x, synthetic 38.0x, uci_har 14.6x); no significantly worse final optimum |
| T08 drift-rewarm | SURVIVES | 3/4 scenarios |
| T09 cheap-seeding | SURVIVES | 15 budget-charged wins |
| T10 adaptive-k | SURVIVES | max CV 1.8% |
| T11 hnsw-scaling-laws | KILLED | 1/6 held-out pairs |
| T12 universal-scaling-law | SURVIVES | LOCO median penalty M1 0.108% vs M0 0.319% (Wilcoxon p = 1.05e-18); LOAO 0.338% vs 0.440% |
| T13 drift-balanced-ivf | descriptive | no kill criterion |

What changed from v2 (budget cap 5 -> 25, 3 -> 5 seeds):
- **ALS is no longer killed.** With a 25-epoch budget, ml10m swept EMA reaches ALS-WR@25 at 5.84 epochs (4.28x, 5 seeds after the seed-2 fill; 4.25x with 4 seeds),
  the untuned law at 8.9 (2.8x), tuned stepwise ALS only at 20.1 (1.25x). On ml1m nothing reaches the reference before
  25 epochs; all three online methods end within 0.0005 RMSE of it (swept 0.8656 = reference; p = 0.21), i.e. ml1m
  ALS-WR is already converged and the "0x" is a tie, not a loss. The pre-registered median (2.14x) passes only
  narrowly: report per-dataset numbers, not the median.
- GMM full covariance needs 2–10 epochs (swept) — the v2 "not reached within 5 epochs" was the budget cap. The tuned
  online-EM baseline is very weak on full covariance (ends 10 nats above the reference on sift1m) — still a weak
  baseline (§7, step 3).
- LDA: the k-means law still does not transfer (never reaches batch VB@25). Swept EMA (alpha0 0.005–0.01) reaches it
  on 20ng (1.13 ep, 22x) and enron (4.4 ep); on nytimes no online method reaches batch VB@25 (3167) within 25 epochs —
  swept EMA ends best (3242 vs tuned online LDA 3317, Hoffman default 3615).
- HMM: the untuned law reaches Baum–Welch@25 in 0.95–2.2 epochs, before tuned stepwise EM (1.3–5.7), but on the v3
  seeds it **ends slightly above** tuned stepwise EM on all three datasets (brown 5.215 vs 5.209, synthetic 3.927 vs
  3.889, uci_har 2.228 vs 2.205 nats/token; each within ~1 seed-std). **This corrects the v2 note below** ("law ends
  lower than tuned stepwise EM") — do not use that claim. Swept EMA ends lowest on all three (5.206, 3.854, 2.108).
  Claim: the law gets there faster without tuning; the tuned recipe also ends best.
- T12 law: the pre-registered test passes, but held-out-algorithm (LOAO) penalties stay large for LDA (M1 27.7%) and
  HMM (4.5%). The exploratory "observations per cell" refit (§5, `refit_coords.log`, 422 cells) on v3 data: Qe0
  LOAO median 0.143% (vs M1 0.289%), LDA 11.9%, HMM 1.5%; Qe1 LOCO 0.072%. Median swept alpha0*: ALS 0.21, dictlearn
  0.26, GMM 0.32, HMM 1.0, LDA 0.029 — LDA's alpha* is ~10x below the others (in q_eff/alpha* terms 6.8 vs 19–148).

**Isolated timing v3 (CPU only, 12 threads, alone, 3 seeds; T ≤ 5 only, §2)** — wall-clock speed-up to reach the
reference@25 vs the reference's own 25 iterations (`—` = no match within 5 epochs):

| group | ref@25 (s) | swept | untuned law |
|---|---|---|---|
| dictlearn cifar_p8, k=128 | 184 | 82x (0.25 ep) | 16.4x |
| dictlearn cifar_p12, k=192 | 161 | 30.8x | 10.1x |
| dictlearn nmf_20ng, k=32 | 1.85 | 6.5x | 7.3x |
| HMM brown_pos | 59.4 | 32.9x | 25.3x |
| HMM synthetic | 1.55 | 24.7x | 13.9x |
| HMM uci_har_vq | 0.68 | 9.9x | 7.5x |
| GMM synthetic diag | 8.8 | 17.6x | — |
| GMM sift1m_pca64 diag | 25.0 | 10.6x | 9.2x |
| GMM dinov2_pca64 diag | 5.4 | 8.7x | 6.6x |
| GMM agnews_pca64 diag | 10.7 | 6.1x | 7.5x |
| GMM synthetic full | 20.9 | 5.4x | 6.2x |
| LDA 20ng | 6.0 | 6.8x | — |
| GMM full (agnews, dinov2, sift1m), LDA enron/nytimes, ALS ml1m/ml10m | | — (match > 5 ep or never) | — |

**2026-11-14 04:48 UTC — final verdicts of the 5090 queue (protocol v2, superseded)** (`results/evaluate_all.log`):

| track | outcome | headline |
|---|---|---|
| T00 core | SURVIVES | 4 datasets reach Lloyd@25 in <= 3 epochs |
| T01 gmm-ema | SURVIVES | 0/4 datasets need > 8 epochs (diag: 2.16, 1.91, 1.00, 1.00) |
| T02 opq-joint-rotation | SURVIVES | joint rotation significantly better in 6/6 groups |
| T03 balanced-ivf | SURVIVES (marginal) | median p99@R0.9 +10.1% vs required +10% |
| T04 dictlearn-scaling | SURVIVES | gain capped at 25x (cifar_p12, cifar_p8), 13.9x nmf_20ng |
| T05 lda-scaling | SURVIVES | fitted cosine law worse than default online LDA on 0/3 corpora |
| T06 als-ema | KILLED (artefact, §2.1) | 0.00x: nothing reaches ALS@25 within 5 epochs |
| T07 hmm-ema | SURVIVES | gain capped at 25x (brown, synthetic), 14.4x uci_har |
| T08 drift-rewarm | SURVIVES | re-warm beats tuned constant alpha in 3/4 scenarios |
| T09 cheap-seeding | SURVIVES | 15 budget-charged wins over random init |
| T10 adaptive-k | SURVIVES | max CV of selected k 1.8% (threshold 20%) |
| T11 hnsw-scaling-laws | KILLED | law wins 1/6 held-out (dataset, recall) pairs; beats plain defaults, loses to defaults + tuned efSearch |
| T12 universal-scaling-law | SURVIVES (pre-registered test) | LOCO median penalty M1 0.080% vs M0 0.280% (Wilcoxon p = 6.5e-18); LOAO 0.225% vs 0.312% |
| T13 drift-balanced-ivf | descriptive | no kill criterion |

**Isolated timing (headline wall-clock; CPU only, 12 threads, alone, 3 seeds)** — time to reach the full-batch
reference@25 loss, as a speed-up over the reference's own 25-iteration time:

| group | reference@25 (s) | swept method | untuned law method |
|---|---|---|---|
| dictlearn cifar_p12, k=192 | 161 | 26.2x (1 epoch) | 10.6x (2.78 ep) |
| dictlearn cifar_p8, k=128 | 184 | 22.5x (1 epoch) | 17.4x (1.68 ep) |
| dictlearn nmf_20ng, k=32 | 1.85 | 8.0x | 7.7x |
| HMM brown_pos, k=12 | 59.4 | 23.8x (1 epoch) | 24.3x (1 epoch) |
| HMM synthetic, k=12 | 1.55 | 19.0x | 11.4x |
| HMM uci_har_vq, k=12 | 0.68 | 10.5x | 8.0x |
| GMM sift1m_pca64 diag | 25.0 | 11.0x | 10.2x |
| GMM dinov2_pca64 diag | 5.41 | 9.4x | 6.9x |
| GMM agnews_pca64 diag | 10.7 | 7.0x | 7.6x |
| GMM synthetic full | 20.9 | 5.9x | 6.8x |
| LDA 20ng, k=50 | 6.0 | 6.8x | not reached |
| LDA enron / nytimes, ALS ml1m / ml10m, GMM full (agnews, dinov2, sift1m) | — | not reached within 5 epochs (§2.1) | |

Interpretation notes for the paper:
- Untuned law vs tuned prior art (HMM): the k-means-law method ends lower than tuned stepwise EM on all three HMM
  datasets (brown 5.226 vs 5.237, synthetic 3.927 vs 3.935, uci_har 2.191 vs 2.271 nats/token).
- The k-means alpha0 law transfers to GMM, dictionary learning, HMM and ALS (median val-loss penalty <= 0.2%) but
  **not to LDA**: `lda_ema_kmlaw` (alpha0 ~0.85–1) gets worse with more epochs (20ng perplexity 3468 -> 3821),
  while the swept optimum is ~0.01–0.04. The LDA-specific fitted cosine law fixes it (beats default online LDA on
  3/3 corpora, tuned online LDA on 2/3).

## 5. Exploratory analyses (not pre-registered — label as such in the paper)

**2026-11-14 — refit of the alpha0 law in "observations per statistic cell" coordinates**
(`tracks/universal-scaling-law/refit_coords.py`, output `results/universal-scaling-law/refit_coords.json`).
- Motivation: literature review (§6) — the step size should depend on how many observations each sufficient-
  statistic cell receives per step, not on n/k. Per-step observations per cell q: GMM B/k; dictionary learning
  B·(code density); ALS B·(ratings/user)/items; LDA and HMM B·(tokens per doc or sequence)/(K·V). q_eff uses the
  effective vocabulary exp(H(unigram)) instead of V.
- Data: 342 sweep cells (GMM 100, HMM 84, dictlearn 60, LDA 60, ALS 38); 23 censored at alpha0 = 0.01.
- Score: val-loss penalty of the predicted alpha0 vs the swept optimum; leave-one-cell-out (LOCO) and
  leave-one-algorithm-out (LOAO).

| model | LOCO median | LOAO median | LOAO penalty on LDA | one global law: R² of log alpha0* |
|---|---|---|---|---|
| k-means law (unchanged) | 0.149% | 0.149% | 18.1% (predicts 0.19 vs best ~0.04) | — |
| M1: n/k, B, T (+ per-alg constant) | 0.089% | 0.265% | 30.6% | 0.17 (global constant) |
| Q1: q, n/B, T (+ per-alg constant) | 0.072% | 0.235% | 6.7% | 0.37 (global constant) |
| Qe1: q_eff, n/B, T (+ per-alg constant) | 0.076% | 0.154% | 5.7% (predicts 0.069) | **0.56** (global constant) |

- Global fit Qe0: alpha0* = 0.438 · q_eff^0.357 · (n/B)^-0.243 · T^-0.256.
- Findings: the per-cell coordinate explains most of the LDA gap (sparsity of topic-word cells) and triples
  explained variance; **but it is not a universal law**: the exponent on q_eff is 0.36, not 1 (so "a fixed number of
  observations per cell held in the EMA" is rejected — held observations range from ~6 for LDA to ~150 for
  dictionary learning), and HMM wants the maximum rate alpha0 = 1 despite sparse cells (~1 observation per step),
  which raises its LOAO penalty from 0.1% to 1.4%. Candidate next feature: the missing-information fraction
  (theory predicts larger optimal steps when more information is missing).
- Caveat: LDA's side of the fit is incomplete while 20/60 LDA cells are censored at the grid floor (see next entry).

**2026-11-14 — LDA alpha0 grid extension below 0.01** (`tracks/lda-scaling/extend_grid.py`). Started 2026-11-14,
7 parallel workers (20ng ×1, enron ×2, nytimes ×4), LDA GPU port on, same seeds as each cell. Continues the grid's
log spacing downward (0.00518, 0.00268, 0.00139, then 0.00072, 0.000373 only while a cell's lowest alpha is still
best), only for the 20 censored cells (20ng 2, enron 6, nytimes 12). Records are appended to
`results/lda-scaling/sweep.jsonl` in the normal sweep format. Note for the paper: the headline `lda_ema_swept` and
`cosine_law` records were computed on the [0.01, 1] grid; if the extension moves their selected alpha0 they must be
re-emitted. *Result: see the next entry.*

**2026-11-14 — LDA grid extension: result, and the refit re-run on uncensored data.**
- All 20 censored cells now have an interior optimum: 14 at 0.01 (it was the true optimum, only unconfirmed) and 6
  one step lower at 0.00518 (20ng 0, enron 2, nytimes 4, one of them a head cell); 0.00268 was never best. Cells still censored at the lower edge overall: 3 (all GMM), down from 23. LDA's median alpha0* stays 0.031.
- Headline cells (k=50, B=256, full n): only **NYTimes T=5** changes, 0.01 -> 0.00518, val perplexity
  3258.8 -> 3251.1 (-0.24%). Every other headline choice is unchanged, so `lda_ema_swept` moves by one point at
  most; `cosine_law` (fitted on the law-sweep cells, several of which moved) will change slightly when re-fitted.
- Refit (`refit_coords.py`, same 342 cells, LDA now fully usable):

| model | LOCO median | LOAO median | LOAO penalty on LDA | one global law: R² of log alpha0* |
|---|---|---|---|---|
| k-means law (unchanged) | 0.149% | 0.149% | 18.8% | — |
| M1: n/k, B, T | 0.088% | 0.286% | 30.6% | 0.22 (global constant) |
| Q1: q, n/B, T | 0.067% | 0.244% | 6.8% | 0.49 (global constant) |
| Qe1: q_eff, n/B, T | 0.070% | 0.190% | 5.9% | **0.66** (global constant) |
| Qe0: one global law, q_eff | 0.106% | 0.154% | 12.8% | 0.66 |

- Global fit Qe0 (uncensored): alpha0* = 0.414 · q_eff^0.422 · (n/B)^-0.279 · T^-0.268.
- Conclusions unchanged in direction, stronger in size: the per-cell coordinate explains two-thirds of the
  variance of log alpha0* with one global law (vs 22% for n/k), and cuts the cost of predicting LDA from data that
  excludes LDA from 18.8% to 5.9%. Still not universal: exponent on q_eff 0.42 (not 1), and HMM (optimum at the grid
  top, alpha0 = 1, with ~1 observation per cell per step) is predicted worse than by the k-means law
  (LOAO 1.6–1.8% vs 0.1%). Next candidate feature: missing-information fraction.

## 6. Positioning and related work (from the literature review, 2026-11-14)

Full report: `reports/Novel directions for EMA statistics.md`; notes per area in
`research_notes/Novel directions for EMA statistics/`.
- The e-means paper: "e-means: outperforming k-means (in one-tenth the epochs)", Bessa, Mazaré, Szilvasy, Touvron,
  Picard, Douze, Jégou (Meta FAIR), arXiv:2611.09501, 6 Nov 2026. Its law was fitted on one dataset (DEEP). It does
  not appear to cite the online-EM lineage below (to be confirmed from the full text).
- **The update rule is prior art**: EMA of sufficient statistics = stepwise / online EM (Sato & Ishii 2000;
  Cappé & Moulines 2009; Liang & Klein 2009, who already showed 2–10x over batch EM across four model families and
  that the best step "depends on the task and dataset") = SVI's natural-gradient step (Hoffman et al. 2013).
  Per-family online methods: Hoffman et al. 2010 (LDA), Mairal et al. 2010 (dictionary learning / NMF), Nguyen,
  Forbes & McLachlan 2019/2020 (mini-batch EM for exponential-family mixtures; lists tuning step and batch size as
  future work).
- Not found in the literature (limited searches — confirm before claiming): a cross-family step/batch law; cosine
  schedules for EM-family models; one dead-component detector and revival rule across models; a cross-family
  time-to-batch-EM benchmark; differentially private mini-batch EMA k-means.
- Expected reviewer objections: "stepwise EM renamed"; Liang & Klein already showed generality; the law is not
  universal by our own LDA result; laws fitted on one dataset; speed-ups are regime-dependent; missing baselines
  (Flash-KMeans, Flash-GMM, variance-reduced EM, tuned stepwise EM as the main baseline).
- Recommended framing: a predictive theory of stochastic-approximation EM at scale (noise-scale / per-cell law,
  failure modes, time-to-baseline benchmark), with stepwise EM named as the lineage and main baseline.

## 7. Open items before writing

**2026-11-16 — paper metadata started:** `paper/METADATA.md` (titles, abstract draft v0, contributions with
evidence and status, setup, headline tables, figure inventory, limitations, open items, reproducibility) for the
flagship paper, from the v3 results at commit 342cd1c. Items below that v3 closed: budgets 10/25 + sub-epoch and
5 seeds (done, §2 protocol v3); LDA re-emit and re-fit (done by `v3_reemit` and the v3 queue); commit (done, 342cd1c).

- [x] LDA grid extension result (§5) -> `refit_coords.py` re-run (2026-11-14).
- [ ] Re-emit `lda_ema_swept` (NYTimes T=5 changed) and re-run the LDA `cosine-law` stage, the LDA timing-swept
      pass, `fit_v2.py` and `evaluate.py`; archive the superseded records first.
- [ ] Budgets 10 and 25 (+ sub-epoch) for ours and tuned baselines; 5 seeds for headline groups (§2.1–2.3).
- [ ] Check the weak-baseline suspicions (§2.6).
- [ ] Missing-information feature for the law (§5).
- [ ] Novelty checks listed at the end of the literature review report.
- [ ] Commit: results live in the WSL clone `~/clasic_evolution`; code changes in `F:\clasic_evolution`.
