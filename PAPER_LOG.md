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
- 2026-11-16: **the tuned stepwise / online-EM baselines ran without revival; the recipe runs with it.** Found while
  checking the "weak GMM baseline" suspicion (§2.6). Diagnostic (`scratchpad oemdiag.py / revdiag.py`, seed 0 or
  seeds 0–2, best tuned (tau0, kappa) and best swept alpha0 at T = 1 and 5; test loss):

  | group | tuned stepwise, no revival → with revival | cosine recipe, no revival → with revival |
  |---|---|---|
  | GMM SIFT1M full, T=5 | 50.35 → **40.84** (57 of 64 components dead without revival) | 51.18 → **40.84** |
  | GMM SIFT1M full, T=1 | 51.46 → 42.56 | 51.51 → 42.17 |
  | GMM DINOv2 diag, T=5 | 71.78 → 71.66 | 71.78 → 71.69 |
  | LDA 20NG, T=5 (perplexity) | 2404 → 2309 | 2701 → 2297 |
  | HMM synthetic / Brown / UCI HAR, T=5 | 3.935 / 5.237 / 2.271 (revival: no effect) | 3.906 / 5.220 / 2.143 (no effect) |
  | DL 20NG NMF, T=5 | 0.44761 (no effect) | 0.44752 (no effect) |
  | ALS | neither uses revival | — |

  Consequence: for GMM and LDA the "ours vs tuned stepwise EM" gap is mostly the revival rule, not the cosine
  schedule; for HMM, dictionary learning (NMF) and ALS it is the schedule. The v3 claim "reaches the reference sooner
  than tuned stepwise EM on 17/19" is therefore confounded for GMM/LDA. Fix: a second tuned baseline,
  **`<stepwise>_rev_tuned` = the same (tau0, kappa) grid and val selection plus the recipe's revival rule**
  (`core/trackkit.stage_stepwise(revival=True)`, `gmm run_v2 stage_online_em(revival=True)`, stage `online_rev`), run for
  GMM, dictionary learning, LDA and HMM (not ALS) at all v3 budgets with 5 seeds (`scripts/make_jobs_rev.py`,
  `results/jobs_rev_v3.txt`, 273 jobs, started 2026-11-16 03:29 UTC). Its long-budget top-3 grid is frozen from its own
  T = 5 cells over all 5 seeds (the no-revival baseline froze from seeds 0–2) — disclose. The original
  no-revival baselines are kept: they are the published prior art; the new one is the schedule-vs-revival ablation.
  **Done 2026-11-16 16:08 UTC**: 273/273 jobs rc=0, every `*_rev_tuned` cell has 5 seeds, `rev_evaluate` re-ran
  evaluate.py + make_figures.py (VERDICT.md now lists `*_rev_tuned`). Results in §4 "Schedule vs revival ablation".

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

### Schedule vs revival ablation (v3, 2026-11-16; ablation, not pre-registered)

Recipe (`*_swept`) vs tuned stepwise/online EM **with revival** (`*_rev_tuned`), both α0 / (τ0, κ) selected on
val, 5 seeds. Paired t-test by seed (4 dof, p < 0.05) at T ∈ {1, 2, 3, 5, 10, 25}; W = recipe significantly lower,
L = significantly higher. Script: `scripts/compare_rev.py`.

| family | W / tie / L (cells) | relative margin (recipe − rev_tuned) | notes |
|---|---|---|---|
| GMM (8 groups) | 26 / 22 / 0 | −0.03% … −1.5% | real-data full cov: W only at T ≤ 2, tie after; synthetic diag W at every T |
| HMM (3) | 14 / 4 / 0 | −0.05% … −10.3% | revival never fires; rev_tuned = tuned |
| DL (3) | 11 / 7 / 0 | −0.03% … −0.30% | revival never fires; rev_tuned = tuned |
| LDA (3) | 0 / 14 / 4 | −0.8% … +3.2% | L: 20NG T=1 (+3.2%), Enron T=1 (+0.6%) and T=25 (+2.5%), NYTimes T=25 (+1.4%) |

Epochs to reach full-batch@25, rev_tuned (swept in brackets): GMM synth diag 3.94 (0.29), synth full 23.0 (1.98),
SIFT1M diag 1.64 (0.88), SIFT1M full 11.8 (7.86; 10.4 (7.24) after the 2026-11-17 reference fill), DINOv2 diag 11.8 (1.88), DINOv2 full 9.01 (8.78), AG News diag 4.49
(2.54), AG News full 16.0 (9.63); LDA 20NG **0.89** (1.13), Enron 7.17 (4.44), NYTimes not reached (final **3198**
vs swept 3242, ref 3167); HMM and DL unchanged from no-revival tuned (UCI HAR 6.73 vs 5.70; noise in the selection).
Recipe sooner on 16/19 (ALS compared with no-revival stepwise ALS as before), slower on LDA 20NG, neither on
NYTimes / ML-1M.

Revival effect on the tuned baseline at T = 25: SIFT1M full 50.68 → 40.53 nats, DINOv2 full 69.64 → 54.61,
AG News full 78.81 → 70.37; LDA NYTimes 3317 → 3198, Enron 2401 → 2240, 20NG 2410 → 2320.

Interpretation for the paper:
- Revival is the component that makes stochastic EM competitive on GMM full covariance and LDA; the old
  "beats tuned stepwise EM" claim was mostly revival there.
- With revival on both sides, the cosine schedule still wins modestly on GMM, clearly on HMM, marginally on DL,
  and **not on LDA**, where a tuned inverse-time schedule + revival is as good or better (also at long budgets,
  where the recipe's test perplexity rises from T = 5 to T = 25: overfitting or cosine floor too high).
- So: claim C2 holds outside LDA; LDA is the honest counter-example, consistent with the LDA α0 anomaly (C4, C6).

### Tuned-baseline timing (2026-11-17; closes "wall-clock for the tuned baselines")

`scripts/timing_tuned.py` re-runs the tuned prior-art baselines (`*_tuned`, `*_rev_tuned`; ALS `stepwise_als_tuned`
only) alone under the v3 timing conditions (CPU, 12 threads, one job, untimed warm-up, 3 seeds, T ∈ {0.25, 0.5, 1, 2,
3, 5}), using the val-selected (τ0, κ) per T from the main runs. Records go to `results/_timing/<track>/runs.jsonl`
with `timing_pass=True`; log `results/timing_tuned.log` (108 seed-runs, no errors, 19:22–20:29 UTC).

Wall-clock speed-up to reach full-batch@25 (reference@25 time / time to match; — = not reached by T = 5):

| group | swept | kmlaw | tuned | rev_tuned |
|---|---|---|---|---|
| DL cifar_p8 | 82.0x | 16.4x | 36.8x | 36.4x |
| DL cifar_p12 | 30.8x | 10.1x | 15.4x | 15.4x |
| DL nmf_20ng | 6.5x | 7.3x | 5.4x | 5.5x |
| HMM brown_pos | 32.9x | 25.3x | 16.8x | 16.6x |
| HMM synthetic | 24.7x | 13.9x | 8.9x | 8.5x |
| HMM uci_har_vq | 9.9x | 7.5x | 4.2x | — (gap at T=5 +0.05%) |
| GMM synthetic diag | 17.6x | — | — | 9.2x |
| GMM sift1m diag | 10.6x | 9.2x | — | 7.3x |
| GMM dinov2 diag | 8.6x | 6.6x | — | — |
| GMM agnews diag | 6.1x | 7.5x | — | 4.3x |
| GMM synthetic full | 5.4x | 6.2x | — | — |
| LDA 20ng | 6.8x | — | 5.2x | 9.5x |
| GMM full (real), LDA enron/nytimes, ALS | — | — | — | — |

Per-epoch cost: at T = 2 the tuned baselines' wall time is within ±5% of the recipe's on all 19 groups (e.g. DL
cifar_p8 17.4 vs 17.5 s, LDA nytimes 27.4 vs 27.5 s, GMM sift1m full 18.2 vs 18.3 s). So wall-clock ranks the methods
as epochs do; the recipe's EMA adds no measurable cost. Recipe (swept) faster than rev_tuned on 11/12 timed groups
(including the three where rev_tuned does not match by T = 5); slower only on LDA 20ng, as in epochs.
Revival never fires for HMM and DL (tuned ≈ rev_tuned, differences are timing noise).
Summary JSON: `results/_timing/timing_tuned_summary.json`.

Interpretation for the paper: most of the speed-up over full batch belongs to stochastic approximation + revival;
the recipe's schedule and law add a further 1.2–2.9x in wall time where both match (DL nmf_20ng 1.2x, DL cifar_p8
2.3x, HMM brown 2.0x, HMM synthetic 2.9x) except on LDA (0.7x).

### Revival on rare, isolated groups (2026-11-17; ablation, not pre-registered)

Question (raised by Weiss 2026, arXiv:2611.18238, for k-means): does the recipe's count-based revival (re-seed any
unit with EMA mass < 0.1·mean every 5 batches) delete components that cover rare, isolated groups?

Script `tracks/gmm-ema/revival_rare.py`. GMM recipe (cosine EMA, B = 1024, ε = 0.1) with three revival modes:
- **off:** no revival.
- **count:** the recipe's rule.
- **dist:** Weiss's rule adapted to GMMs. A count-flagged component is re-seeded only if
  min_i ‖μ_j − μ_i‖² ≤ 4·MSE, with MSE = Σ_j w_j tr Σ_j, or if it has had < 0.5 points of batch mass for 2 epochs.
  Implemented as a `protect` mask on `revive_gmm_components`; the default None leaves the recipe unchanged, and the
  22 GMM tests pass.

Setup:
- α0 from {0.072, 0.139, 0.268, 0.518, 1.0}, selected on validation NLL per (dataset, method, T). Seeds 0–2,
  T ∈ {2, 5}.
- Reference: sklearn full-batch EM, 25 iterations.
- Tail NLL = mean test NLL on rare groups: the synthetic small clusters, or labelled classes with < 2% of training
  points.
- Datasets:
  - synthetic far-groups (d = 16; 30 large clusters + 4 at radius 30 with 0.2% mass each; diag and full, k = 34);
  - Statlog Shuttle (OpenML 40685);
  - KDD Cup 99 10% (38 numeric columns);
  - Covertype;
  - SIFT1M-PCA64 (control, no rare groups);
  - real data standardised; k = 64, diagonal covariance.
- **Deviation:** KDD99 and Covertype use a covariance floor of 0.01 (standardised units) instead of the recipe's
  1e-6 relative floor. With the recipe floor, components collapse onto binary or constant columns and the NLL
  diverges (KDD99 full-batch test NLL 1.7e5, Covertype −246). The first KDD99 and Covertype runs were discarded.
  Logs are in `results/gmm-ema/revival_rare_logs/`, results in `revival_rare.jsonl` (546 rows) and
  `revival_rare.summary.json`.

Test NLL, mean ± sd over 3 seeds, at the val-selected α0 (tail NLL in brackets; lower is better):

| dataset | T | off | count (recipe) | dist (Weiss) |
|---|---|---|---|---|
| SIFT1M control | 2 | 60.02 ± 0.14 | 59.30 ± 0.02 | 59.30 ± 0.02 (identical) |
| SIFT1M control | 5 | 59.92 ± 0.10 | 59.21 ± 0.02 | 59.21 ± 0.02 (identical) |
| synthetic diag | 2 | 27.36 [61.5] | 27.00 [53.8] | 26.98 [53.8] |
| synthetic diag | 5 | 27.40 [62.0] | **26.32** [53.9] | 26.48 [53.9] |
| synthetic full | 2 | 26.48 [34.3] | 26.56 [36.3] | 26.50 [34.7] |
| synthetic full | 5 | 26.47 [34.4] | 26.54 [36.3] | 26.49 [34.8] |
| Shuttle | 2 | −8.35 [19.7] | −11.11 [16.8] | −10.99 [21.8] |
| Shuttle | 5 | −9.18 [16.8] | −12.14 [19.5 ± 5.0] | −12.11 [14.9 ± 1.4] |
| KDD99 | 2 | −45.82 [2.4 ± 7.1] | −46.58 [5.2 ± 4.6] | **−47.07** [**−12.0** ± 4.6] |
| KDD99 | 5 | **−47.41** [**−24.9** ± 1.2] | −46.27 ± 0.48 [15.4 ± 18.7] | −46.93 [−6.9 ± 4.7] |
| Covertype | 2 | −45.29 [−43.25] | −45.25 [−41.59 ± 0.37] | **−45.57** [−43.05] |
| Covertype | 5 | −45.24 [−43.25] | −45.20 [−41.44 ± 0.34] | **−45.47** [−42.29] |

References@25 (test [tail]): SIFT 59.27; synthetic diag 26.55 [55.3], full 26.25 [32.0]; Shuttle −8.11 [17.8];
KDD99 −32.15 [784]; Covertype −45.49 [−43.72]. On KDD99, full-batch EM is stuck far above every stochastic run.

Reading:
- **The caveat is real but data-dependent.**
  - Where rare isolated groups exist, count-based revival costs tail likelihood: Covertype 1.7–1.8 nats on rare
    classes at both budgets; KDD99 tail +3 to +40 nats vs off; synthetic full covariance 2 nats.
  - On KDD99 at T = 5 it is also worse overall than no revival (−46.27 vs −47.41).
  - The distance-aware rule removes most of that cost. It is the best overall on KDD99 T=2 and Covertype, matches
    count on Shuttle and synthetic diag (count better at synthetic diag T=5, 26.32 vs 26.48), and keeps the
    overall gains of revival where they exist (Shuttle −12.1 vs −9.2 off).
- **Our benchmark is unaffected:** on SIFT1M the distance test never fires and the results are bit-identical, as
  Weiss reports for embedding data. The paper's numbers stand.
- **Revival still matters:** off is clearly worse on SIFT1M, the synthetic diag data and Shuttle.
- **Caveats:** 3 seeds; tail sets are small (Shuttle 39 points, KDD99 367, Covertype 431); KDD99 tail variance is
  large; only GMM was tested (HMM / LDA / DL not tested).
- **For the paper:** limitation 8 plus Appendix F; recommend the distance-aware rule for data with rare isolated
  groups.

### Title, abstract and release plan (2026-11-17; decision log)

- Title kept provisionally as METADATA §1 option 1 (the draft's current title); author to confirm.
- Abstract now ends: "Code, job files and all 44,197 run records are released." (67 JSONL files.)
- `paper/RELEASE_PLAN.md`: release code, jobs, JSONL, verdicts, figures; not raw data (MovieLens forbids
  redistribution; AG News / CIFAR unlicensed), rebuilt by `scripts/fetch_data.py`. Console logs contain the local
  home path (~25k occurrences, 199 files) and must be scrubbed; JSONL is clean. Anonymised snapshot for review,
  public repo + Zenodo DOI at arXiv/camera-ready. Open author decisions: licence; same repo vs fresh one.
- Appendix outline reordered (E before F).

### TMLR target and LaTeX manuscript (2026-11-17; decision)

- Venue: TMLR (claims-and-evidence criteria; double-blind; OpenReview; supplementary ZIP up to 100 MB; no fees).
- `paper/tmlr/main.tex` + `references.bib` converted from `paper/draft.md`; builds with tectonic (15 pages incl.
  appendices A–F), no undefined references. Falls back to article+natbib until `tmlr.sty` / `tmlr.bst` are added
  (download from github.com/JmlrOrg/tmlr-style-file pending author OK).
- New in the LaTeX version vs draft.md: Appendix C (repairs, incl. the SIFT1M reference fill), D (full law table;
  states that the unchanged k-means law's LOAO 0.155% is close to Qe0's 0.149%), F (rare-group revival table).
  Appendix B (grids) is still a TODO.
- Checklist and likely reviewer questions: `paper/tmlr/SUBMISSION_CHECKLIST.md`.

### TMLR package completed (2026-11-17)

- Official TMLR style added (`tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` from github.com/JmlrOrg/tmlr-style-file, Apache-2.0;
  downloaded with author OK). Abstract merged into one paragraph (template rule). Builds with no undefined
  references and no overfull boxes.
- Appendix B written from the code: α0 grid (8 log-spaced on [0.01, 1], LDA extended by 5 values to 3.7e-4),
  stepwise grids (κ 0.5–1.0; τ0 {2, 10, 100}, GMM {1, 10, 100, 1000}), head batch sizes, long-budget plan,
  law-fitting cells, ALS λ.
- References: author lists from arXiv for e-means (first author Swann Bessa), Weiss (Eyal Weiss), Flash-KMeans
  (first author Shuo Yang; the old entry "Yang, Xi" was wrong), Flash-GMM; Hoffman 2010 order set to the
  proceedings' (Hoffman, Bach, Blei); Sculley, ANN-Benchmarks, Faiss, DINOv2 checked. 24 entries verified online.
- §5.5 now states that the unchanged k-means law's LOAO median (0.155%) is close to Qe0's (0.149%) and better on
  HMMs; the per-cell law's gain is concentrated on LDA.
- Supplementary ZIP (5.4 MB, 1,309 files) built from the release export; `make_release.sh` now also drops the
  companion figures from `paper/figs`. Scanned: no user, machine, repo or employer names.
- Left to the author: names / affiliation, OpenReview profiles, employer clearance, title, upload.

### SIFT1M full-covariance reference filled to 5 seeds (2026-11-17; deviation repaired)

Found while checking Table 1 against `results/summary.json`: the GMM `sift1m_pca64 | full` reference
(`em_full_batch`, and the `sklearn_em` check) had only seeds 3–4. v2 never ran this group's seeds 0–2 (§2,
2026-11-14 note) and v3 only added seeds 3–4, as for every group. Every other group has 5 reference seeds.
Ran seeds 0–2 with the same v3 command (`run_v2.py --stage reference`, GPU port for GMM, as seeds 3–4); logs
`results/gmm-ema/console_v3_sift1m_pca64_full_ref{0,1,2}.log`, no errors. Effect (5 vs 2 seeds):
- reference@25 40.671 ± 0.104 → 40.685 ± 0.076 nats;
- epochs to match: swept 7.86 → 7.24, law 16.1 → 14.8, tuned + revival 11.8 → 10.4; tuned online EM still never;
- no change to Table 2 (recipe vs rev_tuned, no reference involved), the law fits, or any count in the abstract.
Draft Table 1 / Table 3, METADATA §4 updated; §5.1 now says real-data full-covariance GMMs take 7–10 epochs.

### Release snapshot (2026-11-17)

- Licence MIT (`LICENSE`, copyright "the authors" until names are set). Fresh repo, single commit: private
  `sa-em-scaling-laws`, commit e195097, 1,300 files, 54 MB, built by `scripts/make_release.sh`.
- Export keeps core, the five families, law fits, scripts, tests, results, paper, PAPER_LOG, RESEARCH_PLAN. It
  drops the companion-paper tracks (OPQ, balanced/drift IVF, drift re-warm, seeding, adaptive k, HNSW) with their
  tests, results, figures and job lines, plus internal machine notes and the literature-review notes.
  RESEARCH_PLAN and PAPER_LOG still name the companion tracks (pre-registration kept intact).
- Scrubbed: local home path, the hostname (in 14,422+ run records' `host` field and verdict tables; now
  "desktop"), old macOS checkout paths in two console logs, and the private repo/owner name. Final check:
  no username, hostname, home path, `/Users/` or email string left.
- Verified on the export: `evaluate.py`, `make_figures.py` and `refit_coords.py` run cleanly; verdict tables
  differ from the shipped ones only in the "Generated at" line; Qe0 reproduces (0.419, 0.427, −0.282, −0.326,
  R² 0.659); pytest 71 passed, 15 GPU tests deselected.
- Found while building: the working repo's verdict tables were stale (missing the `*_rev_tuned` rows); refreshed
  in the working repo (a7d4696). Paper numbers came from `compare_rev.py` and are unaffected.

### Title chosen (2026-11-17; author decision)

- Title: **"When Running Averages Are Enough: Few-Epoch EM for Mixtures, Sequences, Topics and Factorizations"**.
  Supersedes the provisional "Scaling laws for stochastic-approximation EM: ..." (entry "Title, abstract and
  release plan" above).
- Reason: the headline result is matching full-batch EM in a few epochs, and the law adds little over the
  unchanged k-means law (§5.5). "Scaling laws" would overclaim. "When" signals that the paper maps where the
  recipe works, including the LDA exception.
- Updated in `paper/tmlr/main.tex` (PDF rebuilt), `paper/draft.md`, `paper/METADATA.md` §1,
  `paper/README_release.md`, `LICENSE`, and the TMLR checklist.
- PDF layout fixes (author reported "some issue with the pdf"; found by rendering every page):
  - the title broke as "Factor-izations"; it now uses explicit line breaks;
  - all five figures had floated past the appendices to pages 14–16, because Figure 1 is page-sized and `[t]`
    only. Figure 1 is now `[p]`, the others `[tbp]`, and `placeins` keeps floats inside their section. Figures
    are now on pages 6 and 9–10, inside Results;
  - Figure 1 was never cited in the text; §5.1 now cites it;
  - the Appendix B α0-grid line was badly stretched (an unbreakable math list); it is now plain text.
  16 pages, no errors, no overfull boxes.

### Framing: lead with our results, not e-means (2026-11-17; author decision)

- The author asked that the paper not open with e-means. Abstract now opens: "How many passes over the data does
  EM need? We show that across five latent-variable families, a few are enough." The method is described
  directly; e-means is no longer named in the abstract (the k-means rate law is described as "a rate law fitted
  for k-means").
- Introduction: paragraph 2 now states our claim and lists the three additions. A single sentence credits e-means
  (Bessa et al., 2026) with proposing them together for k-means. Credit is kept because TMLR rejects incorrect
  novelty claims. The question is reframed as which parts belong to SA-EM in general, which are family-specific,
  and whether one law serves all families.
- Discussion heading "What transfers from e-means, and what does not" → "What carries across families, and what
  does not". Conclusion opens "Within the scope we tested, a few epochs of stochastic-approximation EM are enough."
- No numbers changed. `paper/draft.md` (older Markdown) keeps the old framing; `main.tex` is canonical.

### Methodology review and revision (2026-11-17)

The author supplied a 31-point review judged against Vinod Chandra & Anand Hareendran, *Research Methodology*:
readability 5/10 for a general ML reader. The author approved parts 1–4 of the plan (consistency fixes, statistics,
rewrite, figures).

**New analyses** (`scripts/review_stats.py`, output `results/review_stats.json`; from existing records only, no
new runs):
- **Multiple comparisons** over the 102 (configuration, budget) pairs, ours vs tuned stepwise EM + revival.
  Results are wins / ties / losses:

  | | GMM | HMM | DL | LDA | all |
  |---|---|---|---|---|---|
  | uncorrected | 26/22/0 | 14/4/0 | 11/7/0 | 0/14/4 | 51/47/4 |
  | Benjamini–Hochberg (FDR 5%) | 19/29/0 | 13/5/0 | 10/8/0 | 0/16/2 | 42/58/2 |
  | Holm (FWER 5%) | 0/48/0 | 1/17/0 | 2/16/0 | 0/18/0 | 3/99/0 |
  | all 5 seeds agree (ours / baseline) | 30/0 | 16/0 | 10/0 | 0/5 | 56/5 |

  - With 5 seeds no exact rank or sign-flip test can reach p < 0.05 on one pair (min two-sided p = 2/32), so the
    seed-agreement count stands in for a per-pair Wilcoxon test.
  - **Claim changed**: the paper now leads with the BH count (42 wins / 2 losses), reports Holm (3 wins), and says
    the evidence is the consistent direction, not per-pair significance.
- **Best-iterate reference.** On every GMM, HMM, DL and ALS configuration the full-batch curve is still improving
  or flat at iteration 25, so batch@25 is also the best iterate. On LDA, batch VB is best at iteration 9 (20NG), 11
  (Enron) and 15 (NYTimes), then rises 1.35% / 0.72% / 0.15%. Against the best iterate:
  - our method needs 1.38 epochs on 20NG (was 1.13);
  - our method does not match Enron within 25 epochs (was 4.44);
  - tuned + revival needs 0.95 on 20NG (was 0.89) and 15.0 on Enron (was 7.17);
  - nothing else changes, and the "12 within 3 epochs" claim holds.
- **#29, Enron timing.** It was timed, but on the 3 timing seeds our method does not reach batch@25 within 5 epochs
  (the 5-seed mean is 4.44). The text now says so; before, it said Enron was "not timed".

**Manuscript changes (`paper/tmlr/main.tex`, 20 pages):**
- **Abstract:** about 8 numbers, no undefined jargon; the exploratory 28% → 12% result dropped.
- **New sections and boxes:**
  - new §2, Background (sufficient statistics, E/M-steps with a GMM example, stepwise EM, EMA, SVI, notation);
  - Algorithm 1 (full method) and a cost-per-step paragraph (overhead about d/B of the E-step for a full GMM);
  - intuition for the law exponents (positive on q, negative on n/B and T; unit-free).
- **Terms:**
  - Table 3 defines our method, the law variant, tuned stepwise EM (+ revival), batch@25, epochs-to-match, gain,
    configuration, pair, sweep cell, LOCO/LOFO and law penalty;
  - "LOAO" renamed LOFO everywhere, including Figure 4;
  - C1–C6/C2b labels replaced by section references;
  - the law models M0/M1/Q0/Q1/Qe0/Qe1 are now defined in §3.1, and the law table moved from Appendix D to §6.5.
- **Hypotheses:** Table 4 lists the pre-registered hypotheses with pass criteria and outcomes, from
  `scripts/kill_criteria.py`. The LDA test concerns the LDA-fitted cosine law, now mentioned in §6.4.
- **Statistics paragraph:** bootstrap CIs, paired t-tests (4 dof), BH/Holm, seed agreement, and the Wilcoxon
  signed-rank test for the law test.
- **#15:** pre-registered vs refit M1 numbers (0.108 / 0.103) are explained once, in §6.5 and Appendix D.
- **Related Work** rewritten as prose; the protocol opens with a prose paragraph; the LDA paragraph in §6.2 has
  fewer numbers.
- **Limitations** consolidated into 7 items. They now include the reference definition, seeds and test
  assumptions, and sub-sampled law cells. Repairs are only pointed to (Appendix C).
- **Conclusion:** adds 5 future-work items.
- American spelling throughout. Appendix F (new) holds the full 19-panel convergence figure. The missing-information
  test moved to Appendix D.

**Figures (`scripts/make_figures.py`):**
- `convergence_main` is a new 6-panel Figure 1 (SIFT1M full, DINOv2 diag, Brown POS, LDA 20NG, CIFAR 12x12,
  ML-10M) with y-axis labels. The full grid drops the 4 k-means panels and moves to Appendix F.
- Target line: red dotted, labeled "target: full-batch EM after 25 iterations".
- Method names unified ("ours, α0 tuned on validation", "ours, α0 from the k-means law (untuned)", "tuned
  stepwise EM", "+ revival").
- Math notation in the axis labels; Figure 4 medians shown to 3 decimals and labeled "leave one family out".

**Not yet done:** the supplementary ZIP and the private release repo still have the old manuscript; `draft.md` is
superseded and not updated.

### Literature cross-check before submission (2026-11-27)

New or uncited work found since the TMLR package (searched alphaXiv, arXiv and the web). The manuscript is not yet
changed.

- **Fuzzy e-Means** (Tiwari & Rathore, arXiv 2611.29241, 2026-11-22). Transfers the e-means optimizer to fuzzy
  c-means (soft weights, the closest relative of our GMM track). Findings:
  - the moving average is essential;
  - the k-means laws transfer without refitting on standard data, but not in low dimension;
  - revival on soft masses must be removed: on KDD Cup 99 it costs 26–46%, and even a hard-count trigger does not
    fix it;
  - the cosine schedule barely matters.

  This is concurrent and independent evidence for our transfer claim and our Appendix E rare-group finding (same
  dataset, KDD99). It weakens the "e-means tested no other model" framing only if we imply nobody else has: we must
  cite it as concurrent work.
- **Two Counterexamples for e-means** (Assylbekov, arXiv 2611.22065, 2026-11-17). Proves that mass-based revival,
  which is our rule, can destroy a global optimum, with a loss ratio to Lloyd that is unbounded even under
  k-means++ seeding. It also gives an example where forgetting moves a boundary past an unseen point. It cites
  Faiss issue #5907 / PR #5908 (Sahai, separation-based revival) and PR #5915 (Cabannes, cost-based revival).
  It belongs in Related Work (dead components) and Limitation 7.
- **anemone** (Liu, Shao & Van den Broeck, arXiv 2505.19982 v2, Oct 2025). A mini-batch EM for probabilistic
  circuits, HMMs included, with a per-parameter adaptive step. It reports that plain mini-batch EM (Peharz et al.
  2020) ends below full-batch EM at convergence. It is relevant to step-size adaptation and to our "converged
  reference" limitation.
- **Li & Song** (arXiv 2605.08864, May 2026). Online EM as equilibrium tracking, theory only (latent linear
  Gaussian). An optional citation for the background.
- **e-GMM** (GitHub kei-kochiya/egmm). Indexed by search engines as "EMA-EM for GMMs with a finite-horizon cosine
  schedule and split-based revival", borrowing from e-means, "with Overleaf paper". It now returns 404 (gh: not
  found), so its content and date are unverifiable and it cannot be cited. Risk: a reviewer may know it.
- **Also new, complementary:** ez-means (2611.*, 2026-11-24, faster centroid search), eDIAR (2611.18529,
  application), PISCES (HF model card, principal-axis revival split).
- **No paper found** that applies the e-means recipe to HMMs, LDA, dictionary learning or ALS, or that compares
  against tuned stepwise EM with revival. The cross-family study, the ablation and the law remain distinct.
- **TMLR pages re-read 2026-11-27:**
  - LLMs are allowed as assistive tools, with authors fully responsible;
  - an author submission quota applies ("generalized harmonic quota rule");
  - the author list is fixed at submission;
  - arXiv posting is allowed at any time;
  - the camera-ready must link the OpenReview page.
- **Applied (same day, author OK).** Four BibTeX entries added (assylbekov2026counterexamples, tiwari2026fuzzy,
  liu2025anemone, li2026equilibrium). New citations in `main.tex`:
  - Introduction: a concurrent-work sentence on Fuzzy e-Means;
  - Related Work: Li & Song (online EM), anemone (step-size adaptation), and both counterexamples and Fuzzy
    e-Means (dead components);
  - Limitation 7: the three revival papers;
  - Appendix E: Fuzzy e-Means reports the same failure on KDD99;
  - future work 1: anemone.

  e-GMM is not cited (unverifiable). The PDF is 20 pages with no undefined citations.
- **Supplementary ZIP rebuilt** from commit f398303 by `scripts/make_release.sh`: 1,314 files, 6.0 MB, written to
  `paper/tmlr/supplementary.zip` (gitignored). The personal-string scan passed. The only "IBM" hits are journal
  macros in the official `tmlr.bst`.

### Author block and named build (2026-11-27; author decision)

- The author supplied the name and contact for the paper (single author, "Independent Researcher"). They are kept
  only in `paper/tmlr/authors.tex`.
- `main.tex` now has a `\namedversion` switch, default 0 (anonymous, as submitted). `main_preprint.tex` sets it to
  1, loads `tmlr` with `[preprint]` and inputs `authors.tex`. It produces `main_preprint.pdf` for arXiv and the
  camera-ready version, and must **not** be uploaded to the double-blind submission.
- Checked by text extraction: `main.pdf` contains no name or email and shows "Anonymous authors" and "Under review
  as submission to TMLR"; `main_preprint.pdf` shows the author block. Both are 20 pages with no errors.
- `make_release.sh` removes `authors.tex`, `main_preprint.tex` and `main_preprint.pdf` before its personal-string
  check, so the supplementary stays anonymous.

### Final pre-submission error check (2026-11-27)

Read the full manuscript and checked numbers against `results/review_stats.json`, the verdict tables and this log.
Fixed:
- **LDA step ratio.** The abstract said LDA's best step is "about ten times smaller than the rule predicts", but
  Table 9 gives 0.187 predicted vs 0.037 best (5×). Now "about five times" in the abstract. The conclusion says
  5× below the k-means law and about 10× below the other families (median 0.029 vs 0.21–1.0).
- **Best-iterate paragraph (§6.1).** It said no other number changes, but the tuned LDA baselines change too:
  - tuned stepwise EM, 20NG 1.60 → 2.20;
  - with revival, 20NG 0.89 → 0.95 and Enron 7.17 → 15.0.

  These are now stated.
- **Timing (§6.3).** "Twelve configurations match within 5 epochs" now reads: thirteen on the 5-seed means,
  twelve also on the 3 timing seeds; Enron explained.
- **Statistics paragraph.** It claimed bootstrap CIs are "reported"; they are only in the released verdict tables,
  and it now says so.
- **Table 4.** GMM and LDA criteria reworded to match `kill_criteria.py` (rejection at two-thirds of datasets or
  corpora).
- **Tables 5 and 7.** HMM UCI HAR differs with and without revival (5.70 vs 6.73) although revival never fires,
  because each baseline selects (τ0, κ) separately. Table 5's caption explains this; Table 7 no longer says
  "identical".
- **Related Work.** "Our update is theirs" pointed at Li & Song after the citation insert; the sentence order is
  fixed. "Mini-batch EM ends below full-batch EM" was ambiguous and now reads "reaches a lower final likelihood".
- **Wording.** The abstract says "four families other than LDA"; §3.1 says the k-means law is not fitted by us; B_h
  is defined in Appendix B.

`main.pdf` (anonymous) and `main_preprint.pdf` (named) rebuilt: 20 pages each, no errors, no overfull boxes, no
undefined citations.

**Second check, same day.** Every value in Tables 5, 7 and 8 was recomputed from `results/*/runs.jsonl`,
`results/review_stats.json` and `results/_timing/timing_tuned_summary.json`, and all match. The §6.2 claims were
checked per pair: HMM −10.25% at T = 1 on UCI HAR, significant through T = 25; real-data full covariance
significant only at T ≤ 2. Liang & Klein (2009) were re-read from the ACL Anthology PDF. Fixed:
- **Table 6, GMM range:** −1.5% to +0.09% (some non-significant pairs slightly favor the baseline); it was
  "−0.03% to −1.5%". §6.2 now says "at most 1.5%".
- **Liang & Klein:** "four NLP models" → "four unsupervised NLP tasks"; the best step and batch size depend on the
  task and dataset (their wording).
- **Table 3, "pair":** now explains 102 = 17 configurations × 6 budgets (ALS has no revival baseline).
- **§6.3:** the extra 1.2–2.9× from the schedule now says "where both methods are timed (LDA 20NG excepted)".

**Third check, same day.** An automated scan of the PDF text found:
- no unresolved references;
- no leftover old terms (LOAO, recipe, C1–C6, EM@25);
- no British spellings, and the abstract and body counts agree;
- the repeated-word hits were table and math artifacts.

Undefined abbreviations were fixed: HMM is now defined in the introduction, ALS in the Notation paragraph (it was used
before §5), NMF in Table 1's caption, and DL and PCA64 in Table 2's caption.

### Full sweep: method vs code, numbers, references, supplementary (2026-11-27)

Systematic pass after the author asked for a thorough sweep.

**Method vs code** (`core/plugin.py`, `core/schedule.py`, `core/revival.py`, `tracks/*/`). The text described one
idealized recipe; the code differs in three places, now stated in §2–3:
- **Steps.** `steps_per_epoch = n // B`: each epoch reshuffles and drops the last partial batch, and M =
  round(T⌊n/B⌋), not ⌈Tn/B⌉. The schedule index runs t = 0…M−1, and revival runs when (t+1) mod 5 = 0 (Algorithm 1
  updated).
- **Revival.** GMMs split a donor along its principal covariance axis (δ√λ1, δ = 1/128). HMMs and LDA split a donor
  drawn ∝ mass with a relative ±0.05 perturbation. Dictionary learning re-seeds a dead atom from the residual of the
  worst-reconstructed point of the last batch and resets its statistics to the live-atom mean. It never fires for HMMs
  or DL.
- **Initialization.** Masses start at 1e-10 except LDA (random λ init, n0 = 1).

Confirmed as described: all five families run with `finalize_mode="none"` (no last-epoch averaging), threshold 0.1 ×
mean mass, revival every 5 batches, cosine floor 0.1, baseline α = min(1, (τ0 + t)^−κ).

**Numbers vs source files:**
- Table 9 matches `refit_coords.log` exactly; the missing-information feature matches `refit_coords_mi.log` (R²
  0.673, LOFO 0.167%); Table 10 matches `revival_rare.summary.json`.
- Pre-registered test: M0's LOFO median is **0.439%** (verdict file), not 0.440%; fixed.
- Record count: the supplementary holds **44,203** JSONL records in 67 files. 44,197 was before the SIFT1M
  reference fill (+6). The Reproducibility paragraph and README now say 44,203 and that the records cover runs, sweep
  cells and law-fit penalties.

**References:** all 46 entries re-read. The hmmlearn version (0.3.3) matches `uv.lock`. The internal comment "details
from memory" was removed from `references.bib`, since it ships in the supplementary.

**PDF:** the metadata has no author or title fields and the PDF has no embedded links.

**Supplementary:**
- `paper/draft.md`, `paper/METADATA.md`, `paper/RELEASE_PLAN.md` and `paper/references.md` contradicted the
  manuscript (old title, "51 of 102", LOAO, 44,197). `make_release.sh` now drops them.
- The README now points to `paper/tmlr/main.tex` / `main.pdf`, lists `review_stats.py`, and says PAPER_LOG is an
  append-only notebook in which the manuscript and the generated files are authoritative.

The manuscript is now 21 pages (the method details added half a page).

**Correction, same day (found while refreshing the release repo).** The record count above is wrong. That recount ran
in the working repo, not the export, so 44,203 records in 67 files is the working repo's total including the
companion-paper tracks. The original 44,197 had the same problem. The supplementary and release tree hold **39,185
JSONL records in 52 files**, counted in both `~/sa-em-release-tmlr` and the release checkout with a quoted
`find -name '*.jsonl'`, and matching `git ls-files '*.jsonl'` minus companion tracks. The Reproducibility paragraph
and the README now say 39,185.

### Last review pass (2026-11-27)

- **Appendix B vs code:** batch sizes, the α0 grid and its LDA extension, the τ0/κ grids, the ALS λ grid and the
  long-budget plan all match. The law-fitting cells were incomplete: every family also varies k (GMM diagonal only k
  ∈ {16, 64}; HMM {6, 12}; LDA {20, 50}; DL {k_h/2, k_h}; ALS rank {16, 32}). This is now stated.
- **Visual check of every page:**
  - Captions above tables touched the top rule; `\belowcaptionskip` is now 4pt.
  - Float pages spread their tables apart; they are now top-aligned with a fixed 14pt gap. Float placement only;
    template fonts and margins are unchanged.
  - The Appendix F heading was stranded under Table 10 with its figure alone on the next page. Compute is now
    Appendix F and the curves Appendix G, starting on a fresh page with the figure.
  - Table 10 bolded values inconsistently, with no stated meaning. It now bolds the lowest overall test NLL per row,
    and the caption says so.
- Back to 20 pages; references start on page 15. Both PDFs build cleanly and the anonymous one is name-free.

### Final code-level pass (2026-11-27)

- **Confirmed:**
  - the k-means law formula and constants (`core/scaling.py`, as in Bessa et al.);
  - the law variant takes B from the e-means batch law in every family (`stage_kmlaw`);
  - the Appendix E setup (d = 16, 30 + 4 far clusters at radius 30 with 0.2% mass, k = 34, B = 1024, rare = < 2%
    of training points; `revival_rare.py`);
  - the embedding models (DINOv2-small, all-MiniLM-L6-v2).
- **Fixed, splits:** §5 said splits are "fixed per seed"; in the code every family draws its split once with a
  constant seed (0 or 123), shared by all run seeds, and the seeds change initialization and batch order. The text
  now says so.
- **Fixed, Appendix A:** data sizes and subsamplings were not disclosed. They are now:
  - GMM PCA fitted on ≤ 50,000 training points and rescaled to unit average variance;
  - SIFT1M 200k / 10k train/val from the first 300k base vectors, 10k queries as test;
  - DINOv2 45k / 5k with CIFAR-100 test; AG News 90k / 10k with 7,600 test; synthetic 200k / 10k / 10k (d = 32,
    k = 32);
  - NYTimes restricted to the first 100,000 documents;
  - the Brown vocabulary capped at 5,000 words.

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

*Superseded: first refit, 342 cells with 23 censored. Current values are in the v3 table at the end of the
next entry.*

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
- Refit (`refit_coords.py`, same 342 cells, LDA now fully usable). *Superseded by the v3 refit below; kept for
  the record.*

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

**Current law comparison (v3 refit, 2026-11-16; corrected in this log 2026-11-17).** `refit_coords.py` on the v3
sweeps: 422 cells (GMM 136, HMM 96, dictlearn 72, LDA 72, ALS 46), 15 censored at the lower edge.
Source: `results/universal-scaling-law/refit_coords.log`. These are the numbers the paper uses (draft §5.5,
Figure 3, METADATA C6). They replace the two tables above. Relative to the second table, KM moves
0.149% → 0.155%, M1 LOCO 0.088% → 0.103%, Qe0 LOAO 0.154% → 0.149% and LDA held out 12.8% → 11.6%.

| model | LOCO median | LOAO median | HMM held out | LDA held out (predicted / best α0) | one global law: R² of log α0* |
|---|---|---|---|---|---|
| k-means law (unchanged) | 0.155% | 0.155% | 0.19% | 16.3% (0.187 / 0.037) | — |
| M1: n/k, B, T (+ per-alg constant) | 0.103% | 0.290% | 4.54% | 27.7% (0.27 / 0.037) | 0.20 (M1g) |
| Q1: q, n/B, T (+ per-alg constant) | 0.076% | 0.225% | 4.79% | 6.4% (0.088 / 0.037) | 0.50 (Q0) |
| Q0: one global law, q | 0.193% | 0.249% | 4.54% | 29.9% (0.309 / 0.037) | 0.50 |
| Qe1: q_eff, n/B, T (+ per-alg constant) | 0.073% | 0.183% | 1.81% | 5.3% (0.067 / 0.037) | 0.66 (Qe0) |
| **Qe0: one global law, q_eff** | 0.116% | **0.149%** | 1.52% | 11.6% (0.134 / 0.037) | **0.659** |

- Global fit Qe0 (v3): α0* = 0.419 · q_eff^0.427 · (n/B)^−0.282 · T^−0.326.
- Conclusions match the 2026-11-14 refit. The exponent on q_eff is 0.43, not 1. HMM is predicted worse by every
  coordinate law than by the k-means law (1.5–4.8% vs 0.19%). The missing-information feature (entry below) does
  not change this.

### LDA cosine-floor diagnostic (2026-11-16/17; diagnostic, not pre-registered)

Question: is the recipe's rise in LDA held-out perplexity from T=5 to T=25 caused by the cosine floor (eps=0.1) or
by the long-budget alpha0 grid being too narrow? Script `tracks/lda-scaling/eps_diag.py`; records in
`results/lda-scaling/eps_diag.jsonl` (102 runs; not read by evaluate.py). Headline cells k=50, B=256, full n,
revival on, seeds 0-2. eps=0.1 baselines come from sweep.jsonl plus eps=0.1 runs made here for alphas outside the
sweep. alpha0 picked on mean val; test is the mean over seeds at that alpha0.

| corpus | T | eps=0.1: best a0, test | eps=0.01: best a0, test |
|---|---|---|---|
| 20NG | 5 | 0.0193, 2297.3 | 0.0193, 2292.0 |
| 20NG | 25 | 0.00518, 2321.4 | 0.00518, 2328.1 |
| Enron | 5 | 0.01, 2265.1 | 0.01, 2264.6 |
| Enron | 25 | 0.00518, 2292.8 | 0.00518, 2289.6 |
| NYTimes | 5 | 0.00518, 3224.6 | 0.00518, 3225.3 |
| NYTimes | 25 | 0.00518, 3235.3 | 0.00518, 3234.6 |

- The floor does not matter: |Δ test| ≤ 6.7, same selected alpha0 in every cell.
- Grid edge: at T=25, 20NG was extended to 0.00268 and Enron to 0.00139; both are worse on val (20NG 2569 vs 2466),
  so the optima are interior. NYTimes `--below` crashed (no alpha below 0.00139 in the UP/DOWN map; KeyError in
  eps_diag_logs/nytimes_T25_below_*.log); not needed, since its optimum 0.00518 is interior already.
- The T=5 → T=25 rise persists under both eps on all three corpora (20NG 2297 → 2321, Enron 2265 → 2293, NYTimes
  3225 → 3235). Cause left open; the paper says so (draft §5.2 item 3).

### Missing-information feature (2026-11-17; exploratory, not pre-registered) — negative result

Hypothesis: EM's rate depends on the fraction of missing information (Dempster, Laird & Rubin 1977), so a proxy for
it should explain what q_eff leaves, especially the LDA α0 anomaly. Proxy: h = mean posterior entropy of the latent
assignment / log k on validation data, from one recipe fit per sweep group (seed 0, T = 2, group-median α0* and B);
GMM responsibilities, HMM forward-backward state posteriors, LDA per-token φ; h = 0 for DL and ALS. Script
`tracks/universal-scaling-law/missing_info.py` -> `results/universal-scaling-law/missing_info.json` (44 groups,
log `missing_info.log`); refit `refit_coords.py --missing-info` -> `refit_coords_mi.json`, log `refit_coords_mi.log`.

h by family (median, range): GMM 0.048 (0.014–0.155), HMM 0.170 (0.043–0.469), LDA 0.207 (0.157–0.236), DL / ALS 0.
Within a family h falls with n (HMM synthetic k=6: 0.469 / 0.350 / 0.175 at n = 5k / 10k / 20k).

| model | LOCO | LOAO | HMM held out | LDA held out | R²(log α*) |
|---|---|---|---|---|---|
| Qe1 | 0.073% | 0.183% | 1.81% | 5.30% | — |
| Qe0 | 0.116% | 0.149% | 1.52% | 11.64% | 0.659 |
| Qh1 (+ log(1−h)) | 0.073% | 0.214% | 4.35% | 3.93% | — |
| Qh0 (+ log(1−h)) | 0.099% | 0.167% | 2.45% | 11.72% | 0.673 |

Global fit Qh0: α0* = 0.286 · q_eff^0.466 · (1−h)^−1.433 · (n/B)^−0.254 · T^−0.320.

Reading: h adds almost nothing (R² +0.014, LOCO 0.116 -> 0.099%) and makes LOAO worse (0.149 -> 0.167%; HMM 1.52 ->
2.45%). The fitted sign is the opposite of the hypothesis: more missing information -> *larger* α0*, driven by HMM
(high h, α* = 1), whereas LDA has similar h (0.16–0.24) but needs α* ≈ 0.03. So posterior entropy does not explain
the LDA anomaly; whatever does is specific to LDA (candidates: the Dirichlet prior on topics and the sparse,
heavy-tailed word statistics; the per-document local E-step). Qe0 stays the headline law. For the paper: one
sentence in §5 (exploratory, negative) plus this table in the appendix.

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

**2026-11-16 — novelty checks before drafting** (web searches, extended mode; limited, not a systematic review).
- e-means paper (arXiv 2611.09501v1, read from the HTML): alpha0 law α0 = min(0.824·k^-0.46·B^0.90·n^-0.45·T^-0.11, 1)
  (R² 0.95), batch law B = 0.035·k^0.13·n^0.88·T^0.81 (÷8 for n > 2M), both fitted on DEEP only; revival every 5
  batches when N_j < 0.1·mean(N), donor split (Faiss); its ablation names revival the largest single effect (+2.97%
  train MSE without it). It cites Neal & Hinton 1998, Nowlan 1991, Sculley 2010, Bottou & Bengio 1994 and calls the
  update "a hard-assignment form of EM with decayed statistics", but does **not** cite Cappé & Moulines, Liang & Klein,
  Sato & Ishii, Hoffman et al. or Mairal et al., and tests no model other than k-means. Its own limitations: laws not
  optimal on low-dimensional / tiny data; runs only up to 25 epochs.
- Not found: a step/batch-size law for online EM across model families; cosine-annealed step sizes for online EM;
  a cross-family stochastic-vs-batch EM epochs benchmark. Closest prior art to cite and position against: Liang & Klein
  2009 (stepwise EM, 4 NLP models, "depends on the task"); Ranganath et al. 2013 (adaptive, tuning-free SVI step size);
  Hughes & Sudderth 2013 (memoized online VI — no learning rate, birth/merge moves = a revival-like mechanism);
  Chen et al. 2018 (variance-reduced stochastic EM, constant step); Nguyen, Forbes & McLachlan 2019 (mini-batch
  exponential-family mixtures); Hosseini & Sra (Riemannian SGD for GMM beats EM); Busbridge et al. 2023 (scaling an
  EMA with batch size).
- Possible concurrent work: GitHub issue meyer-lab/DDMC #561 proposes porting e-means (EMA, scaling laws, split
  revival) to EM clustering / diagonal GMM — a proposal, no results. Re-check arXiv for e-means follow-ups before
  submission (the paper is 10 days old).
- Consequence for claims: the revival rule is the part with the least direct prior art in online EM (memoized VI's
  births are the nearest), and our ablation shows it is what makes online EM work on GMM full covariance and LDA. The
  schedule is the part that matters for HMM / DL / ALS.

**2026-11-17 — novelty re-check and dataset licences.**
- arXiv / web search (extended) for e-means follow-ups and for step-size laws or cosine schedules in online EM. One
  follow-up found: Weiss 2026, "Small Is Not Dead: Distance-Aware Cluster Revival for e-means" (arXiv:2611.18238,
  13 Nov 2026). It covers k-means only: a count-flagged cluster is removed only if its squared distance to the
  nearest centre is ≤ 4·MSE, or if it gets no points for 2 epochs. It takes e-means' α0/B laws as given, does not
  refit them, and covers no other model family. No overlap with our contributions. Cited in draft §2 (dead
  components). Relevance: it confirms that count-based revival can harm rare clusters, a possible caveat for our
  shared revival rule (not tested here). Still not found: a step/batch-size law for online EM across families; a
  cosine schedule for EM.
- Dataset licences verified (`paper/datasets.md`), from provider pages or the READMEs shipped with the data:
  - UCI Bag of Words and UCI HAR: CC BY 4.0. The HAR README also prohibits commercial use.
  - MovieLens: research use only, no redistribution, must cite.
  - Brown README: "redistribution permitted"; NLTK's index lists it as non-commercial.
  - AG News card: licence "unknown".
  - CIFAR: no licence stated; cite the 2009 tech report.
  - SIFT1M (TEXMEX) and 20 Newsgroups home pages were unreachable, so their terms are not verified.
  - We release no raw data, so every term is met.

## 7. Open items before writing

**2026-11-16 — paper metadata started:** `paper/METADATA.md` (titles, abstract draft v0, contributions with
evidence and status, setup, headline tables, figure inventory, limitations, open items, reproducibility) for the
flagship paper, from the v3 results at commit 342cd1c. Items below that v3 closed: budgets 10/25 + sub-epoch and
5 seeds (done, §2 protocol v3); LDA re-emit and re-fit (done by `v3_reemit` and the v3 queue); commit (done, 342cd1c).

- [x] LDA grid extension result (§5) -> `refit_coords.py` re-run (2026-11-14).
- [x] (v3 `v3_reemit`, 2026-11-15) Re-emit `lda_ema_swept` (NYTimes T=5 changed) and re-run the LDA `cosine-law` stage, the LDA timing-swept
      pass, `fit_v2.py` and `evaluate.py`; archive the superseded records first.
- [x] (protocol v3) Budgets 10 and 25 (+ sub-epoch) for ours and tuned baselines; 5 seeds for headline groups (§2.1–2.3).
- [x] Check the weak-baseline suspicions (§2.6), 2026-11-17: (a) weak full-covariance online EM was missing
      revival. Fixed by the `*_rev_tuned` baseline and the schedule-vs-revival ablation (§4); the paper compares
      against it. (b) T11 HNSW grid oracle: companion paper, track killed, not used in this paper.
- [x] Missing-information feature for the law (§5): done 2026-11-17, negative (see §5).
- [x] Novelty checks (2026-11-16 and 2026-11-17, §6).
- [x] Commit: results live in the WSL clone `~/clasic_evolution`; code changes in `F:\clasic_evolution`.

**2026-11-16 — paper draft v0 (Markdown, user decision: Markdown first, LaTeX later):** `paper/draft.md`, framing A
("Scaling laws for stochastic-approximation EM"). Sections 1–7 written from METADATA §3–5 and §4 here; claims C1–C6 as
in METADATA, C2 stated against the revival-equipped tuned baseline, LDA as the counter-example. Number corrections made
while drafting: refit_coords.log (current) gives Qe0 LOAO 0.149% and LDA LOAO 11.6% (METADATA had 0.143% / 11.9% from an
earlier run of the refit); METADATA C6 updated. Open TODOs in the draft: related-work citations, LDA ε = 0.01
diagnostic (recipe's LDA perplexity rises T = 5 → 25), Figures 4–5, tuned-baseline wall-clock, Flash-KMeans/GMM.

### Figure restyle (2026-11-17)

All seven figures re-rendered from `scripts/make_figures.py` after the user reported overlapping text.
- Fig 1: 4 columns, short panel titles ("GMM AG News diag, k=64"), x label only on the bottom row, 3 y ticks,
  legend outside below. Method colours now follow roles; cosine_law and full-iteration library checks are not drawn
  (stated in convergence.txt and the caption).
- Fig 2 redefined: x = alpha0 / in-sample Qe0 prediction, y = median val penalty over cells (0.25-decade bins), three
  budget groups (0.25–1, 2–5, 10–25 epochs). The old version drew every cell (unreadable) and its T colours repeated.
- Fig 3 recomputed from refit_coords (KM, M1, Qe1, Qe0; LOCO and LOAO; 422 cells), horizontal boxes, medians printed.
  Medians match refit_coords.log (KM 0.155/0.155, M1 0.103/0.290, Qe1 0.073/0.183, Qe0 0.116/0.149 %).
  Note: the §5 table above lists KM 0.149% and M1 0.088% (an older refit); refit_coords.log is the current value.
  Fixed 2026-11-17: both old §5 tables are now marked superseded, and a v3 table (422 cells) follows them.
- Drift figure: frozen_emeans was drawn in the same blue as rewarm (suffix "emeans" matched first); fixed.
