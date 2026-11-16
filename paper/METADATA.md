# Paper metadata — flagship paper (draft, 2026-11-16)

Working file for the flagship paper ("The e-means recipe is general", RESEARCH_PLAN.md §5: T01, T04–T07, T12).
Every number here is copied from `results/*/VERDICT.md`, `results/evaluate_all.log` or
`results/universal-scaling-law/refit_coords.log` at commit **342cd1c** (protocol v3). PAPER_LOG.md is the
source of record for decisions, deviations and bugs; when this file and the generated results disagree, the generated
results win. Companion papers (vector indexes T02/T03/T08/T13; HNSW T11; seeding/adaptive-k T09/T10) are listed at
the end and are not covered here.

---

## 1. Identity

**Working titles** (pick one; the literature review argues against an "X-means for Y" title, PAPER_LOG §6):
1. *Scaling laws for stochastic-approximation EM: one step-size rule across five model families*
2. *When does an EMA of sufficient statistics beat batch EM? A cross-family time-to-baseline study*
3. *From e-means to stochastic EM at scale: schedules, revival and a step-size law that (almost) transfers*

**Authors / affiliations:** TBD by the author. (Do not fill from tooling.)

**arXiv categories:** primary cs.LG; cross-list stat.ML (and stat.CO if the law/theory part grows).

**Keywords:** online EM, stepwise EM, stochastic approximation, exponential moving average, sufficient statistics,
scaling laws, hyperparameter transfer, k-means, Gaussian mixtures, hidden Markov models, latent Dirichlet
allocation, dictionary learning, matrix factorization.

**Candidate venues:** NeurIPS / ICML main track (empirical-plus-law paper); TMLR (if the contribution is judged
mainly empirical); a datasets-and-benchmarks track if the time-to-batch benchmark becomes the centre.

**Lineage that must be named up front** (reviewers will say "stepwise EM renamed" otherwise): the update is online /
stepwise EM (Sato & Ishii 2000; Cappé & Moulines 2009; Liang & Klein 2009) and SVI's natural-gradient step (Hoffman
et al. 2013); per-family precedents Hoffman et al. 2010 (LDA), Mairal et al. 2010 (dictionary learning / NMF),
Nguyen, Forbes & McLachlan 2019 (mini-batch EM). Starting point: e-means, arXiv:2611.09501 (Meta FAIR, 2026).

## 2. Abstract (draft v0 — numbers from protocol v3)

> e-means trains k-means by keeping an exponential moving average of each cluster's sufficient statistics, with a
> cosine-annealed rate, dead-cluster revival and a law that sets the initial rate from the problem size. The update
> is stochastic-approximation (stepwise) EM, so it applies to any model with a closed-form M-step. We test the full
> recipe on five families — Gaussian mixtures, hidden Markov models, latent Dirichlet allocation, online dictionary
> learning / NMF and alternating least squares — on 19 dataset configurations with 5 seeds and budgets from 0.25 to 25
> epochs, against full-batch EM at 25 iterations and against stepwise / online EM tuned on the same validation grid.
> With its rate selected on validation, the recipe reaches the batch objective in under 3 epochs on 12 of 19
> configurations (HMM median 38x fewer epochs, dictionary learning 30x, diagonal GMM 9.9–85x), reaches it sooner than
> tuned stepwise EM on 17 of 19 (the other two: neither reaches it), and ends at an equal or lower objective on all 19; on a CPU, matched-quality wall-clock speed-ups reach 5–82x.
> The k-means rate law, applied unchanged, transfers to GMM, HMM, dictionary learning and ALS (median validation
> penalty ≤ 0.2%) but fails on LDA, whose optimal rate is about 10x lower. Re-expressing the law in observations per
> sufficient-statistic cell explains two-thirds of the variance of the optimal rate across all families with one
> global fit and cuts the held-out-family penalty on LDA from 28% to 12%. [Final sentence: the claim the paper
> settles on — see §3.]

Counts checked against §5 (2026-11-16). The final-objective margins over tuned stepwise EM are often within one
seed-std (e.g. DL 20NG 0.4473 vs 0.4474) — say "equal or lower", not "better". These are the swept method's
numbers; the untuned law ends slightly above tuned stepwise EM on HMM (§3 C4).

## 3. Contributions and the evidence behind each

| # | Claim | Status | Evidence (v3) | Pre-registered? |
|---|---|---|---|---|
| C1 | The EMA-statistics recipe (cosine rate + revival) reaches full-batch EM@25 in ≤ 3 epochs across families | **Supported** for GMM diag, HMM, dictionary learning, LDA 20ng; **not** for GMM full (2–10 ep), LDA enron (4.4) / nytimes (never), ALS (5.8 ep ml10m; tie ml1m) | T01, T04, T05, T06, T07 VERDICT.md | Yes (kill criteria, `scripts/kill_criteria.py`) |
| C2 | It beats or matches stepwise / online EM tuned on the same grid | **Supported**: reaches the reference sooner on 17/19 groups (LDA NYTimes, ALS ML-1M: neither reaches it); final objective swept ≤ tuned on 19/19, often within one seed-std | `*_tuned` / `online_em_tuned` rows | Partly (tuned baselines were in the plan; "ours vs tuned" as headline is a v3 framing) |
| C3 | Matched-quality wall-clock speed-ups on CPU (isolated timing) | **Supported for T ≤ 5 only**: 5.4–82x (swept), 6.2–25x (law) | §5 timing table; `results/_timing/` | Measurement protocol fixed 2026-11-14 |
| C4 | The untuned k-means α0 law transfers to other families | **Supported except LDA**: median penalty ≤ 0.2% for GMM/HMM/DL/ALS; LDA law never reaches batch VB@25. HMM: law converges fastest but ends slightly *above* tuned stepwise EM (within 1 seed-std) | `*_kmlaw` rows; `refit_coords.log` | Yes (T12) |
| C5 | One cross-family law (T12, M1) beats a per-algorithm constant | **Supported** (pre-registered): LOCO median penalty 0.108% vs 0.319%, Wilcoxon p = 1.05e-18; LOAO 0.338% vs 0.440% | `results/universal-scaling-law/VERDICT.md` | Yes |
| C6 | "Observations per statistic cell" coordinates explain the LDA anomaly | **Exploratory**: one global law R² 0.66 (Qe0) vs 0.20 for n/k; LOAO median 0.143%; LDA LOAO 11.9% (vs 27.7% M1). Exponent on q_eff 0.43, not 1 — not a universal constant | `refit_coords.log` (422 cells) | **No — label exploratory** |

Claims **not** to make (contradicted or unsupported by v3 data):
- "The untuned law ends lower than tuned stepwise EM on HMM" — v2 claim, false on 5 seeds (PAPER_LOG §4).
- Any wall-clock speed-up for matches after 5 epochs (timing pass covers T ≤ 5 only, PAPER_LOG §2).
- ALS as a clear win: the pre-registered median 2.14x passes narrowly; ml1m is a tie (report per dataset).
- "Universal" law: LDA and HMM held-out penalties stay large; the q_eff exponent is not 1.
- The 100x on dictlearn cifar_p8 is a floor effect (matched at the first budget, 0.25 epochs) — report as "≤ 0.25 epochs".

## 4. Experimental setup (for the methods section)

**Families, datasets and groups (19 headline groups, 5 seeds each).**

| family | track | datasets (group) | k | test metric | full-batch reference@25 |
|---|---|---|---|---|---|
| Gaussian mixture | T01 | synthetic, SIFT1M-PCA64, DINOv2-PCA64, AG News-PCA64 (MiniLM) — diag and full covariance (8 groups) | 32 (synthetic), 64 | test NLL / point (nats) | EM 25 iterations |
| Dictionary learning / NMF | T04 | CIFAR 8×8 patches (k=128), CIFAR 12×12 patches (k=192), 20 Newsgroups NMF (k=32) | 128 / 192 / 32 | objective (recon + λ‖α‖₁) | full-batch DL, 25 iterations |
| LDA | T05 | 20 Newsgroups (V=10k), Enron (V=20k), NYTimes (V=20k) (UCI bag-of-words) | 50 topics | held-out perplexity | batch variational Bayes, 25 iterations |
| ALS | T06 | MovieLens 1M, MovieLens 10M | rank 32 | test RMSE | ALS-WR, 25 iterations |
| HMM | T07 | synthetic (V=200, L=40), Brown POS (V=5000, mean L=20.2), UCI HAR vector-quantised (V=64, L=128) | 12 states | held-out NLL / token (nats) | Baum–Welch, 25 iterations |

Supporting: T00 core reproduction (k-means on SIFT1M k=1024/4096, AG News MiniLM k=1024, DINOv2-small CIFAR-100
k=512; reference Faiss Lloyd@25).

**Methods compared per group.**
- `*_swept` (ours, tuned): recipe with α0 selected on validation per budget from the 8-point log grid [0.01, 1]
  (LDA grid extended to 0.00518 and below where the optimum was at the floor).
- `*_kmlaw` (ours, untuned): α0 and B from the k-means law, no tuning.
- Tuned prior art: stepwise / online EM with (τ0, κ) selected on the same validation protocol
  (`stepwise_em_tuned`, `online_em_tuned`, `online_lda_tuned`, `stepwise_dl_tuned`, `stepwise_als_tuned`).
- Library defaults: Hoffman online LDA default, sklearn MiniBatchDictionaryLearning, sklearn GaussianMixture,
  hmmlearn Baum–Welch.

**Protocol v3** (PAPER_LOG §1–2): budgets (0.25, 0.5, 1, 2, 3, 5, 10, 25) epochs; seeds 0–4; epochs-to-match = first
(linearly interpolated) epoch where the seed-mean test curve reaches the seed-mean reference@25; gain = 25 /
epochs-to-match; 95% CI by seed bootstrap; paired t-test by seed (α = 0.05). At T = 10 and 25 the search is a frozen
3–4-point local search around the T = 5 optimum for both ours and the tuned baselines
(`results/<track>/long_budget_plan.json`) — disclose.

**Recipe constants:** cosine schedule α(t) = α0·(ε + (1−ε)·½(1+cos(πt/M))), ε = 0.1; revival when
N_j < 0.1·mean(N) checked every 5 batches; statistics initialised at N = 1e-10.

**Hardware / software.**
- AMD Ryzen 9 9900X (12C/24T), NVIDIA RTX 5090 32 GB (driver 610.88), Windows 11 + WSL2 Ubuntu 24.04.
- Python 3.12.14, torch 2.15.0 + CUDA 13.2, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, numba 0.68.1,
  faiss 1.16.0, hmmlearn 0.3.3, hnswlib. Locked in `uv.lock`.
- GPU ports (float64, parity-tested in `tests/test_gpu_ports.py`) change wall-clock only, never the algorithm.
- Headline wall-clock: isolated timing pass only — CPU only, 12 threads, one job at a time, 3 seeds, after an
  untimed warm-up (`results/_timing/`).

**Compute.** Protocol-v3 queue: 686 jobs (+22 HMM repair, +4 ALS repair), 2026-11-14 05:42 → 2026-11-16 01:15 UTC
(~43.5 h wall), ~575 job-hours of concurrent CPU/GPU jobs on the one desktop. Sweep rows recorded: GMM 9,958,
HMM 4,268, dictionary learning 3,702, LDA 3,391, ALS 2,420. Earlier v1/v2 runs on an Apple-silicon Mac and the
desktop are archived (`superseded_*` folders) and not used for v3 numbers.

## 5. Headline numbers (v3, for tables)

**Epochs to reach full-batch@25** (seed means; gain = 25 / epochs). `—` = not reached within 25 epochs.

| group | swept (ours) | law (ours, untuned) | tuned stepwise / online EM |
|---|---|---|---|
| GMM synthetic diag | 0.29 | 17.2 | — |
| GMM synthetic full | 1.98 | 2.88 | — |
| GMM SIFT1M diag | 0.88 | 0.97 | — |
| GMM SIFT1M full | 7.86 | 16.1 | — |
| GMM DINOv2 diag | 1.88 | 2.04 | — |
| GMM DINOv2 full | 8.78 | 23.1 | — |
| GMM AG News diag | 2.54 | 2.04 | — |
| GMM AG News full | 9.63 | 12.0 | — |
| DL CIFAR 8×8 | ≤ 0.25 | 1.68 | 0.54 |
| DL CIFAR 12×12 | 0.83 | 2.83 | 1.66 |
| DL 20NG NMF | 2.32 | 2.36 | 3.70 |
| LDA 20NG | 1.13 | — | 1.60 |
| LDA Enron | 4.44 | — | — |
| LDA NYTimes | — (ends best: 3242 vs tuned 3317, ref 3167) | — | — |
| ALS ML-10M | 5.84 | 8.88 | 20.1 |
| ALS ML-1M | — (tie: 0.8656 = ref) | — | — |
| HMM Brown POS | 0.64 | 0.95 | 1.33 |
| HMM synthetic | 0.66 | 1.44 | 1.95 |
| HMM UCI HAR | 1.72 | 2.21 | 5.70 |

GMM note: tuned online EM never reaches EM@25 on any GMM group and is far off on full covariance (e.g. SIFT1M full
50.7 vs reference 40.7 nats) — **suspected weak baseline**, must be checked before submission (§8).

**Isolated wall-clock speed-up (CPU, T ≤ 5 only):** see PAPER_LOG §4 "Isolated timing v3" (5.4–82x swept,
6.2–25.3x law; HMM Brown 32.9x / 25.3x; DL CIFAR 12×12 30.8x / 10.1x).

**T12 law:** see C5/C6 above; per-family median swept α0*: ALS 0.21, DL 0.26, GMM 0.32, HMM 1.0, LDA 0.029.

## 6. Figure and table inventory

| id | content | source | status |
|---|---|---|---|
| Fig 1 | convergence curves (test objective vs epochs, ours vs tuned vs batch) per family | `results/figures/convergence.{pdf,png,txt}` | generated (v3); needs final styling |
| Fig 2 | α0 sweeps: val loss vs α0 per budget, law prediction marked | `results/figures/alpha0_sweeps.*` | generated (v3) |
| Fig 3 | law penalties: LOCO / LOAO penalty per model (M0, M1, KM, Q*, Qe*) | `results/figures/law_penalties.*` | generated (v3) |
| Fig 4 (planned) | optimal α0* vs observations per cell q_eff, all families, global fit | from `refit_coords.json` | **to make** |
| Fig 5 (planned) | wall-clock speed-up bar chart (isolated timing) | `results/_timing/*/runs.jsonl` | **to make** |
| Tab 1 | families × datasets × metric × reference | §4 above | draft |
| Tab 2 | epochs-to-match, ours vs tuned vs law | §5 above / VERDICT.md | draft |
| Tab 3 | isolated timing | PAPER_LOG §4 | draft |
| Tab 4 | T12 model comparison (LOCO, LOAO, per held-out family) | `refit_coords.log`, T12 VERDICT.md | draft |
| App. | per-group full tables with CIs and p-values | every `results/<track>/VERDICT.md` | generated |
| (other papers) | `ivf_recall_latency.*`, `drift_window_mse.*` | T03 / T08 / T13 | belong to the index paper |

## 7. Limitations to disclose (from PAPER_LOG §2–3)

1. Datasets narrower than planned: ALS on MovieLens 1M/10M (plan: 25M, Netflix); HMM on Brown, not Penn Treebank;
   LDA without PubMed/Wikipedia; DL without audio.
2. Long budgets (10, 25) use a frozen 3–4-point local search for both ours and baselines.
3. Wall-clock speed-ups only for matches within 5 epochs; CPU only; one machine.
4. The reference is batch EM at 25 iterations, not converged batch EM.
5. Weak-baseline suspicions: GMM tuned online EM (full covariance); exact GPU baselines (Flash-KMeans,
   Flash-GMM) not compared.
6. Operational repairs (all disclosed, none changed a method): LDA E-step aliasing bug (all LDA re-run), scheduler
   adoption bug (73 jobs re-run), 17 HMM jobs re-run after CUDA errors, one ALS cell filled.
7. T12's refit in per-cell coordinates is exploratory (chosen after seeing the LDA failure).

## 8. Open items before submission

- [ ] Check / strengthen weak baselines: GMM full-covariance online EM; add Flash-KMeans / Flash-GMM or justify.
- [ ] Missing-information fraction as a law feature (PAPER_LOG §5); statistic-space noise-scale test.
- [ ] Novelty checks (literature review, "Novelty checks" table): cross-family step law, cosine schedule for EM,
      unified revival rule.
- [ ] Read the e-means PDF for exact law exponents; confirm it does not cite the online-EM lineage.
- [ ] Decide headline framing (title choice, §1) and finalise the abstract's last sentence.
- [ ] Make Fig 4 and Fig 5; restyle Figs 1–3.
- [ ] Optional: time T = 10/25 matches (GMM full, LDA Enron, ALS ML-10M) if wall-clock for them is wanted.
- [ ] Dataset licences / citations for every dataset (SIFT1M, MovieLens, UCI BoW, UCI HAR, Brown, 20NG, CIFAR-100,
      AG News, DINOv2 / MiniLM models).
- [ ] Code/data release plan (repo is private at `shashanka300/clasic_evolution`).

## 9. Reproducibility

- Code + results: commit `342cd1c`, branch `desktop-5090`.
- Data: `python3 scripts/fetch_data.py` (downloads and embeds; embeddings recomputed on the 5090).
- Full v3 regeneration: `python3 scripts/make_jobs_v3.py && python3 scripts/pqueue.py --jobs results/jobs_budget_v3.txt
  --done results/pqueue_done_v3.txt --budget 32 --max-jobs 16`; repairs: `results/jobs_hmm_fix_v3.txt`,
  `results/jobs_als_fix_v3.txt`.
- Verdicts and figures: `scripts/evaluate.py`, `scripts/make_figures.py`; law fits:
  `tracks/universal-scaling-law/fit_v2.py`, `refit_coords.py`.

## 10. Companion papers (not covered here)

| paper | tracks | v3 status |
|---|---|---|
| Fast, balanced, live vector indexes | T02 OPQ (6/6 groups), T03 balanced IVF (marginal +10.1%), T08 drift re-warm (3/4), T13 (descriptive) | results in place; T03 marginal |
| No-tuning ANN indexes via scaling laws | T11 HNSW | **killed** (1/6 held-out pairs); oracle grid suspect |
| Seeding and adaptive k for streaming k-means | T09 (15 wins), T10 (CV 1.8%) | results in place |
