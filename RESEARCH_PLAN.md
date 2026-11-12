# Modernizing Classic Algorithms — Parallel Research Plan

Inspired by **e-means** (Bessa et al., arXiv 2611.09501). e-means turned Lloyd's k-means into a
mini-batch, EMA-of-sufficient-statistics optimizer with a cosine-annealed step, dead-unit revival,
a cheap finalization pass, and scaling laws that remove hyperparameter tuning. Result: Lloyd's
25-epoch loss in ~2 epochs.

**Research question for this whole program:** *Is the e-means recipe a general method?* That is, does
"EMA of sufficient statistics + cosine anneal + dead-unit revival + finalization + scaling law"
give order-of-magnitude speedups for other classic full-pass algorithms, and where does it break?

---

## 1. How the work is laid out

```
main ──► core (Phase 0, blocks everything)
           ├── T01 gmm-ema            ┐
           ├── T02 opq-joint-rotation │
           ├── T03 balanced-ivf       │
           ├── T04 dictlearn-scaling  │
           ├── T05 lda-scaling        │  Phase 1: all in parallel,
           ├── T06 als-ema            │  one git worktree each
           ├── T07 hmm-ema            │
           ├── T08 drift-rewarm       │
           ├── T09 cheap-seeding      │
           ├── T10 adaptive-k         │
           └── T11 hnsw-scaling-laws  ┘
                       │
                       ▼
           T12 universal-scaling-law (Phase 2, reads results from T01–T07)
           T13 drift-balanced-ivf     (Phase 2, merges T03 + T08)
```

### Rules that keep 11 parallel branches from colliding

1. **`core/` is owned by the `core` branch only.** Tracks import it and never edit it. A track that
   needs a core change opens a small PR into `core`, then every track runs `git rebase core`.
2. **Each track writes only inside `tracks/<slug>/` and `results/<slug>/`.** So merges never conflict.
3. **One results format for everyone** (`results/<slug>/*.jsonl`, one line per run):
   `{track, method, dataset, n, d, k_or_units, epochs, batch, alpha0, seed, train_loss, test_loss, wall_s, gpu_s, extra:{...}}`
   T12 depends on this format, so it is frozen once Phase 0 ends.
4. **Fixed compute-matched comparison.** Every track reports (a) loss at equal epochs (1, 2, 3, 5, 10, 25)
   and (b) epochs and wall-clock time to reach the classic baseline's 25-epoch loss. This is the e-means headline metric, kept identical.
5. **3 seeds minimum, 5 for headline numbers.** Report mean ± std.
6. **Kill criterion per track** (below). A dead track writes a one-page `NEGATIVE.md`. Negative results feed T12.

---

## 2. Phase 0 — shared core (blocks all tracks)

<!-- track:T00 slug:core phase:0 deps:none -->
### T00 · core

**Goal:** a reusable "e-means engine" that any classic algorithm can plug into.

**Build:**
- `core/stats.py` — EMA accumulator for arbitrary sufficient statistics (`N_j`, `S_j`, plus optional second moments and matrix stats). Initialized with `N=1e-10` so init centroids are returned exactly (as in the paper).
- `core/schedule.py` — cosine step `α(t) = α0·(ε + (1-ε)·½(1+cos(πt/M)))`, `ε=0.1`. Also constant, 1/t (Sculley) and warm-restart variants.
- `core/revival.py` — dead-unit detection (`N_j < τ·mean(N)`, `τ=0.1`, every 5 batches) and Faiss-style donor split.
- `core/finalize.py` — last-epoch-mean and slow-EMA (half-life `h`).
- `core/scaling.py` — fit log-linear laws `log α0*`, `log B*` = linear in `(log n, log k, log T, log B)`. Includes the sweep driver.
- `core/emeans.py` — a reference e-means reimplemented from the paper. **Acceptance test:** reproduce the
  paper's "beats Lloyd@25 within ~1.4–2.6 epochs" on at least SIFT1M and one DINO/text embedding set. If faiss ships the official `e_means`, test against it as well.
- `core/bench.py` — dataset loaders and cache (SIFT1M, GIST1M, Deep10M, a DINOv2/v3 embedding subset, a text-embedding set such as MS MARCO), timers, and the JSONL writer.
- `core/plugin.py` — the interface each track implements:
  `init(X0)`, `assign(batch) -> stats`, `params_from_stats(S)`, `revive(S)`, `loss(X)`.

**Done when:** e-means reproduction passes and a toy plugin (plain k-means through `plugin.py`) gives
identical numbers to `core/emeans.py`. Tag `core-v1`. Every track branches from that tag.

---

## 3. Phase 1 — parallel tracks

Each block below is copied verbatim into that worktree as `TASK.md`.

<!-- track:T01 slug:gmm-ema phase:1 deps:core -->
### T01 · gmm-ema — EM for Gaussian mixtures

- **Hypothesis:** EM is k-means with soft weights and variances. EMA of `(Σγ, Σγx, Σγxxᵀ or diag)` with a cosine anneal and component revival reaches full-batch EM's 25-iteration log-likelihood in ≤3 epochs.
- **Baselines:** sklearn `GaussianMixture` (full EM), online EM (Cappé & Moulines 2009, step `t^-κ`), SGD/Adam on the GMM log-likelihood.
- **Variants:** diagonal vs full covariance. Hard (CEM) vs soft responsibilities. Covariance floor `1e-6·trace/d`.
- **Data:** the core embedding sets PCA'd to 64/128 dims, plus synthetic mixtures with known parameters (to measure parameter recovery, not just likelihood).
- **Metrics:** test log-likelihood per point, epochs to EM@25, parameter error on synthetic data, number of collapsed components.
- **Open questions:** Does the e-means scaling law transfer unchanged? Does revival need a covariance-aware donor rule (split along the top eigenvector)?
- **Kill if:** after the sweep, best variant needs >8 epochs to match EM@25 on 2 of 3 datasets.

<!-- track:T02 slug:opq-joint-rotation phase:1 deps:core -->
### T02 · opq-joint-rotation — Optimized Product Quantization

- **Hypothesis:** OPQ alternates "fit rotation R" and "run PQ k-means". Updating R online from an EMA of the cross-covariance `Σ x·(decoded x)ᵀ` (orthogonal Procrustes via SVD every `m` batches), in step with e-means codebooks, matches OPQ's distortion in a fraction of the passes.
- **Baselines:** Faiss `OPQMatrix + PQ` (default iterations), plain PQ with e-means codebooks (ablation), learned rotation via gradient (as in some neural quantizers).
- **Knobs:** Procrustes refresh period `m ∈ {1, 10, 100}` batches. Rotation EMA decay tied to or separate from the codebook decay. Also extend to residual quantization (RQ/LSQ) as a stretch goal.
- **Data:** SIFT1M, Deep10M, a DINO subset, text embeddings. M ∈ {8, 16, 32} subquantizers, 8 bits.
- **Metrics:** reconstruction MSE, recall@10 for ADC search, training wall-clock.
- **Kill if:** joint online rotation is no better than "e-means PQ + one final OPQ refit".

<!-- track:T03 slug:balanced-ivf phase:1 deps:core -->
### T03 · balanced-ivf — balanced k-means for inverted-file indexes

- **Hypothesis:** IVF query latency is set by the biggest lists. Within each mini-batch, assignment can be balanced with a few Sinkhorn iterations toward the target sizes (as in SwAV), while the centroids remain EMAs. This gives near-uniform lists at almost no MSE cost, faster than balanced Lloyd variants.
- **Baselines:** Faiss IVF (Lloyd), e-means IVF (no balancing), balanced k-means (Malinen & Fränti), the size-penalized assignment that some vector databases use.
- **Knobs:** Sinkhorn iterations ∈ {1, 3, 10}, entropy ε, soft vs hard target (allow ±x% slack). Balancing only in the early schedule vs throughout.
- **Data:** SIFT1M/10M, Deep10M, text embeddings (these are the most imbalanced). nlist ∈ {1k, 16k, 65k}.
- **Metrics:** imbalance factor (Σ|L|²·k/n²), recall@10 vs nprobe, p50/p99 query latency at fixed recall, training time.
- **Kill if:** p99 latency at recall 0.9 improves by <10% over e-means IVF.

<!-- track:T04 slug:dictlearn-scaling phase:1 deps:core -->
### T04 · dictlearn-scaling — online dictionary learning / NMF

- **Hypothesis:** Mairal-style online dictionary learning already keeps the sufficient statistics `A = Σαα^T` and `B = Σxα^T`, but its forgetting factor is hand-set. A cosine-annealed EMA on A and B, unused-atom revival and a fitted scaling law remove the tuning and speed up convergence.
- **Baselines:** sklearn `MiniBatchDictionaryLearning`, `MiniBatchNMF`, full-batch NMF (multiplicative updates / HALS).
- **Data:** 8×8 / 16×16 natural-image patches, MNIST/CIFAR patches, a bag-of-words matrix for NMF, and audio spectrograms.
- **Metrics:** objective (reconstruction + λ‖α‖₁), epochs to baseline@25, number of dead atoms.
- **Kill if:** gain < 2× in epochs-to-match.

<!-- track:T05 slug:lda-scaling phase:1 deps:core -->
### T05 · lda-scaling — online LDA topic models

- **Hypothesis:** Online LDA (Hoffman 2010) step `ρ_t = (τ0+t)^-κ` has two hand-tuned knobs. Replacing them with a cosine schedule plus a scaling law in `(docs, topics, vocab, epochs, batch)` matches tuned online LDA without a sweep. Topic revival fixes dead topics.
- **Baselines:** batch variational LDA, online LDA with paper defaults and with the best grid-searched (τ0, κ), Gensim LdaMulticore.
- **Data:** 20 Newsgroups, NYTimes and PubMed (UCI bag-of-words), Wikipedia subset.
- **Metrics:** held-out perplexity, topic coherence (NPMI), epochs to batch@25.
- **Kill if:** the fitted law does worse than default online LDA on 2 of 3 corpora.

<!-- track:T06 slug:als-ema phase:1 deps:core -->
### T06 · als-ema — alternating least squares for recommenders

- **Hypothesis:** each ALS item update needs `Σ u·uᵀ` and `Σ r·u` over the users who rated it. EMA versions of these statistics, updated from user mini-batches with a cosine anneal, converge in far fewer passes than full ALS.
- **Baselines:** implicit-library ALS, Spark ALS, SGD matrix factorization.
- **Data:** MovieLens-25M, Netflix Prize (if available), Amazon reviews.
- **Metrics:** test RMSE (explicit), recall@k / NDCG (implicit), epochs and time to ALS@25.
- **Edge to watch:** cold items with very few ratings. Treat "dead" here as stale statistics and decide whether to revive or to freeze.
- **Kill if:** gain < 2× in epochs-to-match.

<!-- track:T07 slug:hmm-ema phase:1 deps:core -->
### T07 · hmm-ema — Baum–Welch for hidden Markov models

- **Hypothesis:** expected transition and emission counts from forward–backward are sufficient statistics. EMA over sequence mini-batches with a cosine anneal and unused-state revival beats full Baum–Welch and online EM (Cappé 2011) in passes.
- **Baselines:** hmmlearn (full Baum–Welch), online EM for HMMs, stepwise EM (Liang & Klein 2009).
- **Data:** synthetic HMMs with known parameters, unsupervised POS induction (Penn Treebank), activity-recognition sensor data.
- **Metrics:** held-out log-likelihood, parameter recovery (synthetic), many-to-one accuracy (POS).
- **Kill if:** gain < 2× or worse local optima than full Baum–Welch.

<!-- track:T08 slug:drift-rewarm phase:1 deps:core -->
### T08 · drift-rewarm — e-means for data that changes over time

- **Hypothesis:** e-means anneals once and then freezes. Monitoring the batch loss (Page–Hinkley test) and re-raising the step size with a short warm-restart cosine when drift is detected keeps an index's centroids current without a full rebuild.
- **Baselines:** frozen e-means, constant-α EMA (VQ-VAE style), periodic full retrain, Spark streaming k-means with forgetfulness.
- **Drift scenarios:** (a) gradual class-mix shift, (b) abrupt arrival of new clusters, (c) embedding-model upgrade (old → new encoder), (d) periodic/seasonal.
- **Metrics:** loss on the current window over time, regret vs an oracle that retrains at every step, compute spent.
- **Kill if:** no scenario beats "constant α tuned per scenario".

<!-- track:T09 slug:cheap-seeding phase:1 deps:core -->
### T09 · cheap-seeding — fast starting points

- **Motivation:** the paper reports that k-means++ seeding helps at T=3 but is too expensive at low epochs.
- **Hypothesis:** a seeding cost of ≤0.1 epoch (AFK-MC², k-means|| on one batch, or "split-as-you-go" from the revival mechanism with k growing during the first epoch) captures most of the k-means++ gain.
- **Baselines:** random init, k-means++, greedy k-means++, k-means||.
- **Metrics:** final loss at T ∈ {1, 2, 3} *including* seeding cost in the epoch budget.
- **Kill if:** nothing beats random init once seeding time is charged.

<!-- track:T10 slug:adaptive-k phase:1 deps:core -->
### T10 · adaptive-k — choosing the number of clusters during training

- **Hypothesis:** the revival machinery already splits and kills clusters. Adding split and merge rules driven by an information criterion (BIC / X-means style, computed from the EMA statistics) lets e-means choose k within one run.
- **Baselines:** X-means, G-means, a k sweep with the elbow and silhouette methods, DP-means.
- **Data:** synthetic data with known k, real embeddings (no true k, so evaluate downstream: IVF recall at a fixed memory budget).
- **Metrics:** error in k on synthetic data, total compute vs a sweep, downstream quality.
- **Kill if:** the selected k is unstable across seeds (coefficient of variation >20%).

<!-- track:T11 slug:hnsw-scaling-laws phase:1 deps:core -->
### T11 · hnsw-scaling-laws — no-tuning graph indexes

- **Hypothesis:** the e-means idea of setting hyperparameters from the problem size carries over to HNSW. Fit `M*`, `efConstruction*`, `efSearch*` as log-linear laws in `(n, d, target recall, intrinsic dimension)`. The law then picks settings within 5% of a full grid search's cost-recall frontier.
- **Baselines:** library defaults (hnswlib, Faiss), full grid search, existing auto-tuners (Faiss autotune).
- **Data:** all ann-benchmarks datasets plus held-out ones (to test the law on data it was not fitted on).
- **Metrics:** build time, memory, QPS at recall 0.9/0.95/0.99, distance to the grid-search Pareto front.
- **Note:** this track uses `core/scaling.py` only, not the EMA engine.
- **Kill if:** the law loses to library defaults on held-out datasets.

---

## 4. Phase 2 — synthesis tracks (start once their inputs land)

<!-- track:T12 slug:universal-scaling-law phase:2 deps:T01,T02,T03,T04,T05,T06,T07 -->
### T12 · universal-scaling-law — is there one law for all EMA-of-statistics optimizers?

- **Question:** for T01–T07, is the optimal `α0` predicted by one shared form with an algorithm-specific constant? For example, is it governed by "effective points per unit" (n/k, n/topics, ratings/item)?
- **Input:** every `results/*/` JSONL file. No new training except to fill gaps.
- **Output:** a cross-algorithm fit, the residual of each algorithm, a "when the recipe fails" taxonomy built from the NEGATIVE.md files, and the main paper figure.
- **This is the flagship result:** it turns several "X-means for Y" papers into one general method paper.

<!-- track:T13 slug:drift-balanced-ivf phase:2 deps:T03,T08 -->
### T13 · drift-balanced-ivf — a live vector index that stays balanced and current

- **Combine:** T03 balancing + T08 drift rewarm. Maintain an IVF index under streaming inserts and an encoder upgrade, while keeping list sizes balanced.
- **Metrics:** recall and p99 latency over time, re-indexing compute vs periodic rebuild.

---

## 5. Suggested paper outputs

| Paper | Tracks | Venue type |
|---|---|---|
| "The e-means recipe is general" (flagship) | T01, T04–T07, T12 | ML (NeurIPS/ICML) |
| Fast, balanced, live vector indexes | T02, T03, T08, T13 | Systems / IR (VLDB, SIGIR, SIGMOD) |
| No-tuning ANN indexes via scaling laws | T11 | Datasets & benchmarks / systems |
| Seeding and adaptive k for streaming k-means | T09, T10 | Workshop or short paper |

## 6. Check-in cadence

- **Daily:** each worktree appends a line to `results/<slug>/LOG.md` (what ran, the best number so far).
- **Weekly:** `scripts/status.sh` prints the latest row of every track's log. Rebase all tracks onto `core`.
- **Kill review at week 2:** every track either shows its hypothesis holding on one dataset or writes NEGATIVE.md.
