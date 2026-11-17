# Prior-art positioning: mini-batch / online algorithms driven by an EMA of sufficient statistics

Research date: 2026-11-14. Scope: what is already claimed around the e-means recipe (EMA of per-component sufficient statistics + cosine-annealed rate + dead-component revival + size-based hyperparameter laws), what is open, and how a "the recipe is general" paper can establish its own identity.

Verification legend: "[verified]" means I opened the primary source (or its abstract/full text) this session. "[not opened]" means the citation is standard and the link is given, but I did not read the paper this session; the report writer should treat specific details from those as unverified.

---

## Q1. What exactly is the e-means paper, and who cites it?

### Takeaway
e-means is arXiv:2611.09501 (Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou; Meta FAIR; v1 submitted 6 Nov 2026). It is eight days old at the time of writing: I found no academic papers citing it, only a blog, a third-party reproduction critique, and a GitHub issue in an unrelated project adopting it. The paper does not mention extending the recipe to GMM/EM, PQ, dictionary learning or topic models, so the "recipe is general" claim is not pre-empted by the authors themselves.

### Cited Findings
- Exact title "e-means: outperforming k-means (in one-tenth the epochs)"; authors Swann Bessa, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Hugo Touvron, David Picard, Matthijs Douze, Hervé Jégou; submitted 6 Nov 2026; cs.LG/cs.AI [verified] — [arXiv abs](https://arxiv.org/abs/2611.09501)
- Headline claims: matches Lloyd's 25-epoch loss in 1.4-2.6 epochs on embedding datasets; 10-45x GPU wall-clock speedups over state-of-the-art k-means (9.5x-45.5x vs Flash-KMeans); clusters DINO10B (10B DINO embeddings) into 16M clusters in about 2.5 epochs with loss below Lloyd@25; often beats Lloyd's converged loss including with greedy k-means++ init over 40 seeds [verified] — [arXiv abs](https://arxiv.org/abs/2611.09501), [arXiv HTML](https://arxiv.org/html/2611.09501v1)
- Method: per-cluster count N_j and sum S_j, updated N_j <- gamma N_j + (1-gamma) n_j, S_j <- gamma S_j + (1-gamma) s_j; centroid c_j = S_j/N_j; step alpha = 1-gamma follows a cosine schedule alpha(t) = alpha0 (eps + (1-eps) * 0.5(1+cos(pi t/M))), eps = 0.1; the EMA rule is credited to van den Oord et al. (2017) VQ-VAE [verified] — [arXiv HTML](https://arxiv.org/html/2611.09501v1)
- Revival: every 5 batches, clusters with count < tau = 0.1 x mean count are rebuilt by splitting a donor chosen with probability proportional to count (Faiss procedure); without revival 26.3% of clusters stay dead on DEEP* at 1M points [verified] — [arXiv HTML (Appendix H)](https://arxiv.org/html/2611.09501v1)
- Init: N_j = 1e-10, S_j = N_j c_j^(0); uniform random data sampling (not k-means++). Finalization: "last-epoch mean" for train-optimized, slow EMAs with half-life h for out-of-sample [verified] — [arXiv HTML](https://arxiv.org/html/2611.09501v1)
- Scaling laws: alpha0 = min(a_c k^a_k B^a_B n^a_n T^a_T, 1) and B = b_c k^b_k n^b_n T^b_T (rounded to power of 2; targets the "knee", the largest batch within +0.2% of min validation MSE; divided by 8 for n > 2M). No d term (exponent unstable). Fit only on DEEP with 75k runs; R^2 = 0.95 (step) and 0.96 (batch); OOD regret 0.07% / 0.19%. Exact exponent values could not be extracted from the HTML rendering [verified, exponents = gap] — [arXiv HTML](https://arxiv.org/html/2611.09501v1)
- Stated limitations: laws tuned for high-dimensional data, not optimal for very low-d or tiny datasets; local intrinsic dimension matters but is unknown; test MSE can rise in late epochs in some high-d synthetic cases (overfitting relative to Lloyd); fit only on DEEP [verified] — [arXiv HTML](https://arxiv.org/html/2611.09501v1)
- Appendix E: Adam, SparseAdam and SGD centroid updates all degrade relative to EMA at large batch; SGD collapses beyond a critical learning rate, attributed to steps leaving the convex hull of a cluster's points [verified] — [arXiv HTML, offset 100k](https://arxiv.org/html/2611.09501v1)
- Related work cited by e-means: MacQueen 1967, Sculley 2010, Newling & Fleuret 2016 (nested mini-batch), Spark StreamingKMeans, Bidaurrazaga et al. 2021, Zhu et al. 2024 (staleness reduction); Bottou & Bengio 1994/95, So et al. 2022 (online k-means convergence); Robbins & Monro 1951; van den Oord 2017; neural codec revival (Zeghidour 2021, Défossez 2022, Dhariwal 2020, ...); Łańcucki et al. 2020; Neal & Hinton 1998 and Nowlan 1991 (EMA centroids as hard-EM with decayed statistics); Bjorck et al. 2025 (LR scaling across token horizons). The paper says none of the mini-batch k-means variants gives a way to set batch size or step size for a given problem [verified] — [arXiv HTML](https://arxiv.org/html/2611.09501v1)
- Notably absent from e-means related work (as extracted): Cappé & Moulines 2009, Sato & Ishii 2000, Liang & Klein 2009, Hoffman et al. 2010/2013, Mairal et al. 2010 [inferred from the extracted related-work summary; a full-text grep should confirm] — [arXiv HTML](https://arxiv.org/html/2611.09501v1)
- The paper mentions IVF only as a way to accelerate assignment (Appendix M.2), not as an extension; GMM/EM/PQ/dictionary learning/topic models are not mentioned [verified] — [arXiv HTML, offset 100k](https://arxiv.org/html/2611.09501v1)
- Origin story (blog): grew out of Touvron/Jégou finding that gradient-descent mini-batch k-means could beat Lloyd but was unstable at large LR, while EMA updates were not — [e-means blog](https://e-means.github.io/)
- Third-party reproduction (dev.to, unofficial NumPy reimplementation): 1.51x (not 10x) faster on 100k x 128-d synthetic blobs, k=1024, with 1.42% worse MSE; 1.03x and slightly worse on structureless data; first attempt failed due to data-point init and missing revival ("the update rule is not the algorithm"). Author reported faiss/contrib/e_means returned 404 at the time — [dev.to critique](https://dev.to/frankchu/10x-faster-k-means-i-re-ran-it-and-got-15x-24bb); conflicts with search results listing [faiss contrib/e_means on GitHub](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means) (likely published after the critique; not verified)
- Adoption signal: a GitHub issue in an unrelated pathology project proposes using e-means for consensus k-means with large k — [GitHub issue](https://github.com/Honzawa-Ryu/toxpatho-patch-selector/issues/16)
- Douze LinkedIn post calls it "the most significant iteration of k-means in decades" (promotional, not evidence) — [LinkedIn](https://www.linkedin.com/posts/matthijs-douze-750980177_e-means-is-the-most-significant-iteration-activity-7525980070767730688-leQ6)

### Inferences
- Because e-means omits the online-EM lineage (Sato-Ishii, Cappé-Moulines, Liang-Klein), a paper that explicitly reconnects e-means to stepwise EM has room to contribute; but the same omission means reviewers will see "e-means generalized to EM" as rediscovering stepwise EM unless the new paper states the equivalence up front.
- The project's own results (stepwise EM close behind; alpha0 law fails on LDA) are consistent with the dev.to finding that gains are regime-dependent; the paper should report regime boundaries rather than a blanket 10x.

### Gaps
- Exact fitted exponents for the alpha0 and B laws (table garbled in HTML extraction; read the PDF).
- Citation graph: paper is 8 days old; Semantic Scholar/Google Scholar citations not checked and almost certainly ~0. Re-check before submission.
- Whether faiss/contrib/e_means is now live (conflicting reports).

---

## Q2. Which prior work on EMA / online sufficient statistics already covers our claims?

### Takeaway
The core mechanism (stochastic-approximation update of expected sufficient statistics, s <- (1-eta) s + eta s_batch, followed by a closed-form M-step) is stepwise / online EM, established by Sato & Ishii 2000, generalized by Cappé & Moulines 2009, and shown empirically across four model families by Liang & Klein 2009, who already reported 2-10x speedups, sometimes better optima than batch EM, and that the best (step power, mini-batch size) is task- and dataset-dependent. "e-means for EM" is therefore not new as an algorithm; the cosine schedule, the revival rule, and the size-based hyperparameter laws are the only plausibly new ingredients.

### Cited Findings
- Liang & Klein, "Online EM for Unsupervised Models", NAACL-HLT 2009, pp. 611-619. Studies incremental EM (Neal & Hinton 1998) and stepwise EM (Sato & Ishii 2000; Cappé & Moulines 2009) on four tasks: POS tagging (bigram HMM, WSJ), document classification (multinomial mixture, 20 Newsgroups), word segmentation (English and Chinese), word alignment (IBM Model 1) [verified] — [ACL Anthology](https://aclanthology.org/N09-1069/)
- Liang & Klein: stepwise EM update with step eta_k = (k+2)^(-alpha); mini-batches of size m "interpolate between a pure online (m=1) and a pure batch (m=n)"; default rule picks (alpha, m) from {0.5,...,1.0} x {1,...,10K} to maximize log-likelihood (no labels needed) [verified] — [ACL Anthology PDF](https://aclanthology.org/N09-1069.pdf)
- Liang & Klein: across all five datasets stepwise EM reaches at least comparable likelihood/accuracy as batch EM "anywhere from about 2 (word alignment) to 10 (POS tagging) times faster"; POS: batch EM 57.3% after 100 iterations vs stepwise 65.4% after two [verified] — [ACL Anthology PDF](https://aclanthology.org/N09-1069.pdf)
- Liang & Klein: "In general, the best (alpha, m) depends on the task and dataset" (large m critical for document classification; small alpha and m better for POS tagging); incremental EM and (alpha,m)=(1,1) are "prone to catastrophic failure" [verified] — [ACL Anthology PDF](https://aclanthology.org/N09-1069.pdf)
- Liang & Klein: sEM improves accuracy on "clustering" tasks but only modestly on "structural" tasks (segmentation, alignment), left as an open problem [verified] — [ACL Anthology PDF](https://aclanthology.org/N09-1069.pdf)
- Cappé & Moulines 2009 (JRSS-B), "Online EM algorithm for latent data models": online EM interpreted as a stochastic approximation recursion on sufficient statistics; compare gamma_n = n^-0.6 vs 1/n, with 1/n showing large bias; descends from Titterington 1984 [verified via PDF snippet] — [arXiv 0712.4273](https://arxiv.org/pdf/0712.4273), [Berkeley copy](https://people.eecs.berkeley.edu/~jordan/sail/readings/cappe-moulines.pdf)
- Karimi, Wai, Moulines, Lavielle (NeurIPS 2019): EM as scaled-gradient method in sufficient-statistics space; first global non-asymptotic analysis of online/stochastic EM, plus fast incremental EM (fiEM) — [NeurIPS 2019](https://papers.NeurIPS.cc/paper_files/paper/2019/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
- Fort, Moulines, Wai (NeurIPS 2020) SPIDER-EM: unified stochastic-approximation framework covering online EM, sEM-vr, FIEM, SPIDER-EM for exponential-family models; experiments use constant step 5e-3 and suggest a few online-EM epochs before switching to variance-reduced methods — [arXiv 2012.01929](https://arxiv.org/pdf/2012.01929)
- Nguyen, Forbes, McLachlan (arXiv 2019), "Mini-batch learning of exponential family finite mixture models": mini-batch EM built on the online EM framework, a stochastic stabilization scheme, convergence results; applied to normal mixtures and MNIST [verified abstract] — [arXiv 1902.03335](https://arxiv.org/abs/1902.03335)
- Hoffman, Blei, Bach (NeurIPS 2010), online LDA: natural-gradient stochastic optimization of the VB objective, converges to a local optimum; fit 100 topics on 3.3M Wikipedia articles in a single pass; "as good or better" than batch VB in a fraction of the time [verified abstract]; step rho_t = (tau0 + t)^(-kappa) with kappa in (0.5, 1] (secondary sources; paper's own sensitivity numbers not verified) — [NeurIPS 2010](https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation), [metricgate secondary](https://metricgate.com/docs/online-lda-topic-streaming/)
- Fast online EM for big topic modeling (Zeng et al.; arXiv 1210.2179) and a WWW 2015 comparison of batch/incremental/online EM, CVB0, BP for LDA — [arXiv 1210.2179](https://arxiv.org/abs/1210.2179), [WWW 2015](https://archives.iw3c2.org/www2015/documents/proceedings/proceedings/p669.pdf)
- Mairal, Bach, Ponce, Sapiro (JMLR 2010), online dictionary learning / matrix factorization: accumulates past sufficient statistics (A = sum alpha alpha^T, B = sum x alpha^T) with optional down-weighting; replaces seldom-used atoms with randomly chosen training samples (snippet-level, exact rule not verified) — [JMLR PDF](https://www.di.ens.fr/~fbach/mairal10a.pdf), [arXiv 0908.0050](https://arxiv.org/pdf/0908.0050v1.pdf)
- Scalable Extreme Deconvolution (2019): GPU minibatch GMM via Cappé-Moulines online EM and via direct gradient — [arXiv 1911.11663](https://arxiv.org/pdf/1911.11663)
- Schwartzman (ICLR 2023): mini-batch k-means terminates within O(d/eps) iterations with batch Omega~((d/eps)^2) without LR decaying to zero; k-means++ init gives O(log k) approximation; follow-up mini-batch kernel k-means (arXiv 2410.05902) compares his LR to sklearn's — [arXiv 2304.00419](https://arxiv.org/pdf/2304.00419), [arXiv 2410.05902](https://arxiv.org/pdf/2410.05902)
- [not opened] Neal & Hinton 1998 "A view of the EM algorithm that justifies incremental, sparse, and other variants" (cited by Liang & Klein, verified as a citation) — [ACL PDF reference list](https://aclanthology.org/N09-1069.pdf)
- [not opened] Sato & Ishii 2000, "On-line EM algorithm for the normalized Gaussian network", Neural Computation 12:407-432 (cited by Liang & Klein, verified as a citation); Sato 2001 online VB model selection not opened — [ACL PDF reference list](https://aclanthology.org/N09-1069.pdf)
- [not opened] Sculley 2010 "Web-scale k-means clustering" (WWW) — per-center learning rate 1/count; Bottou & Bengio 1995 "Convergence properties of the k-means algorithms" (NIPS); Hoffman, Blei, Wang, Paisley 2013 "Stochastic variational inference" (JMLR); van den Oord et al. 2017 VQ-VAE EMA. Cited inside e-means (Sculley, Bottou-Bengio, van den Oord verified as citations there) — [e-means HTML](https://arxiv.org/html/2611.09501v1)

### Claim-by-claim status (our claims vs literature)
| Our claim | Status | Nearest prior art |
|---|---|---|
| EMA of sufficient statistics + closed-form M-step works for GMM, HMM, mixtures, LDA | TAKEN | Sato & Ishii 2000; Cappé & Moulines 2009; Liang & Klein 2009 (HMM, multinomial mixture, segmentation, IBM1); Hoffman 2010 (LDA); Nguyen et al. 2019 (exp-family mixtures) |
| Same for dictionary learning / NMF | TAKEN | Mairal et al. 2010 (statistics A, B accumulated online) |
| Online variant reaches batch quality in ~1-3 passes, sometimes better optimum | LARGELY TAKEN | Liang & Klein 2009 (2-10x faster, better accuracy on clustering tasks); Hoffman 2010 (single pass); e-means for k-means |
| Hyperparameters matter and differ per task | TAKEN (as negative result) | Liang & Klein 2009: best (alpha, m) "depends on the task and dataset" |
| Cosine-annealed EMA rate for EM-family models | NOT FOUND (absence of evidence) | e-means (k-means only); classic online EM uses power-law decay (t+tau0)^-kappa |
| Dead-component revival as a model-agnostic plug-in for EM-family online learners | NOT FOUND as unified treatment | per-model fixes (Faiss/e-means split, Mairal atom replacement, VQ restarts, SAE AuxK) |
| Closed-form size-based law alpha0, B = f(n, k, T) across model families | NOT FOUND | e-means (k-means only); LLM step laws (Li et al. 2025); symbolic defaults (Gijsbers et al. 2021) |
| Cross-family benchmark with time-to-baseline-loss metric | NOT FOUND | Liang & Klein (4 NLP tasks, 2009); per-model papers |
| EMA-trained balanced IVF / OPQ / drift re-warm | Partially (balanced k-means, streaming k-means with forgetting exist; not opened this session) | Spark StreamingKMeans, Bidaurrazaga et al. 2021 (cited in e-means) |

### Inferences
- The project's finding "tuned stepwise EM (Liang & Klein) is close behind" is the most important positioning fact: the delta attributable to e-means-specific ingredients (cosine schedule, revival, finalization, laws) over a well-tuned 2009 baseline is what a paper can legitimately claim, and it is small.
- The finding that the k-means alpha0 law does not transfer to LDA echoes Liang & Klein's task-dependence observation; a paper claiming a "universal law" will be checked against this.

### Gaps
- Did not verify Hoffman 2010's own kappa/tau0/batch-size sensitivity figures, nor SVI 2013 details.
- Did not find a 2023-2026 "stochastic EM revisited" paper with empirical cross-model focus; searches returned only 2019-2020 theory (Karimi 2019, Fort 2020). Absence of evidence, not proof.
- Did not open Sculley 2010 or Bottou & Bengio 1995 directly.

---

## Q3. Is there prior work on scaling laws / universal step-size laws for classical (non-neural) algorithms, or hyperparameter transfer for classic ML?

### Takeaway
Hyperparameter scaling laws as closed-form functions of problem size are well established for neural pretraining (Kaplan 2020; Step Law 2025) and e-means introduces them for k-means; I found no prior work fitting such laws for online EM, LDA, HMM, GMM, dictionary learning, or across classical model families. Data-dependent / symbolic defaults for classic ML exist (Probst et al. 2019; Gijsbers et al. 2021) but are learned from dataset meta-features over a benchmark, not derived from (n, k, T, B) scaling, and do not target optimizer step sizes of EM-family algorithms.

### Cited Findings
- Step Law (Li et al., 2025): optimal batch size depends primarily on dataset size D; optimal LR depends jointly on N and D; fit from 120 LLMs; positioned as "plug-and-play" tuning-free formula — [arXiv 2503.04715](https://arxiv.org/html/2503.04715v1)
- Kaplan et al. 2020: critical batch size as compute/time tradeoff, roughly a power law in loss — [arXiv 2001.08361](https://arxiv.org/pdf/2001.08361)
- A 2025 paper cautions that batch-size-with-dataset-size laws may fail because limited fitting ranges bias the fits — [arXiv 2509.25049](https://arxiv.org/pdf/2509.25049)
- e-means fits alpha0 and B laws on DEEP only and evaluates transfer on held-out datasets; it tried Bjorck et al. 2025's polynomial-valley LR fit and did not adopt it — [e-means HTML](https://arxiv.org/html/2611.09501v1)
- Probst, Boulesteix, Bischl (JMLR 2019) "Tunability": data-based defaults and tunability measures, 38 OpenML datasets, 6 algorithms — [JMLR PDF](https://jmlr2020.csail.mit.edu/papers/volume20/18-444/18-444.pdf)
- Gijsbers, Pfisterer, van Rijn, Bischl, Vanschoren (2021) "Meta-Learning for Symbolic Hyperparameter Defaults": formulas over dataset meta-features found by evolutionary search; six algorithms, 100+ datasets; notes prior symbolic defaults were hand-crafted and single-hyperparameter — [arXiv 2106.05767](https://arxiv.org/abs/2106.05767v1)
- Search for scaling laws for SVM/RF/GBM/k-NN hyperparameters returned only LLM work — [search result summary; Step Law](https://step-law.github.io/)
- Liang & Klein 2009 report task-dependent optimal (alpha, m), i.e., no universal setting, but propose an unsupervised selection rule (maximize log-likelihood) — [ACL PDF](https://aclanthology.org/N09-1069.pdf)
- Cappé & Moulines 2009 and SPIDER-EM use fixed schedules (n^-0.6; constant 5e-3) chosen by hand — [arXiv 0712.4273](https://arxiv.org/pdf/0712.4273), [arXiv 2012.01929](https://arxiv.org/pdf/2012.01929)
- SAE scaling (Gao et al. 2024, ICLR 2025 oral): scaling laws in latent count and k for sparse autoencoders (neural, but dictionary-learning-like) — [arXiv 2406.04093](https://arxiv.org/html/2406.04093v1)

### Inferences
- A "universal" law is unclaimed, but our own evidence (LDA non-transfer) argues against universality in raw (n, k, T) coordinates. A defensible novel claim is a law in a dimensionless/normalized coordinate, e.g., alpha0 expressed via effective number of samples per component per update (B/k), statistic noise-to-signal ratio, or "EMA memory in epochs" (1/(alpha0 * n/B)), that collapses across families where raw laws do not. This is untested in literature (absence of evidence).
- The natural framing for reviewers: "e-means laws = Step-Law-style hyperparameter scaling for stochastic-approximation EM"; cite Li et al. 2025 and Kaplan 2020 as analogies, Probst/Gijsbers as the classical-ML meta-learning alternative to beat.

### Gaps
- No search coverage of AutoML for clustering hyperparameters (e.g., AutoClust-type systems) or of "learning-rate-free" stochastic EM; possible prior art there unverified.
- No theory found that predicts optimal online-EM step size from (n, k, T); Karimi 2019/Fort 2020 give rates, not optimal constants (not checked in detail).

---

## Q4. Is there a unified treatment of dead components across models (empty clusters, GMM collapse, dead topics, dead atoms, VQ codebook collapse, SAE dead latents)?

### Takeaway
Each community has its own fix: Faiss/e-means split-a-donor revival for k-means, Mairal's replacement of unused atoms with training samples, random restart / resampling from features for VQ codebooks (CVQ-VAE, VQ-EMA restart), auxiliary losses (AuxK) and resampling for SAEs. I found no paper that treats these as one phenomenon with a shared detector and a shared, model-agnostic revival operator on sufficient statistics. This is a credible novel framing.

### Cited Findings
- e-means revival: count < 0.1 x mean every 5 batches; split donor sampled proportional to count; 26.3% dead clusters without revival on DEEP* 1M — [e-means HTML](https://arxiv.org/html/2611.09501v1)
- e-means cites neural audio codecs (Zeghidour 2021, Défossez 2022, Dhariwal 2020, Zhang 2024, Ji 2025) and Łańcucki et al. 2020 for dead-centroid revival in VQ — [e-means HTML](https://arxiv.org/html/2611.09501v1)
- Mairal et al.: some atoms are never/seldom used, typically after bad initialization; replaced during optimization by randomly chosen training elements (snippet-level) — [JMLR PDF](https://www.di.ens.fr/~fbach/mairal10a.pdf), [ar5iv](https://ar5iv.arxiv.org/html/0908.0050)
- CVQ-VAE / Online Clustered Codebook (Zheng & Vedaldi 2023): dynamically re-initializes unused codes by resampling from learned features — [arXiv 2307.15139](https://huggingface.co/papers/2307.15139.md)
- VQ-EMA with random codebook restart used in recent tokenizers (InsightTok) — [arXiv 2605.14333](https://arxiv.org/pdf/2605.14333)
- Gao et al. 2024 SAEs: share of dead latents grows with size; fix = encoder init as decoder transpose + AuxK auxiliary loss on top-k_aux dead latents; dead = no activation in 10M tokens — [arXiv 2406.04093](https://web3.arxiv.org/pdf/2406.04093)
- Critiques that SAE fixes (resampling, aux losses) treat symptoms rather than geometric root cause; AuxK sometimes barely moves dead fraction — [arXiv 2603.18056 / 2601.22244 via search summary](https://arxiv.org/pdf/2603.18056), [LessWrong](https://www.lesswrong.com/posts/EC5H7xakTgxxNXm5g/sanity-checks-for-sanity-checks-for-sparse-autoencoders)
- Liang & Klein: incremental EM and aggressive stepwise settings can "fail catastrophically in face of local optima" (a related but distinct failure mode) — [ACL PDF](https://aclanthology.org/N09-1069.pdf)
- Search for a single paper unifying dead SAE latents and dead VQ codes returned none — [search summary sources: arXiv 2601.12391, 2307.15139](https://huggingface.co/papers/2307.15139.md)

### Inferences
- In EMA-of-statistics learners the dead-component signal is uniform: the zeroth-order statistic N_j (EMA count / responsibility mass / topic mass / atom usage A_jj) falls below a threshold. A revival operator defined on (N_j, first-order stats) works identically for k-means, GMM (also needs covariance reset), LDA topics (lambda rows), dictionary atoms (A, B columns), and VQ codebooks. This gives a clean, model-agnostic statement that I did not find in the literature.
- GMM component collapse (variance -> 0 singularities) is a different failure (over-concentration, not starvation); a unified paper should distinguish "starvation" (dead) from "degeneracy" (collapsed) components. I did not search GMM-degeneracy literature this session (e.g., Bayesian regularization of covariances), so cite with care.

### Gaps
- Did not verify Mairal's exact replacement rule, nor Łańcucki 2020 details.
- Did not search "topic death" in online LDA specifically; no source found either way.
- Did not search GMM covariance-degeneracy literature (Fraley & Raftery etc.).

---

## Q5. Is there a standard benchmark for classic unsupervised algorithms at scale (time-to-baseline-loss across families)? Would a "ClassicBench" be novel?

### Takeaway
Clustering benchmark batteries exist (Gagolewski's framework, SoftwareX 2022) but evaluate partition quality on mostly small datasets for clustering only; per-model online-EM comparisons exist (Liang & Klein 2009 for NLP models; WWW 2015 for LDA; fiEM GMM experiments) but use incompatible protocols. I found no benchmark measuring time/epochs-to-batch-baseline objective across mixture models, topic models, HMMs, dictionary learning and vector-index training. A cross-family "time-to-baseline" suite would be novel (absence of evidence).

### Cited Findings
- Gagolewski 2022, "A framework for benchmarking clustering algorithms" (SoftwareX 20, 101270): aggregates and standardises clustering batteries, new datasets of varying dimensionality/size, consistent evaluation; notes many papers use few datasets — [GitHub](https://www.github.com/gagolews/clustering-benchmarks), [DOAJ](https://doaj.org/article/4c4a2231cc4b4a14bb1dc6cfd5adf59d)
- No source found describing a time-to-target benchmark for unsupervised algorithms — [search summary](https://pypi.org/project/clustering-benchmarks)
- Cross-model online-EM results are scattered and not directly comparable (different datasets/metrics/hyperparameters) — [ACL N09-1069](https://aclanthology.org/N09-1069.pdf), [arXiv 1210.2179](https://arxiv.org/pdf/1210.2179), [WWW 2015](https://archives.iw3c2.org/www2015/documents/proceedings/proceedings/p669.pdf), [arXiv 2212.05402](https://arxiv.org/pdf/2212.05402)
- e-means' metric (epochs and wall-clock to reach Lloyd@25 loss) on a fixed suite (SIFT, DEEP, SPACEV, SSNPP, DINO10B, VIBE sets) is effectively a k-means-only time-to-baseline benchmark — [e-means HTML](https://arxiv.org/html/2611.09501v1)
- ANN benchmarks exist for vector search (VIBE datasets used by e-means), but those measure search quality, not training-time-to-loss — [e-means HTML](https://arxiv.org/html/2611.09501v1)

### Inferences
- The analog that reviewers will recognize is AlgoPerf (time-to-target for neural optimizers). A "time-to-batch-EM-objective" suite for latent-variable models, with fixed baselines (batch EM @ 25 iterations, tuned stepwise EM per Liang & Klein, Hoffman online VB, Mairal ODL, Flash-KMeans/Flash-GMM), would be a distinct contribution even if algorithmic novelty is small. I did not verify AlgoPerf details this session.

### Gaps
- Did not check OpenML/UCI-scale unsupervised benchmarks or "TopicBench"-style topic-model benchmarks.

---

## Q6. Recent (2023-2026) papers and venues revisiting classic algorithms with modern ML engineering

### Takeaway
2025-2026 shows a clear "classic algorithms on modern GPUs" wave: Flash-KMeans (NeurIPS 2026 poster), Flash-GMM (2026), sublinear variational GMM at billions of parameters (2025), and e-means itself (Meta FAIR). These are mostly systems/kernels papers (exact algorithms, faster dataflow); e-means is the algorithmic one. None generalizes the EMA recipe across model families.

### Cited Findings
- Flash-KMeans (Yang, Xu et al., arXiv 2603.09229, NeurIPS 2026 poster): exact Lloyd with FlashAssign (fused distance + online argmin) and Sort-Inverse Update; up to 17.9x end-to-end on H200, 33x vs cuML, >200x vs FAISS; out-of-core to 1B points — [arXiv 2603.09229](https://arxiv.org/pdf/2603.09229), [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/151963)
- Flash-GMM (arXiv 2606.10896, 2026): fused Triton kernel for GMM in one GPU pass, avoids materializing responsibilities; notes TorchGMM OOMs beyond 10M points; ~20x speedup reported (secondary) — [HF papers](https://huggingface.co/papers/2606.10896.md)
- Sublinear variational GMM (arXiv 2501.12299, 2025): >10B-parameter GMM on billion-scale image data in ~9 h on one CPU, via mixtures of factor analyzers — [arXiv 2501.12299](https://arxiv.org/pdf/2501.12299)
- e-means compares against Flash-KMeans (9.5-45.5x faster on GPU) — [e-means HTML](https://arxiv.org/html/2611.09501v1)
- Schwartzman ICLR 2023 and mini-batch kernel k-means 2024 show ICLR/NeurIPS acceptance of mini-batch classic-algorithm theory — [arXiv 2304.00419](https://arxiv.org/pdf/2304.00419), [arXiv 2410.05902](https://arxiv.org/pdf/2410.05902)
- Improved seeding for k-means and k-GMM (arXiv 2506.21291, 2025) — [arXiv 2506.21291](https://arxiv.org/pdf/2506.21291)

### Inferences
- Venues: NeurIPS/ICML (Flash-KMeans, Karimi 2019, Fort 2020 at NeurIPS), ICLR (Schwartzman 2023; Gao SAE 2025), JMLR/TMLR for a benchmark-plus-theory paper; AISTATS for stochastic-EM theory; SIGMOD/VLDB or MLSys for vector-index tracks.
- Flash-GMM is a direct competitor for any "GMM at scale on GPU" claim: the GMM track must compare against it (EMA mini-batch on top of a fused kernel is complementary, which is a selling point).

### Gaps
- Did not find a 2023-2026 paper titled like "EM in the era of deep learning" or "stochastic EM revisited" with empirical focus; absence of evidence.

---

## Q7. Positioning map: ranked identity-defining framings and likely reviewer objections

### Takeaway
"The e-means recipe is general" alone is weak: the mechanism is stepwise EM (2000/2009) and its generality across model families was shown by Liang & Klein (2009). Identity must come from what 2009-era work did not have: (a) a principled, transferable hyperparameter law (and an honest account of where it fails), (b) a model-agnostic dead-component theory/operator, (c) a cross-family time-to-baseline benchmark at modern scale, and (d) the vector-index applications. The strongest single paper combines (a)+(c) with stepwise EM as the named, tuned baseline.

### Cited Findings (nearest prior art per framing)
Ranked framings (novelty argument -> nearest prior art):

1. **"Stochastic-approximation EM, re-tuned: a cross-family scaling law in normalized coordinates (and where it breaks)."** Claim: alpha0 and B laws expressed in dimensionless quantities (samples per component per step, EMA memory in epochs, statistic SNR) collapse k-means, GMM, HMM, ODL, LDA onto one curve, or provably cannot (LDA as the counterexample with mechanism). Novelty: no prior closed-form step-size law for online EM across families found. Nearest: e-means laws (k-means only) [arXiv 2611.09501](https://arxiv.org/html/2611.09501v1); Liang & Klein task-dependent (alpha, m) [ACL](https://aclanthology.org/N09-1069.pdf); Step Law for LLMs [arXiv 2503.04715](https://arxiv.org/html/2503.04715v1); symbolic defaults [arXiv 2106.05767](https://arxiv.org/abs/2106.05767v1).
2. **"EMBench / time-to-batch-EM: a benchmark for online latent-variable learning at scale."** Fixed protocol (epochs and wall-clock to batch-EM@25 objective), datasets from 1M to 1B, baselines incl. tuned stepwise EM, online VB, ODL, Flash-KMeans/Flash-GMM. Novelty: no cross-family time-to-target benchmark found. Nearest: Gagolewski clustering framework [GitHub](https://www.github.com/gagolews/clustering-benchmarks); Liang & Klein's 4-task study; e-means' k-means suite.
3. **"Dead-component dynamics in EMA-of-statistics learners: one detector, one revival operator."** Unifies empty clusters, starved GMM components, dead topics, unused atoms, dead VQ codes, (and by analogy) SAE dead latents, via the zeroth-order statistic; includes theory of when starvation occurs as a function of alpha and B/k. Novelty: no unified treatment found. Nearest: e-means/Faiss split revival; Mairal atom replacement [JMLR](https://www.di.ens.fr/~fbach/mairal10a.pdf); CVQ-VAE [2307.15139](https://huggingface.co/papers/2307.15139.md); SAE AuxK [2406.04093](https://web3.arxiv.org/pdf/2406.04093).
4. **"Why does stepwise/EMA EM find better optima than batch EM?"** Liang & Klein left this open (better accuracy on clustering tasks, not structural ones); e-means also beats Lloyd's converged loss. A mechanistic answer (noise-as-annealing + revival, as a function of alpha0) is open. Nearest: Liang & Klein 2009 [ACL](https://aclanthology.org/N09-1069.pdf); e-means 40-seed results.
5. **"Schedule matters: cosine vs Robbins-Monro power decay for online EM."** Classic online EM uses (t+tau0)^-kappa satisfying Robbins-Monro; e-means uses cosine to a floor (eps=0.1), which violates sum-of-squares conditions yet works in finite budgets. A finite-budget analysis/ablation across families is open. Nearest: Cappé & Moulines 2009 [arXiv 0712.4273](https://arxiv.org/pdf/0712.4273); Karimi 2019 [NeurIPS](https://papers.NeurIPS.cc/paper_files/paper/2019/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf); Schwartzman 2023 (non-vanishing LR) [arXiv 2304.00419](https://arxiv.org/pdf/2304.00419).
6. **"EMA statistics for vector-index construction" (balanced IVF via mini-batch Sinkhorn, online OPQ rotation from EMA cross-covariance, drift re-warm).** Application-domain identity aimed at SIGMOD/VLDB/MLSys. Nearest: e-means (IVF only as accelerator); Spark StreamingKMeans and Bidaurrazaga 2021 (drift) cited in e-means [HTML](https://arxiv.org/html/2611.09501v1). Balanced k-means and OPQ prior art not verified this session.
7. **"Hyperparameter laws for ANN graph indices (HNSW M, efConstruction)."** Separate identity; no prior law-based tuning found in this session's searches (not specifically searched; treat as unverified).
8. **"A field guide: when does online EM beat batch EM?"** Negative-results-forward paper reporting regimes where gains vanish (low-d, small n, structural models, LDA law failure), consistent with the dev.to reproduction (1.5x, not 10x) [dev.to](https://dev.to/frankchu/10x-faster-k-means-i-re-ran-it-and-got-15x-24bb) and Liang & Klein's clustering-vs-structural split. Good TMLR/JMLR fit.

Likely reviewer objections (with citations):
- "This is stepwise EM (Sato & Ishii 2000; Cappé & Moulines 2009) under a new name" — the EMA update is exactly the stochastic-approximation recursion on sufficient statistics [arXiv 0712.4273](https://arxiv.org/pdf/0712.4273). Mitigation: name it stepwise EM, credit lineage, isolate e-means-specific ingredients by ablation.
- "Liang & Klein 2009 already showed generality and 2-10x speedups across four models" [ACL](https://aclanthology.org/N09-1069.pdf). Mitigation: tuned stepwise EM as a primary baseline (already observed "close behind"); claim only the measured delta.
- "LDA online is Hoffman et al. 2010; dictionary learning is Mairal et al. 2010" [NeurIPS 2010](https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation), [JMLR](https://www.di.ens.fr/~fbach/mairal10a.pdf). Mitigation: use them as the per-family baselines.
- "Your 'universal' law doesn't hold — your own LDA results show it" (and Liang & Klein: optimal settings are task-dependent). Mitigation: reframe as normalized-coordinate law with explicit failure analysis.
- "Scaling laws fit on one dataset may not extrapolate" — e-means fit only on DEEP [HTML](https://arxiv.org/html/2611.09501v1); limited-range fits bias laws [arXiv 2509.25049](https://arxiv.org/pdf/2509.25049).
- "Speedups are regime-dependent / implementation-dependent" — independent reproduction found ~1.5x [dev.to](https://dev.to/frankchu/10x-faster-k-means-i-re-ran-it-and-got-15x-24bb). Mitigation: report wall-clock with fused kernels and epochs separately; include low-d/small-n regimes.
- "Compare against fast exact GPU baselines" — Flash-KMeans [arXiv 2603.09229](https://arxiv.org/pdf/2603.09229), Flash-GMM [HF](https://huggingface.co/papers/2606.10896.md).
- "Variance-reduced stochastic EM (sEM-vr, FIEM, SPIDER-EM) is the modern baseline, not plain online EM" [arXiv 2012.01929](https://arxiv.org/pdf/2012.01929), [Karimi 2019](https://papers.NeurIPS.cc/paper_files/paper/2019/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf).
- "Mini-batch EM for exponential-family mixtures with convergence theory already exists" — Nguyen, Forbes, McLachlan 2019 [arXiv 1902.03335](https://arxiv.org/abs/1902.03335).
- "Concurrency/novelty window" — e-means is 8 days old; a follow-up generalizing it may be seen as incremental unless it brings theory or a benchmark.

### Inferences
- Recommended flagship identity: framings 1 + 2 together ("a benchmark and a normalized hyperparameter law for stochastic-approximation EM across latent-variable models"), with framing 3 as a key mechanism section and 4 as the scientific question. Present e-means and stepwise EM as two points in one design space (schedule shape x step scale x batch x revival), and make the "close behind" baseline a feature, not a threat.
- Retitle away from "the e-means recipe is general" toward something like "Stochastic-approximation EM at scale: laws, failure modes, and a benchmark", which pre-empts the "rename of stepwise EM" objection.

### Gaps
- Not checked: Semantic Scholar citation graphs of Liang & Klein 2009 and Cappé & Moulines 2009 for 2023-2026 follow-ups that might already fit step-size laws (most important remaining check before claiming novelty for framing 1).
- Not checked: AlgoPerf details, balanced-k-means/OPQ/HNSW-tuning prior art, GMM degeneracy literature, topic-death literature, Sato 2001 online VB.
- e-means exponent values not extracted.
