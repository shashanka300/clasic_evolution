# Theory of step sizes, averaging and variance reduction for online / mini-batch EM (stochastic approximation of sufficient statistics)

Scope note: the e-means recipe (EMA of per-cluster sums and counts, cosine-annealed EMA rate alpha0, dead-cluster revival, fitted law alpha0, B = f(n, k, epochs)) is, formally, Cappé-Moulines online EM run in sufficient-statistic space with a constant-then-annealed step size and mini-batches. Everything below is organized around that identification. Sources marked "(not fetched this session)" are well-known primary papers whose URLs are given for the report writer's convenience; I did not re-verify their content in this session. Their claims are kept to standard, widely cited statements.

Notation used in inferences: n = dataset size, B = batch size, alpha = EMA rate per step, s = sufficient statistics, f = fraction of missing information (eigenvalues of I_c^{-1} I_mis), Sigma_s = per-example covariance of the E-step statistic.

---

## Q1. Online EM theory: what is proven about online / stepwise / incremental EM, and how does it relate to natural gradient / mirror descent?

### Takeaway
Online EM in sufficient-statistic space (Cappé & Moulines 2009) is a Robbins-Monro recursion whose mean field, near the optimum, is a gradient step preconditioned by the inverse complete-data Fisher information; its Jacobian is H = I_c^{-1} (Hessian of KL), so its eigenvalues are exactly 1 minus the missing-information fractions. Step gamma_0 n^{-1} is rate-optimal only if gamma_0 exceeds a curvature-dependent threshold "usually impossible to check"; slowly decaying steps plus Polyak-Ruppert averaging give Fisher-efficient estimates without that condition. Batch EM is mirror descent with KL geometry (Kunstner et al. 2021), which is why its stochastic version behaves like a natural-gradient method.

### Cited Findings
- Online EM recursion: a stochastic-approximation E-step on sufficient statistics, followed by the usual M-step map; does not require the complete-data information matrix and is analyzed also for misspecified models (data not from the model). Converges to stationary points of KL(data || model) — [Cappé & Moulines 2009, JRSS-B, arXiv:0712.4273](https://arxiv.org/abs/0712.4273)
- Near convergence, online EM is equivalent to the "online gradient algorithm" theta <- theta + gamma I_c(theta)^{-1} grad log g(Y; theta), i.e. Titterington's (1984) recursion; for canonical exponential families the two coincide. "It is remarkable that the online EM algorithm can achieve a convergence performance similar to that of the online gradient algorithm without explicit matrix approximation nor inversion." — [Cappé & Moulines, Sec. 3](https://arxiv.org/pdf/0712.4273)
- Theorem 7 (Cappé & Moulines): with gamma_n = gamma_0 n^{-alpha}, gamma_0 is free in (0,1) for alpha in (1/2,1) but must exceed a curvature threshold when alpha = 1; then n^{1/2}-asymptotic normality with covariance solving a Lyapunov equation involving H = I_c^{-1} * Hessian(KL). The mean-field Jacobian H has real, strictly negative eigenvalues. Using alpha = 1 "provides the optimal convergence rate of 1/sqrt(n) but only at the price of a constraint on the scale gamma_0, which is usually impossible to check in practice." — [Cappé & Moulines](https://arxiv.org/pdf/0712.4273)
- With alpha in (1/2,1) and Polyak-Ruppert averaging of iterates, the averaged estimator attains the optimal sandwich covariance; under correct specification it equals the inverse observed Fisher information (asymptotically efficient). The authors recommend gamma_n ~ n^{-0.5-eps} plus averaging as "very robust"; in their mixture-of-regressions experiment, step n^{-1} showed significant bias, n^{-0.6} with averaging (started at n/2) matched the asymptotic interquartile range, while 5 iterations of batch EM were not sqrt(n)-consistent. — [Cappé & Moulines, Sec. 3-4](https://arxiv.org/pdf/0712.4273)
- Stepwise EM (Liang & Klein 2009): interpolates statistics with eta_k = (k+2)^{-alpha}, 0.5 <= alpha <= 1, with mini-batches; choice of step size and mini-batch size matters, and the authors report it can be chosen reasonably without supervision; stepwise EM outperformed incremental EM; on POS tagging, stepwise reached ~65.4% after 2 passes vs batch EM ~57.3% after 100 iterations (figures from a preview/secondary summary, verify in paper). — [Liang & Klein, NAACL 2009](https://aclanthology.org/N09-1069/); step-size form as quoted in [arXiv:1702.04938](https://arxiv.org/pdf/1702.04938)
- EM for exponential families is mirror descent with a KL Bregman divergence; this yields non-asymptotic convergence rates in KL that are invariant to any homeomorphic reparametrization (EM steps are identical under continuous invertible reparametrizations); gradient-descent-style analyses of EM are argued to be loose. Also covers local linear/superlinear rates and generalized EM. — [Kunstner, Kumar & Schmidt, AISTATS 2021](https://proceedings.mlr.press/v130/kunstner21a.html); [arXiv:2011.01170](https://arxiv.org/abs/2011.01170)
- Online variational Bayes (Sato 2001) is derived as a stochastic approximation to maximizing the free energy, with a convergence proof; the technical report discusses the link to Amari's natural gradient (the exact equivalence statement was not verified in this session). — [Sato, Neural Computation 13(7), 2001](https://mlanthology.org/neco/2001/sato2001neco-online); [ATR tech report](https://results.atr.jp/35th/techreport/TR-H-0282/index.html)
- Online LDA (Hoffman, Blei & Bach 2010) is online VB using a natural-gradient step, proven to converge to a local optimum of the VB objective; fit a 100-topic model to 3.3M Wikipedia articles in a single pass. — [Hoffman, Blei & Bach, NeurIPS 2010](https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation). scikit-learn's implementation defaults: learning_decay (kappa) = 0.7, learning_offset (tau0) = 10, batch_size = 128 — [sklearn docs mirror](https://www.ocaml.org/p/sklearn/latest/doc/Sklearn/Decomposition/LatentDirichletAllocation/index.html)
- Karimi, Wai, Moulines & Lavielle (2019) give the first global non-asymptotic analysis of incremental/stochastic EM, with two views (parameter space and sufficient-statistics space), treating EM as a scaled-gradient method; iEM rates correspond to vanilla gradient descent. — [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html); [arXiv:1910.12521](https://arxiv.org/pdf/1910.12521)
- Neal & Hinton (1998) incremental EM (free-energy view; per-example statistic replacement) — "A view of the EM algorithm that justifies incremental, sparse, and other variants" (not fetched this session) — [Learning in Graphical Models, 1998](https://www.cs.toronto.edu/~radford/ftp/emk.pdf)
- Hoffman, Blei, Wang & Paisley (2013) Stochastic Variational Inference: for conditionally conjugate exponential-family models, the natural gradient of the ELBO w.r.t. global natural parameters equals (expected sufficient statistics + prior) minus current parameter, so a natural-gradient step with rate rho is exactly an EMA of rescaled statistics (not fetched this session) — [JMLR 2013, arXiv:1206.7051](https://arxiv.org/abs/1206.7051)

### Inferences
- The e-means update S <- (1-alpha) S + alpha * (n/B) s_B is precisely SVI / stepwise EM with constant step alpha (then cosine-annealed). It is therefore a natural-gradient (mirror-descent-in-KL) method with step alpha; the "EMA rate" is a natural-gradient learning rate. This puts the whole recipe inside Cappé-Moulines / Hoffman et al. theory and means LR theory for natural-gradient SGD applies.
- Cappé-Moulines Theorem 7 implies the local contraction rate of the mean ODE in direction i is alpha * (1 - f_i), where f_i is the i-th missing-information fraction (eigenvalue of I_c^{-1} I_mis). With well-separated clusters (k-means limit, f -> 0) the mean field is a pure contraction with rate alpha; that is why alpha up to ~1 is stable for k-means.
- The cosine-annealed constant step is a practical stand-in for "large step then averaging"; Cappé-Moulines' own recommendation (slow decay + Polyak-Ruppert averaging) is the theoretically efficient alternative and is a direct baseline the paper should include.

### Gaps
- I did not find a paper that derives the optimal *finite-horizon constant* step for online EM (as opposed to asymptotic rate conditions). Cappé-Moulines only give asymptotic conditions.
- Exact Liang & Klein experimental sweep values (optimal alpha and minibatch per task) were not retrieved.
- Hoffman et al. 2010 tau0 / kappa sensitivity tables were not retrieved; a third-party page claims tau0 = 64-256 and kappa ~0.7 recommended, unverified.

---

## Q2. Variance-reduced and incremental EM: what do sEM-VR, FIEM, SPIDER-EM and averaging/momentum offer, and what is new (2022-2026)?

### Takeaway
Variance-reduced EM (sEM-vr, FIEM, SPIDER-EM, Geom-SPIDER-EM) lets a *constant* step reach the batch-EM optimum with provable complexity (SPIDER-EM: n + sqrt(n)/eps E-step evaluations), but all require periodic full or large anchor passes, so their advantage in a 1-3 epoch budget is unclear. Averaging (Polyak-Ruppert) is the proven route to efficiency for plain online EM; I found no 2023-2026 paper on momentum/schedule-free variants of online EM for mixtures specifically.

### Cited Findings
- sEM-vr (Chen, Zhu, Teh & Zhang 2018): SVRG-style control variate on the E-step statistics; same exponential asymptotic rate as batch EM and needs only a constant step size, reducing tuning. — [NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/aba22f748b1a6dff75bda4fd1ee9fe07-Abstract.html); [PDF](https://bigml.cs.tsinghua.edu.cn/~jianfei/static/sem_vr.pdf)
- FIEM (Karimi et al. 2019): SAGA-style fast incremental EM; a review summary gives O(n^{2/3}/eps) iteration complexity; fiEM and sEM-vr were faster than iEM in expectation. — [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html); [reviews](https://papers.neurips.cc/paper_files/paper/2019/file/a14ac55a4f27472c5d894ec1c3c743d2-Reviews.html)
- SPIDER-EM (Fort, Moulines & Wai 2020): SPIDER estimator applied to the E-step; K_Opt(n, eps) = O(1/eps) and K_CE(n, eps) = n + sqrt(n)/eps to reach an eps-stationary point, matching or improving sEM-vr and FIEM. — [NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/c589c3a8f99401b24b9380e86d939842-Paper.pdf); [arXiv:2012.01929](https://arxiv.org/pdf/2012.01929)
- Geom-SPIDER-EM: geometrically distributed outer-loop length, same bounds, plus linear convergence under a Polyak-Lojasiewicz condition; an ICASSP 2021 variant avoids the full data scan at each outer loop. — [arXiv:2011.12392](https://arxiv.org/abs/2011.12392v1)
- Fast incremental EM for finite sums with nonasymptotic bounds relies on a random termination rule and constant SA step (Fort, Gach & Moulines). — [arXiv:2012.14670](https://www.emergentmind.com/articles/2012.14670)
- Dieuleveut, Fort, Moulines & Wai (2023) give a unified stochastic-approximation framework with non-asymptotic results covering stochastic EM. — referenced via search summary of [arXiv:2302.11147](https://pdf.arxiv.org/pdf/2302.11147) (content not verified in this session)
- Karimi et al. biased stochastic approximation analysis covers online EM with potentially biased mean field. — [arXiv:1902.00629](https://pdf.arxiv.org/pdf/1902.00629)
- Mini-batch learning of exponential-family finite mixtures (Nguyen, Forbes & McLachlan 2019/2020) uses Polyak averaging for its learning-rate sequence. — [arXiv:1902.03335](https://arxiv.org/pdf/1902.03335)
- Polyak-averaged SGD with momentum: finite-sample rates and asymptotic normality (general SGD, not EM). — [arXiv:2305.17665](https://arxiv.org/abs/2305.17665)
- Constant-step SGD is a Markov chain; averaged iterates have an explicit expansion in step size (bias O(step) for non-quadratic objectives), and Richardson-Romberg extrapolation (two runs at steps gamma and 2*gamma, combined) cancels the leading bias. — [Dieuleveut, Durmus & Bach, Annals of Statistics 2020, arXiv:1707.06386](https://arxiv.org/abs/1707.06386v1)

### Inferences
- In the 1-3 epoch regime the e-means paper targets, SPIDER/SVRG anchors cost a large fraction of the budget; the theory (n + sqrt(n)/eps) is about asymptotic finite-sum complexity, not few-epoch loss. This is a plausible reason that VR methods are not standard in practice, but I found no paper comparing VR-EM vs constant-step EMA at matched 1-3 epochs. That is a cheap, publishable empirical check.
- The EMA of statistics is itself a "memory" control variate with geometric forgetting (SAGA with exponential discounting). The constant-step bias (Dieuleveut et al.) is the reason cosine annealing is needed at the end; Richardson-Romberg on two alpha values, or tail Polyak averaging of the M-step parameters, are principled replacements for the annealing tail.
- Speculation: a STORM-style recursive momentum estimator (Cutkosky & Orabona 2019) applied to E-step statistics would give variance reduction without anchor passes; I did not find this published for EM (gap; not verified exhaustively).

### Gaps
- No 2024-2026 paper found on momentum, schedule-free, or adaptive-step online EM for mixtures. Searches surfaced only general SA papers and a 2026 Scale-Mixture EM paper ([arXiv:2602.14280](https://www.arxiv.org/pdf/2602.14280)) and a 2026 paper on gradient methods for overparameterized GMMs ([arXiv:2605.30936](https://arxiv.org/pdf/2605.30936)), neither fetched.
- Whether SPIDER-EM's sqrt(n) term is competitive at 1-3 epochs is unknown.

---

## Q3. What determines the optimal step? Noise-to-curvature, Fisher information, and the fraction of missing information. Does missing information explain why LDA needs ~50-100x smaller alpha0 than k-means?

### Takeaway
Established: batch EM's local linear rate equals the largest fraction of missing information (Dempster-Laird-Rubin), and for Gaussian mixtures that fraction is controlled by component overlap (Ma, Xu & Jordan 2000), vanishing for well-separated clusters (k-means regime). Inferred: missing information slows the mean dynamics (rate alpha*(1-f)), which under standard SA theory calls for a *larger*, not smaller, step; so missing information alone does not explain LDA's much smaller alpha0. The more likely driver is the statistic-space noise-to-signal ratio per coordinate (sparse topic-word counts), i.e. a "gradient noise scale" for sufficient statistics.

### Cited Findings
- Batch EM local rate is linear and given by the amount of missing information under regularity conditions; recent work shows how the ratio of missing information affects non-asymptotic linear/superlinear rates near minimizers. — [Kunstner et al. arXiv:2011.01170](https://arxiv.org/pdf/2011.01170.pdf)
- Dempster, Laird & Rubin (1977): EM's rate matrix is I_c^{-1} I_mis (fraction of missing information) (not fetched this session) — [JRSS-B 1977](https://www.jstor.org/stable/2984875)
- For Gaussian mixtures, the local EM rate near the true solution is governed by an average overlap measure e(Theta*): the rate is o(e(Theta*)^{0.5-eps}), so EM becomes asymptotically superlinear (quasi-Newton-like) as overlap -> 0. — [Ma, Xu & Jordan, Neural Computation 12(12), 2000](https://mlanthology.org/neco/2001/ma2001neco-asymptotic); [PDF](https://www.math.pku.edu.cn/teachers/jwma/homepage/papers/nc-00.pdf)
- Xu & Jordan (1996): the EM effective Hessian (preconditioned) has smaller condition number than the raw log-likelihood Hessian. Ma & Xu (2005) extend to exponential-family mixtures. — summarized in search results for [Ma, Xu & Jordan 2000](https://mlanthology.org/neco/2001/ma2001neco-asymptotic) and [Ma & Xu 2005](https://www.math.pku.edu.cn/teachers/jwma/homepage/papers/neurocom2005_mx.pdf)
- Online EM's Jacobian H = I_c^{-1} * Hessian(KL); for a well-specified model Hessian(KL) = I_obs, so H = I_c^{-1} I_obs = Id - I_c^{-1} I_mis; with alpha = 1 schedule the scale gamma_0 must exceed a threshold set by H's smallest eigenvalue. — [Cappé & Moulines Theorem 7](https://arxiv.org/pdf/0712.4273)
- Larger online-EM steps speed convergence but may land in worse local maxima; smaller steps converge slower but help reach the global basin; alpha close to 1 often converges too slowly. — snippet from [stochastic approximation EM for item factor analysis, arXiv:1912.12755](https://arxiv.org/pdf/1912.12755)
- Gradient noise scale B_noise (a gradient signal-to-noise statistic, roughly tr(Sigma)/|G|^2) predicts the largest efficient (critical) batch size to within an order of magnitude across tasks; critical batch ranged from 20 (SVHN autoencoder) to >10M; noise scale grows as loss falls. — [McCandlish, Kaplan, Amodei et al. 2018, arXiv:1812.06162](https://arxiv.org/abs/1812.06162)

### Inferences
- Linearized model of e-means/stepwise EM (quadratic SA in the I_c metric): per eigendirection i with contraction c_i = 1 - f_i and per-step statistic noise variance sigma_i^2 / B, constant step alpha gives stationary error ~ alpha * sigma_i^2 / (2 c_i B) and bias decay exp(-alpha c_i t). With T = epochs * n / B steps the finite-horizon optimum is roughly alpha* ~ log(SNR_i * c_i * T) / (c_i * T). This (speculative, standard-SA-style) formula says: (i) alpha* grows as B grows (fewer steps) — consistent with a joint (alpha0, B) law; (ii) alpha* depends on the noise only through log(SNR), and on missing information through 1/c_i (so more missing information -> larger alpha); (iii) the bias-variance balance in statistic space, not curvature, is what can push alpha down by 50-100x.
- Why LDA plausibly needs much smaller alpha0: the per-coordinate signal of a topic-word statistic is ~ (n * L) / (K * V) expected tokens (L = doc length, V = vocab), typically orders of magnitude smaller than k-means' per-cluster count n/k aggregated over d dense dims; batch estimates of rare-word counts are Poisson-like with relative variance ~ 1/count. The EMA must therefore average over many more batches (memory B/alpha must hold enough tokens per (topic, word)) -> smaller alpha. A second LDA-specific factor: the M-step is the nonlinear exp(digamma(.)) map and the local E-step is itself iterative, so batch statistics are biased as well as noisy. This is a hypothesis to test, not a published result.
- Testable prediction: if alpha0 is reparametrized as an effective sample size per sufficient-statistic coordinate (n_eff = B(2-alpha)/alpha samples in the EMA, times expected support per coordinate), the k-means and LDA optima should collapse onto one curve. If they do not, missing information / M-step nonlinearity is the residual.
- Saturation at alpha0 = 1 while sweeps choose 0.07-0.27: in the noise-scale picture, alpha = 1 (no memory, pure mini-batch EM) is only optimal when B >> B_noise^S; the fitted law hitting the boundary indicates the functional form is mis-specified near the boundary (e.g. fitting alpha0 directly instead of log-timescale), or that the sweeps are noise-dominated where the law extrapolates.

### Gaps
- I found no paper that relates the optimal online-EM step to the fraction of missing information or to effective sample size per parameter (n/k). This appears to be an open gap (searched "online EM step size missing information"; only asymptotic rate conditions surfaced).
- No published "statistic-space gradient noise scale" for EM found.

---

## Q4. Hyperparameter-free / self-tuning schedules: schedule-free, D-adaptation/Prodigy, coin betting, Kesten, adaptive EMA. Could a sufficient-statistic noise scale predict B and alpha0?

### Takeaway
Schedule-free optimization (Defazio et al. 2024) replaces schedules with an interpolation between primal averaging and Polyak-Ruppert averaging, needs no stopping time, and keeps worst-case guarantees; it maps cleanly onto statistic-space EMA because e-means' EMA already is a primal average. Parameter-free methods (D-adaptation, Prodigy, coin betting) and Kesten-style sign tests are mature for SGD but I found no application to online EM. McCandlish's noise scale is computable from per-batch statistics and is the most direct candidate to predict B and alpha0.

### Cited Findings
- Schedule-Free: no LR schedule, no need to know stopping time T; interpolates between primal averaging (beta=1) and Polyak-Ruppert averaging (beta=0); retains worst-case PR-averaging rates while matching/exceeding schedules; no extra hyperparameters beyond momentum methods; Schedule-Free AdamW was core of the winning AlgoPerf 2024 self-tuning entry. Caveat from the package page: still requires tuning LR and regularization. — [Defazio et al., NeurIPS 2024, arXiv:2405.15682](https://arxiv.org/html/2405.15682v2); [code](https://github.com/facebookresearch/schedule_free/blob/main/README.md); [PyPI](https://pypi.org/project/schedulefree/1.2.7)
- Critical batch size from the gradient noise scale; adapting batch size during training improves efficiency. — [McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)
- D-adaptation (Defazio & Mishchenko 2023) and Prodigy (Mishchenko & Defazio 2023) estimate the distance-to-solution online to set the LR without tuning (not fetched this session) — [arXiv:2301.07733](https://arxiv.org/abs/2301.07733); [arXiv:2306.06101](https://arxiv.org/abs/2306.06101)
- Coin-betting parameter-free online learning (Orabona & Pal 2016) (not fetched this session) — [arXiv:1602.04128](https://arxiv.org/abs/1602.04128)
- Kesten (1958): decrease the Robbins-Monro step only when successive increments change sign (not fetched this session) — [Annals of Math. Stat. 29(1), 1958](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-29/issue-1/Accelerated-Stochastic-Approximation/10.1214/aoms/1177706705.full)
- Robust adaptive online EM updating sufficient statistics by stochastic approximation with a non-asymptotic analysis depending on the step size and an outlier threshold (Fukushima, Nitanda & Yamanishi 2020). — [arXiv:2007.12160](https://arxiv.org/pdf/2007.12160v1.pdf)

### Inferences
- Statistic-space noise scale (proposal): B_noise^S = tr(I_c^{-1/2} Sigma_s I_c^{-1/2}) / ||I_c^{-1/2}(s_bar(theta) - S)||^2, measured in the Fisher (mirror) metric so it is reparametrization-invariant in the Kunstner sense. Everything needed (per-batch statistics, running EMA) is already computed by e-means, so it costs one extra variance accumulator. Predict B* ~ B_noise^S and alpha* ~ min(1, B / B_noise^S) (speculative; the SGD analogue is that optimal LR scales ~ linearly with B below the critical batch, McCandlish et al.). This would explain saturation: alpha* = 1 only once B exceeds the statistic noise scale.
- Schedule-free EM (proposal): keep two statistic sequences, z_t (EMA with constant alpha, the "fast" iterate) and x_t (uniform/PR average of z), run E-steps at the interpolation y_t = (1-beta) z_t + beta x_t. This removes the cosine schedule and its dependence on the epoch budget, which is currently an input to the law.
- Kesten / Pflug-style rule in statistic space: shrink alpha when the sign of successive changes in M-step parameters (or of <Delta S_t, Delta S_{t-1}>) flips more than half the time. Cheap and hyperparameter-light; a natural "self-tuning alpha0" baseline.
- Per-cluster exact running means (alpha_k,t = B_k,t / N_k,t, i.e. the 1/count learning rate) are the "parameter-free" baseline for k-means (Sculley 2010 uses per-center 1/count rates; see Q6). A global alpha is mis-specified for imbalanced clusters; a per-component alpha derived from component-specific noise scale is a principled middle ground.

### Gaps
- No published application of schedule-free, D-adaptation, Prodigy, or coin-betting to online EM / sufficient statistics found.
- No published estimate of a noise scale in sufficient-statistic space found.

---

## Q5. Scaling laws and transfer for hyperparameters (muP, LR/batch/EMA-timescale laws) and analogues for classic algorithms

### Takeaway
The closest modern analogue to an "alpha0, B = f(n, k, epochs)" law is the AdamW EMA-timescale literature: Wang & Aitchison (ICML 2025) show AdamW weights are an EMA with timescale 1/(eta*lambda) whose optimum, measured in epochs, is roughly constant across model/data size; Bergsma et al. (NeurIPS 2025) refine this to tau = B/(eta*lambda*D) following a power law in tokens-per-parameter, with optimal and critical batch sizes scaling as power laws in D independent of N. The e-means EMA has the identical structure: memory in epochs = B/(alpha*n).

### Cited Findings
- AdamW weights can be viewed as an EMA of recent updates; lambda = 1/(eta*tau_iter); the optimal timescale in epochs is roughly constant as model and dataset size change; implies weight decay should fall as N grows under fixed schedule, and rise with width under muP LR scaling. — [Wang & Aitchison, ICML 2025, arXiv:2405.13698](https://arxiv.org/html/2405.13698v3); [PMLR](https://proceedings.mlr.press/v267/wang25b.html)
- Power Lines: tau = B/(eta*lambda*D); optimal tau follows a precise power law in tokens-per-parameter D/N (not constant); optimal and critical batch size scale as power laws in D, independent of model size N; fitted on hundreds of muP-trained models; new critical-batch estimator that works for any schedule/optimizer. — [Bergsma et al., NeurIPS 2025, arXiv:2505.13738](https://arxiv.org/abs/2505.13738)
- muP / hyperparameter transfer: parametrization under which optimal LR transfers across width (Yang et al. 2022, "Tensor Programs V") (not fetched this session) — [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
- Critical batch from gradient noise scale, rising as loss falls. — [McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)

### Inferences
- Direct mapping: e-means EMA memory tau_epochs = B / (alpha * n) (since the EMA averages ~1/alpha batches of size B). Wang-Aitchison predicts tau_epochs ~ constant; Bergsma predicts tau ~ power of "data per parameter", whose analogue is n / (number of sufficient-statistic coordinates) = n / (k * d) for k-means/GMM, n*L / (K * V) tokens per topic-word for LDA, n / (r * (m+u)) for ALS, etc. Re-fitting the e-means law as log(tau_epochs) = a + b * log(data per statistic coordinate) is a concrete, falsifiable reparametrization that could (i) remove the alpha0 = 1 boundary problem (tau has no upper bound issue, alpha = min(1, B/(tau n))), and (ii) explain the LDA 50-100x gap, since LDA has far fewer tokens per (topic, word) coordinate than k-means has points per (cluster, dim).
- muP-style transfer for classic algorithms would ask: under which normalization of statistics (per-cluster mean vs sum, per-token vs per-document) does alpha* become independent of k, d, V? The SVI normalization (rescale batch statistics by n/B) is the analogue of a "parametrization"; choosing per-coordinate effective-count normalization is a candidate muP analogue. Speculative.

### Gaps
- I found no published hyperparameter scaling/transfer law for classic EM-type algorithms (k-means, GMM, LDA, NMF, HMM). Searches returned only DL literature.
- Exact Bergsma exponents not retrieved.

---

## Q6. Mini-batch k-means convergence theory and dead-cluster / revival theory

### Takeaway
Schwartzman (ICLR 2023) proves mini-batch k-means with a termination test stops within O(d/eps) iterations w.h.p. if B = Omega~((d/eps)^2), regardless of initialization, and applies to sklearn's implementation; Sculley (2010) introduced per-center 1/count learning rates. Dead-cluster handling is mostly heuristic (VQ-VAE EMA codebook restarts when EMA count < 1); I found no theory linking the EMA decay to the probability of cluster death.

### Cited Findings
- Mini-batch k-means terminates within O(d/eps) iterations w.h.p. when batch size is Omega((d/eps)^2 log(nkd/eps)); holds for any initialization; with k-means++ seeding retains O(log k) approximation; results apply to sklearn's MiniBatchKMeans. — [Schwartzman, ICLR 2023, arXiv:2304.00419](https://arxiv.org/pdf/2304.00419)
- Extension to mini-batch kernel k-means (Jourdan & Schwartzman). — [arXiv:2410.05902](https://www.arxiv.org/abs/2410.05902?context=cs)
- Sculley (2010) "Web-scale k-means clustering": mini-batch k-means with per-center learning rate 1/v[c] (count of points assigned so far) (not fetched this session) — [WWW 2010](https://dl.acm.org/doi/10.1145/1772690.1772862)
- Bottou & Bengio (1995): online k-means as Newton's method / stochastic gradient with 1/count rates (not fetched this session) — [NeurIPS 1994](https://papers.nips.cc/paper/1994/hash/a1140a3d0df1c81e24ae954d935e8926-Abstract.html)
- VQ-VAE EMA codebooks: only codes selected in a mini-batch receive EMA updates; restart heuristics re-initialize dead codes with active encoder outputs (Razavi et al. 2019; Zheng & Vedaldi 2023); one recent rule re-initializes code k whenever its running usage count N_k < 1. — [VP-VAE arXiv:2602.17133](https://arxiv.org/pdf/2602.17133); [InsightTok arXiv:2605.14333](https://arxiv.org/pdf/2605.14333); [CVQ-VAE arXiv:2307.15139](https://arxiv.org/pdf/2307.15139v1.pdf)

### Inferences
- Under EMA with rate alpha, an unused cluster's count decays as (1-alpha)^t; with the N_k < 1 restart rule, a cluster of steady-state mass p_k*n is declared dead after ~log(p_k n)/alpha batches without hits. Probability of a run of zero hits for a cluster with true mass p_k is (1-p_k)^{B t}. Combining gives a closed-form false-revival / true-revival tradeoff in (alpha, B, p_k): revival threshold should scale with alpha and B, not be a constant. Provable and cheap; not found in literature.
- Schwartzman's batch requirement (d/eps)^2 is for memoryless mini-batch updates. EMA statistics average over ~2B/alpha points, so the same argument with "effective batch" 2B/alpha (and a bias term from stale assignments) should give termination with much smaller B. This is a natural theorem for the e-means paper: "EMA mini-batch k-means terminates within O(d/eps) iterations when B/alpha = Omega~((d/eps)^2)" (speculative statement; the staleness bias needs handling, e.g. via Lipschitzness of assignments).

### Gaps
- No theory of dead-cluster formation probability under EMA found.
- No convergence/termination analysis of EMA (constant-step) mini-batch k-means found; Schwartzman's analysis covers the plain variant.

---

## Q7. Concrete theory-driven research directions (with novelty gaps)

### Takeaway
Seven directions follow; the strongest combination for a paper is (D1) a statistic-space noise scale plus (D2) a timescale reparametrization, which together predict alpha0 and B per algorithm from measurable quantities, explain the LDA gap and the alpha0 = 1 saturation, and can be tested on existing sweep data. (D4) schedule-free EM and (D6) an EMA termination theorem are the most "provable novelty" candidates.

### Cited Findings
- Building blocks: online EM = natural-gradient SA with Jacobian I_c^{-1} I_obs [Cappé & Moulines](https://arxiv.org/abs/0712.4273); EM = KL mirror descent [Kunstner et al.](https://arxiv.org/abs/2011.01170); gradient noise scale [McCandlish et al.](https://arxiv.org/abs/1812.06162); EMA timescale laws [Wang & Aitchison](https://arxiv.org/html/2405.13698v3), [Bergsma et al.](https://arxiv.org/abs/2505.13738); schedule-free averaging [Defazio et al.](https://arxiv.org/html/2405.15682v2); VR-EM [SPIDER-EM](https://arxiv.org/pdf/2012.01929); termination [Schwartzman](https://arxiv.org/pdf/2304.00419); constant-step bias and Richardson-Romberg [Dieuleveut et al.](https://arxiv.org/abs/1707.06386v1)

### Inferences
- D1. Sufficient-statistic noise scale (explains B and alpha0). Define B_noise^S in the Fisher/mirror metric from per-batch statistic variance and the EMA's drift; predict B* ~ B_noise^S and alpha* ~ min(1, B/B_noise^S); measure it on k-means, GMM, LDA, ALS, HMM. Novelty gap: McCandlish's noise scale exists only for gradients; no EM / sufficient-statistic version found. Provable piece: in the linearized exponential-family model, derive the finite-horizon optimal constant alpha as a function of noise scale, horizon and missing-information spectrum (extends Cappé-Moulines Thm 7 from asymptotic to finite-horizon). Status: hypothesis.
- D2. Timescale reparametrization of the e-means law (explains transfer failure and saturation). Fit tau_epochs = B/(alpha n) as a power law in "data per sufficient-statistic coordinate" (n/(kd), nL/(KV), ...), following Bergsma et al. Test whether one law covers k-means and LDA. Novelty gap: no hyperparameter scaling law for classic EM algorithms found; the AdamW-EMA analogy is unexploited. Status: directly testable on existing sweeps.
- D3. Missing-information diagnostics (separates "epochs needed" from "step size"). Show theoretically (linear SA) that missing information sets the contraction rate alpha(1-f) and therefore the number of epochs, while noise sets alpha; estimate f online from the ratio of successive statistic changes (as in batch-EM rate estimation) and from Ma-Xu-Jordan overlap for GMMs. Novelty gap: no paper relating online-EM optimal step to missing information found. Prediction to test: more cluster overlap -> more epochs, not smaller alpha.
- D4. Schedule-free / averaged EM (hyperparameter-light). Replace cosine annealing with Defazio-style interpolation between constant-alpha EMA statistics and their uniform (Polyak-Ruppert) average, or with tail averaging of M-step outputs (Cappé-Moulines' recommendation), plus optional Richardson-Romberg across two alphas. Novelty gap: schedule-free not applied to EM in anything found; provable via the mirror-descent view in complete-data mean-parameter space (where M-step objective is convex) for the online-to-batch argument, local analysis otherwise. Removes "epochs" from the law's inputs.
- D5. Self-tuning alpha in statistic space. Kesten/Pflug sign tests on successive statistic increments, per-component alpha from per-component noise scale, or a D-adaptation-like distance estimate in KL. Baseline: per-cluster 1/count (Sculley). Novelty gap: no adaptive-step online EM found in 2024-2026 searches. Risk: may be dominated by D1+D2 if the law is accurate.
- D6. Termination / convergence theorem for EMA mini-batch k-means (provable novelty). Extend Schwartzman's O(d/eps) termination analysis to EMA statistics with effective batch ~2B/alpha and a staleness bias term; yields a theory-backed lower bound on B/alpha. Novelty gap: existing analysis covers memoryless mini-batch k-means only.
- D7. Dead-cluster theory. Closed-form probability of spurious vs genuine death under EMA counts as a function of (alpha, B, cluster mass), giving a principled revival threshold scaled by alpha and B instead of a constant; connects to VQ-VAE codebook-collapse literature where only heuristics exist. Novelty gap: no theory found.
- D8 (empirical-theory check, lower priority). Matched-budget comparison of VR-EM (sEM-vr, SPIDER-EM, Geom-SPIDER-EM ICASSP variant) against EMA statistics at 1-3 epochs. Novelty gap: VR-EM theory is asymptotic finite-sum complexity; no few-epoch comparison found.

### Gaps
- All directions above are proposals; none is an established result. The finite-horizon optimal-alpha formula in Q3 is a standard-SA heuristic I derived, not a cited theorem.
- I could not verify whether Sato (2001) formally proves equivalence to stochastic natural gradient; Hoffman et al. 2013 (SVI) is the standard citation for that equivalence (not fetched here).
- Literature search for 2024-2026 adaptive/momentum online EM was limited (a few queries); a dedicated arXiv listing search (stat.ML, "online EM", "stochastic EM", 2024-2026) is recommended before claiming novelty.
