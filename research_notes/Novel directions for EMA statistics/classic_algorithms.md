# Classic ML/statistics algorithms that could take the e-means recipe (EMA of statistics, cosine-annealed rate, dead-component revival, size-based scaling law)

Scope note: candidates here are NOT among the 13 existing tracks (GMM-EM, OPQ, balanced IVF, online dictlearn/NMF, online LDA, ALS, HMM Baum-Welch, drift re-warming, cheap seeding, adaptive k, HNSW scaling laws, universal scaling law, live drifting IVF). Citations of classic papers below rely on canonical venue/arXiv pages; the items explicitly confirmed by web search in this session are marked [verified]. Claims about "what the recipe would add", novelty and risk are my own judgement and are marked as inferences.

## Q1. Prior art that frames the whole class (what a "general tool" paper must position against)

### Takeaway
The recipe is mathematically a member of stochastic-approximation EM / stochastic variational inference (SVI): an EMA of sufficient statistics with step rho_t is exactly online EM (Sato 2000; Cappe-Moulines 2009) and, for conjugate-exponential models, SVI's natural-gradient step (Hoffman et al. 2013). EMA codebook k-means with dead-code reset is also standard in VQ-VAE practice. The novelty is therefore NOT the EMA update itself but (a) the cosine schedule + revival + scaling-law package, (b) systematic cross-model evidence, and (c) a law for (alpha0, B) as a function of (n, k, epochs) — none of which the prior art supplies.

### Cited Findings
- Incremental/sparse EM variants that update sufficient statistics from subsets of data were justified via the free-energy view of EM — Neal & Hinton 1998, [A view of the EM algorithm that justifies incremental, sparse, and other variants](https://www.cs.toronto.edu/~radford/ftp/emk.pdf).
- Online EM replaces the E-step by a stochastic-approximation update of the expected complete-data sufficient statistics, s_t = (1-gamma_t) s_{t-1} + gamma_t E[S(x_t)], then applies the usual M-step — Cappe & Moulines 2009 (JRSS-B), [arXiv:0712.4273](https://arxiv.org/abs/0712.4273). Extended to HMMs by Cappe 2011, [arXiv:0908.2359](https://arxiv.org/abs/0908.2359).
- Stepwise EM with mini-batches and step size (t+2)^-kappa was shown to be fast and competitive with batch EM on NLP models — Liang & Klein 2009, [Online EM for unsupervised models](https://aclanthology.org/N09-1069/).
- SVI: for conjugate-exponential models, the global natural-gradient step with step rho_t is a convex combination of the old global parameter and an intermediate estimate computed from a mini-batch — i.e. an EMA of (scaled) sufficient statistics — Hoffman, Blei, Wang & Paisley 2013, [JMLR 14](https://jmlr.org/papers/v14/hoffman13a.html).
- Sparse variational GPs with natural-gradient SVI over inducing-variable parameters scale GPs to millions of points — Hensman, Fusi & Lawrence 2013, [Gaussian Processes for Big Data, arXiv:1309.6835](https://arxiv.org/abs/1309.6835); natural gradients for GP variational parameters revisited by Salimbeni, Eleftheriadis & Hensman 2018, [arXiv:1803.09151](https://arxiv.org/abs/1803.09151).
- [verified] Mini-batch EM for exponential-family mixtures, built on the online EM framework, with convergence results; authors list "tuning learning rates and batch sizes" and extension to mixtures of regressions / mixtures of experts as future work — Nguyen, Forbes & McLachlan 2020, Statistics and Computing 30:731-748, [arXiv:1902.03335](https://arxiv.org/abs/1902.03335).
- [verified] Variance-reduced stochastic EM (sEM-vr) reaches batch-EM-like exponential asymptotic rate with a constant step size; tested on GMM and PLSA — Chen, Zhu, Teh & Zhang 2018, [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/aba22f748b1a6dff75bda4fd1ee9fe07-Abstract.html).
- Mini-batch k-means with per-center learning rate 1/count — Sculley 2010, [Web-scale k-means clustering (WWW)](https://dl.acm.org/doi/10.1145/1772690.1772862).
- [verified] VQ-VAE codebooks are trained by "an exponentially smoothed version of k-means" — Dieleman, van den Oord & Simonyan 2018, [arXiv:1806.10474](https://arxiv.org/pdf/1806.10474); EMA codebook update introduced in van den Oord, Vinyals & Kavukcuoglu 2017, [arXiv:1711.00937](https://arxiv.org/abs/1711.00937).
- [verified] Dead-code replacement (codes whose EMA cluster size falls below a threshold are re-initialised from a random vector of the current batch, SoundStream-style) is a default feature of the widely used vector-quantize-pytorch library — [vector-quantize-pytorch](https://github.com/GallagherCommaJack/vector-quantize-pytorch); random restarts for codebooks in Jukebox, Dhariwal et al. 2020, [arXiv:2005.00341](https://arxiv.org/abs/2005.00341); SoundStream, Zeghidour et al. 2021, [arXiv:2107.03312](https://arxiv.org/abs/2107.03312).

### Inferences
- A reviewer will say "this is online EM / SVI with a schedule". The paper's defensible contribution is the empirical scaling law and the cross-model recipe; frame it as "a practical, tuned SVI/online-EM recipe with a predictive (alpha0, B) law", and include Cappe-Moulines / Liang-Klein / sEM-vr as baselines (the project already does the first two).
- The finding that the k-means alpha0 law does not transfer to LDA is consistent with SVI theory: the effective step depends on how many data points each global statistic aggregates (the n/|B| rescaling in SVI) and on the per-statistic noise. A "universal" law likely needs a model-specific normaliser such as effective sample count per component (n/k for mixtures, tokens per topic for LDA). This is speculation that the candidates below could test.
- Nguyen-Forbes-McLachlan explicitly leave learning-rate/batch-size tuning open; the scaling law directly answers that open item.

### Gaps
- Did not verify whether any 2024-2026 paper fits a scaling law for online-EM step size vs. (n, k, epochs); I found none, but searched only indirectly.
- Fast incremental EM (FIEM, Karimi/Fort/Moulines line, ~2019-2020) not checked in this session.

## Q2. Linear-Gaussian latent models: PCA / PPCA / FA / MFA, incremental PCA, Oja, Frequent Directions, CCA / PLS

### Takeaway
The best candidate in this family is the mixture of factor analysers / mixture of PPCA: it is a GMM-like EM with richer per-component statistics, has dead-component problems, and has no widely used online-EM implementation (the recent large-scale work uses SGD instead). Single-component PCA/FA is largely solved by streaming PCA theory (Oja, block power method, Frequent Directions), so the payoff there is low except as a sanity check.

### Cited Findings
- EM for PCA/sensible PCA (alternating projection/re-estimation, sufficient statistics = E[z], E[zz^T], sum x E[z]^T) — Roweis 1998, [EM algorithms for PCA and SPCA](https://cs.nyu.edu/home/people/in_memoriam/roweis/papers/empca.pdf).
- EM for factor analysis — Rubin & Thayer 1982 (Psychometrika 47:69-76); EM for mixtures of factor analysers — Ghahramani & Hinton 1996, [tech report CRG-TR-96-1](http://mlg.eng.cam.ac.uk/zoubin/papers/tr-96-1.pdf); mixtures of probabilistic PCA — Tipping & Bishop 1999 (Neural Computation 11(2)).
- [verified] Gepperth 2023 trains MFA by SGD from random starts because batch EM is impractical at large n and k-means initialisation is a bottleneck; uses precision-matrix parameterisation; tested on SVHN/MNIST — [arXiv:2308.13778](https://arxiv.org/abs/2308.13778). The search found no dedicated online/stochastic EM for MFA.
- Oja's rule (1982, J. Math. Biol.) is the classic online PCA neuron; [verified] Mitliagkas, Caramanis & Jain 2013 give a block-stochastic power method (average within mini-batch blocks, update once per block) with O(kp) memory and O(p log p) sample complexity — [NeurIPS 2013](https://proceedings.nips.cc/paper_files/paper/2013/file/76cf99d3614e23eabab16fb27e944bf9-Paper.pdf).
- Frequent Directions deterministic streaming sketch — Liberty 2013, [arXiv:1206.0594](https://arxiv.org/abs/1206.0594); Ghashami, Liberty, Phillips & Woodruff 2016, [arXiv:1501.01711](https://arxiv.org/abs/1501.01711).
- Incremental PCA with forgetting factor for visual tracking — Ross, Lim, Lin & Yang 2008 (IJCV 77) (basis of scikit-learn's IncrementalPCA).
- Stochastic CCA — Arora, Marinov, Mianjy & Srebro 2017, [arXiv:1702.06818](https://arxiv.org/abs/1702.06818); stochastic PCA/PLS — Arora, Cotter, Livescu & Srebro 2012 (Allerton).

### Inferences (candidate cards)
- **Mixture of factor analysers / mixture of PPCA (HIGH payoff).** Statistics: per component r_k, sum r x, sum r E[z], sum r x E[z]^T, sum r E[zz^T]. Existing online: only generic online EM (Cappe-Moulines covers it in principle) and SGD (Gepperth 2023). Recipe adds: EMA of these statistics, cosine alpha, dead-component revival (MFA components collapse like GMM ones), scaling law with k and latent dim q. Why it matters: density modelling of embeddings, anomaly detection, a classic "GMM with low-rank covariances" that scales to high d. Novelty: first systematic online-EM MFA at million-scale with a step/batch law; also a direct extension of the GMM track (easy engineering). Risk of being done: low-moderate (generic online EM papers mention it, but no prominent large-scale benchmark found).
- **Streaming PCA / FA (LOW payoff).** EMA of covariance is literally a forgetting-factor covariance estimator; Mitliagkas' block power method is already "mini-batch + average". Little room for novelty except as a positive control for the scaling law (alpha0 vs. eigengap). Risk: high (well studied, with theory).
- **CCA / PLS (LOW-MEDIUM).** Statistics are cross-covariances; EMA of Cxx, Cyy, Cxy + periodic whitening is straightforward. Stochastic CCA has theory (Arora 2017). Novelty limited; could be a short appendix.

### Gaps
- Did not confirm whether an incremental-EM MFA implementation exists in R packages (e.g. EMMIXmfa) for large n.

## Q3. Temporal / structured models: Kalman smoother EM (LDS), switching state-space, CRFs

### Takeaway
Online EM for linear Gaussian state-space models exists (filter-based EM, stochastic-approximation EM with particle/forward smoothing), but a mini-batch-of-sequences EMA recipe for LDS is under-benchmarked and is the natural sibling of the HMM track. CRFs are discriminatively trained by gradient, not by sufficient statistics, so they are a poor fit.

### Cited Findings
- EM for LDS with an RTS-smoother E-step — Shumway & Stoffer 1982 (J. Time Series Analysis 3:253-264); Ghahramani & Hinton 1996, [Parameter estimation for linear dynamical systems, CRG-TR-96-2](http://mlg.eng.cam.ac.uk/zoubin/papers/tr-96-2.pdf); nonlinear extension Ghahramani & Roweis 1999, [Learning nonlinear dynamical systems using an EM algorithm](https://cs.nyu.edu/home/people/in_memoriam/roweis/papers/nlds.pdf).
- [verified] Elliott & Krishnamurthy (1999, IEEE TAC) derive finite-dimensional filters so the LDS EM can use filtered rather than smoothed quantities, enabling recursive EM — [summary/PDF](https://typeset.io/pdf/new-finite-dimensional-filters-for-parameter-estimation-of-2wz6d5irbf.pdf).
- [verified] Particle-based online EM for general state-space models (PaRIS), noting linear-Gaussian and finite-state models have tractable likelihoods — Olsson & Westerborn, [arXiv:1502.04822](https://arxiv.org/abs/1502.04822).
- Switching state-space models with variational EM — Ghahramani & Hinton 2000 (Neural Computation 12(4)).

### Inferences (candidate cards)
- **LDS / Kalman-smoother EM over mini-batches of sequences (MEDIUM-HIGH).** Statistics: sum E[x_t], E[x_t x_t^T], E[x_t x_{t-1}^T], y_t E[x_t]^T. Recipe: EMA over sequence mini-batches (multi-sequence datasets: sensor fleets, neural recordings, finance panels). Existing: online/recursive EM (per time step) and SVI for HMMs (Foti et al. 2014, not verified here). Novelty: the sequence-batch EMA + scaling law, revival is N/A (no components) unless using switching LDS. Risk: moderate.
- **Switching LDS / SLDS (MEDIUM, high risk of instability).** Combines HMM (already a track) with LDS; dead regimes are a known failure mode, so revival is meaningful. Engineering cost higher.
- **CRFs (LOW).** Training is gradient-based on conditional likelihood (no closed-form M-step); EMA-of-statistics framing does not apply cleanly. Skip.

### Gaps
- Did not verify SVI-for-HMM/LDS papers (Foti, Xu, Laird, Fox 2014 "Stochastic variational inference for hidden Markov models"; Johnson & Willsky 2014 SVI for HSMMs) in this session; they are likely relevant prior art for both the HMM track and an LDS track.

## Q4. Clustering relatives: kernel k-means, Nystrom spectral, mean-shift, DP-means, fuzzy c-means, co-clustering, PLSA, mixture of experts, vMF / spherical k-means, Bernoulli/multinomial mixtures

### Takeaway
The closest-to-e-means and highest-payoff candidates are (1) spherical k-means / vMF mixtures for normalised embeddings (direct users: vector search, embedding clustering; the e-means machinery transfers almost verbatim), (2) DP-means / Bayesian nonparametric hard clustering (revival and adaptive-k connect naturally), (3) kernel k-means, where a 2024 mini-batch paper exists but uses no EMA/schedule law, and (4) Bernoulli/multinomial mixtures and mixtures of experts, which are textbook EM cases with sparse prior online work.

### Cited Findings
- [verified] Mini-batch kernel k-means: per-iteration cost O~(k b^2) vs O(n^2), 10-100x speedups with little quality loss, O(log k) approximation with k-means++ init — Jourdan & Schwartzman, [arXiv:2410.05902](https://arxiv.org/abs/2410.05902) (listed as a NeurIPS 2026 poster: [neurips.cc](https://neurips.cc/virtual/2026/poster/156028)).
- Nystrom approximation for spectral grouping — Fowlkes, Belongie, Chung & Malik 2004 (IEEE TPAMI 26(2)).
- Mean shift — Comaniciu & Meer 2002 (IEEE TPAMI 24(5)).
- [verified] DP-means: small-variance limit of DP mixture Gibbs sampling, a k-means-like objective with a per-cluster penalty; batch algorithm — Kulis & Jordan 2012, [arXiv:1111.0352](https://arxiv.org/abs/1111.0352). Dynamic means for streaming/temporally evolving clusters — Campbell, Liu, Kulis, How & Carin 2013, [NeurIPS PDF](https://people.bu.edu/bkulis/pubs/nips_dynmeans.pdf). Parallel DP-means with optional online mode — Dinari & Freifeld 2022, [pdc-dp-means](https://github.com/BGU-CS-VIL/pdc-dp-means).
- [verified] Online spherical k-means (winner-take-all, annealed learning rate, forgetting factor) for text streams; online version reported significantly better than batch, especially with an annealing-type learning rate — Zhong 2005 (Neural Networks 18:790-798; IJCNN 2005); vMF mixture EM and spherical k-means as its limit — Banerjee, Dhillon, Ghosh & Sra 2005, [JMLR 6](https://jmlr.org/papers/v6/banerjee05a.html). [verified] Frequency-sensitive competitive learning for balanced clustering on hyperspheres (Banerjee & Ghosh) — [UMN record](https://experts.umn.edu/en/publications/frequency-sensitive-competitive-learning-for-scalable-balanced-cl/).
- Fuzzy c-means — Bezdek 1981 (book) / Bezdek, Ehrlich & Full 1984 (Computers & Geosciences); single-pass and online FCM for large data — Hore, Hall & Goldgof 2007 (FUZZ-IEEE).
- Information-theoretic co-clustering (alternating row/column assignment) — Dhillon, Mallela & Modha 2003 (KDD).
- PLSA via (tempered) EM — Hofmann 1999, [arXiv:1301.6705](https://arxiv.org/abs/1301.6705); sEM-vr evaluates stochastic EM on PLSA (Chen et al. 2018, above).
- Hierarchical mixtures of experts trained by EM — Jordan & Jacobs 1994 (Neural Computation 6(2)).
- Semi-supervised EM for multinomial naive-Bayes mixtures of text — Nigam, McCallum, Thrun & Mitchell 2000 (Machine Learning 39).

### Inferences (candidate cards)
- **Spherical k-means / vMF mixtures for embeddings (HIGH).** Statistics: per-cluster sum of unit vectors and counts (+ resultant length for kappa). Online versions exist (Zhong 2005, competitive learning) but with ad hoc annealing. Recipe adds: EMA sums, cosine alpha, revival, scaling law; and kappa estimation from EMA resultant length for the vMF (soft) case. Why it matters: cosine-similarity embeddings (LLM/sentence embeddings) are the dominant clustering workload; IVF indexes for inner-product search use spherical centroids. Novelty: modest technically but very high practical value; tightly couples to existing IVF tracks. Risk: moderate (spherical mini-batch k-means is in libraries like FAISS spherical option, but not with the law).
- **DP-means / BNP hard clustering (HIGH).** Statistics as k-means plus a "new cluster" rule with penalty lambda. Recipe: EMA statistics, revival = re-spawning, plus the adaptive-k track becomes principled (lambda instead of k); scaling law becomes alpha0, B = f(n, lambda, epochs). Existing: dynamic means (2013), PDC-DP-means online mode; streaming variational BNP (Campbell et al. 2015, not verified). Novelty: decent. Risk: low-moderate.
- **Kernel k-means (MEDIUM-HIGH).** Statistics live in feature space; with random Fourier features or Nystrom landmarks, they become explicit vectors so EMA applies directly. Jourdan & Schwartzman 2024 already do mini-batch kernel k-means with a Sculley-style learning rate; the recipe's schedule/law would be an incremental but clean comparison. Risk: moderate-high for "mini-batch", low for "EMA + law".
- **Spectral clustering via Nystrom (LOW-MEDIUM).** After embedding, it is k-means (already covered); the eigen part is streaming PCA. Little new.
- **Mean shift (LOW).** Mode-seeking on a KDE; no fixed set of components or global sufficient statistics; mini-batch KDE estimates are possible but the recipe does not fit naturally.
- **Fuzzy c-means (MEDIUM).** Membership-weighted sums and counts are exactly EMA-able (identical to soft GMM-style statistics). Large user base in remote sensing/medical imaging. Novelty modest (online FCM exists since 2007); risk moderate. Cheap to add.
- **Co-clustering / biclustering (MEDIUM).** Alternating row and column assignment with block-sum statistics; EMA over row-batches and column-batches is non-trivial (two coupled statistic sets) — a genuinely new test of whether the law handles two "n"s. Risk: low (little online co-clustering literature found, but not searched exhaustively).
- **PLSA (LOW as a separate track).** Very close to LDA (already a track) and used in sEM-vr. Useful only as an LDA ablation (EM vs. VB) to probe the 50-100x alpha0 discrepancy.
- **Mixture of experts EM (MEDIUM).** Gating is a multinomial logistic regression (no closed-form M-step), so the EMA applies only to experts' statistics (e.g. linear-Gaussian experts) while the gate needs IRLS/gradient steps. Explicitly listed as future work by Nguyen-Forbes-McLachlan 2020. Interesting bridge to modern MoE routers but heavier engineering.
- **Bernoulli / multinomial mixtures (MEDIUM, cheap).** Pure exponential-family mixtures — the cleanest test of a "universal" law on discrete data (binary images, market baskets, genotypes, survey answers). Low novelty per se, but valuable for the universality claim. Risk: low.

### Gaps
- Did not check FAISS/cuML APIs for spherical mini-batch k-means options in this session.
- Streaming variational BNP (Campbell, Straub, Fisher & How 2015) not verified here.

## Q5. Factorisation / quantisation / tensors / optimal transport: ICA, archetypal analysis, SOM / neural gas / LBG, CP-ALS / Tucker, matrix completion, Wasserstein barycenters, Sinkhorn clustering

### Takeaway
CP-ALS for tensors and free-support Wasserstein barycenters (which reduce to Lloyd when weights are uniform) are the most promising: both are alternating schemes with accumulable statistics, both have only partial online prior art. SOM/neural gas/LBG are historically the origin of the recipe (annealed learning rates, neighbourhood shrinkage) — useful as related work but low novelty. ICA (FastICA) uses fixed-point iterations on expectations that can be EMA-ed, but the field has mature online versions.

### Cited Findings
- [verified] OnlineCP: ALS-style online CP decomposition for higher-order tensors, projecting new time slices onto fixed factors; running time constant in history length vs. linear for batch — Zhou, Nguyen, Bailey, Jia & Davidson 2016, [KDD, DOI 10.1145/2939672.2939763](https://dblp1.uni-trier.de/rec/conf/kdd/ZhouVBJD16.html); [verified] a 2020 paper notes OnlineCP updates only the temporal mode, not non-temporal modes — [arXiv:2003.04497](https://arxiv.org/abs/2003.04497).
- Tensor decompositions survey (CP-ALS, Tucker/HOOI) — Kolda & Bader 2009 (SIAM Review 51(3)); randomized/sketched CP-ALS — Battaglino, Ballard & Kolda 2018, [arXiv:1701.06600](https://arxiv.org/abs/1701.06600).
- [verified] Free-support Wasserstein barycenters by alternating weight assignment and re-centring; with uniform weights equivalent to Lloyd — Cuturi & Doucet 2014, [arXiv:1310.4375](https://arxiv.org/abs/1310.4375). Stochastic barycenters with adjustable support — Claici, Chien & Solomon 2018, [ICML](https://proceedings.mlr.press/v80/claici18a.html). SGD in Wasserstein space incl. mini-batch variant — Backhoff-Veraguas, Fontbona, Rios & Tobar 2024, [arXiv:2201.04232](https://arxiv.org/abs/2201.04232).
- [verified] Online dictionary learning for approximate archetypal analysis with mini-batches; 400 archetypes from 300K pose samples — Mei, Wang & Zeng 2018, [ECCV](https://openaccess.thecvf.com/content_ECCV_2018/html/Jieru_Mei_Online_Dictionary_Learning_ECCV_2018_paper.html). Archetypal analysis introduced by Cutler & Breiman 1994 (Technometrics 36(4)).
- FastICA fixed-point algorithm — Hyvarinen 1999 (IEEE TNN 10(3)).
- Self-organising maps — Kohonen 1982 (Biological Cybernetics 43); neural gas — Martinetz, Berkovich & Schulten 1993 (IEEE TNN 4(4)); LBG vector quantiser design — Linde, Buzo & Gray 1980 (IEEE Trans. Commun. 28(1)).

### Inferences (candidate cards)
- **CP-ALS / Tucker-ALS on large tensors (HIGH).** Each ALS factor update solves normal equations with statistics MTTKRP (X_(n) (KR product)) and Hadamard of Gram matrices; with fibre/slice sampling these are mini-batch-estimable and EMA-able for ALL modes (fixing OnlineCP's temporal-only limitation). Users: recommender tensors, spatio-temporal sensor data, knowledge-graph tensors, chemometrics. Novelty: EMA-of-MTTKRP with cosine schedule + scaling law in (tensor size, rank, epochs) is not something I found. Close to the ALS track, so engineering is cheap. Risk: moderate (stochastic/sketched CP-ALS exists: Battaglino 2018 uses sampling but not EMA).
- **Free-support Wasserstein barycenters / OT-Lloyd (MEDIUM-HIGH).** Alternating Sinkhorn-plan assignment and re-centring = soft k-means with marginal constraints. EMA of transport-weighted sums per support point is natural; dead support points (zero mass) are exactly the revival case. Also covers balanced clustering via OT (relates to balanced IVF track). Novelty: decent; Claici 2018 and Backhoff 2024 are gradient/stochastic but no EMA-schedule-law. Risk: moderate.
- **Archetypal analysis (MEDIUM).** Mei et al. 2018 already put AA into the online dictionary-learning (Mairal) framework, which itself accumulates EMA-like matrices A_t, B_t. Since dictlearn is already a track, AA is mostly a variant. Risk: moderate-high.
- **SOM / neural gas / LBG (LOW novelty, HIGH historical relevance).** These are the earliest "online k-means with annealed rate" algorithms; SOM's batch version is EM-like with neighbourhood-weighted statistics. Must be cited as precursors; a short experiment showing the law also predicts SOM neighbourhood/learning-rate schedules would be a nice universality point but not headline.
- **FastICA (LOW-MEDIUM).** The fixed-point update w <- E[x g(w^T x)] - E[g'(w^T x)] w uses expectations that can be EMA-ed over batches; however the nonlinearity makes the statistic depend on current w (not a fixed exponential-family statistic). Online ICA has a long history (Bell-Sejnowski infomax, natural-gradient ICA by Amari et al. 1996, not verified here). Risk high.
- **Matrix completion beyond ALS (LOW).** Soft-Impute / nuclear-norm methods do not decompose into sufficient statistics cleanly; ALS is already a track.

### Gaps
- Did not check whether "stochastic ALS / mini-batch CP" papers 2021-2026 (e.g. CP-stream, SamBaTen) use EMA of MTTKRP; a targeted search is needed before claiming novelty.
- No verified source for online/mini-batch co-clustering.

## Q6. Robust / missing-data EM, IRT, mixed-membership and SBM community detection

### Takeaway
Two strong "new-audience" candidates: item response theory (psychometrics: massive online assessment logs; classic Bock-Aitkin EM is a quadrature-based sufficient-statistic EM) and stochastic block models / MMSB for graphs (SVI exists since 2013 but with Robbins-Monro steps and no size law). t-mixtures and missing-data EM are cheap extensions of the GMM track that support the "general tool" claim.

### Cited Findings
- EM for incomplete data (general framework) — Dempster, Laird & Rubin 1977 (JRSS-B 39).
- Robust mixtures of multivariate t via EM/ECM (latent scale weights as extra statistics) — Peel & McLachlan 2000 (Statistics and Computing 10).
- Marginal-ML EM for IRT item parameters (Bock-Aitkin), whose E-step accumulates expected counts at quadrature nodes — Bock & Aitkin 1981 (Psychometrika 46).
- [verified] Variational IRT (VIBO) for fast scalable inference, and a stochastic VI version applied to five large-scale item response datasets with better log-likelihood and imputation than alternatives — Wu, Davis, Domingue, Piech & Goodman 2020, [arXiv:2002.00276](https://arxiv.org/abs/2002.00276); 2021, [arXiv:2108.11579](https://arxiv.org/abs/2108.11579). The search found no stochastic/online EM work for IRT.
- MMSB — Airoldi, Blei, Fienberg & Xing 2008, [JMLR 9](https://jmlr.org/papers/v9/airoldi08a.html). [verified] SVI for overlapping communities: repeatedly subsample the network and update community estimates; scales to networks with hundreds of thousands to millions of nodes; svinet code — Gopalan & Blei 2013, [PNAS 110(36)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3767539); [svinet](https://github.com/premgopalan/svinet).

### Inferences (candidate cards)
- **IRT via quadrature EM (HIGH, new audience).** Statistics: per item, expected correct/incorrect counts at each quadrature node (r_ik, n_ik) — an exact exponential-family-like accumulation, so the EMA is natural; M-step is a small per-item logistic fit. Users: educational testing, adaptive learning platforms (Duolingo, MOOCs), LLM benchmark analysis (IRT is increasingly used to analyse benchmark items). Prior online work: SVI/VIBO (2020-21), no online EM found. Novelty: high in that community; risk low-moderate. Dead components do not apply, but "dead items/persons" do not either — revival is N/A; law becomes f(#persons, #items, epochs).
- **SBM / MMSB / degree-corrected SBM community detection (MEDIUM-HIGH).** Statistics: block-pair edge counts and node-community soft counts; mini-batches are node or edge samples. Dead communities are common (revival applies). Prior: Gopalan & Blei 2013 SVI with Robbins-Monro steps. Novelty: the law (alpha0, B vs. #nodes, #edges, K) and cosine schedule; risk moderate (SVI-MMSB is established, so must benchmark against svinet).
- **t-mixtures (MEDIUM, cheap).** Adds latent precision weights u_i; EMA of u-weighted sums. Shows recipe handles robust models and outliers (revival may be triggered by outliers — interesting failure mode). Risk low.
- **Missing-data GMM/FA EM (MEDIUM, cheap).** Conditional expectations of missing coordinates enter the statistics; EMA works unchanged. Practical for tabular imputation at scale. Risk low.

### Gaps
- Did not search for online EM for IRT under names like "stochastic approximation EM IRT" (MH-RM, Cai 2010, uses Robbins-Monro stochastic approximation — likely relevant prior art, not verified here).

## Q7. Ranking by expected payoff for a "the recipe is a general tool" paper

### Takeaway
Rank (my judgement, not a sourced fact): 1) mixtures of factor analysers / PPCA, 2) spherical k-means / vMF mixtures, 3) CP-ALS tensors, 4) IRT quadrature EM, 5) DP-means, 6) SBM/MMSB, 7) LDS Kalman-EM, 8) OT/Wasserstein barycenter Lloyd, 9) Bernoulli/multinomial + t-mixtures + missing-data EM (cheap universality points), 10) kernel k-means, 11) fuzzy c-means, co-clustering, mixture of experts; low: streaming PCA, CCA/PLS, ICA, SOM, mean-shift, CRFs, matrix completion, archetypal analysis.

### Cited Findings
- Prior-art density supporting the ranking: MFA has only an SGD large-scale method ([Gepperth 2023](https://arxiv.org/abs/2308.13778)); IRT has SVI but no online EM ([Wu et al. 2021](https://arxiv.org/abs/2108.11579)); OnlineCP handles only the temporal mode ([arXiv:2003.04497](https://arxiv.org/abs/2003.04497)); mini-batch kernel k-means exists ([Jourdan & Schwartzman 2024](https://arxiv.org/abs/2410.05902)); streaming PCA is well-theorised ([Mitliagkas et al. 2013](https://proceedings.nips.cc/paper_files/paper/2013/file/76cf99d3614e23eabab16fb27e944bf9-Paper.pdf)); SVI-MMSB is established ([Gopalan & Blei 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3767539)); EMA + dead-code reset for k-means codebooks is standard in VQ-VAE practice ([vector-quantize-pytorch](https://github.com/GallagherCommaJack/vector-quantize-pytorch)).

### Inferences
- Ranking criteria: (a) does the algorithm have per-component/global sufficient statistics with a closed-form M-step (needed for EMA to be exact), (b) does it have "components" that can die (revival matters), (c) user base and data scale, (d) thinness of existing online prior art, (e) engineering cost given the existing codebase (GMM, ALS, dictlearn, HMM, LDA tracks).
- Strategic suggestion: pick 3-4 tracks that span different statistic types to stress the universal law — continuous low-rank (MFA), directional (vMF), multilinear (CP-ALS), discrete/psychometric (IRT) — plus 2-3 cheap EM variants (Bernoulli mixture, t-mixture, missing-data GMM) as universality data points. These would directly test the hypothesis that a model-specific "effective samples per statistic" normaliser rescues the law (given the LDA 50-100x discrepancy).
- The alpha0 saturation at 1 while sweeps pick 0.07-0.27 suggests the fitted law's functional form is wrong at small n/k or high epochs; adding diverse models will either expose a common normaliser or show the law is model-specific — both are publishable outcomes but the paper must not claim universality before this test.
- Biggest single novelty risk for the whole paper: reviewers equating the recipe with SVI (Hoffman 2013) + VQ-VAE EMA/dead-code reset. The distinguishing claim must be the predictive scaling law and the cross-model benchmark at matched compute (epochs), not the update rule.

### Gaps
- No exhaustive 2025-2026 literature check per candidate; each "risk" rating should be re-checked with a targeted search before committing a track.
- Did not quantify user base sizes (e.g. downloads of IRT or tensor packages); the "why it matters" statements are qualitative.
