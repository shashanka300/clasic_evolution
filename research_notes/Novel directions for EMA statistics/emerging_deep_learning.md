# Emerging deep-learning methods with k-means / EM / dictionary-learning inner loops: where "EMA of sufficient statistics done right, with scaling laws" fits

Scope note: e-means recipe = mini-batch EMA of per-cluster sums/counts, cosine-annealed EMA rate alpha0, dead-cluster revival, scaling law (alpha0, B) = f(n, k, epochs). Citations marked [verified] were checked against the page during this research session (Nov 2026). Citations marked [known] are well-established primary papers cited from background knowledge with their canonical arXiv IDs; their specific details were not re-fetched here. Search budget was limited (~16 calls), so 2025-2026 coverage is a sample, not exhaustive.

## Q1. VQ-VAE / VQ-GAN / RVQ codec tokenizers: how codebooks are updated, failure modes, what our recipe changes, novelty gap

### Takeaway
VQ codebooks with EMA updates are literally online k-means with EMA sufficient statistics (cluster sums + counts) plus ad-hoc dead-code restarts. The field still uses a **fixed** EMA decay (0.8-0.99), a fixed dead-code threshold, and no batch/codebook-size scaling rule. A live 2026 practitioner bug report shows that the dead-code threshold silently breaks across batch sizes. This is the highest-impact, most direct transfer of our recipe. The novelty gap: nobody has published a schedule plus scaling law for (decay, threshold, batch) as a function of (codebook size K, vectors/step, steps). Most 2024-2026 work instead abandons EMA (FSQ, LFQ, SimVQ, rotation trick, IBQ, NS-VQ/TransVQ, VP-VAE). That gives a clear contrarian thesis: "EMA done right is competitive with the reparameterization fixes."

### Cited Findings
**Current update rule**
- VQ-VAE (van den Oord, Vinyals, Kavukcuoglu, 2017) introduced the codebook/commitment losses and an appendix EMA variant of codebook updates [known] — [arXiv 1711.00937](https://arxiv.org/abs/1711.00937)
- VQ-VAE-2 (Razavi, van den Oord, Vinyals, 2019) used EMA codebook updates (EMA of per-code counts N_k and sums m_k, e_k = m_k/N_k, decay gamma ~0.99) [known] — [arXiv 1906.00446](https://arxiv.org/abs/1906.00446)
- Łańcucki et al. (2020), "Robust Training of Vector Quantized Bottleneck Models", proposed periodic k-means re-initialization of the codebook to fight underuse. This is a direct precursor to "revival" [verified in search listing] — [arXiv 2005.08520](https://arxiv.org/pdf/2005.08520)
- SoundStream (Zeghidour et al., 2021) uses residual VQ with EMA codebooks, k-means init on the first batch, and replacement of codes unused for several batches with random batch vectors [known; "dynamic code replacement" confirmed as cited in NS-VQ, below] — [arXiv 2107.03312](https://arxiv.org/abs/2107.03312)
- EnCodec (Défossez et al., 2022) and AudioCraft/MusicGen use the same RVQ with EMA decay and `threshold_ema_dead_code` style expiry [known; code file listed in search] — [arXiv 2210.13438](https://arxiv.org/abs/2210.13438); [audiocraft core_vq.py](https://huggingface.co/spaces/facebook/MusicGen/blob/a31f19b770bcc507e5ebb468cf2077a73951bc99/audiocraft/quantization/core_vq.py)
- The widely used vector-quantize-pytorch library defaults to EMA decay 0.8. Published tokenizer setups use 0.9, 0.96 or 0.99 with no stated rule [verified via search summary] — [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)

**Failure modes and their importance**
- **Batch-size-dependent dead-code threshold (2026).** GitHub issue #257 (24 Sept 2026) shows that `threshold_ema_dead_code` is compared against an EMA of *per-batch* counts. Its steady-state scale is vectors_per_step / K, so a threshold of 2 copied from SoundStream/EnCodec large-batch configs reset 867-1011 of 1024 codes per layer *per step* in an 8x1024 RVQ toy at 800 vectors/step. In a real 9.8M-param speech codec, threshold 0.2 vs 2 raised layer-1 perplexity from 563 to 842 and lowered Whisper WER at 0.5 kbps from 14.5% to 11.9% (about 18% relative). No maintainer response was visible. The proposed fixes are documentation, a warning, or a "relative threshold" — [Issue #257](https://github.com/lucidrains/vector-quantize-pytorch/issues/257) [verified]
- VQGAN-LC (Zhu et al., NeurIPS 2024) reports that the optimal codebook size for VQGAN, VQGAN-FC and VQGAN-EMA is 16,384, versus 100,000 for its frozen-feature lookup variant. That means EMA VQ stops scaling with K [verified via search snippet] — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/1716d022edeac750e57a2986a7135e13-Paper-Conference.pdf)
- A 2026 video-compression preprint reports that gradient-based VQ collapsed at K<=512 while EMA plus restarts kept full utilization. Its restart rule re-initializes any code with N_k < 1.0 from a random batch encoder output [verified via search snippet] — [arXiv 2607.02562](https://arxiv.org/html/2607.02562)
- A large-codebook speech tokenizer paper attributes index collapse to sparse codebook gradients and uses product quantization [verified via search listing] — [arXiv 2406.02940](https://arxiv.org/pdf/2406.02940)

**Fixes that avoid or replace EMA (nearest prior art / competitors)**
- ViT-VQGAN (Yu et al., 2021): low-dimensional, l2-normalized factorized codes [known] — [arXiv 2110.04627](https://arxiv.org/abs/2110.04627). DAC (Kumar et al., 2023) adopts factorized and l2-normalized codes for RVQ [known] — [arXiv 2306.06546](https://arxiv.org/abs/2306.06546)
- CVQ-VAE (Zheng & Vedaldi, ICCV 2023), "Online Clustered Codebook": re-anchors under-used codes toward encoder features, using a running average of code usage. **This is the closest prior art to our revival component** [known] — [arXiv 2307.15139](https://arxiv.org/abs/2307.15139)
- Huh et al. (2023), affine reparameterization plus alternating optimization for VQ [known] — [arXiv 2305.08842](https://arxiv.org/abs/2305.08842)
- FSQ (Mentzer et al., 2023) removes the codebook entirely [known] — [arXiv 2309.15505](https://arxiv.org/abs/2309.15505). LFQ / MAGVIT-v2 (Yu et al., 2023) also removes it [known] — [arXiv 2310.05737](https://arxiv.org/abs/2310.05737)
- SimVQ (Zhu et al., 2024): a linear reparameterization of the codebook [known] — [arXiv 2411.02038](https://arxiv.org/abs/2411.02038). Rotation trick (Fifty et al., 2024) [known] — [arXiv 2410.06424](https://arxiv.org/abs/2410.06424). IBQ (Shi et al., 2024) [known] — [arXiv 2412.02692](https://arxiv.org/abs/2412.02692)
- **2026, NS-VQ / TransVQ** (Lu, Koyun, Guo, Zhu, Alili, Gurcan; arXiv 2602.18896, 21 Feb 2026) argues that encoder non-stationarity is the root cause of collapse. NS-VQ applies kernel-weighted updates to *non-selected* codes; TransVQ maps the codebook through a small transformer. Its toy EMA baseline uses a *fixed* alpha = 0.3 and never updates unassigned codes. It uses no annealed EMA decay and no explicit restarts. It shows rFID worsening with batch size (Fig. 4). Results on CelebA-HQ (K=8912): TransVQ rFID 13.70, NS-VQ 14.01, SimVQ 14.37 [verified] — [arXiv 2602.18896](https://arxiv.org/html/2602.18896v1)
- 2026 "NSVQ: Mitigating Codebook Collapse by Stabilizing Encoder Drift" (arXiv 2606.11363) is another drift-centric fix [title only, verified in listing] — [arXiv 2606.11363](https://arxiv.org/html/2606.11363)
- 2026 "Early Quantization Shrinks Codebook" (Zhao, Zou, Shah, Wu, Lin, Liu; 17 Mar 2026) identifies "token representation shrinkage", distinct from dead codes, and fixes it by deferred quantization (pretrain the AE, k-means init, then enable VQ). It uses fixed EMA decay (0.9 / 0.99), no schedule and no restarts [verified] — [arXiv 2603.17052](https://arxiv.org/html/2603.17052)
- 2026 VP-VAE drops the explicit codebook during training. It describes EMA plus restart heuristics as "early works" [verified via snippet] — [arXiv 2602.17133](https://arxiv.org/html/2602.17133)
- MGVQ (2025) uses multi-group quantization to reduce dead codes at large K [verified listing] — [arXiv 2507.07997](https://arxiv.org/html/2507.07997v2)
- DiVeQ (2025) uses differentiable VQ via reparameterization [listing only] — [arXiv 2509.26469](https://arxiv.org/pdf/2509.26469)

### Inferences
- **What our recipe changes:**
  1. Replace the fixed decay gamma with cosine-annealed alpha(t) = 1 - gamma(t). This lets codes track fast early encoder drift, which NS-VQ identifies as the root cause, and then freeze late.
  2. Express the dead-code threshold in *units of expected count* (n_k / (B/K)), which is exactly the "relative threshold" the #257 author suggests, and revive from high-error batch vectors.
  3. Use the scaling law to set alpha0 and B from (K, vectors/step, total steps).
  The non-stationarity diagnosis of NS-VQ is effectively a statement that the EMA rate is wrong at the start of training, and our drift re-warming track addresses this directly.
- Our scaling law is the natural extension of Busbridge et al.'s EMA rule (decay^kappa when batch is scaled by kappa; see Q6) to *count-weighted* cluster statistics, where the effective per-code batch is B/K rather than B. Speculation: this per-code view could explain why EMA-VQ peaks at K ~16k (VQGAN-LC), since the per-code effective sample rate falls as K grows at fixed B.
- **Paper angle:** a benchmark of "EMA-VQ done right" against FSQ, SimVQ, rotation trick and IBQ at matched compute on ImageNet tokenizers and an RVQ speech codec (EnCodec/DAC-scale). If EMA done right closes the gap at K >= 64k, that is a strong, contrarian, high-visibility result.
- **Risk:** reparameterization methods (SimVQ, IBQ) already report ~100% utilization, so the win must come from rFID/bitrate, scaling to larger K, or tuning-free robustness across batch sizes. The robustness angle is well evidenced by #257.

### Gaps
- I did not find any paper proposing a cosine/annealed EMA decay schedule for VQ codebooks or a K- and batch-dependent decay law. Absence in a limited search is not proof of absence. A targeted search of 2025-2026 audio-codec papers (Mimi/Moshi, SNAC, WavTokenizer, BigCodec, X-codec) is still needed.
- I did not verify what the Moshi/Mimi codec (Kyutai, 2024) uses for codebook updates.
- "Online Vector Quantized Attention" (Alonso, Figliolia, Millidge; arXiv 2602.03922): only metadata could be extracted, so its codebook update rule is unknown (see Q4).

## Q2. Sparse autoencoders (SAEs) for LLM interpretability: dead latents, resampling (= revival), scaling laws

### Takeaway
SAEs are the direct deep-learning analogue of our dictionary-learning track. Dead latents are a first-order problem: Gao et al. (2024) report up to 90% dead latents without mitigation. Mitigations are heuristics (resampling, AuxK loss, decoder-transpose init, and in 2026 mean-centering). Dead-ness is measured by fixed "steps since fired" windows, not by EMA statistics with batch-scaled thresholds. SAEs are trained by Adam on the reconstruction loss, *not* by sufficient-statistic updates. Gao et al. already own the "SAE scaling laws" headline. Our contribution would have to be (a) a principled EMA-of-firing-frequency revival rule with batch/dictionary-size scaling, or (b) a closed-form, EMA-statistics dictionary update (online dictionary learning à la Mairal 2009 with our schedule) as a cheap SAE alternative. ITDA (2025) already shows that cheap non-SGD dictionaries are of interest.

### Cited Findings
- Anthropic, "Towards Monosemanticity" (Bricken et al., 2023) introduced neuron resampling: periodically re-initialize dead features toward poorly reconstructed inputs [known] — [transformer-circuits.pub 2023](https://transformer-circuits.pub/2023/monosemantic-features)
- Anthropic, "Scaling Monosemanticity" (Templeton et al., 2024) trained SAEs up to ~34M features and reported a large fraction of dead features at the largest size [known] — [transformer-circuits.pub 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity)
- Gao, Dupré la Tour et al. (OpenAI; arXiv 2406.04093; ICLR 2025 oral), "Scaling and evaluating sparse autoencoders", introduced TopK SAEs. In larger autoencoders an increasingly large share of latents die, up to 90% without mitigation. Encoder init from the decoder transpose plus the AuxK auxiliary loss leaves only 7% dead in the 16M-latent model. Scaling laws are clean once dead latents are controlled. A latent counts as dead after a fixed window (e.g. 256,000 examples / ~10M tokens), with AuxK coefficient 1/32 [verified via search] — [arXiv 2406.04093](https://arxiv.org/pdf/2406.04093); [ICLR 2025 oral](https://iclr.cc/virtual/2025/oral/31752)
- JumpReLU SAEs (Rajamanoharan et al., 2024) and Gemma Scope (Lieberum et al., 2024) released 400+ SAEs on Gemma 2 [known] — [arXiv 2407.14435](https://arxiv.org/abs/2407.14435); [arXiv 2408.05147](https://arxiv.org/abs/2408.05147). Gated SAEs (Rajamanoharan et al., 2024) [listing] — [arXiv 2404.16014](https://arxiv.org/pdf/2404.16014)
- BatchTopK and Matryoshka SAEs still produce dead or rarely active features because features can consistently miss the top-k. One study calls features with frequency below 1e-5 "almost dead" and finds that BatchTopK accumulates them faster than a hierarchical SAE. Dead features are defined by batch-count windows (e.g. inactive for 25 batches), not EMA [verified via search snippets] — [arXiv 2502.20578](https://arxiv.org/html/2502.20578v2); [AdaptiveK arXiv 2508.17320](https://arxiv.org/html/2508.17320v2)
- **2026, ICML:** Simon, Adams, Zou (arXiv 2605.31518, 29 May 2026), "On the Relationship Between Activation Outliers and Feature Death in SAEs". Activation outliers (large mean relative to per-token variation) cause feature death at initialization. Outlier severity predicts the initial death rate across 454 model-layer pairs (Spearman 0.89 TopK, 0.82 ReLU). Dead features revive only if the bias learns the mean, which is too slow at high severity. Their fix is mean-centering, with no resampling or EMA [verified] — [arXiv 2605.31518](https://arxiv.org/abs/2605.31518)
- Expander SAEs (2026) disable dead-feature resampling when more than 80% of features are dead. This is another fixed-threshold heuristic [verified via snippet] — [arXiv 2607.01799](https://arxiv.org/pdf/2607.01799)
- SAE scaling theory in 2025-2026: a capacity-allocation model with feature manifolds [verified listing] — [arXiv 2509.02565](https://arxiv.org/abs/2509.02565); "The Geometric Wall" on layerwise SAE scaling laws [listing] — [arXiv 2605.09887](https://arxiv.org/pdf/2605.09887)
- ITDA (Leask, Nanda, Al Moubayed; ICML 2025) builds dictionaries greedily from activations worst-approximated by matching pursuit. It trains in ~1% of SAE time and data, scales to Llama-3.1 405B on one GPU, but generally reconstructs worse than SAEs [verified] — [arXiv 2505.17769](https://arxiv.org/abs/2505.17769)
- A unified theory of sparse dictionary learning for mech-interp (piecewise biconvexity, spurious minima) credits K-SVD as foundational [listing] — [arXiv 2512.05534](https://arxiv.org/pdf/2512.05534)

### Inferences
- **What our recipe changes:**
  1. Track an EMA of per-latent firing frequency with a cosine-annealed rate.
  2. Declare a latent dead relative to its *expected* rate (k/num_latents per token for TopK), with a threshold that scales with tokens per step. This is the same unit bug as VQ issue #257.
  3. Revive from high-residual tokens, mirroring Anthropic's resampling but continuous and schedule-aware.
  4. Optionally replace Adam decoder updates with an EMA-of-sufficient-statistics block-coordinate update (Mairal-style A = E[zz^T], B = E[xz^T]). This could give a "1-3 epochs to full-batch quality" result. Activation datasets are huge, so fewer passes matter.
- **Novelty gap:** a recipe that removes AuxK and resampling hyperparameters and comes with a law for (lr / EMA rate, batch) vs (dictionary size, k, tokens). Gao et al. give loss-vs-size scaling laws, not optimizer-hyperparameter laws. Speculative risk: the mech-interp community optimizes for interpretability metrics (SAEBench), so a pure reconstruction or speed gain may be undervalued unless dead/dense-latent counts and SAEBench scores improve.
- Mean-centering (Simon et al. 2026) is complementary: it removes initial death, while our revival targets death during training.

### Gaps
- I could not verify Anthropic's 2024 circuits-update changes to resampling (e.g. whether resampling was replaced by other techniques). Check transformer-circuits.pub monthly updates.
- I found no paper applying Mairal-style online dictionary learning with EMA statistics to LLM-scale SAEs. This looks open but is not confirmed.
- I did not survey SAELens / dictionary_learning default dead-feature handling.

## Q3. Mixture-of-Experts routing and load balancing as balanced assignment

### Takeaway
DeepSeek's auxiliary-loss-free balancing is an online dual (price) update for a balanced-assignment / balanced k-means problem. It uses a sign-based bias step with a fixed rate u, no annealing, and no scaling rule in (experts, tokens/batch). A Dec 2025 / Apr 2026 theory paper (Han & Zhong) frames it as primal-dual with log-regret "under certain step-size choices", which partly covers the theory angle. An empirical schedule and scaling law for u (and for EMA-smoothed loads) does not appear to exist. This is high impact because MoE is mainstream, but the competition is heavy and our "k-means" framing is less obviously necessary.

### Cited Findings
- Switch Transformer (Fedus, Zoph, Shazeer, 2021): auxiliary load-balancing loss plus capacity factor; dropped tokens [known] — [arXiv 2101.03961](https://arxiv.org/abs/2101.03961)
- BASE layers (Lewis et al., 2021): token-to-expert routing solved as a linear (balanced) assignment problem [known] — [arXiv 2103.16716](https://arxiv.org/abs/2103.16716)
- Hash layers (Roller et al., 2021) [known] — [arXiv 2106.04426](https://arxiv.org/abs/2106.04426). Expert-choice routing (Zhou et al., 2022) [known] — [arXiv 2202.09368](https://arxiv.org/abs/2202.09368)
- Wang et al. (DeepSeek, 2024), "Auxiliary-Loss-Free Load Balancing": a per-expert bias b_i is added only for top-k *selection* and updated after each step by b_i <- b_i + u * sign(e_i), where e_i is mean load minus actual load. u is a fixed update rate [verified] — [arXiv 2408.15664](https://arxiv.org/pdf/2408.15664)
- DeepSeek-V3 (2024) adopts this bias balancing with bias update speed 0.001 for most of training, plus a small sequence-wise auxiliary loss [known] — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- Han & Zhong (arXiv 2512.03915, 3 Dec 2025, rev. 26 Apr 2026) frame ALF-LB as a single-step primal-dual method for an assignment problem. They prove monotone Lagrangian improvement, overloaded-to-underloaded token moves, and approximate balancing in the deterministic case, and log expected regret online under certain step sizes. They run experiments on 1B DeepSeekMoE. The abstract gives no step-size scaling with experts or batch [verified] — [arXiv 2512.03915](https://arxiv.org/abs/2512.03915)
- 2026: "phi-Balancing for Mixture-of-Experts Training" [listing only] — [arXiv 2605.15403](https://arxiv.org/pdf/2605.15403). "Slicing and Dicing: Configuring Optimal Mixtures of Experts" [listing only] — [arXiv 2605.11689](https://arxiv.org/pdf/2605.11689)

### Inferences
- **Mapping:** experts = clusters, bias b = dual variable / "price" from balanced k-means, per-step token load = cluster count. Our recipe would:
  1. Replace sign(e_i) with an EMA of load imbalance and a cosine-annealed rate u(t).
  2. Use a scaling law u0 = f(#experts E, tokens per step, top-k, steps). The per-expert effective sample is tokens*k/E per step, analogous to B/K.
  3. Treat "dead experts" (persistently under-loaded or collapsed) with revival, e.g. re-initializing router rows toward high-loss tokens.
  Our balanced-IVF track is the closest internal evidence.
- The Han & Zhong theory gives a ready-made framing. An empirical contribution showing that a fixed u is suboptimal across E in {64, 256} and batch sizes, with a transferable law, seems novel but must be checked against phi-Balancing (2026) and other 2025-2026 MoE reports (Qwen3-MoE, Kimi K2, GLM-4.5, Llama 4). Speculation: frontier labs likely tune u internally.
- **Risk:** MoE experiments are expensive. An impact claim needs at least ~1B-scale MoE runs, which may exceed a single 5090.

### Gaps
- I did not read the phi-Balancing paper or the full Han & Zhong step-size results.
- I did not verify whether any 2025-2026 MoE report already uses an annealed or batch-scaled bias update rate.

## Q4. KV-cache compression and attention via online clustering / VQ

### Takeaway
Many 2024-2026 KV-cache methods run k-means (ClusterKV, CentroidKV, LouisKV, Multipole Attention, STARC) or VQ/PQ (PQCache, Transformer-VQ, OVQ-attention). Updates are either full k-means on blocks or incremental "cluster only new tokens" schemes, with no EMA-statistics recipe and no revival. This is a streaming, drifting, per-head clustering problem, an ideal fit for our live-drifting-IVF and drift re-warming tracks. Novelty is moderate: the field is crowded with systems papers, but none frame the problem as "EMA statistics with a principled rate".

### Cited Findings
- Routing Transformer (Roy, Saffar, Vaswani, Grangier, 2020) routes attention with online k-means whose centroids are updated by EMA (decay ~0.999) [known] — [arXiv 2003.05997](https://arxiv.org/abs/2003.05997). Reformer (Kitaev et al., 2020) uses LSH buckets [known] — [arXiv 2001.04451](https://arxiv.org/abs/2001.04451)
- Transformer-VQ (Lingle, 2023): VQ keys for linear-time attention, with an EMA-updated codebook [known] — [arXiv 2309.16354](https://arxiv.org/abs/2309.16354)
- ClusterKV (2024) runs k-means over keys. During decoding it re-clusters every m steps on new tokens and computes centroid means with a CUDA kernel [verified via search snippet] — [arXiv 2412.03213](https://arxiv.org/html/2412.03213)
- CentroidKV (2025): online KV cache clustering, up to 75% memory reduction and up to 1.92x faster decoding [verified snippet, not full text] — [arXiv 2506.11418](https://arxiv.org/abs/2506.11418)
- Multipole Attention (2025): sequential MacQueen-style assignment of new tokens plus a few full k-means iterations on the final block [verified snippet] — [arXiv 2506.13059](https://arxiv.org/abs/2506.13059)
- STARC (PIM, 2025) re-clusters every 128 decoding steps on new tokens only and never revisits old clusters [verified snippet] — [arXiv 2505.05772](https://arxiv.org/pdf/2505.05772). LouisKV (2025) uses segment centroids [snippet] — [arXiv 2510.11292](https://arxiv.org/pdf/2510.11292). CommunityKV (2026) uses graph partitioning with constant-time local updates [snippet] — [arXiv 2610.00418](https://arxiv.org/pdf/2610.00418)
- Online Vector Quantized Attention (Alonso, Figliolia, Millidge; arXiv 2602.03922, 2026). Its cited keys include PQCache, ClusterKV and CommVQ. The update rule is unverified — [arXiv 2602.03922](https://arxiv.org/pdf/2602.03922)
- Flash-KMeans (2026): fast exact k-means on GPU [listing] — [arXiv 2603.09229](https://arxiv.org/pdf/2603.09229)
- PQCache (2024) [known] — [arXiv 2407.12820](https://arxiv.org/abs/2407.12820). RetrievalAttention (2024) [known] — [arXiv 2409.10516](https://arxiv.org/abs/2409.10516). KIVI 2-bit KV quantization (2024) [known] — [arXiv 2402.02750](https://arxiv.org/abs/2402.02750). Token Merging (Bolya et al., 2022) [known] — [arXiv 2210.09461](https://arxiv.org/abs/2210.09461)

### Inferences
- **What our recipe changes:** decode-time centroid maintenance becomes EMA of sums/counts per centroid with a rate tied to tokens generated per refresh vs number of clusters, with revival of empty clusters (new topics during long reasoning traces). This should reduce periodic re-clustering cost and keep cluster quality as the KV distribution drifts during long generation.
- Moderate impact. The measurable win is recall/accuracy at fixed memory on LongBench/RULER with less re-clustering compute. Most competitors are systems papers, so a principled-statistics angle could differentiate. Speculation: gains may be small because block re-clustering is already cheap.

### Gaps
- I did not read full texts, so whether CentroidKV or OVQ already use EMA/running-mean centroids with decay is unverified. This is critical for novelty and should be checked before committing.

## Q5. Self-supervised online clustering (SwAV, DINO prototypes) and k-means data curation / dedup

### Takeaway
DINO-family prototypes suffer *partial prototype collapse*: only 1.6% of 65,536 DINO prototypes are unique (Govindarajan et al., BMVC 2024). An ICLR 2026 paper (Arteaga et al.) already fixes this by modeling prototypes as a GMM updated with **online EM decoupled from the encoder**. That is very close to our GMM-EM track and **partly pre-empts** the idea, but leaves room for schedule, revival and scaling-law contributions. Data curation by hierarchical k-means (Vo et al. 2024) and SemDeDup (Abbas et al. 2023) run billion-scale k-means offline, where our recipe offers 1-3-epoch mini-batch k-means at full-batch quality, which is a cost/scale contribution.

### Cited Findings
- SwAV (Caron et al., 2020): online clustering with Sinkhorn-Knopp equipartition (balanced assignment) [known] — [arXiv 2006.09882](https://arxiv.org/abs/2006.09882). SeLa (Asano et al., 2019) [known] — [arXiv 1911.05371](https://arxiv.org/abs/1911.05371). DeepCluster (Caron et al., 2018) re-assigns empty clusters [known] — [arXiv 1807.05520](https://arxiv.org/abs/1807.05520)
- DINO (Caron et al., 2021): teacher EMA plus centering with an EMA of teacher outputs [known] — [arXiv 2104.14294](https://arxiv.org/abs/2104.14294). DINOv2 (Oquab et al., 2023) adds Sinkhorn centering and KoLeo [known] — [arXiv 2304.07193](https://arxiv.org/abs/2304.07193)
- Govindarajan et al. (BMVC 2024), "On Partial Prototype Collapse in the DINO family": DINO ViT-S/16 initializes 65,536 prototypes, of which only 1,078 (1.6%) are unique. They propose a KoLeo-Proto regularizer [verified via search snippet] — [arXiv 2410.14060](https://arxiv.org/abs/2410.14060)
- **Arteaga, Aasan, Chakraborty, Hjelkrem-Tan, Silva, Kampffmeyer, Ramírez Rivera (arXiv 2510.20108; ICLR 2026)**, "Why Prototypes Collapse": joint encoder-prototype optimization causes shortcut learning. The fix is fully decoupled prototypes modeled as a Gaussian mixture and updated with an online EM-style procedure, removing collapse without regularization [verified] — [arXiv 2510.20108](https://arxiv.org/abs/2510.20108)
- Vo et al. (2024), "Automatic Data Curation for SSL": successive hierarchical k-means plus balanced sampling. Models trained on curated data beat those on unbalanced data, for vision and text [verified via snippet] — [arXiv 2405.15613](https://arxiv.org/abs/2405.15613)
- SemDeDup (Abbas et al., 2023): k-means on embeddings, then within-cluster near-duplicate removal, at LAION scale [known] — [arXiv 2303.09540](https://arxiv.org/abs/2303.09540)
- ClusterClip sampling for LLM training uses cluster-uniform sampling with a repetition clip [snippet] — [themoonlight review](https://www.themoonlight.io/review/balanced-data-sampling-for-language-model-training-with-clustering)

### Inferences
- Arteaga et al. (ICLR 2026) is the **nearest prior art** to "EMA/online-EM statistics for SSL prototypes". Our differentiation is cosine-annealed EM rate, revival, and a scaling law for (EM rate, batch) vs (#prototypes, dataset size). It is likely publishable as a follow-up or ablation, not as a headline.
- Data curation: hierarchical k-means on 10^9-10^10 embeddings is compute-dominant. Showing that our 1-3-epoch mini-batch recipe matches full-batch Lloyd quality and downstream curation quality at a fraction of the cost is a clean applied contribution. Our cheap-seeding and scaling-law tracks apply directly. Medium impact, low novelty risk on the method, but practical value is high.

### Gaps
- I did not check whether Meta's DINOv3 (2025) or later curation pipelines already use mini-batch or EMA k-means.
- I found no 2025 LLM-pretraining paper directly extending Vo et al.'s hierarchical k-means, based on limited search.

## Q6. EMA of parameters / normalization statistics and EMA-rate scaling laws (nearest prior art for the "scaling law" component)

### Takeaway
"How to scale your EMA" (Busbridge et al., NeurIPS 2023) is the canonical prior art for scaling an EMA rate with batch size (momentum rho -> rho^kappa). Our scaling law must be positioned as its generalization to *count-weighted sufficient statistics with K components*: effective per-component batch B/K, plus annealing and revival. BN running stats (test-time adaptation) and power-EMA (Karras 2024) are adjacent.

### Cited Findings
- Busbridge, Ramapuram, Ablin et al. (NeurIPS 2023): when batch size is scaled by kappa, set EMA momentum to rho^kappa. Validated on BYOL (batch 24,576, 6x wall-clock), DINO and pseudo-labeling [verified] — [arXiv 2307.13813](https://arxiv.org/abs/2307.13813)
- A 2026 optimizer survey generalizes this to converting beta at batch B into beta^(B'/B), preserving half-life in examples, and cites Marek et al. (2025) small-batch results [secondary, verified snippet] — [arXiv 2608.28557](https://arxiv.org/pdf/2608.28557)
- Karras et al. (2024), EDM2: post-hoc power-function EMA whose length scales relative to training time [known] — [arXiv 2312.02696](https://arxiv.org/abs/2312.02696)
- BN statistics adaptation at test time (Schneider et al., 2020; TENT, Wang et al., 2021) [known] — [arXiv 2006.16971](https://arxiv.org/abs/2006.16971); [arXiv 2006.10726](https://arxiv.org/abs/2006.10726)

### Inferences
- Busbridge's rule is per-step for a *single* average. For K-component statistics, each component sees ~B*p_k samples per step, so the "right" decay is component-specific: rho_k ~ rho^(B*p_k / B_ref). That is the formal reason the VQ dead-code threshold (#257) and SAE dead-latent windows break across batch sizes. Writing this down plus validating it empirically is a crisp unifying theory contribution across VQ, SAE, MoE and SSL prototypes.

### Gaps
- I did not verify Marek et al. 2025 directly.

## Q7. Embedding quantization for vector search (RaBitQ, ScaNN, Matryoshka), plus ranking by impact and novelty

### Takeaway
Vector-search quantizers (ScaNN anisotropic VQ, RaBitQ, PQ/OPQ) are mostly trained offline and refreshed rarely. Our OPQ, balanced-IVF and live-drifting-IVF tracks already cover this space, so the marginal novelty is low-to-medium. The ranking below synthesizes Q1-Q7.

### Cited Findings
- ScaNN (Guo et al., ICML 2020), anisotropic vector quantization [known] — [arXiv 1908.10396](https://arxiv.org/abs/1908.10396). RaBitQ (Gao & Long, SIGMOD 2024), a randomized binary quantizer with error bounds and no codebook training [known] — [arXiv 2405.12497](https://arxiv.org/abs/2405.12497). Matryoshka representations (Kusupati et al., 2022) [known] — [arXiv 2205.13147](https://arxiv.org/abs/2205.13147)

### Inferences
**Ranking by expected impact x novelty, for a paper identity of "EMA of sufficient statistics done right, with scaling laws". This ranking is our judgment, not sourced fact:**
1. **VQ / RVQ tokenizers (images, video, audio codecs).** Highest. The method is literally EMA k-means, fixed decays are used everywhere, and there is concrete 2026 evidence of batch-scaling breakage (#257: 18% relative WER change from a threshold unit error). Heavy use in LLM-style generation and speech LMs. The novelty gap is the schedule plus scaling law for (decay, threshold, B, K). Closest prior art: CVQ-VAE 2023 (usage-based re-anchoring), Łańcucki 2020 (k-means re-init), NS-VQ 2026 (drift diagnosis), Busbridge 2023 (EMA batch scaling).
2. **Unifying theory: per-component EMA scaling (Busbridge rule generalized to K count-weighted statistics) applied across VQ, SAE, MoE and SSL prototypes.** High novelty and a strong identity piece. Validate on two or three of these domains.
3. **SAEs.** High visibility. A dead-latent EMA revival rule with batch/dictionary scaling, plus an optional sufficient-statistics (Mairal-style) decoder update. Partial pre-emption: Gao 2024 (AuxK, scaling laws), Simon et al. 2026 (mean-centering), ITDA 2025 (cheap dictionaries).
4. **MoE bias balancing.** Large impact if it works, but compute-heavy. Theory partly covered by Han & Zhong 2025/26. Offer a schedule plus scaling law for u.
5. **SSL prototypes / data curation k-means.** The prototype angle is largely pre-empted by Arteaga et al. (ICLR 2026, online-EM GMM prototypes). The data-curation cost angle is solid but incremental.
6. **KV-cache clustering.** A good fit for the drift tracks, but crowded with systems papers and modest expected gains.
7. **Vector-search quantizers.** Already covered by existing tracks, so low marginal novelty.

### Gaps
- The 2025-2026 coverage of audio codecs (Mimi, SNAC, WavTokenizer), MoE tech reports (Qwen3, Kimi K2, Llama 4) and SAE tooling defaults was not fully searched. Any of these could already use annealed EMA decay or relative dead-code thresholds. Check before claiming novelty.
- I found no source applying a size-based scaling law for EMA rate and batch to any clustering-type inner loop in deep learning. This is a negative result from a limited search.
