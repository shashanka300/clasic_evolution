# Systems, privacy, streaming, federated and hardware directions for EMA-of-sufficient-statistics clustering (e-means)

Scope: directions NOT already covered by existing tracks (balanced IVF p99, OPQ joint rotation, HNSW parameter laws, drift re-warming, live drifting IVF, cheap seeding, adaptive k). Research date: 2026-11-14. About 15 searches/fetches. Labelling used below: "Fact" = backed by a cited source; "Training knowledge" = from the model's prior knowledge and NOT checked in this session (verify before citing); "Speculation" = my own reasoning.

The core point of e-means used throughout: the state is per-cluster (sum S_j, count N_j), updated as S_j <- (1-a_t) S_j + a_t * batchsum_j, with a cosine-annealed a_t, dead-cluster revival, and a scaling law for a0 and B in terms of (n, k, epochs). Every direction below is about where that (S, N, a_t) state is a better systems object than Lloyd's full-pass sums, Sculley's 1/count rate, or a coreset.

---

## Q1. Differentially private k-means / EM / PCA via noisy sufficient statistics: does an EMA over subsampled mini-batches give a better privacy-utility trade-off?

### Takeaway
Central DP k-means is still built on "noisy Lloyd", which releases a noisy sum and noisy count per cluster each iteration over the full dataset. Its privacy cost grows linearly in the number of iterations T. I found no paper that combines mini-batch (Poisson-subsampled) k-means, an EMA of noisy statistics, and an RDP/PLD accountant with amplification by subsampling. This is the clearest novelty gap in this note. It is also cheap to test on one GPU.

### Cited Findings
- Fact: a Lloyd step can be written only in terms of per-cluster sums and counts. DP-Lloyd adds noise to both, and its privacy loss after T iterations is epsilon = (r/beta_sum + 1/beta_count) * T, i.e. linear in T. Lloyd-style DP methods are sensitive to initialisation. — [Su et al., "Differentially Private k-Means Clustering" (arXiv 1504.05998, 2015; CODASPY 2016)](https://arxiv.org/html/1504.05998v1)
- Fact: recent federated DP k-means papers (2025) also use Gaussian noise on sums and Laplace noise on counts, with each centre set to noisy-sum / noisy-count. — [DP Federated k-Means with Server-Side Data (arXiv 2506.05408, 2025)](https://arxiv.org/pdf/2506.05408)
- Fact: FastLloyd (Diaa, Humphries, Kerschbaum; USENIX Security 2025) is a DP Lloyd variant with cluster-size constraints, lightweight secure aggregation, and computational DP. It reports runtime several orders of magnitude faster than prior secure federated k-means (v1 said 4, later versions say 5), with utility matching central-DP state of the art. — [arXiv 2405.02437](https://arxiv.org/abs/2405.02437v2); [USENIX Sec '25 page](https://www.usenix.org/conference/usenixsecurity25/presentation/diaa)
- Fact: the FastLloyd bibliography cites Su 2016, Mohassel 2020, Chang 2021, Stemmer 2021, Ghazi 2020, Nissim 2018 and Balcan 2017. These are the prior-art baselines a DP-EMA paper would need to compare against. — [FastLloyd PDF (USENIX)](https://www.usenix.org/system/files/usenixsecurity25-diaa.pdf) (citation keys only were readable)
- Fact: "Locally Private k-Means in One Round" (Chang, Ghazi, Kumar, Manurangsi; 2021) is the LDP baseline. — [arXiv 2104.09734](https://arxiv.org/pdf/2104.09734)
- Fact: subsampling amplification has been used for DP k-means through importance sampling (Fay, Mair, Sjölund; TMLR 2024). That work subsamples the dataset once. It is not mini-batch iterative training. — [arXiv 2307.10187](https://arxiv.org/abs/2307.10187v3)
- Fact: tighter mechanism-specific amplification-by-subsampling bounds are available (Schuchardt et al., NeurIPS 2024). — [arXiv 2403.04867](https://arxiv.org/abs/2403.04867v3)
- Fact: standard RDP accountants assume Poisson subsampling, which gives variable-size batches. Fixed-size without-replacement accounting has been analysed (2024). — [arXiv 2408.10456](https://arxiv.org/html/2408.10456v1). The sampled-Gaussian RDP reference is Mironov, Talwar, Zhang 2019 (arXiv 1908.10530), which appeared only as a reference in search results.
- Fact: DP-PCA (Liu, Kong, Jain, Oh; NeurIPS 2022) is a single-pass minibatch method that privatises each update with the Gaussian mechanism and is near-optimal for sub-Gaussian data, error ~ sqrt(d/n) + d/(eps n), with a matching lower bound. — [arXiv 2205.13709](https://arxiv.org/pdf/2205.13709); [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c150ebe1b9d1ca0eb61502bf979fa87d-Abstract.html)
- Fact: a 2026 preprint reports private guarantees for streaming Oja-style PCA beyond rank-one updates. — [arXiv 2609.26508](https://arxiv.org/pdf/2609.26508) (not checked for peer review)

### Inferences
- Speculation: DP-Lloyd pays T full-data releases. DP-e-means with Poisson rate q = B/n pays T' subsampled Gaussian releases, and amplification makes each one cost roughly O(q * eps) in the small-eps regime. The EMA then averages the noise across batches. With rate a_t, the stationary noise variance in S_j is reduced by about a_t/(2-a_t) relative to a single batch release. So the cosine-annealed a_t acts as a noise-averaging schedule: a large rate early gives fast progress, and a small rate late averages out the DP noise. This mirrors DP-SGD learning-rate decay, but here applied to sufficient statistics. That gives a concrete, testable claim: at a fixed (eps, delta), DP-e-means reaches lower k-means cost than DP-Lloyd, Su 2016, and FastLloyd's central variant.
- Speculation: there is a subtle point a paper must handle. The EMA itself is post-processing, so privacy is unaffected. But assignment uses current centres, which are released (or at least internal), so the whole trajectory composes. Using an accountant over T' Poisson-subsampled sampled-Gaussian mechanisms (sum + count jointly, with sensitivity bounded by clipping ||x|| <= r) is standard and sound.
- Speculation: dead-cluster revival must be made private. Re-seeding from a data point leaks that point. A private alternative is to re-seed at the noisy centre of the largest cluster plus a random offset, or at a point drawn from a DP histogram. This is a small but real technical contribution.
- Speculation: the scaling law (a0, B) = f(n, k, epochs) gets a privacy axis: (a0, B, sigma) = f(n, k, eps). The optimal batch size under DP is typically larger than without DP (as in DP-SGD), and that can be checked empirically.
- Speculation: DP-EM for GMMs (Park et al. 2017, training knowledge) and DP-PCA via noisy covariance (Liu et al. 2022) fit the same template: EMA of noisy (N, S, second moment). The repo already has GMM-EMA results (results/gmm-ema), so a DP-GMM-EMA variant could reuse that code.
- Feasibility: high. One 5090 is enough. Datasets: MNIST/Fashion embeddings, SIFT1M subsample, synthetic GMMs; Opacus/`prv_accountant` for accounting. Days, not weeks.

### Gaps
- I found no paper doing DP mini-batch k-means with subsampling amplification and an EMA. Searches for "DP mini-batch k-means Poisson RDP" returned only components. This is a gap in what I found, not proof of absence: a targeted Google Scholar check is still needed before claiming novelty.
- I could not find a 2024–2026 DP-EM-for-GMM paper. Park et al. 2017 (DP-EM, AISTATS) is from training knowledge and was not verified here.
- FastLloyd's exact algorithm (constraint form, budget split) was not readable. The PDF was binary-only via fetch.

---

## Q2. Federated clustering / federated EM: communicating sufficient statistics

### Takeaway
Federated k-means splits into one-shot schemes (k-FED and its 2025–2026 descendants) and secure-aggregation Lloyd (FastLloyd). An EMA of (S, N) maps naturally onto asynchronous, partial-participation federated rounds, where server-side momentum on sufficient statistics plays the role of FedAvgM. Under privacy, though, it collapses into Q1. The standalone non-private federated angle looks incremental.

### Cited Findings
- Fact: k-FED (Dennis, Li, Smith; ICML 2021) is one-shot. Each device runs local k-means and sends its local means. Heterogeneity (k' < k clusters per device) weakens the separation requirements. — [arXiv 2103.00697](https://arxiv.org/abs/2103.00697); [PMLR v139](http://proceedings.mlr.press/v139/dennis21a/dennis21a.pdf)
- Fact: recent one-shot federated clustering (2025–2026) includes FedBWC, where clients fit a variational Bayesian GMM and the server runs weighted k-means in one round, a TNNLS 2025 density-core one-shot secure federated k-means, and arXiv 2601.06404 / 2601.17512. — [FedBWC (J. Shanghai Univ., 2026)](https://www.jas.shu.edu.cn/EN/10.3969/j.issn.0255-8297.2026.05.012); [arXiv 2601.06404](https://arxiv.org/pdf/2601.06404); [arXiv 2601.17512](https://arxiv.org/pdf/2601.17512)
- Fact: there is a 2025 arXiv preprint on LDP federated clustering via gravitational potential fields. — [arXiv 2512.00849](https://arxiv.org/pdf/2512.00849)
- Fact: there is DP vertical federated clustering (Li et al., VLDB 2023). — [PVLDB 16](https://www.vldb.org/pvldb/vol16/p1277-li.pdf)
- Fact: the search found no 2024–2025 federated clustering paper that explicitly frames communication as streaming per-cluster (count, sum) with server-side decay. — [search summary of k-FED and descendants above](https://arxiv.org/abs/2103.00697)

### Inferences
- Speculation: e-means' server state (S, N) with an EMA is exactly "FedAvg with server momentum" for k-means. Clients send their per-batch (sum_j, count_j) for the current centres. The server applies S <- (1-a_t)S + a_t * sum over clients. Benefits: robustness to partial participation and stragglers (stale contributions decay), clients do no local iteration (stateless), and it composes with secure aggregation, because sums and counts are additive, which is the same property FastLloyd uses.
- Speculation: the novel piece is dead-cluster revival under non-IID clients. A cluster can look "dead" in a round only because the clients that hold it did not participate. Revival must use participation-aware counts (an EMA of N, not the round's N). This is a real failure mode and fits the existing revival code naturally.
- Feasibility: high. Simulate 100–1000 clients on one machine. Novelty: medium-low on its own; strongest packaged as "DP + secure aggregation + EMA" (Q1 + Q2 together), compared directly with FastLloyd.

### Gaps
- FedEM (Marfoq et al., NeurIPS 2021) is federated multi-task learning under mixture distributions, not federated GMM fitting. I did not re-verify it in this session.
- I did not obtain full texts for the 2026 one-shot papers.

---

## Q3. Streaming / sliding-window clustering with guarantees vs EMA forgetting

### Takeaway
Streaming k-means with guarantees uses coresets with merge-and-reduce (StreamKM++, BICO). DP continual-release streaming clustering now exists (Epasto, Mukherjee, Zhong; AISTATS 2026). An EMA forgets exponentially, not by a hard window, so it has no worst-case approximation guarantee. Its distinct contribution would be an empirical and theoretical comparison of exponential forgetting vs sliding-window coresets at a matched memory budget, plus a DP continual-release variant. Drift re-warming is already a track, so the new part is the guarantee and continual-release angle, not drift handling itself.

### Cited Findings
- Fact: Epasto, Mukherjee, Zhong, "Differentially Private Clustering in Data Streams" (arXiv 2023; AISTATS 2026, PMLR v300). It gives the first DP k-means and k-median over a stream of length T, with space sublinear in T and output at every timestep. It wraps any offline DP coreset or DP clustering algorithm as a black box. Guarantees: an O(1)-multiplicative approximation with about k^1.5 * poly(d, log T) space and poly(k, d, log T) additive error, and a (1+gamma) variant at higher space. Union of outputs over all timesteps is DP (continual release). — [PMLR v300](https://proceedings.mlr.press/v300/epasto26a.html); [arXiv 2307.07449](https://arxiv.org/html/2307.07449v3); [AISTATS 2026 poster](https://virtual.aistats.org/virtual/2026/poster/13802)
- Fact: the search found no DP sliding-window clustering result. — [same search; Epasto et al.](https://arxiv.org/html/2307.07449v3)

### Inferences
- Training knowledge (not verified here): StreamKM++ (Ackermann et al., JEA 2012) and BICO (Fichtenberger et al., ESA 2013) are the standard coreset-streaming baselines. Sliding-window k-means coresets go back to Braverman et al. (SODA 2016) and Borassi et al. (ICML 2020).
- Speculation: an EMA with rate a acts like a soft window of effective length about B/a. A useful result would be "at an equal memory of k*(d+1) floats, EMA tracks the window optimum within X% on drifting benchmarks, while coresets need O(k log n * d) memory". That is memory-constrained streaming, an edge or IoT story.
- Speculation: DP continual release maps naturally onto an EMA. A noisy (S, N) update every step, combined with the binary-tree mechanism or matrix-factorisation mechanisms for prefix sums, gives continual DP. An exponential-decay sum is a linear operator on the stream, so DP matrix-factorisation mechanisms (from DP-FTRL / DP-MF work) apply directly. Specifically, decayed-sum continual counting is a studied problem (training knowledge). This could be a neat theoretical-plus-empirical paper: "continual-release DP k-means via decayed-sum matrix mechanisms".
- Feasibility: high (CPU is enough). Novelty: medium. Impact: medium.

### Gaps
- I did not verify specific numbers for StreamKM++ or BICO, or the sliding-window coreset papers.
- I did not check the literature on decayed-sum continual counting under DP in this session.

---

## Q4. Vector-database index maintenance under inserts/deletes (SPFresh, Ada-IVF, Quake); multi-vector; RaBitQ; IVF training at billion scale

### Takeaway
The 2023–2025 systems line (SPFresh/LIRE, Ada-IVF, Quake) maintains IVF partitions with local split/merge plus local k-means re-clustering, driven by cost models and access "temperature". None of them keeps a global EMA of per-partition statistics that turns centroid drift, revival (to replace merges), and balance into one continuous update. The existing "live drifting IVF" track overlaps here, so the new angle must be specific. I suggest (a) query-temperature-weighted EMA statistics, that is, an EMA over the query stream rather than the data stream, and (b) deletes as negative-weight EMA updates.

### Cited Findings
- Fact: SPFresh (Xu et al., SOSP 2023) introduces LIRE, a lightweight incremental rebalancing protocol that splits partitions and reassigns only boundary vectors in neighbouring partitions. It is built on SPANN (disk). At a 1% daily update rate on a billion-scale index, it reports better latency and accuracy than global rebuilds using 1% of DRAM and under 10% of cores at peak. — [arXiv 2410.14452](https://arxiv.org/abs/2410.14452); [MSR SOSP PDF](https://microsoft.com/en-us/research/uploads/prod/2023/08/SPFresh_SOSP.pdf)
- Fact: FreshDiskANN (graph-based) batches deletions and consolidates them periodically. A later paper (LSM-VEC) argues that SPFresh's coarse partitioning reduces recall. — [search summary incl. SPFresh sources](https://arxiv.org/html/2410.14452v1)
- Fact: Ada-IVF (Mohoney et al., arXiv 2411.00970, Nov 2024; preprint) has an adaptive policy for choosing which partitions to repartition, plus local re-clustering, driven by per-partition read temperature. It reports 2x average and up to 5x higher update throughput than prior dynamic IVF maintenance at equal query throughput, and notes that rebuilding a billion-scale IVF from scratch can take days. — [arXiv 2411.00970](https://arxiv.org/abs/2411.00970v1)
- Fact: Quake (Mohoney et al., OSDI 2025) is a multi-level partitioned index. A cost model predicts query latency from partition sizes and access frequencies, splits hot or large partitions (by k-means) and merges cold or small ones. Adaptive Partition Scanning picks nprobe per query for a recall target. It is NUMA-aware. Reported results: 1.5–38x lower query latency and 4.5–126x lower update latency than SVS, DiskANN, HNSW and ScaNN on dynamic workloads (authors' own Wikipedia-based benchmark). — [arXiv 2506.03437](https://arxiv.org/abs/2506.03437); [USENIX OSDI '25](https://www.usenix.org/conference/osdi25/presentation/mohoney)
- Fact: cuVS IVF-Flat and IVF-PQ use balanced hierarchical k-means for the coarse quantizer, and IVF-PQ also uses k-means for PQ codebooks. Training is usually subsampled (for example, kmeans_trainset_fraction=0.1). Billion-scale examples include 1B x 96-d fp32 (360 GiB). — [NVIDIA cuVS IVF-PQ deep dive pt1](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/); [IVF-Flat blog](https://developer.nvidia.com/blog/accelerated-vector-search-approximating-with-nvidia-cuvs-ivf-flat)
- Fact: IVF+RaBitQ needs k-means centroids on raw vectors (RaBitQ itself needs no codebook, only a random rotation and per-vector factors). The library suggests about 4*sqrt(n) clusters following Faiss. Extended RaBitQ (SIGMOD 2025) supports multi-bit codes, but sources disagree on the bit range (2–9 vs 1–8). — [RaBitQ Library IVF docs](https://vectordb-ntu.github.io/RaBitQ-Library/index/ivf); [Extended-RaBitQ README](https://github.imc.re/gaoj0017/Extended-RaBitQ/blob/main/README.md)

### Inferences
- Speculation (direction 4a, query-weighted EMA partitions): Quake and Ada-IVF use access frequency to decide when to split or merge, but centroids are still trained on data-distribution MSE. An EMA over the query stream, weighting each data point's contribution to (S_j, N_j) by an EMA of how often it is retrieved, gives continuously adapting, query-cost-aware partitions without discrete split/merge events. Existing balanced-IVF p99 results suggest balance matters, and query-skew-aware balance is the natural next step. Compare against Quake's open-source code at matched recall. Novelty: medium-high. Feasibility: high (Quake is CPU-based, so it runs on 12 cores).
- Speculation (direction 4b, deletes as negative EMA mass): deletions are rarely handled by centroid updates. Systems mostly tombstone the vectors and rebuild later. Subtracting a deleted vector's contribution from (S_j, N_j), scaled by its decayed weight, keeps statistics exact under deletes. Low novelty on its own, but it is a cheap ablation inside the live-IVF track.
- Speculation (multi-vector / PLAID): ColBERTv2/PLAID trains about 2^16–2^18 centroids over all token embeddings (hundreds of millions of tokens). That is a large-k, huge-n k-means where mini-batch EMA with revival plus the (a0, B) scaling law could cut training cost. It is a clean "scaling law at extreme k" test. I did not verify PLAID training-cost numbers (see Gaps).
- Speculation (IVF training at billion scale): the cuVS/Faiss practice is to subsample about 10% and run full k-means. An e-means "train while streaming the whole dataset once" (out-of-core, single epoch) with a scaling-law-chosen B is a direct, measurable alternative. Report wall-clock time and recall@10 vs nprobe against cuVS balanced hierarchical k-means on DEEP-100M or a 1B subsample on the 5090 (32 GB VRAM means streaming from host or NVMe).

### Gaps
- PLAID/ColBERT centroid count and k-means training cost: no source found this session. The PLAID paper is Santhanam et al. 2022, arXiv 2205.09707 (training knowledge).
- I could not determine whether the cuVS IVF-PQ build exposes a training fraction for the codebooks.
- Filtered ANN (for example, Filtered-DiskANN, ACORN) was not researched in this session.

---

## Q5. GPU / accelerator systems: billion-scale k-means, mixed precision, out-of-core, low-precision accumulation of sufficient statistics

### Takeaway
Flash-KMeans (March 2026; NeurIPS 2026) shows that exact GPU k-means is limited by memory traffic and atomic contention. It fuses distance computation with argmin and turns scatter-atomic centroid updates into sort-based segment reductions. Mixed-precision k-means has been analysed for Lloyd (Carson, Chen, Liu 2024). Nobody I found has analysed low-precision (bf16/fp8) accumulation of EMA statistics, where the decay factor itself changes the error dynamics. That is a small but genuine and very 5090-relevant niche, because Blackwell has FP8/FP4 tensor cores.

### Cited Findings
- Fact: Flash-KMeans (Yang, Xi, ..., Stoica; arXiv 2603.09229, March 2026; NeurIPS 2026 poster). FlashAssign fuses distance computation with an online argmin, so the N x K distance matrix is never materialised. The sort-inverse update replaces atomic scatter with segment reductions. Reported on H200: up to 17.9x end to end over the best baselines, 33x over cuML and over 200x over FAISS. Exact, no approximation. Code: github.com/svg-project/flash-kmeans. — [arXiv 2603.09229](https://arxiv.org/abs/2603.09229); [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/151963)
- Fact: "Computing k-means in mixed precision" (Carson, Chen, Liu; arXiv 2407.12208, 2024) studies the numerical stability of Lloyd's distance and centre-update stages and proposes a mixed-precision framework, including fp16. Secondary summaries say centre updates are kept in higher precision. — [arXiv 2407.12208](https://export.arxiv.org/pdf/2407.12208); [secondary review](https://www.themoonlight.io/en/review/computing-k-means-in-mixed-precision)
- Fact: an earlier (2019) CPU-GPU k-means used two-step summation so that single-precision centroid updates reach near double-precision accuracy. — [cited in search summary](https://marcbab.github.io/cpe6621.pdf)
- Fact: cuVS balanced hierarchical k-means is the production GPU baseline for IVF. — [NVIDIA cuVS blog](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/)

### Inferences
- Speculation: EMA accumulation in low precision has a specific failure mode. With S <- (1-a)S + a*b and small late-schedule a (cosine annealing toward 0), the increment a*b falls below half an ulp of S in bf16 (8 mantissa bits) and gets absorbed, so the statistics freeze. The cosine schedule makes this worse exactly at the end of training. Remedies to evaluate: stochastic rounding, Kahan compensation, storing (S/N) in low precision and N in fp32, or an "a_t floor = f(precision)". This is a crisp numerical-analysis contribution combined with a 5090 benchmark (fp32 vs bf16 vs fp8 for both the distance GEMM and accumulation). I found no prior work on this. Novelty: medium-high (narrow). Feasibility: very high.
- Speculation: Flash-KMeans' sort-inverse update is exactly the kernel e-means needs per mini-batch, since the EMA only changes the final blend. Porting e-means onto FlashAssign plus segment reduction would make the method competitive on speed as well as quality. Comparing "e-means one pass over 1B points out-of-core" against "Flash-KMeans exact Lloyd on a 10% subsample" is a strong systems experiment. Out-of-core matters on a 32 GB card.
- Feasibility: 1B x 96-d fp32 is 360 GiB, so it must stream from NVMe or host RAM. It is feasible if the host has enough RAM or fast NVMe, and is otherwise limited to 100M scale (about 36 GiB).

### Gaps
- No source quantifies bf16 or fp8 error in k-means centroid accumulation.
- Flash-KMeans speedups are the authors' own H200 numbers. I found no Blackwell or consumer-GPU numbers.

---

## Q6. Learning partitions for query cost / routing quality instead of MSE

### Takeaway
Learned routing (BLISS 2022, learning-to-rank routing 2024, LIRA 2025) learns which partitions to probe, mostly on top of fixed k-means partitions. Few methods change the partition training objective itself. An EMA over (query, true-neighbour) statistics, where a centroid moves toward the queries that need its members, is a cheap, online form of query-aware partitioning. It sits between Quake's discrete cost model and fully learned routers.

### Cited Findings
- Fact: LIRA (arXiv 2503.23409, 2025) is a learning-based, query-aware partition framework. Its related work notes that BLISS uses a fixed nprobe, and that another line builds hierarchical balanced clusters with neural adaptive nprobe. — [arXiv 2503.23409](https://arxiv.org/pdf/2503.23409)
- Fact: "A Learning-to-Rank Formulation of Clustering-Based ANN Search" (arXiv 2404.11731, 2024) treats routing as ranking. Training data is cheap (queries plus exact-search labels), and learned routing improves ANN accuracy. — [arXiv 2404.11731](https://arxiv.org/html/2404.11731v1)
- Fact: "Unleashing Graph Partitioning for Large-Scale NN Search" (arXiv 2403.01797, 2024) proposes combinatorial routing usable with any partitioning, trained in about 30 minutes on billion-point data, in contrast to expensive neural routers. — [arXiv 2403.01797](https://arxiv.org/pdf/2403.01797)
- Fact: LoRANN (2024) notes that centroid routing remains the most common approach despite exploratory alternatives. — [arXiv 2410.18926](https://arxiv.org/pdf/2410.18926)

### Inferences
- Training knowledge: Neural LSH (Dong, Indyk, Razenshteyn, Wagner; ICLR 2020, arXiv 1901.08544) learns balanced partitions via graph partitioning plus a neural classifier. BLISS (Gupta et al., VLDB 2022) iteratively learns balanced partitions with query feedback.
- Speculation: add a third sufficient statistic per cluster, Q_j = an EMA of the query vectors whose true neighbours lie in j but which ranked j low. Use it to pull the routing centroid (kept separate from the data centroid) toward the queries it misses. This is a dual-centroid IVF: a data centroid for assignment and an EMA-trained routing centroid for probing. It is cheap, online, and compatible with balanced IVF. Measure recall@10 at fixed nprobe vs LTR routing (2404.11731) and LIRA. Novelty: medium-high. Feasibility: high. The required labels (exact kNN of training queries) are cheap on a 5090.

### Gaps
- I did not verify LIRA's authors, venue or numbers. BLISS and Neural LSH were not re-verified here.

---

## Q7. Data deduplication and curation at scale (SemDeDup, D4, hierarchical k-means curation) as a systems workload

### Takeaway
Clustering-based curation (SemDeDup 2023, D4 2023, Vo et al. 2024 hierarchical k-means curation) runs k-means over billions of embeddings as a preprocessing step. Its quality depends on balanced, well-populated clusters, which is exactly what EMA plus revival plus scaling-law hyperparameters target. This is a high-impact applied direction with an easy story ("a cheaper, one-pass k-means for curation at the same downstream accuracy"), but downstream evaluation (training a model on the curated data) is expensive.

### Cited Findings
- Fact: Vo et al., "Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach" (FAIR/INRIA/Google; arXiv 2405.15613; TMLR 2024). It applies k-means successively and hierarchically to obtain clusters spread uniformly over concepts, then samples in a balanced way. Features trained on the curated data beat uncurated data and match or beat manually curated data on web images, satellite images and text. Code: facebookresearch/ssl-data-curation. — [arXiv 2405.15613](https://arxiv.org/abs/2405.15613); [TMLR](https://mlanthology.org/tmlr/2024/vo2024tmlr-automatic)

### Inferences
- Training knowledge (not verified this session): SemDeDup (Abbas et al., 2023, arXiv 2303.09540) clusters CLIP/LLM embeddings with k-means and removes near-duplicates within clusters. D4 (Tirumala et al., NeurIPS 2023, arXiv 2308.12284) combines SemDeDup with SSL-prototype-based diversification. All of them use Faiss k-means.
- Speculation: a small, CPU/GPU-feasible proxy is to measure duplicate-pair recall (fraction of true near-duplicate pairs that land in the same cluster) and per-cluster size balance on LAION-subset or C4-subset embeddings (10–100M). Compare e-means vs Faiss k-means vs hierarchical k-means at equal wall-clock time. That avoids downstream model training. Vo et al.'s hierarchical scheme needs k-means at several levels, and the scaling law (a0, B) = f(n, k) is directly useful for choosing per-level settings. Novelty: medium (applied). Impact: high if it holds.

### Gaps
- SemDeDup/D4 scale numbers (dataset size, k, k-means cost) were not retrieved in this session.

---

## Q8. Additional directions beyond the candidate list (ranking and synthesis)

### Takeaway
Ranked by feasibility x novelty x impact on one RTX 5090 plus a 12-core CPU, with the key evidence for each:
1. **DP-e-means: subsampled, EMA-averaged noisy sufficient statistics with a privacy-aware scaling law and private revival.** I found no direct prior work. DP-Lloyd cost is linear in T ([Su et al.](https://arxiv.org/html/1504.05998v1)). The state of the art is FastLloyd ([USENIX Sec '25](https://www.usenix.org/conference/usenixsecurity25/presentation/diaa)). Feasibility very high, novelty high, impact medium-high. Extends naturally to DP-GMM-EMA and DP-PCA-EMA ([DP-PCA](https://arxiv.org/pdf/2205.13709)), and to federated with secure aggregation (Q2).
2. **Low-precision (bf16/fp8) EMA statistic accumulation plus a Flash-KMeans-style kernel.** The absorption failure under an annealed rate is new as far as I found. Baselines: [Carson et al. 2024](https://export.arxiv.org/pdf/2407.12208), [Flash-KMeans 2026](https://arxiv.org/abs/2603.09229). Feasibility very high, novelty medium-high, impact medium.
3. **Query-weighted EMA partitions / dual (data, routing) centroids.** A continuous alternative to Quake/Ada-IVF split-merge and learned routers ([Quake](https://arxiv.org/abs/2506.03437), [Ada-IVF](https://arxiv.org/abs/2411.00970v1), [LTR routing](https://arxiv.org/html/2404.11731v1), [LIRA](https://arxiv.org/pdf/2503.23409)). Feasibility high, novelty medium-high, impact high (vector DBs). Watch the overlap with the live drifting IVF track.
4. **One-pass, out-of-core IVF / PQ / PLAID centroid training at 100M–1B, against cuVS balanced hierarchical k-means on 10% subsamples** ([cuVS](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/)). Feasibility medium (I/O bound), novelty medium, impact high.
5. **DP continual-release streaming k-means via decayed-sum (matrix) mechanisms**, compared with the [Epasto et al. AISTATS 2026](https://proceedings.mlr.press/v300/epasto26a.html) framework. Feasibility high, novelty medium-high (theory-heavy), impact medium.
6. **Curation/dedup k-means at scale** ([Vo et al. 2024](https://arxiv.org/abs/2405.15613)). Feasibility medium, novelty medium, impact high.
7. **Federated EMA-statistics clustering with participation-aware revival (non-private)**, after [k-FED](https://arxiv.org/abs/2103.00697) and one-shot descendants. Feasibility high, novelty low-medium, impact medium.

### Cited Findings
- Fact: Quake explicitly notes that as partitions change, nprobe must adapt or recall degrades. This supports coupling any EMA partition update with an adaptive nprobe policy in evaluations. — [Quake arXiv 2506.03437](https://arxiv.org/html/2506.03437v2)
- Fact: Ada-IVF notes that a full billion-scale IVF rebuild can take days, which motivates cheap incremental centroid updates. — [arXiv 2411.00970](https://arxiv.org/abs/2411.00970v1)

### Inferences
- Speculation (extra direction, machine unlearning): an EMA of (S, N) supports approximate deletion of a point's influence (subtract its decayed weight). DP-e-means gives certified unlearning for free. Combining "deletion = negative EMA update" with DP noise as a certified-unlearning clustering method is a possible small paper, related to the "Making AI Forget You" k-means deletion work (Ginart et al., NeurIPS 2019, training knowledge, not verified).
- Speculation (extra direction, edge/embedded): the memory footprint is k(d+1) floats with no coreset. An "EMA k-means on microcontrollers / on-device keyboard clustering" angle overlaps with federated (Q2) and is lower priority.
- Speculation (extra direction, secure-aggregation-friendly counts): EMA weights are public scalars, so secure aggregation of (sum, count) composes with the EMA exactly. No new crypto is needed, which keeps the Q1+Q2 combination simple to implement.
- Interim results to connect: balanced IVF (+10% p99) and the OPQ joint-rotation wins suggest the e-means machinery is already competitive in quality. The HNSW parameter law losing to tuned efSearch suggests that "scaling laws for hyperparameters" pay off most where tuning is expensive or impossible. That is exactly the private setting (tuning spends privacy budget) and the federated setting (tuning spends communication rounds). This argues for prioritising #1. — internal context from the assignment, not an external source.

### Gaps
- Novelty claims ("no prior work found") come from about 15 searches, not a systematic literature review. Run targeted Google Scholar / Semantic Scholar queries before writing. In particular, check "differentially private mini-batch k-means", "stochastic DP k-means", "DP online k-means", and "bf16 k-means".
- Several baselines (StreamKM++, BICO, BLISS, Neural LSH, SemDeDup, D4, PLAID, Park et al. DP-EM, Ginart et al.) are from training knowledge and were not verified this session.
