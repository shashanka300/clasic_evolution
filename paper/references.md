# References (draft bibliography, author–year)

Status: **v** = venue/details checked by web search (2026-11-16/17); unmarked = standard references (classic papers,
datasets, libraries, well-known preprints) whose details are from memory. BibTeX: `paper/tmlr/references.bib`.

## Starting point and fast exact baselines
- Bessa (Swann), Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou (2026). e-means: outperforming k-means (in one-tenth the
  epochs). arXiv:2611.09501. **v** (PAPER_LOG §6)
- Weiss, Eyal (2026). Small Is Not Dead: Distance-Aware Cluster Revival for e-means. arXiv:2611.18238. **v**
  (2026-11-17; k-means only, keeps small isolated clusters that count-based revival removes)
- Yang, Shuo; Xi, Haocheng; et al. (13 authors) (2026). Flash-KMeans: Fast and memory-efficient exact k-means. arXiv:2603.09229; NeurIPS 2026
  poster. **v**
- Bloch (Gal), Gera (Ariel), Orbach (Matan), Eytan (Ohad), Toledo (Assaf) (2026). Flash-GMM: A memory-efficient kernel for scalable soft clustering.
  arXiv:2606.10896. **v**
- Johnson, Douze, Jégou (2019). Billion-scale similarity search with GPUs. IEEE Trans. Big Data 7(3):535–547.
  doi:10.1109/TBDATA.2019.2921572. (Faiss) **v**
- Douze, Guzhva, Deng, Johnson, Szilvasy, Mazaré, Lomeli, Hosseini, Jégou (2024). The Faiss library.
  arXiv:2401.08281. **v**
- Sculley (2010). Web-scale k-means clustering. WWW 2010, pp. 1177–1178. doi:10.1145/1772690.1772862. **v**

## Online / stepwise EM, stochastic approximation, SVI
- Robbins, Monro (1951). A stochastic approximation method. Ann. Math. Stat. 22(3):400–407.
- Neal, Hinton (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. In
  Learning in Graphical Models (Jordan, ed.), Kluwer.
- Delyon, Lavielle, Moulines (1999). Convergence of a stochastic approximation version of the EM algorithm. Ann.
  Stat. 27(1):94–128. doi:10.1214/aos/1018031103. **v**
- Sato, Ishii (2000). On-line EM algorithm for the normalized Gaussian network. Neural Computation 12(2):407–432.
  doi:10.1162/089976600300015853. **v** (includes unit production / deletion / division: prior art for revival)
- Cappé, Moulines (2009). On-line expectation–maximization algorithm for latent data models. JRSS-B 71(3):593–613.
  doi:10.1111/j.1467-9868.2009.00698.x. **v**
- Liang, Klein (2009). Online EM for unsupervised models. NAACL-HLT 2009, pp. 611–619 (ACL N09-1069). **v**
- Hoffman, Bach, Blei (2010; order as in the NeurIPS proceedings). Online learning for latent Dirichlet allocation. NeurIPS 2010, pp. 856–864. **v** (proceedings list the
  authors as Hoffman, Bach, Blei; check the order against the PDF at the BibTeX stage)
- Hoffman, Blei, Wang, Paisley (2013). Stochastic variational inference. JMLR 14:1303–1347. **v**
- Mairal, Bach, Ponce, Sapiro (2010). Online learning for matrix factorization and sparse coding. JMLR 11:19–60. **v**
- Nguyen, Forbes, McLachlan (2020). Mini-batch learning of exponential family finite mixture models. Statistics and
  Computing 30:731–748. doi:10.1007/s11222-019-09919-4; arXiv:1902.03335. **v**
- Ranganath, Wang, Blei, Xing (2013). An adaptive learning rate for stochastic variational inference. ICML 2013,
  PMLR 28(2):298–306. **v**
- Hughes, Sudderth (2013). Memoized online variational inference for Dirichlet process mixture models. NeurIPS 2013, pp. 1133–1141. **v**
- Chen, Zhu, Teh, Zhang (2018). Stochastic expectation maximization with variance reduction. NeurIPS 2018. **v**
- Hosseini, Sra (2020). An alternative to EM for Gaussian mixture models: batch and stochastic Riemannian
  optimization. Mathematical Programming 181:187–223 (2020). doi:10.1007/s10107-019-01381-4. **v**

## Schedules, scaling laws, hyperparameter transfer
- Loshchilov, Hutter (2017). SGDR: Stochastic gradient descent with warm restarts. ICLR 2017. (cosine schedule)
- McCandlish, Kaplan, Amodei, OpenAI Dota team (2018). An empirical model of large-batch training.
  arXiv:1812.06162. (gradient noise scale)
- Kaplan et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
- DeepSeek-AI (Bi et al.) (2024). DeepSeek LLM: Scaling open-source language models with longtermism.
  arXiv:2401.02954. (learning-rate / batch-size laws)
- Busbridge, Ramapuram, Ablin, Likhomanenko, Dhekane, Suau, Webb (2023). How to scale your EMA. NeurIPS 2023 (spotlight); arXiv:2307.13813. **v**

## Datasets and models
- Jégou, Douze, Schmid (2011). Product quantization for nearest neighbor search. TPAMI 33(1). (SIFT1M)
- Aumüller, Bernhardsson, Faithfull (2020). ANN-Benchmarks. Information Systems 87:101374.
  doi:10.1016/j.is.2019.02.006. (SIFT1M / GloVe distribution) **v**
- Oquab et al. (2024). DINOv2: Learning robust visual features without supervision. TMLR. **v**
- Reimers, Gurevych (2019). Sentence-BERT. EMNLP 2019; Wang et al. (2020). MiniLM. NeurIPS 2020.
  (all-MiniLM-L6-v2 embeddings)
- Zhang, Zhao, LeCun (2015). Character-level convolutional networks for text classification. NeurIPS 2015.
  (AG News)
- Krizhevsky (2009). Learning multiple layers of features from tiny images. Tech. report. (CIFAR-100)
- Lang (1995). NewsWeeder: Learning to filter netnews. ICML 1995. (20 Newsgroups)
- Newman (2008). Bag of Words data set. UCI Machine Learning Repository. (Enron, NYTimes)
- Harper, Konstan (2015). The MovieLens datasets: History and context. ACM TiiS 5(4).
- Francis, Kučera (1979). Brown Corpus manual; Petrov, Das, McDonald (2012). A universal part-of-speech tagset.
  LREC 2012.
- Anguita, Ghio, Oneto, Parra, Reyes-Ortiz (2013). A public domain dataset for human activity recognition using
  smartphones. ESANN 2013. (UCI HAR)
- Zhou, Wilkinson, Schreiber, Pan (2008). Large-scale parallel collaborative
  filtering for the Netflix prize. AAIM 2008, pp. 337–348. **v**

## Libraries
- Pedregosa et al. (2011). Scikit-learn. JMLR 12. — hmmlearn (software, v0.3.3). — Paszke et al. (2019). PyTorch.
  NeurIPS 2019. — Lam, Pitrou, Seibert (2015). Numba. LLVM-HPC.
