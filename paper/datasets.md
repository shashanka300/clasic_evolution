# Datasets: sources, citations, licences (draft, 2026-11-16)

Every licence line is from memory of the provider's page and **must be checked against that page before
submission**. We redistribute none of the raw data. `scripts/fetch_data.py` downloads it from the original
provider, and we release only derived results: per-run JSONL with losses, no data points.

| dataset | used in | source (fetch_data.py) | cite | licence / terms (to verify) |
|---|---|---|---|---|
| SIFT1M | GMM (PCA-64), T00 | ANN-Benchmarks HDF5 (TEXMEX corpus, INRIA) | Jégou et al. 2011; Aumüller et al. 2020 | no formal licence stated; research use |
| DINOv2-small embeddings of CIFAR-100 | GMM (PCA-64), T00 | computed by us (facebook/dinov2-small) | Oquab et al. 2024; Krizhevsky 2009 | model Apache-2.0; CIFAR-100 no formal licence |
| AG News (MiniLM embeddings) | GMM (PCA-64), T00 | Hugging Face `ag_news`; all-MiniLM-L6-v2 | Zhang et al. 2015; Reimers & Gurevych 2019 | AG corpus: non-commercial / academic use; model Apache-2.0 |
| CIFAR-100 patches | dictionary learning | CIFAR-100 python version | Krizhevsky 2009 | no formal licence stated |
| 20 Newsgroups | LDA, NMF | scikit-learn `fetch_20newsgroups` | Lang 1995 | no formal licence; widely redistributed for research |
| Enron, NYTimes (bag of words) | LDA | UCI ML Repository "Bag of Words" | Newman 2008 | UCI repository: CC BY 4.0 |
| MovieLens 1M, 10M | ALS | GroupLens | Harper & Konstan 2015 | GroupLens usage licence: research use, no redistribution, must cite |
| Brown corpus + universal tagset | HMM | NLTK data | Francis & Kučera 1979; Petrov et al. 2012 | NLTK notes non-commercial use |
| UCI HAR | HMM (vector-quantised) | UCI ML Repository | Anguita et al. 2013 | CC BY 4.0 |
| synthetic (GMM, HMM) | GMM, HMM | generated from fixed seeds (code) | — | ours |

Notes for the paper's data statement:
- Text data (20NG, Enron, NYTimes, AG News, Brown) may contain personal names and email content. We use only
  bag-of-words counts or embeddings, and we report only aggregate losses.
- The NYTimes corpus is restricted to the first 100,000 documents and the 20,000 most frequent words. Documents with
  fewer than 10 tokens are dropped from all LDA corpora (`tracks/lda-scaling/run_v2.py`).
