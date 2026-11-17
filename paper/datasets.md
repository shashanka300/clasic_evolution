# Datasets: sources, citations, licences (checked 2026-11-17)

Licence terms were checked on 2026-11-17 against the provider's page or the README shipped with the downloaded data
(`~/clasic_evolution/data/raw/`). The **checked against** column gives the source. Where neither could be read, the
entry says so. We redistribute none of the raw data: `scripts/fetch_data.py` downloads it from the provider, and we
release only derived results (per-run JSONL with losses, no data points). That satisfies every term below, including
MovieLens' no-redistribution clause.

| dataset | used in | source (fetch_data.py) | cite | licence / terms | checked against |
|---|---|---|---|---|---|
| SIFT1M | GMM (PCA-64), T00 | ANN-Benchmarks HDF5 (TEXMEX corpus, INRIA) | Jégou, Douze & Schmid 2011 (TPAMI); Aumüller et al. 2020 | no licence found; the TEXMEX page (corpus-texmex.irisa.fr) was unreachable (TLS error) and no mirror states the original terms | not verified: cite and state "research use" |
| DINOv2-small embeddings of CIFAR-100 | GMM (PCA-64), T00 | computed by us (facebook/dinov2-small) | Oquab et al. 2024; Krizhevsky 2009 | model Apache-2.0 (Hugging Face card; Meta release note); CIFAR: no licence stated | HF model card; CIFAR page |
| AG News (MiniLM embeddings) | GMM (PCA-64), T00 | Hugging Face `ag_news`; all-MiniLM-L6-v2 | Zhang, Zhao & LeCun 2015; Reimers & Gurevych 2019 | dataset card: licence "unknown"; its summary describes the AG corpus as provided for research and non-commercial use. Model Apache-2.0 | HF dataset card (fancyzhx/ag_news); model licence from secondary sources |
| CIFAR-100 patches | dictionary learning | CIFAR-100 python version | Krizhevsky 2009 (tech report; the page asks for this citation) | no licence stated on the page | cs.toronto.edu/~kriz/cifar.html |
| 20 Newsgroups | LDA, NMF | scikit-learn `fetch_20newsgroups` (20news-bydate) | Lang 1995 | no licence known; the home page (qwone.com/~jason/20Newsgroups) refused connections | not verified |
| Enron, NYTimes (bag of words) | LDA | UCI ML Repository "Bag of Words" | Newman 2008, doi:10.24432/C5ZG6P | **CC BY 4.0**; document names withheld for copyright reasons | UCI dataset page 164 |
| MovieLens 1M, 10M | ALS | GroupLens | Harper & Konstan 2015 | research use only; must acknowledge in publications; no implied endorsement; **no redistribution** without permission; no commercial use without permission | READMEs shipped with ml-1m and ml-10M100K |
| Brown corpus | HMM | NLTK data | Francis & Kučera 1979 | README: distributed with the copyright holder's permission, "redistribution permitted"; NLTK's index lists it as non-commercial use | NLTK `corpora/brown/README`; NLTK index (secondary) |
| universal tagset mapping | HMM (Brown tags) | NLTK `universal_tagset` | Petrov, Das & McDonald 2012 | CC BY-SA 4.0 per the original project page (secondary source; the local README states no licence) | not verified at source |
| UCI HAR | HMM (vector-quantised) | UCI ML Repository | Anguita et al. 2013 (ESANN); dataset doi:10.24432/C54S4K | UCI page: **CC BY 4.0**. The shipped README also says "Any commercial use is prohibited". Both are compatible with research use; cite the ESANN paper as the README requires | UCI dataset page 240; shipped README |
| synthetic (GMM, HMM) | GMM, HMM | generated from fixed seeds (code) | — | ours | — |

Open: SIFT1M and 20 Newsgroups terms could not be read (sites down); both are standard research benchmarks with no
known restriction on publishing results.

Notes for the paper's data statement:
- Text data (20NG, Enron, NYTimes, AG News, Brown) may contain personal names and email content. We use only
  bag-of-words counts or embeddings, and we report only aggregate losses.
- The NYTimes corpus is restricted to the first 100,000 documents and the 20,000 most frequent words. Documents with
  fewer than 10 tokens are dropped from all LDA corpora (`tracks/lda-scaling/run_v2.py`).
