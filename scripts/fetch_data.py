#!/usr/bin/env python3
"""Download + preprocess every dataset the job queue needs into data/ (or $CLASSIC_ALGOS_DATA_DIR), then touch data/.ready.

  1. network   (parallel threads)  ann-benchmarks SIFT1M / GloVe-100, MovieLens 1M / 10M, UCI BoW Enron / NYTimes,
                                   UCI HAR, NLTK Brown + universal tagset, 20 Newsgroups, HF CIFAR-100 / AG News
  2. GPU       (sequential)        DINOv2-S CIFAR-100, MiniLM and BGE AG News embeddings (val_size=0, as on the Mac:
                                   the cache path ignores val_size, so the first builder fixes the split)
  3. CPU       (parallel procs)    each track's processed cache (dictlearn patches / TF-IDF, LDA CSR, ALS splits, HAR VQ),
                                   built once here so concurrent jobs never race to write the same .npz
Idempotent: finished pieces are skipped. Run from the repo root:  .venv/bin/python scripts/fetch_data.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.bench import data_root, download_file  # noqa: E402

D = data_root()
RAW = D / "raw"


def log(msg):
    print(f"{time.strftime('%T')} {msg}", flush=True)


def fetch(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return
    download_file(url, dest)


def unzip(zpath: Path, into: Path, marker: Path):
    if marker.exists():
        return
    with zipfile.ZipFile(zpath) as z:
        z.extractall(into)


# ------------------------------------------------------------------ 1. network

def ann(name):
    fetch(f"https://ann-benchmarks.com/{name}.hdf5", RAW / f"{name}.hdf5")


def movielens(slug, folder):
    z = RAW / "movielens" / f"{slug}.zip"
    fetch(f"https://files.grouplens.org/datasets/movielens/{slug}.zip", z)
    unzip(z, RAW / "movielens", RAW / "movielens" / folder / "ratings.dat")


def uci_bow(name):
    fetch(f"https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.{name}.txt.gz",
          RAW / "uci_bow" / f"docword.{name}.txt.gz")


def uci_har():
    base = RAW / "uci_har"
    marker = base / "UCI HAR Dataset" / "train" / "Inertial Signals" / "body_acc_x_train.txt"
    if marker.exists():
        return
    outer = base / "uci_har_outer.zip"
    fetch("https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip", outer)
    with zipfile.ZipFile(outer) as z:
        inner = [n for n in z.namelist() if n.endswith("UCI HAR Dataset.zip")]
        if inner:  # the UCI download wraps the dataset zip in another zip
            z.extract(inner[0], base)
            with zipfile.ZipFile(base / inner[0]) as zi:
                zi.extractall(base)
        else:
            z.extractall(base)


def nltk_brown():
    import nltk
    dest = str(RAW / "nltk_data")
    for pkg in ("brown", "universal_tagset"):
        nltk.download(pkg, download_dir=dest, quiet=True, raise_on_error=True)


def news20():
    from sklearn.datasets import fetch_20newsgroups
    fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), data_home=str(RAW / "sk20ng"))


def hf_cifar():
    from datasets import load_dataset
    for split in ("train", "test"):
        load_dataset("uoft-cs/cifar100", split=split, cache_dir=str(RAW / "hf_cache"))


def hf_agnews():
    from datasets import load_dataset
    load_dataset("fancyzhx/ag_news")


def hf_models():
    from huggingface_hub import snapshot_download
    for m in ("facebook/dinov2-small", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"):
        snapshot_download(m)


NETWORK = {
    "sift1m": lambda: ann("sift-128-euclidean"),
    "glove100": lambda: ann("glove-100-angular"),
    "ml1m": lambda: movielens("ml-1m", "ml-1m"),
    "ml10m": lambda: movielens("ml-10m", "ml-10M100K"),
    "enron": lambda: uci_bow("enron"),
    "nytimes": lambda: uci_bow("nytimes"),
    "uci_har": uci_har,
    "brown": nltk_brown,
    "20ng": news20,
    "cifar100": hf_cifar,
    "ag_news": hf_agnews,
    "hf_models": hf_models,
}

# ------------------------------------------------------------------ 3. processed caches (one subprocess each)

CACHE_JOBS = {
    "dl_cifar_p8": ("tracks/dictlearn-scaling/run_v2.py", "load('cifar_p8')"),
    "dl_cifar_p12": ("tracks/dictlearn-scaling/run_v2.py", "load('cifar_p12')"),
    "dl_nmf_20ng": ("tracks/dictlearn-scaling/run_v2.py", "load('nmf_20ng')"),
    "lda_20ng": ("tracks/lda-scaling/run_v2.py", "load_corpus('20ng')"),
    "lda_enron": ("tracks/lda-scaling/run_v2.py", "load_corpus('enron')"),
    "lda_nytimes": ("tracks/lda-scaling/run_v2.py", "load_corpus('nytimes')"),
    "ml1m_split": ("tracks/als-ema/run_v2.py", "load('ml1m')"),
    "ml10m_split": ("tracks/als-ema/run_v2.py", "load('ml10m')"),
    "uci_har_vq": ("tracks/hmm-ema/run_v2.py", "har()"),
    "brown_check": ("tracks/hmm-ema/run_v2.py", "brown()"),
}


def cache_job(name, script, call):
    code = (f"import importlib.util,sys; sys.argv=['x']; "
            f"s=importlib.util.spec_from_file_location('m', {str(ROOT / script)!r}); m=importlib.util.module_from_spec(s); "
            f"s.loader.exec_module(m); m.{call}")
    env = dict(os.environ, OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{name}: {r.stderr[-1500:]}")


def run_pool(jobs: dict, fn, workers):
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, k, v): k for k, v in jobs.items()}
        for f in as_completed(futs):
            k = futs[f]
            try:
                f.result()
                log(f"  ok   {k}")
            except Exception as e:  # keep going; report at the end
                failed.append(k)
                log(f"  FAIL {k}: {type(e).__name__}: {str(e)[-800:]}")
    return failed


def main():
    t0 = time.time()
    log(f"data dir: {D}")
    log("1/3 network downloads (parallel)")
    failed = run_pool(NETWORK, lambda k, fn: fn(), workers=8)

    log("2/3 embeddings on GPU")
    from core.bench import load_dinov2_cifar100, load_sift1m, load_text_embeddings
    from core.device import accel
    log(f"  device: {accel()}")
    steps = {
        "sift1m_load": lambda: load_sift1m(val_size=0),
        "dinov2_cifar100": lambda: load_dinov2_cifar100(val_size=0),
        "agnews_minilm": lambda: load_text_embeddings(dataset_name="ag_news", max_train=100_000, max_test=7_600, val_size=0),
        "agnews_bge": lambda: load_text_embeddings(dataset_name="ag_news", model_name="BAAI/bge-small-en-v1.5",
                                                   max_train=100_000, max_test=7_600, val_size=0),
    }
    for k, fn in steps.items():
        if k in ("dinov2_cifar100",) and "cifar100" in failed or k.startswith("agnews") and "ag_news" in failed:
            continue
        try:
            tic = time.time()
            s = fn()
            log(f"  ok   {k}: train {s.train.shape} test {s.test.shape} ({time.time() - tic:.0f}s)")
        except Exception as e:
            failed.append(k)
            log(f"  FAIL {k}: {type(e).__name__}: {e}")

    log("3/3 processed caches (parallel processes)")
    failed += run_pool(CACHE_JOBS, lambda k, v: cache_job(k, *v), workers=6)

    log(f"done in {(time.time() - t0) / 60:.1f} min; failed: {failed or 'none'}")
    if failed:
        sys.exit(1)
    (D / ".ready").touch()
    log(f"touched {D / '.ready'}: data-dependent queue jobs start on the scheduler's next poll")


if __name__ == "__main__":
    main()
