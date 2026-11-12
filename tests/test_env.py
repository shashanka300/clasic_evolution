"""Smoke tests verifying the Apple Silicon uv benchmark environment across all tracks."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np


def test_core_and_torch_mps() -> None:
    import torch
    import torchvision

    assert torch.backends.mps.is_available(), "Expected Apple Silicon MPS backend"
    x = torch.randn(128, 32, device="mps", dtype=torch.float32)
    y = x @ x.T
    torch.mps.synchronize()
    assert y.shape == (128, 128)
    assert hasattr(torchvision, "__version__")


def test_faiss_kmeans_opq_ivf() -> None:
    import faiss

    rng = np.random.default_rng(0)
    xb = rng.standard_normal((500, 32), dtype=np.float32)
    xq = rng.standard_normal((10, 32), dtype=np.float32)

    # Lloyd k-means (T00 / T03 / T09 / T10)
    km = faiss.Kmeans(d=32, k=16, niter=5, verbose=False)
    km.train(xb)
    assert km.centroids.shape == (16, 32)

    # OPQ + PQ (T02)
    opq = faiss.OPQMatrix(32, 4)
    opq.train(xb)
    xb_rot = opq.apply_py(xb)
    assert xb_rot.shape == (500, 32)

    # IVF flat (T03 / T13)
    quantizer = faiss.IndexFlatL2(32)
    index = faiss.IndexIVFFlat(quantizer, 32, 16)
    index.train(xb)
    index.add(xb)
    index.nprobe = 4
    dists, ids = index.search(xq, 5)
    assert dists.shape == (10, 5)
    assert ids.shape == (10, 5)


def test_hnswlib_build_and_query() -> None:
    import hnswlib

    rng = np.random.default_rng(1)
    xb = rng.standard_normal((300, 16)).astype(np.float32)
    idx = hnswlib.Index(space="l2", dim=16)
    idx.init_index(max_elements=300, ef_construction=50, M=8)
    idx.add_items(xb)
    idx.set_ef(20)
    labels, dists = idx.knn_query(xb[:5], k=3)
    assert labels.shape == (5, 3)
    assert dists.shape == (5, 3)


def test_sklearn_gmm_and_dictlearn() -> None:
    from sklearn.decomposition import MiniBatchDictionaryLearning, MiniBatchNMF
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 12))
    gmm = GaussianMixture(n_components=4, covariance_type="diag", max_iter=5, random_state=0).fit(X)
    assert gmm.means_.shape == (4, 12)

    dl = MiniBatchDictionaryLearning(n_components=8, max_iter=5, batch_size=32, random_state=0).fit(X)
    assert dl.components_.shape == (8, 12)

    nmf = MiniBatchNMF(n_components=4, max_iter=5, batch_size=32, random_state=0).fit(np.abs(X))
    assert nmf.components_.shape == (4, 12)


def test_gensim_lda_and_coherence() -> None:
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel, LdaModel

    texts = [
        ["human", "interface", "computer", "system"],
        ["survey", "user", "computer", "system", "response", "time"],
        ["eps", "user", "interface", "system"],
        ["trees", "graph", "minors", "survey"],
        ["graph", "minors", "trees", "system"],
    ]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=2, random_state=0)
    cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_npmi")
    score = cm.get_coherence()
    assert np.isfinite(score)


def test_implicit_als() -> None:
    import implicit
    import scipy.sparse as sp

    rng = np.random.default_rng(3)
    mat = sp.random(60, 40, density=0.15, format="csr", dtype=np.float32, random_state=rng)
    model = implicit.als.AlternatingLeastSquares(factors=8, iterations=3, num_threads=2, random_state=0)
    model.fit(mat, show_progress=False)
    assert model.user_factors.shape == (60, 8)
    assert model.item_factors.shape == (40, 8)


def test_hmmlearn_baum_welch() -> None:
    from hmmlearn import hmm

    rng = np.random.default_rng(4)
    X = rng.standard_normal((150, 3))
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=5, random_state=0)
    model.fit(X, lengths=[50, 50, 50])
    assert np.isfinite(model.score(X, lengths=[50, 50, 50]))


def test_audio_numba_hdf5_and_data_libs() -> None:
    import datasets
    import h5py
    import huggingface_hub
    import librosa
    import matplotlib
    import nltk
    import numba
    import pandas as pd
    import pyarrow
    import sentence_transformers
    import statsmodels.api as sm
    import transformers

    @numba.njit
    def sum_sq(arr: np.ndarray) -> float:
        acc = 0.0
        for i in range(arr.shape[0]):
            acc += arr[i] * arr[i]
        return acc

    assert abs(sum_sq(np.array([3.0, 4.0])) - 25.0) < 1e-6

    y = np.sin(2 * np.pi * 440 * np.linspace(0, 0.25, 4000, dtype=np.float32))
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=16)
    assert mel.shape[0] == 16

    with tempfile.TemporaryDirectory() as tmp:
        h5_path = Path(tmp) / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("train", data=np.zeros((10, 4), dtype=np.float32))
        with h5py.File(h5_path, "r") as f:
            assert f["train"].shape == (10, 4)

    for mod in (datasets, huggingface_hub, matplotlib, nltk, pd, pyarrow, sentence_transformers, sm, transformers):
        assert hasattr(mod, "__version__") or hasattr(mod, "__name__")


def test_worktree_isolation_and_data_dir() -> None:
    root = Path.cwd()
    if (root / "core").is_dir():
        import core

        assert Path(core.__file__).resolve().parent == (root / "core").resolve()
    if (root / "data").exists():
        assert (root / "data").is_dir()


