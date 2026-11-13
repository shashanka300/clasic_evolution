"""T05 lda-scaling, v2 protocol. JSONL only; verdicts from scripts/evaluate.py.

Corpora (real): 20 Newsgroups (sklearn, headers/footers/quotes removed, top-10k vocab, English
stop words removed), UCI Enron (top-20k vocab), UCI NYTimes (first 100k docs, top-20k vocab).
Split per corpus: 2k val + 2k test docs (document-completion perplexity), rest train.

Methods per corpus (K = 50, B = 256, same Gamma(100, .01) init):
  reference  batch_vb               25 full passes
  baseline   hoffman_default        rho = (1024 + t)^-0.7  (Hoffman et al. 2010 best setting)
  baseline   sklearn_online_lda     sklearn defaults (offset 10, decay 0.7, batch 128); final point
  baseline   online_lda_tuned       (tau0, kappa) grid chosen on val per T
  ours       lda_ema_swept          cosine EMA + revival, alpha0 on val per T
  ours       cosine_law             cosine EMA, alpha0 from a law fitted ONLY on the other two corpora's
                                    law-sweep cells:  log a0* = c + a log(D/K) + b log B + t log T
Kill criterion: cosine_law worse than hoffman_default on 2 of 3 corpora (final, T = 5).
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.harness import make_run, read_runs, results_path, write_run  # noqa: E402
from core.plugin import run_ema_engine  # noqa: E402
from core.schedule import CosineSchedule, InverseTimeSchedule  # noqa: E402
from core.sweep import sweep_cells  # noqa: E402
from core.trackkit import BUDGETS, Problem, run_all  # noqa: E402
from lda_v2 import ETA, Corpus, LDAPlugin, batch_vb_step, init_lambda, npmi, perplexity  # noqa: E402

TRACK = "lda-scaling"
RUNS = results_path(TRACK)
SWEEP = results_path(TRACK, "sweep.jsonl")
DATA = ROOT / "data"
K = 50
HEAD_B = 256
CORPORA = ("20ng", "enron", "nytimes")


def _from_coo(doc, word, cnt, n_docs, V, keep_vocab):
    tot = np.bincount(word, weights=cnt, minlength=V)
    keep = np.argsort(-tot)[:keep_vocab]
    remap = -np.ones(V, np.int64)
    remap[keep] = np.arange(len(keep))
    m = remap[word] >= 0
    doc, word, cnt = doc[m], remap[word[m]], cnt[m]
    order = np.lexsort((word, doc))
    doc, word, cnt = doc[order], word[order], cnt[order]
    ip = np.zeros(n_docs + 1, np.int64)
    np.add.at(ip, doc + 1, 1)
    return Corpus(np.cumsum(ip), word, cnt, len(keep))


def load_corpus(name):
    cache = DATA / "processed" / f"lda_{name}.npz"
    if cache.exists():
        z = np.load(cache)
        return Corpus(z["indptr"], z["ids"], z["cts"], int(z["V"]))
    if name == "20ng":
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer
        d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), data_home=str(DATA / "raw" / "sk20ng"))
        X = CountVectorizer(max_features=10_000, stop_words="english", min_df=5).fit_transform(d.data).tocsr()
        c = Corpus(X.indptr, X.indices, X.data, X.shape[1])
    else:
        path = DATA / "raw" / "uci_bow" / f"docword.{name}.txt.gz"
        max_docs = 100_000 if name == "nytimes" else None
        with gzip.open(path, "rt") as f:
            D, V, _ = (int(f.readline()) for _ in range(3))
            arr = np.loadtxt(f, dtype=np.int64, max_rows=None if max_docs is None else 25_000_000)
        doc, word, cnt = arr[:, 0] - 1, arr[:, 1] - 1, arr[:, 2].astype(np.float64)
        if max_docs:
            m = doc < max_docs
            doc, word, cnt, D = doc[m], word[m], cnt[m], max_docs
        c = _from_coo(doc, word, cnt, D, V, 20_000)
    lens = np.diff(c.indptr)
    c = c.subset(np.flatnonzero(lens >= 10))  # drop near-empty docs
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, indptr=c.indptr, ids=c.ids, cts=c.cts, V=c.V)
    return c


def make_problem(name):
    c = load_corpus(name)
    p = np.random.default_rng(0).permutation(c.n)
    te, va, tr = c.subset(p[:2000]), c.subset(p[2000:4000]), c.subset(p[4000:])
    splits = {"test": te.completion_split(1), "val": va.completion_split(2)}
    docs = np.arange(tr.n, dtype=np.float64)[:, None]

    class M:
        def __init__(self, lam):
            self.lam = lam

    def fit(alpha0, T, B, seed, schedule=None, revival=True, n=None, k=None):
        sub = tr if n is None or n >= tr.n else tr.subset(np.arange(n))
        kk = k or K
        plug = LDAPlugin(sub, kk, seed=seed)
        steps = max(int(round(T * max(sub.n // B, 1))), 1)
        res = run_ema_engine(plug, docs[: sub.n], epochs=T, batch_size=B,
                             schedule=schedule or CosineSchedule(alpha0, steps, eps=0.1),
                             revival_every=5 if revival else 0, finalize_mode="none", seed=seed)
        m = M(plug.lam)
        m.revivals = res.revivals
        m.K = kk
        return m

    def loss(m, split):
        o, h = splits[split]
        return perplexity(m.lam, o, h, 1.0 / m.lam.shape[0])

    def extras(m):
        return {"npmi": npmi(m.lam, tr), "revivals": getattr(m, "revivals", None)}

    def reference(seed):
        lam = init_lambda(K, tr.V, seed) + ETA
        pts, wall = [], 0.0
        for it in range(1, 26):
            t0 = time.perf_counter()
            lam = batch_vb_step(tr, lam, 1.0 / K)
            wall += time.perf_counter() - t0
            mm = M(lam)
            pts.append({"epoch": it, "test_loss": loss(mm, "test"), "val": loss(mm, "val"), "wall_s": wall})
            print(f"   batch_vb it={it} test_ppl={pts[-1]['test_loss']:.1f} ({wall:.0f}s)", flush=True)
        return pts, M(lam)

    prob = Problem(track=TRACK, dataset=name, k=K, d=tr.V, n=tr.n, metric="test_perplexity", fit=fit, loss=loss,
                   reference=reference, extras=extras, config={"V": tr.V, "eta": ETA, "alpha": "1/K"})
    prob.train_corpus = tr
    return prob


def _budget_curve(p, method, seeds, make_sched, role, a0src, alpha0_fn=None, extra=None):
    if any(r["method"] == method and r["extra"]["group"] == p.group for r in read_runs(RUNS)):
        return
    for s in seeds:
        curve = []
        m = None
        for T in BUDGETS:
            a0 = alpha0_fn(T) if alpha0_fn else 1.0
            t0 = time.perf_counter()
            m = p.fit(a0, T, HEAD_B, s, schedule=make_sched(T, a0), revival=(role == "ours"))
            curve.append({"epoch": T, "test_loss": p.loss(m, "test"), "wall_s": time.perf_counter() - t0, "alpha0": a0})
        write_run(RUNS, make_run(track=TRACK, method=method, dataset=p.dataset, n=p.n, d=p.d, k_or_units=K,
                  epochs=BUDGETS[-1], batch=HEAD_B, alpha0=curve[-1]["alpha0"] if alpha0_fn else None, seed=s,
                  train_loss=None, test_loss=curve[-1]["test_loss"], wall_s=curve[-1]["wall_s"], role=role,
                  alpha0_source=a0src, group=p.group, metric=p.metric, curve=curve, curve_kind="budget",
                  config=p.config, npmi=npmi(m.lam, p.train_corpus), **(extra or {})))
        print(f"[{method}] {p.dataset} s={s} " + " ".join(f"T{c['epoch']:g}={c['test_loss']:.1f}" for c in curve), flush=True)


def fit_loo_law(target):
    """log a0* = c + a log(D/K) + b log B + t log T, fitted on the OTHER corpora's law-sweep cells."""
    rows = []
    for (g, T, m), cell in sweep_cells(SWEEP, "lda_ema").items():
        ds = g.split("|")[0]
        if ds == target or cell["at_grid_edge"]:
            continue
        kv = dict(part.split("=") for part in g.split("|")[1:4])
        rows.append((float(kv["n"]) / float(kv["k"]), float(kv["B"]), T, cell["alpha_star"]))
    if len(rows) < 6:
        raise RuntimeError(f"only {len(rows)} usable law-sweep cells for LOO law (run --stage law-sweep first)")
    X = np.array([[1.0, np.log(a), np.log(b), np.log(t)] for a, b, t, _ in rows])
    y = np.log([r[3] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(CORPORA))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    for name in args.datasets.split(","):
        p = make_problem(name)
        print(f"\n##### {name} D={p.n} V={p.d} K={K}", flush=True)
        law_cfgs = [{"n": n, "B": B, "T": T} for n in (p.n // 4, p.n) for B in (64, 512) for T in (1.0, 2.0)]
        if args.stage in ("all", "reference", "sweep", "online", "emit", "law-sweep"):
            run_all(p, RUNS, SWEEP, seeds[:2] if args.stage == "law-sweep" else seeds, ref_method="batch_vb",
                    ema_method="lda_ema", stepwise_method="online_lda", head_B=HEAD_B, stage=args.stage,
                    law_sweep_configs=law_cfgs, law_ks=(20, 50))
        if args.stage in ("all", "hoffman"):
            _budget_curve(p, "hoffman_default", seeds, lambda T, a0: InverseTimeSchedule(1.0, tau0=1024.0, kappa=0.7),
                          "baseline", "na")
        if args.stage in ("all", "sklearn"):
            stage_sklearn(p, seeds)
        if args.stage == "cosine-law":
            beta, nrows = fit_loo_law(name)
            law = lambda T: float(np.clip(np.exp(beta @ [1.0, np.log(p.n / K), np.log(HEAD_B), np.log(T)]), 1e-3, 1.0))
            print(f"[law] {name}: LOO coeffs {np.round(beta, 3)} from {nrows} cells; a0(T)=" +
                  ", ".join(f"{law(T):.3f}" for T in BUDGETS), flush=True)
            steps = lambda T: max(int(round(T * max(p.n // HEAD_B, 1))), 1)
            _budget_curve(p, "cosine_law", seeds, lambda T, a0: CosineSchedule(a0, steps(T), eps=0.1), "ours", "law",
                          alpha0_fn=law, extra={"law_coeffs": beta.tolist(), "law_cells": nrows, "law_fit": "leave-one-corpus-out"})


def stage_sklearn(p, seeds):
    from sklearn.decomposition import LatentDirichletAllocation
    from scipy.sparse import csr_matrix
    if any(r["method"] == "sklearn_online_lda" and r["extra"]["group"] == p.group for r in read_runs(RUNS)):
        return
    tr = p.train_corpus
    X = csr_matrix((tr.cts, tr.ids, tr.indptr), shape=(tr.n, tr.V))
    for s in seeds:
        t0 = time.perf_counter()
        lda = LatentDirichletAllocation(K, learning_method="online", max_iter=5, batch_size=128, random_state=s,
                                        doc_topic_prior=1.0 / K, topic_word_prior=ETA, n_jobs=1).fit(X)
        w = time.perf_counter() - t0

        class M:
            lam = lda.components_
        write_run(RUNS, make_run(track=TRACK, method="sklearn_online_lda", dataset=p.dataset, n=p.n, d=p.d, k_or_units=K,
                  epochs=5, batch=128, alpha0=None, seed=s, train_loss=None, test_loss=p.loss(M, "test"), wall_s=w,
                  role="baseline", alpha0_source="na", group=p.group, metric=p.metric, config=p.config,
                  npmi=npmi(lda.components_, tr), note="sklearn defaults, 5 passes; final point only"))


if __name__ == "__main__":
    main()
