"""T07 hmm-ema, v2 protocol. JSONL only; verdicts from scripts/evaluate.py.

Datasets
  synthetic   S=12 sticky HMM, V=200, known parameters (emission recovery error reported)
  brown_pos   Brown corpus (NLTK), lower-cased, top-5000 vocab + UNK, S=12 (universal tagset size),
              many-to-one accuracy against gold universal tags reported
  uci_har_vq  UCI HAR inertial signals: each 128-step window is a sequence; the 9-channel frame
              at each step is vector-quantised to 64 codes (k-means on train), S=12

Methods (same Dirichlet init everywhere): baum_welch_full_batch (reference, 25 it),
hmmlearn_bw (sanity), stepwise_em_tuned (Liang & Klein 2009; tau0, kappa tuned on val per T),
hmm_ema_swept, hmm_ema_kmlaw.

NOTE: Cappe (2011) filter-recursive online EM for HMMs is not implemented; stepwise EM over
sequence mini-batches is the comparator (documented gap).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.harness import results_path  # noqa: E402
from core.plugin import run_ema_engine  # noqa: E402
from core.schedule import CosineSchedule  # noqa: E402
from core.trackkit import Problem, run_all  # noqa: E402
from hmm_v2 import HMMPlugin, em_step, many_to_one, nll_per_token, pad, random_init  # noqa: E402

TRACK = "hmm-ema"
RUNS = results_path(TRACK)
SWEEP = results_path(TRACK, "sweep.jsonl")
DATA = ROOT / "data"
HEAD_B = 256


def synthetic(S=12, V=200, n_train=20_000, seed=7):
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(S))
    A = np.array([rng.dirichlet(np.ones(S) * 0.3 + 8.0 * np.eye(S)[i]) for i in range(S)])
    B = rng.dirichlet(np.ones(V) * 0.05, size=S)

    def sample(m):
        out = []
        for _ in range(m):
            T = int(rng.integers(20, 61))
            z = rng.choice(S, p=pi)
            o = np.empty(T, np.int64)
            for t in range(T):
                o[t] = rng.choice(V, p=B[z])
                z = rng.choice(S, p=A[z])
            out.append(o)
        return pad(out, 60)
    return {"train": sample(n_train), "val": sample(2000), "test": sample(2000)}, {"true_B": B, "S": S, "V": V}


def brown(vocab=5000, seed=0):
    import nltk
    nltk.data.path.insert(0, str(DATA / "raw" / "nltk_data"))
    from nltk.corpus import brown as bc
    sents = bc.tagged_sents(tagset="universal")
    words = [[w.lower() for w, _ in s] for s in sents]
    from collections import Counter
    cnt = Counter(w for s in words for w in s)
    itos = ["<unk>"] + [w for w, _ in cnt.most_common(vocab - 1)]
    stoi = {w: i for i, w in enumerate(itos)}
    tagset = sorted({t for s in sents for _, t in s})
    ttoi = {t: i for i, t in enumerate(tagset)}
    X = [np.array([stoi.get(w, 0) for w in s], np.int64) for s in words]
    Y = [np.array([ttoi[t] for _, t in s], np.int64) for s in sents]
    p = np.random.default_rng(seed).permutation(len(X))
    L = 80
    Xp, Yp = pad([X[i] for i in p], L), pad([Y[i] for i in p], L)
    nte, nva = 5000, 5000
    unigram = np.bincount(np.concatenate(X), minlength=vocab).astype(float)
    data = {"test": Xp[:nte], "val": Xp[nte:nte + nva], "train": Xp[nte + nva:]}
    return data, {"S": 12, "V": vocab, "tags_test": Yp[:nte], "unigram": unigram / unigram.sum()}


def har(codes=64):
    import faiss
    base = DATA / "raw" / "uci_har" / "UCI HAR Dataset"
    chans = [f"{a}_{x}" for a in ("body_acc", "body_gyro", "total_acc") for x in "xyz"]

    def read(split):
        arr = [np.loadtxt(base / split / "Inertial Signals" / f"{c}_{split}.txt", dtype=np.float32) for c in chans]
        return np.stack(arr, -1)  # (n, 128, 9)
    cache = DATA / "processed" / "uci_har_vq.npz"
    if cache.exists():
        z = np.load(cache)
        return {"train": z["train"], "val": z["val"], "test": z["test"]}, {"S": 12, "V": codes}
    tr, te = read("train"), read("test")
    mu, sd = tr.reshape(-1, 9).mean(0), tr.reshape(-1, 9).std(0)
    km = faiss.Kmeans(9, codes, niter=25, seed=0)
    km.train(((tr.reshape(-1, 9) - mu) / sd).astype(np.float32))
    q = lambda a: km.index.search(((a.reshape(-1, 9) - mu) / sd).astype(np.float32), 1)[1].reshape(a.shape[0], 128).astype(np.int64)
    qtr, qte = q(tr), q(te)
    p = np.random.default_rng(0).permutation(len(qtr))
    val, train = qtr[p[:1000]], qtr[p[1000:]]
    np.savez(cache, train=train, val=val, test=qte)
    return {"train": train, "val": val, "test": qte}, {"S": 12, "V": codes}


def make_problem(name):
    data, meta = {"synthetic": synthetic, "brown_pos": brown, "uci_har_vq": har}[name]()
    S, V = meta["S"], meta["V"]
    prior = meta.get("unigram")
    perm = np.random.default_rng(123).permutation(len(data["train"]))
    train = data["train"][perm]

    def fit(alpha0, T, B, seed, schedule=None, revival=True, n=None, k=None):
        x = train[: n or len(train)]
        plug = HMMPlugin(k or S, V, seed=seed, emis_prior=prior)
        steps = max(int(round(T * max(len(x) // B, 1))), 1)
        sched = schedule or CosineSchedule(alpha0, steps, eps=0.1)
        res = run_ema_engine(plug, x, epochs=T, batch_size=B, schedule=sched, revival_every=5 if revival else 0,
                             finalize_mode="none", seed=seed)
        plug.revivals = res.revivals
        return plug

    def loss(m, split):
        return nll_per_token(data[split], m.log_pi, m.A, m.B)

    def extras(m):
        out = {"revivals": getattr(m, "revivals", None)}
        if "tags_test" in meta:
            out["many_to_one"] = many_to_one(data["test"], meta["tags_test"], m.log_pi, m.A, m.B)
        if "true_B" in meta:
            from scipy.optimize import linear_sum_assignment
            c = np.abs(m.B[:, None, :] - meta["true_B"][None]).sum(-1)
            r, cidx = linear_sum_assignment(c)
            out["emission_l1_err"] = float(c[r, cidx].mean())
        return out

    class Frozen:  # wrap params for loss/extras
        def __init__(self, lp, A, B):
            self.log_pi, self.A, self.B = lp, A, B

    def reference(seed):
        lp, A, B = random_init(S, V, seed, prior)
        pts, wall = [], 0.0
        for it in range(1, 26):
            t0 = time.perf_counter()
            lp, A, B = em_step(train, lp, A, B)
            wall += time.perf_counter() - t0
            f = Frozen(lp, A, B)
            pts.append({"epoch": it, "test_loss": loss(f, "test"), "val": loss(f, "val"), "wall_s": wall})
        return pts, Frozen(lp, A, B)

    def sanity(seed):
        from hmmlearn.hmm import CategoricalHMM
        lp, A, B = random_init(S, V, seed, prior)
        m = CategoricalHMM(S, n_iter=25, init_params="", params="ste", tol=-1e9, implementation="scaling", n_features=V)
        m.startprob_, m.transmat_, m.emissionprob_ = np.exp(lp), A, B
        seqs = [r[r >= 0] for r in train]
        t0 = time.perf_counter()
        m.fit(np.concatenate(seqs)[:, None], [len(s) for s in seqs])
        w = time.perf_counter() - t0
        return nll_per_token(data["test"], np.log(m.startprob_), m.transmat_, m.emissionprob_), w

    p = Problem(track=TRACK, dataset=name, k=S, d=V, n=len(train), metric="test_nll_per_token", fit=fit, loss=loss,
                reference=reference, extras=extras, config={"V": V})
    return p, sanity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="synthetic,brown_pos,uci_har_vq")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--no-sanity", action="store_true")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    for name in args.datasets.split(","):
        p, sanity = make_problem(name)
        print(f"\n##### {name} n={p.n} S={p.k} V={p.d}", flush=True)
        law_cfgs = [{"n": n, "B": B, "T": T} for n in (p.n // 4, p.n // 2, p.n) for B in (64, 512) for T in (1.0, 2.0, 5.0)]
        run_all(p, RUNS, SWEEP, seeds, ref_method="baum_welch_full_batch", ema_method="hmm_ema",
                stepwise_method="stepwise_em", head_B=HEAD_B, sanity=None if args.no_sanity else sanity,
                sanity_method="hmmlearn_bw", stage=args.stage, law_sweep_configs=law_cfgs, law_ks=(6, 12))


if __name__ == "__main__":
    main()
