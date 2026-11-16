"""Exploratory (2026-11-17, not pre-registered): does the fraction of missing information explain what q_eff leaves?

EM's convergence rate is governed by the fraction of missing information, the share of the complete-data information
lost because the latent variables are unobserved (Dempster, Laird & Rubin 1977). A cheap per-configuration proxy is the
normalised posterior entropy of the latent assignment under a fitted model,

    h = E_x[ H(z | x) ] / log k        (0 = latents determined by the data, 1 = posterior uniform)

computed on held-out data: GMM component responsibilities, HMM per-token state posteriors, LDA per-token topic
posteriors (phi). Dictionary learning and ALS have no latent posterior (deterministic codes / factors): h = 0.

For every sweep group of GMM, HMM and LDA (dataset, n, k, covariance) this fits the recipe once (seed 0, T = 2, the
group's median swept alpha0*, the group's batch size) and stores h in results/universal-scaling-law/missing_info.json.
The law refit with h as a feature is `refit_coords.py --missing-info`.

  .venv/bin/python -u tracks/universal-scaling-law/missing_info.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.harness import results_path  # noqa: E402
import refit_coords as rc  # noqa: E402

OUT = results_path("universal-scaling-law", "missing_info.json")
MAX_VAL = 20000


def runner(track):
    tdir = ROOT / "tracks" / track
    sys.path.insert(0, str(tdir))
    spec = importlib.util.spec_from_file_location(f"{track.replace('-', '_')}_rv", tdir / "run_v2.py")
    m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [str(tdir / "run_v2.py")]
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


def ent_rows(p):
    p = np.clip(p, 1e-300, 1.0)
    return -(p * np.log(p)).sum(-1)


group_key = rc.group_key


# ---------------------------------------------------------------- per family

def h_gmm(rv, cache, ds, n, k, cov, alpha, B):
    if ds not in cache:
        cache[ds] = rv.load(ds)[0]
    data = cache[ds]
    from gmm import compute_responsibilities
    prm, _ = rv.gmm_ema(data["train"][:n], k, cov, alpha, 2.0, B, 0)
    xv = data["val"][:MAX_VAL]
    g, _ = compute_responsibilities(xv, prm.weights, prm.means, prm.covars, prm.cov_type, eps_cov=prm.eps_cov)
    return float(ent_rows(np.asarray(g)).mean() / math.log(k))


def h_hmm(rv, cache, ds, n, k, alpha, B):
    if ds not in cache:
        cache[ds] = (rv.make_problem(ds), {"synthetic": rv.synthetic, "brown_pos": rv.brown, "uci_har_vq": rv.har}[ds]()[0])
    (p, _), data = cache[ds]
    m = p.fit(alpha, 2.0, B, 0, n=n, k=k)
    pi, A, Bm = np.exp(m.log_pi), m.A, m.B
    tot, toks = 0.0, 0
    seqs = np.asarray(data["val"])
    for s in seqs[: max(1, MAX_VAL // max(seqs.shape[1], 1))]:
        o = s[s >= 0]
        if len(o) == 0:
            continue
        T = len(o)
        al = np.zeros((T, len(pi))); be = np.ones((T, len(pi)))
        a = pi * Bm[:, o[0]]; a /= a.sum(); al[0] = a
        for t in range(1, T):
            a = (al[t - 1] @ A) * Bm[:, o[t]]; a /= a.sum(); al[t] = a
        for t in range(T - 2, -1, -1):
            b = A @ (Bm[:, o[t + 1]] * be[t + 1]); be[t] = b / b.sum()
        gam = al * be
        gam /= gam.sum(1, keepdims=True)
        tot += ent_rows(gam).sum(); toks += T
    return float(tot / toks / math.log(k))


def h_lda(rv, cache, ds, n, k, alpha, B):
    from lda_v2 import exp_elog_beta, e_step
    from scipy.special import digamma as _digamma
    if ds not in cache:
        c = rv.load_corpus(ds)
        perm = np.random.default_rng(0).permutation(c.n)
        cache[ds] = (rv.make_problem(ds), c.subset(perm[2000:4000]))
    p, va = cache[ds]
    m = p.fit(alpha, 2.0, B, 0, n=n, k=k)
    eeb = exp_elog_beta(m.lam)
    docs = np.arange(va.n, dtype=np.int64)
    gamma, _ = e_step(docs, va.indptr, va.ids, va.cts, eeb, 1.0 / k, 50, 1e-3, False)
    gamma = np.asarray(gamma)
    eth = np.exp(_digamma(gamma) - _digamma(gamma.sum(1, keepdims=True)))
    tot, toks = 0.0, 0.0
    for d in range(va.n):
        lo, hi = va.indptr[d], va.indptr[d + 1]
        phi = eth[d][None, :] * eeb[:, va.ids[lo:hi]].T
        phi /= phi.sum(1, keepdims=True)
        tot += float((ent_rows(phi) * va.cts[lo:hi]).sum()); toks += float(va.cts[lo:hi].sum())
    return float(tot / toks / math.log(k))


def main():
    cells = rc.load_cells(rc.dataset_stats())
    out = json.load(open(OUT)) if Path(OUT).exists() else {}
    groups = {}
    for c in cells:
        if c["alg"] in ("gmm-ema", "hmm-ema", "lda-scaling"):
            groups.setdefault(group_key(c), []).append(c)
    print(f"{len(groups)} groups ({len(out)} done)", flush=True)
    rvs, caches = {}, {}
    for key, cs in sorted(groups.items()):
        if key in out:
            continue
        alg, ds = key.split("|")[:2]
        n, k = int(cs[0]["n"]), int(cs[0]["k"])
        B = int(np.median([c["B"] for c in cs]))
        alpha = float(np.median([c["a_star"] for c in cs]))
        rv = rvs.setdefault(alg, runner(alg))
        cache = caches.setdefault(alg, {})
        if alg == "gmm-ema":
            h = h_gmm(rv, cache, ds, n, k, key.split("|")[-1], alpha, B)
        elif alg == "hmm-ema":
            h = h_hmm(rv, cache, ds, n, k, alpha, B)
        else:
            h = h_lda(rv, cache, ds, n, k, alpha, B)
        out[key] = {"h": h, "alpha0": alpha, "B": B, "cells": len(cs)}
        print(f"{key:55s} h={h:.3f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
