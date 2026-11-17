"""Does count-based revival hurt rare, isolated groups? (added 2026-11-17; ablation, not pre-registered)

Weiss 2026 (arXiv:2611.18238) shows that e-means' count-based revival (re-seed any unit with EMA mass below
0.1 x mean) deletes small clusters that cover rare, isolated groups, and proposes a distance-aware rule. Our recipe
uses the same count rule for every family. This script compares, for the GMM recipe (cosine EMA, B = 1024):

  off    no revival
  count  the recipe's rule (PAPER_LOG §1; tracks/gmm-ema/gmm.py revive_gmm_components)
  dist   Weiss's rule adapted to GMMs: a count-flagged component is re-seeded only if the squared distance from its
         mean to the nearest other mean is <= TAU_NEAR * MSE, where MSE = sum_j w_j tr(Sigma_j) (expected squared
         distance of a point to its own component mean), or if it received < 0.5 points of batch mass for
         WINDOW_EPOCHS epochs ("abandoned"). TAU_NEAR = 4, WINDOW_EPOCHS = 2 as in Weiss 2026.

Datasets (standardised with train statistics): synthetic far-groups mixture (30 large + 4 small far clusters,
0.2% mass each), Statlog Shuttle (OpenML 40685), KDD Cup 99 10% (numeric columns), Covertype, and SIFT1M-PCA64 as a
control without rare groups. alpha0 is selected on validation NLL per (dataset, cov, method, T) from ALPHAS, seeds
0-2. Scores: test NLL, its relative gap to full-batch EM@25, and tail NLL = mean NLL on test points of rare groups
(synthetic: the small clusters; labelled data: classes with < 2% of training points).

  .venv/bin/python -u tracks/gmm-ema/revival_rare.py --dataset shuttle
  .venv/bin/python -u tracks/gmm-ema/revival_rare.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.plugin import run_ema_engine  # noqa: E402
from core.schedule import CosineSchedule  # noqa: E402
from gmm import evaluate_gmm_nll, revive_gmm_components  # noqa: E402
from plugin import GMMEmaPlugin, fit_full_batch_em  # noqa: E402

OUT = ROOT / "results" / "gmm-ema" / "revival_rare.jsonl"
DATA = ROOT / "data" / "raw"
ALPHAS = (0.072, 0.139, 0.268, 0.518, 1.0)
BUDGETS = (2.0, 5.0)
SEEDS = (0, 1, 2)
B = 1024
TAU_NEAR = 4.0
WINDOW_EPOCHS = 2.0
EPS_COV_REL = 1e-6
# KDD99 and Covertype have many binary / near-constant columns: with the recipe's 1e-6 relative floor, components
# collapse onto them and the NLL diverges (KDD99 full-batch test NLL 1.7e5). Floor 0.01 in standardised units.
EPS_FLOOR = {"kdd99": 1e-2, "covtype": 1e-2}
METHODS = ("off", "count", "dist")
CONFIGS = {"synthetic_far": (("diag", 34), ("full", 34)), "shuttle": (("diag", 64),), "kdd99": (("diag", 64),),
           "covtype": (("diag", 64),), "sift_control": (("diag", 64),)}


class DistRevivalPlugin(GMMEmaPlugin):
    """GMM recipe with Weiss-style distance-aware revival."""

    def __init__(self, *a, window_steps: int, **kw):
        super().__init__(*a, **kw)
        self.window = int(window_steps)
        self.step = 0
        self.last_mass = np.zeros(self.k, dtype=np.int64)
        self.protected_events = 0

    def assign(self, batch):
        bs = super().assign(batch)
        self.last_mass[np.asarray(bs.counts) >= 0.5] = self.step
        self.step += 1
        return bs

    def revive(self, stats, step, rng=None):
        p = self.params
        mu = p.means
        tr = p.covars.sum(1) if p.cov_type == "diag" else np.trace(p.covars, axis1=1, axis2=2)
        mse = float((p.weights * tr).sum())
        d2 = ((mu[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        far = d2.min(1) > TAU_NEAR * mse
        abandoned = (self.step - self.last_mass) > self.window
        protect = far & ~abandoned
        flagged = stats.counts < self.tau * float(stats.counts.mean())
        self.protected_events += int((protect & flagged).sum())
        return revive_gmm_components(stats, cov_type=self.cov_type, tau=self.tau, delta=self.delta,
                                     eps_cov=self.eps_cov, rng=rng, protect=protect)


def fit(x, k, cov, alpha0, T, seed, method, ec):
    n, d = x.shape
    spe = max(n // B, 1)
    sched = CosineSchedule(alpha0=alpha0, total_steps=max(int(round(T * spe)), 1), eps=0.1)
    kw = dict(k=k, d=d, cov_type=cov, tau=0.1, delta=1.0 / 128.0, eps_cov=ec, seed=seed)
    plug = DistRevivalPlugin(**kw, window_steps=int(WINDOW_EPOCHS * spe)) if method == "dist" else GMMEmaPlugin(**kw)
    res = run_ema_engine(plugin=plug, x=x, epochs=T, batch_size=B, schedule=sched,
                         revival_every=0 if method == "off" else 5, finalize_mode="none", seed=seed)
    return plug.params, res.revivals, getattr(plug, "protected_events", 0)


# ------------------------------------------------------------------ data

def _standardise(tr, *rest):
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd < 1e-6] = 1.0
    return [((a - mu) / sd).astype(np.float64) for a in (tr, *rest)]


def _split_labelled(x, y, seed=0, n_val=10_000, n_test=20_000, max_train=400_000):
    perm = np.random.default_rng(seed).permutation(len(x))
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:][:max_train]
    xtr, xva, xte = _standardise(x[tr], x[va], x[te])
    cls, cnt = np.unique(y[tr], return_counts=True)
    rare = set(cls[cnt < 0.02 * len(tr)].tolist())
    tail = np.array([v in rare for v in y[te]])
    return {"train": xtr, "val": xva, "test": xte, "tail": tail, "n_rare_classes": len(rare)}


def load(name):
    if name == "synthetic_far":
        rng = np.random.default_rng(123)
        d, n_big, n_small, w_small = 16, 30, 4, 0.002
        big = rng.normal(0, 4.0, (n_big, d))
        dirs = rng.normal(size=(n_small, d))
        small = 30.0 * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        means = np.vstack([big, small])
        w = np.r_[np.full(n_big, (1 - n_small * w_small) / n_big), np.full(n_small, w_small)]

        def draw(m):
            z = rng.choice(len(w), size=m, p=w)
            return means[z] + rng.normal(size=(m, d)), z >= n_big
        xtr, _ = draw(100_000)
        xva, _ = draw(10_000)
        xte, tail = draw(40_000)
        return {"train": xtr, "val": xva, "test": xte, "tail": tail, "n_rare_classes": n_small}
    from sklearn import datasets as skd
    home = str(DATA / "sklearn")
    if name == "shuttle":
        b = skd.fetch_openml(data_id=40685, as_frame=False, parser="liac-arff", data_home=home)
        return _split_labelled(np.asarray(b.data, np.float64), np.asarray(b.target).astype(str),
                               n_val=5_000, n_test=10_000)
    if name == "kdd99":
        b = skd.fetch_kddcup99(percent10=True, data_home=home)
        x = np.delete(np.asarray(b.data, dtype=object), [1, 2, 3], axis=1).astype(np.float64)
        return _split_labelled(x, np.asarray(b.target).astype(str))
    if name == "covtype":
        b = skd.fetch_covtype(data_home=home)
        return _split_labelled(np.asarray(b.data, np.float64), np.asarray(b.target).astype(str))
    if name == "sift_control":
        import importlib.util
        tdir = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location("gmm_rv", tdir / "run_v2.py")
        m = importlib.util.module_from_spec(spec)
        argv, sys.argv = sys.argv, [str(tdir / "run_v2.py")]
        spec.loader.exec_module(m)
        sys.argv = argv
        data, _ = m.load("sift1m_pca64")
        return {"train": np.asarray(data["train"], np.float64), "val": np.asarray(data["val"], np.float64),
                "test": np.asarray(data["test"], np.float64), "tail": None, "n_rare_classes": 0}
    raise ValueError(name)


# ------------------------------------------------------------------ run

def done_keys():
    if not OUT.exists():
        return set()
    return {(r["dataset"], r["cov"], r["k"], r["method"], r["T"], r["alpha0"], r["seed"]) for r in map(json.loads, open(OUT))}


def run(name):
    data = load(name)
    x = data["train"]
    ec = max(EPS_COV_REL * float(np.trace(np.cov(x[:20_000].T))) / x.shape[1], EPS_FLOOR.get(name, 0.0))
    tail = data["tail"]
    have = done_keys()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name}: n={len(x)} d={x.shape[1]} rare classes={data['n_rare_classes']} "
          f"tail test points={0 if tail is None else int(tail.sum())}", flush=True)
    for cov, k in CONFIGS[name]:
        key = (name, cov, k, "ref", 25.0, None, 0)
        if key not in have:
            t0 = time.perf_counter()
            p = fit_full_batch_em(x, k, cov_type=cov, max_iter=25, eps_cov=ec, seed=0)
            rec = {"dataset": name, "cov": cov, "k": k, "method": "ref", "T": 25.0, "alpha0": None, "seed": 0,
                   "val": evaluate_gmm_nll(data["val"], p), "test": evaluate_gmm_nll(data["test"], p),
                   "tail": None if tail is None else evaluate_gmm_nll(data["test"][tail], p),
                   "wall_s": time.perf_counter() - t0}
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  ref {cov} k={k}: test {rec['test']:.4f} tail {rec['tail']}", flush=True)
        for T in BUDGETS:
            for method in METHODS:
                for a in ALPHAS:
                    for s in SEEDS:
                        if (name, cov, k, method, T, a, s) in have:
                            continue
                        t0 = time.perf_counter()
                        p, nrev, nprot = fit(x, k, cov, a, T, s, method, ec)
                        rec = {"dataset": name, "cov": cov, "k": k, "method": method, "T": T, "alpha0": a, "seed": s,
                               "val": evaluate_gmm_nll(data["val"], p), "test": evaluate_gmm_nll(data["test"], p),
                               "tail": None if tail is None else evaluate_gmm_nll(data["test"][tail], p),
                               "revivals": nrev, "protected": nprot,
                               "min_weight_ratio": float(p.weights.min() * k), "wall_s": time.perf_counter() - t0}
                        with open(OUT, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                print(f"  {cov} k={k} T={T} {method} done", flush=True)


def summary():
    rows = [json.loads(l) for l in open(OUT)]
    out = {}
    groups = sorted({(r["dataset"], r["cov"], r["k"]) for r in rows})
    for g in groups:
        rs = [r for r in rows if (r["dataset"], r["cov"], r["k"]) == g]
        ref = [r for r in rs if r["method"] == "ref"]
        if not ref:
            continue
        ref = ref[0]
        print(f"\n{g[0]} {g[1]} k={g[2]}: ref@25 test {ref['test']:.4f} tail {ref['tail']}")
        for T in BUDGETS:
            line = []
            for m in METHODS:
                cell = [r for r in rs if r["method"] == m and r["T"] == T]
                if not cell:
                    continue
                by_a = {}
                for r in cell:
                    by_a.setdefault(r["alpha0"], []).append(r)
                a_best = min(by_a, key=lambda a: np.mean([r["val"] for r in by_a[a]]))
                sel = by_a[a_best]
                te = np.array([r["test"] for r in sel])
                tl = None if sel[0]["tail"] is None else np.array([r["tail"] for r in sel])
                gap = 100 * (te.mean() - ref["test"]) / abs(ref["test"])
                out[f"{g[0]}|{g[1]}|k={g[2]}|T={T}|{m}"] = {
                    "alpha0": a_best, "test": te.mean(), "test_sd": te.std(), "gap_pct": gap,
                    "tail": None if tl is None else tl.mean(), "tail_sd": None if tl is None else tl.std(),
                    "revivals": np.mean([r["revivals"] for r in sel]), "protected": np.mean([r["protected"] for r in sel]),
                    "n_seeds": len(sel), "test_by_seed": te.tolist(), "tail_by_seed": None if tl is None else tl.tolist()}
                line.append(f"{m}: a={a_best} test={te.mean():.4f}±{te.std():.4f} ({gap:+.2f}%)"
                            + ("" if tl is None else f" tail={tl.mean():.3f}±{tl.std():.3f}")
                            + f" rev={np.mean([r['revivals'] for r in sel]):.0f} prot={np.mean([r['protected'] for r in sel]):.0f}")
            print(f"  T={T}: " + " | ".join(line))
    json.dump(out, open(OUT.with_suffix(".summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=tuple(CONFIGS))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run(a.dataset)
