"""T04 dictlearn-scaling, v2 protocol. JSONL only; verdicts from scripts/evaluate.py.

Datasets (real):
  cifar_p8    8x8 grayscale CIFAR-100 patches, mean-removed, unit L2 norm; K=128 (2x overcomplete), lambda=0.1
  cifar_p12   12x12 patches, same preprocessing; K=192, lambda=0.1
  nmf_20ng    20 Newsgroups tf-idf (top-2000 vocab, rows L2-normalised); NMF, K=32, lambda=0
Train patches from CIFAR train images, val from held-out train images, test from CIFAR test images.

Loss = test objective 0.5||x - a D||^2 + lambda ||a||_1 per sample, codes by FISTA with D fixed.

Methods (same initial dictionary = random normalised train samples per seed):
  reference  full_batch_dl          code ALL data (FISTA), exact A,B, 5 BCD sweeps; 25 iterations
  baseline   sklearn_minibatch      MiniBatchDictionaryLearning / MiniBatchNMF (Mairal online; default
                                    forgetting), same dict init, max_iter = T epochs, batch B
  baseline   stepwise_dl_tuned      (tau0+t)^-kappa on EMA stats, tuned on val per T
  ours       dl_ema_swept / dl_ema_kmlaw
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

from core.harness import make_run, read_runs, results_path, write_run  # noqa: E402
from core.plugin import run_ema_engine  # noqa: E402
from core.schedule import CosineSchedule  # noqa: E402
from core.trackkit import BUDGETS, Problem, run_all  # noqa: E402
from dictlearn import block_coordinate_descent_update, fista_sparse_code  # noqa: E402
from plugin import DictionaryLearningPlugin  # noqa: E402

TRACK = "dictlearn-scaling"
RUNS = results_path(TRACK)
SWEEP = results_path(TRACK, "sweep.jsonl")
DATA = ROOT / "data"
HEAD_B = 256
SPECS = {
    "cifar_p8": dict(p=8, K=128, lam=0.1, nn=False, n_train=200_000),
    "cifar_p12": dict(p=12, K=192, lam=0.1, nn=False, n_train=100_000),
    "nmf_20ng": dict(K=32, lam=0.0, nn=True),
}


def _cifar_images(split):
    from datasets import Dataset
    base = DATA / "raw" / "hf_cache" / "uoft-cs___cifar100" / "cifar100" / "0.0.0"
    f = next(base.glob(f"*/cifar100-{split}.arrow"))
    ds = Dataset.from_file(str(f))
    imgs = np.stack([np.asarray(im.convert("L"), dtype=np.float32) / 255.0 for im in ds["img"]])
    return imgs


def _patches(imgs, p, n, rng):
    out = []
    while sum(len(o) for o in out) < n:
        idx = rng.integers(0, len(imgs), 20_000)
        r = rng.integers(0, imgs.shape[1] - p + 1, 20_000)
        c = rng.integers(0, imgs.shape[2] - p + 1, 20_000)
        P = np.stack([imgs[i, a:a + p, b:b + p].reshape(-1) for i, a, b in zip(idx, r, c)])
        P -= P.mean(1, keepdims=True)
        nrm = np.linalg.norm(P, axis=1)
        P = P[nrm > 0.05] / nrm[nrm > 0.05, None]   # drop flat patches, unit norm
        out.append(P)
    return np.concatenate(out)[:n].astype(np.float32)


def load(name):
    cache = DATA / "processed" / f"dl_{name}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in ("train", "val", "test")}
    rng = np.random.default_rng(0)
    if name.startswith("cifar"):
        s = SPECS[name]
        tr_imgs, te_imgs = _cifar_images("train"), _cifar_images("test")
        perm = rng.permutation(len(tr_imgs))
        data = {"train": _patches(tr_imgs[perm[5000:]], s["p"], s["n_train"], rng),
                "val": _patches(tr_imgs[perm[:5000]], s["p"], 10_000, rng),
                "test": _patches(te_imgs, s["p"], 10_000, rng)}
    else:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), data_home=str(DATA / "raw" / "sk20ng"))
        X = TfidfVectorizer(max_features=2000, stop_words="english", min_df=5).fit_transform(d.data).toarray().astype(np.float32)
        X = X[X.sum(1) > 0]
        p = rng.permutation(len(X))
        data = {"test": X[p[:2000]], "val": X[p[2000:4000]], "train": X[p[4000:]]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def objective(x, D, lam, nn, chunk=4096):
    tot = 0.0
    for s in range(0, len(x), chunk):
        xb = np.asarray(x[s:s + chunk], np.float64)
        a = fista_sparse_code(xb, D, lam, max_iter=100, tol=1e-5, non_negative=nn).astype(np.float64)
        tot += 0.5 * float(((xb - a @ D) ** 2).sum()) + lam * float(np.abs(a).sum())
    return tot / len(x)


def make_problem(name):
    data = load(name)
    s = SPECS[name]
    K, lam, nn = s["K"], s["lam"], s["nn"]
    x = data["train"]
    n, d = x.shape

    class M:
        def __init__(self, D):
            self.D = np.asarray(D, np.float64)

    def plugin(seed, k=None):
        return DictionaryLearningPlugin(n_components=k or K, alpha_reg=lam, non_negative=nn, seed=seed)

    def fit(alpha0, T, B, seed, schedule=None, revival=True, n=None, k=None):
        xx = x[: n or len(x)]
        pl = plugin(seed, k)
        steps = max(int(round(T * max(len(xx) // B, 1))), 1)
        res = run_ema_engine(pl, xx, epochs=T, batch_size=B, schedule=schedule or CosineSchedule(alpha0, steps, eps=0.1),
                             revival_every=5 if revival else 0, finalize_mode="none", seed=seed)
        m = M(pl.D)
        m.revivals = res.revivals
        return m

    def loss(m, split):
        return objective(data[split], m.D, lam, nn)

    def extras(m):
        a = fista_sparse_code(np.asarray(data["test"][:2000], np.float64), m.D, lam, max_iter=100, non_negative=nn)
        used = (np.abs(a) > 1e-8).mean(0)
        return {"dead_atoms": int((used < 1e-3).sum()), "code_density": float((np.abs(a) > 1e-8).mean()),
                "revivals": getattr(m, "revivals", None)}

    def init_D(seed):
        pl = plugin(seed)
        pl.init(x)
        return np.asarray(pl.D, np.float64)

    def reference(seed):
        D = init_D(seed)
        pts, wall = [], 0.0
        for it in range(1, 26):
            t0 = time.perf_counter()
            A = np.zeros((K, K))
            Bm = np.zeros((K, d))
            for st in range(0, n, 8192):
                xb = np.asarray(x[st:st + 8192], np.float64)
                a = fista_sparse_code(xb, D, lam, max_iter=50, tol=1e-4, non_negative=nn).astype(np.float64)
                A += a.T @ a
                Bm += a.T @ xb
            D = block_coordinate_descent_update(D, A / n, Bm / n, non_negative=nn, n_iter=5).astype(np.float64)
            wall += time.perf_counter() - t0
            mm = M(D)
            pts.append({"epoch": it, "test_loss": loss(mm, "test"), "val": loss(mm, "val"), "wall_s": wall})
            print(f"   full_batch_dl it={it} test={pts[-1]['test_loss']:.5f} ({wall:.0f}s)", flush=True)
        return pts, M(D)

    p = Problem(track=TRACK, dataset=name, k=K, d=d, n=n, metric="test_objective", fit=fit, loss=loss,
                reference=reference, extras=extras, config={"lambda": lam, "non_negative": nn})
    p.init_D, p.data = init_D, data
    return p


def stage_sklearn(p, seeds):
    if any(r["method"] == "sklearn_minibatch" and r["extra"]["group"] == p.group for r in read_runs(RUNS)):
        return
    from sklearn.decomposition import MiniBatchDictionaryLearning, MiniBatchNMF
    x = p.data["train"]
    nn = p.config["non_negative"]

    class M:
        def __init__(self, D):
            self.D = np.asarray(D, np.float64)
    for s in seeds:
        D0 = p.init_D(s)
        curve = []
        for T in BUDGETS:
            t0 = time.perf_counter()
            if nn:
                W0 = np.full((len(x), p.k), 1e-3, np.float32)
                est = MiniBatchNMF(p.k, init="custom", batch_size=HEAD_B, max_iter=int(T), tol=0, max_no_improvement=None,
                                   random_state=s).fit(x, W=W0, H=np.abs(D0).astype(np.float32))
            else:
                est = MiniBatchDictionaryLearning(p.k, alpha=p.config["lambda"], batch_size=HEAD_B, max_iter=int(T), tol=0,
                                                  max_no_improvement=None, dict_init=D0, random_state=s, fit_algorithm="cd",
                                                  shuffle=True).fit(x)
            w = time.perf_counter() - t0
            D = est.components_
            if nn:  # sklearn NMF rows are not unit norm: rescale atoms (codes absorb scale) to match our constraint
                D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
            curve.append({"epoch": T, "test_loss": p.loss(M(D), "test"), "wall_s": w})
        write_run(RUNS, make_run(track=TRACK, method="sklearn_minibatch", dataset=p.dataset, n=p.n, d=p.d, k_or_units=p.k,
                  epochs=BUDGETS[-1], batch=HEAD_B, alpha0=None, seed=s, train_loss=None, test_loss=curve[-1]["test_loss"],
                  wall_s=curve[-1]["wall_s"], role="baseline", alpha0_source="na", group=p.group, metric=p.metric,
                  curve=curve, curve_kind="budget", config=p.config,
                  note="sklearn Mairal online DL / MiniBatchNMF, default forgetting; NMF atoms renormalised"))
        print(f"[sklearn] {p.dataset} s={s} " + " ".join(f"T{c['epoch']:g}={c['test_loss']:.5f}" for c in curve), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cifar_p8,nmf_20ng,cifar_p12")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    for name in args.datasets.split(","):
        p = make_problem(name)
        print(f"\n##### {name} n={p.n} d={p.d} K={p.k}", flush=True)
        law = [{"n": n, "B": B, "T": T} for n in (p.n // 4, p.n) for B in (64, 512) for T in (1.0, 2.0, 5.0)]
        run_all(p, RUNS, SWEEP, seeds, ref_method="full_batch_dl", ema_method="dl_ema", stepwise_method="stepwise_dl",
                head_B=HEAD_B, stage=args.stage, law_sweep_configs=law, law_ks=(p.k // 2, p.k))
        if args.stage in ("all", "sklearn"):
            stage_sklearn(p, seeds)


if __name__ == "__main__":
    main()
