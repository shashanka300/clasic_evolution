"""T01 gmm-ema, v2 protocol. Writes JSONL only; verdicts come from scripts/evaluate.py.

Comparisons (per dataset x covariance type, same k-means++ init for every method):
  reference  em_full_batch     exact EM (same E/M-step code as the EMA), 25-iteration trajectory
  baseline   sklearn_em        sklearn GaussianMixture(init=our means, max_iter=25) sanity check
  baseline   online_em_tuned   Cappe & Moulines step (tau0+t)^-kappa, (tau0, kappa) chosen on val per T
  ours       gmm_ema_swept     cosine EMA + revival, alpha0 chosen on val per T from ALPHA_GRID
  ours       gmm_ema_kmlaw     cosine EMA, alpha0 and B from the e-means *k-means* law, untuned

    python tracks/gmm-ema/run_v2.py --stage all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.bench import load_dinov2_cifar100, load_sift1m, load_text_embeddings  # noqa: E402
from core.harness import done_hashes, config_hash, make_run, read_runs, results_path, write_run  # noqa: E402
from core.scaling import emeans_alpha0, emeans_batch_size  # noqa: E402
from core.schedule import InverseTimeSchedule  # noqa: E402
from core.stats import EMAStats  # noqa: E402
from core.sweep import emit_swept_curves, run_sweep, sweep_group  # noqa: E402
from gmm import compute_batch_stats, compute_responsibilities, evaluate_gmm_nll, init_gmm_parameters, m_step_from_stats  # noqa: E402
from plugin import fit_gmm_ema  # noqa: E402

TRACK = "gmm-ema"
RUNS = results_path(TRACK)
SWEEP = results_path(TRACK, "sweep.jsonl")
EPS_COV_REL = 1e-6
BUDGETS = (1.0, 2.0, 3.0, 5.0)


# ------------------------------------------------------------------ data

def _split(x_pool, x_test, n_train, n_val, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(x_pool))
    return {"train": x_pool[p[:n_train]], "val": x_pool[p[n_train:n_train + n_val]], "test": x_test}


def _pca(data, dim=64):
    pca = PCA(dim, random_state=0).fit(data["train"][:50_000])
    out = {k: pca.transform(v).astype(np.float64) for k, v in data.items()}
    scale = float(np.sqrt(out["train"].var(axis=0).mean()))
    return {k: v / scale for k, v in out.items()}  # unit average variance so eps_cov means the same everywhere


def synthetic(n_train=200_000, d=32, k=32, seed=1234):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 1.0, (k, d))  # overlapping components: EM needs many iterations
    sds = rng.uniform(0.5, 1.5, (k, d))
    w = rng.dirichlet(np.full(k, 5.0))
    def sample(m):
        z = rng.choice(k, m, p=w)
        return means[z] + sds[z] * rng.standard_normal((m, d))
    return {"train": sample(n_train), "val": sample(10_000), "test": sample(10_000)}, {"true_means": means}


def load(name):
    if name == "synthetic":
        return synthetic()
    if name == "sift1m_pca64":
        s = load_sift1m(val_size=0)
        return _pca(_split(np.asarray(s.train[:300_000], np.float32), np.asarray(s.test, np.float32), 200_000, 10_000)), {}
    if name == "dinov2_pca64":
        s = load_dinov2_cifar100(val_size=0)
        return _pca(_split(np.asarray(s.train, np.float32), np.asarray(s.test, np.float32), 45_000, 5_000)), {}
    if name == "agnews_pca64":
        s = load_text_embeddings(dataset_name="ag_news", max_train=100_000, max_test=7_600, val_size=0)
        return _pca(_split(np.asarray(s.train, np.float32), np.asarray(s.test, np.float32), 90_000, 10_000)), {}
    raise ValueError(name)


# ------------------------------------------------------------------ methods

def eps_cov(x):
    return EPS_COV_REL * float(np.trace(np.cov(x[:20_000].T))) / x.shape[1]


def init_params(x, k, cov, seed):
    return init_gmm_parameters(x, k, cov_type=cov, eps_cov=eps_cov(x), rng=np.random.default_rng(seed))


def em_trajectory(data, k, cov, seed, iters=25, chunk=8192):
    x = data["train"]
    ec = eps_cov(x)
    p = init_params(x, k, cov, seed)
    pts, wall = [], 0.0
    for it in range(1, iters + 1):
        t0 = time.perf_counter()
        tot = None
        for s in range(0, len(x), chunk):
            xb = x[s:s + chunk]
            g, _ = compute_responsibilities(xb, p.weights, p.means, p.covars, cov_type=cov, eps_cov=ec)
            bs = compute_batch_stats(xb, g, cov_type=cov)
            if tot is None:
                tot = bs
            else:
                tot.counts += bs.counts
                tot.sums += bs.sums
                for key in tot.extras:
                    tot.extras[key] += bs.extras[key]
        p = m_step_from_stats(EMAStats(tot.counts, tot.sums, tot.extras), cov_type=cov, eps_cov=ec)
        wall += time.perf_counter() - t0
        pts.append({"epoch": it, "test_loss": evaluate_gmm_nll(data["test"], p), "val": evaluate_gmm_nll(data["val"], p),
                    "wall_s": wall})
    return pts, p


def gmm_ema(x, k, cov, alpha0, T, B, seed, schedule=None, revival=True):
    plug, res = fit_gmm_ema(x, k, epochs=T, batch_size=B, alpha0=alpha0, cov_type=cov, eps_cov=eps_cov(x),
                            revival_every=5 if revival else 0, tau=0.1 if revival else 0.0,
                            schedule=schedule, seed=seed)
    return plug.params, res


def param_error(p, meta):
    if "true_means" not in meta:
        return None
    tm = meta["true_means"]
    cost = ((p.means[:, None, :] - tm[None]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(cost[r, c]).mean())


def collapsed(p):
    return int((p.weights < 0.1 / len(p.weights)).sum())


# ------------------------------------------------------------------ stages

def stage_reference(name, data, meta, k, cov, seeds):
    group = f"{name}|k={k}|{cov}"
    have = {(r["method"], r["seed"]) for r in read_runs(RUNS) if r["extra"]["group"] == group}
    n, d = data["train"].shape
    for s in seeds:
        if ("em_full_batch", s) not in have:
            pts, p = em_trajectory(data, k, cov, s)
            write_run(RUNS, make_run(track=TRACK, method="em_full_batch", dataset=name, n=n, d=d, k_or_units=k,
                      epochs=25, batch=n, alpha0=None, seed=s, train_loss=None, test_loss=pts[-1]["test_loss"],
                      wall_s=pts[-1]["wall_s"], role="baseline", alpha0_source="na", group=group, metric="test_nll",
                      reference=True, curve=pts, curve_kind="trajectory", config={"cov": cov},
                      param_error=param_error(p, meta), collapsed=collapsed(p)))
            print(f"[ref] {group} s={s} EM@25 test={pts[-1]['test_loss']:.4f} ({pts[-1]['wall_s']:.0f}s)", flush=True)
        if ("sklearn_em", s) not in have:
            p0 = init_params(data["train"], k, cov, s)
            prec = 1.0 / p0.covars if cov == "diag" else np.linalg.inv(p0.covars)
            t0 = time.perf_counter()
            gm = GaussianMixture(k, covariance_type=cov, max_iter=25, tol=0, reg_covar=eps_cov(data["train"]),
                                 weights_init=p0.weights, means_init=p0.means, precisions_init=prec, random_state=s)
            gm.fit(data["train"])
            w = time.perf_counter() - t0
            write_run(RUNS, make_run(track=TRACK, method="sklearn_em", dataset=name, n=n, d=d, k_or_units=k, epochs=25,
                      batch=n, alpha0=None, seed=s, train_loss=None, test_loss=float(-gm.score(data["test"])), wall_s=w,
                      role="baseline", alpha0_source="na", group=group, metric="test_nll", config={"cov": cov},
                      note="sanity check of em_full_batch; final point only"))


def stage_sweep(name, data, meta, k, cov, seeds, configs):
    n_all, d = data["train"].shape

    def fit_eval(x, kk, a0, T, B, s):
        p, res = gmm_ema(x, kk, cov, a0, T, B, s)
        return {"val": evaluate_gmm_nll(data["val"], p), "test": evaluate_gmm_nll(data["test"], p),
                "revivals": res.revivals, "collapsed": collapsed(p), "param_error": param_error(p, meta)}

    run_sweep(track=TRACK, method=f"gmm_ema_{cov}", dataset=name, data=data, k=k, d=d, fit_eval=fit_eval,
              configs=configs, seeds=seeds, path=SWEEP, metric="test_nll", tag=cov, extra_config={"cov": cov})


def stage_online_em(name, data, k, cov, seeds, B):
    """Online EM tuned on val per T over a (tau0, kappa) grid: the closest prior-art baseline."""
    group = f"{name}|k={k}|{cov}"
    x = data["train"]
    n, d = x.shape
    done = done_hashes(SWEEP)
    grid = [(t0, kp) for t0 in (1.0, 10.0, 100.0, 1000.0) for kp in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    for T in BUDGETS:
        for t0, kp in grid:
            cfg = {"method": "online_em", "dataset": name, "cov": cov, "T": T, "B": B, "tau0": t0, "kappa": kp, "n": n}
            h = config_hash(cfg)
            for s in seeds:
                if (h, s) in done:
                    continue
                tic = time.perf_counter()
                p, _ = gmm_ema(x, k, cov, 1.0, T, B, s, schedule=InverseTimeSchedule(1.0, tau0=t0, kappa=kp), revival=False)
                w = time.perf_counter() - tic
                rec = make_run(track=TRACK, method="online_em", dataset=name, n=n, d=d, k_or_units=k, epochs=T, batch=B,
                               alpha0=None, seed=s, train_loss=None, test_loss=evaluate_gmm_nll(data["test"], p), wall_s=w,
                               role="baseline", alpha0_source="na", group=f"oem|{group}", metric="test_nll",
                               config={k2: v for k2, v in cfg.items() if k2 not in ("method", "dataset", "n")},
                               val_loss=evaluate_gmm_nll(data["val"], p), tau0=t0, kappa=kp, sweep_oem=True)
                rec["extra"]["config_hash"] = h
                write_run(SWEEP, rec)
    # pick (tau0, kappa) per T on val; emit one curve per seed
    rows = [r for r in read_runs(SWEEP) if r["extra"].get("sweep_oem") and r["extra"]["group"] == f"oem|{group}"]
    if any(r["method"] == "online_em_tuned" and r["extra"]["group"] == group for r in read_runs(RUNS)):
        return
    for s in seeds:
        curve = []
        for T in BUDGETS:
            cands = {}
            for r in rows:
                if r["epochs"] == T:
                    cands.setdefault((r["extra"]["tau0"], r["extra"]["kappa"]), []).append(r)
            best = min(cands, key=lambda key: np.mean([r["extra"]["val_loss"] for r in cands[key]]))
            r = next(r for r in cands[best] if r["seed"] == s)
            curve.append({"epoch": T, "test_loss": r["test_loss"], "wall_s": r["wall_s"], "tau0": best[0], "kappa": best[1]})
        write_run(RUNS, make_run(track=TRACK, method="online_em_tuned", dataset=name, n=n, d=d, k_or_units=k,
                  epochs=BUDGETS[-1], batch=B, alpha0=None, seed=s, train_loss=None, test_loss=curve[-1]["test_loss"],
                  wall_s=curve[-1]["wall_s"], role="baseline", alpha0_source="swept", group=group, metric="test_nll",
                  curve=curve, curve_kind="budget", selected_on="val", config={"cov": cov}))


def stage_kmlaw(name, data, k, cov, seeds):
    """Untuned: alpha0 and B straight from the e-means k-means law. Tests 'does the law transfer unchanged?'"""
    group = f"{name}|k={k}|{cov}"
    if any(r["method"] == "gmm_ema_kmlaw" and r["extra"]["group"] == group for r in read_runs(RUNS)):
        return
    x = data["train"]
    n, d = x.shape
    for s in seeds:
        curve = []
        for T in BUDGETS:
            B = emeans_batch_size(n, k, T)
            a0 = emeans_alpha0(n, k, T, B)
            tic = time.perf_counter()
            p, _ = gmm_ema(x, k, cov, a0, T, B, s)
            curve.append({"epoch": T, "test_loss": evaluate_gmm_nll(data["test"], p), "wall_s": time.perf_counter() - tic,
                          "alpha0": a0, "B": B})
        write_run(RUNS, make_run(track=TRACK, method="gmm_ema_kmlaw", dataset=name, n=n, d=d, k_or_units=k,
                  epochs=BUDGETS[-1], batch=curve[-1]["B"], alpha0=curve[-1]["alpha0"], seed=s, train_loss=None,
                  test_loss=curve[-1]["test_loss"], wall_s=curve[-1]["wall_s"], role="ours", alpha0_source="law",
                  group=group, metric="test_nll", curve=curve, curve_kind="budget", config={"cov": cov}))


HEAD_K = 64
HEAD_B = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="synthetic,sift1m_pca64,dinov2_pca64,agnews_pca64")
    ap.add_argument("--covs", default="diag,full")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", default="all", choices=("all", "reference", "sweep", "law-sweep", "online", "kmlaw", "emit"))
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    for name in args.datasets.split(","):
        data, meta = load(name)
        n = len(data["train"])
        k = 32 if name == "synthetic" else HEAD_K
        for cov in args.covs.split(","):
            print(f"\n##### {name} cov={cov} n={n} k={k}", flush=True)
            if args.stage in ("all", "reference"):
                stage_reference(name, data, meta, k, cov, seeds)
            if args.stage in ("all", "sweep"):
                stage_sweep(name, data, meta, k, cov, seeds, [{"n": n, "B": HEAD_B, "T": T} for T in BUDGETS])
            if args.stage in ("all", "online"):
                stage_online_em(name, data, k, cov, seeds, HEAD_B)
            if args.stage in ("all", "kmlaw"):
                stage_kmlaw(name, data, k, cov, seeds)
            if args.stage == "law-sweep" and cov == "diag":
                # scale grid for T12: vary n, K, B at fixed T in {1, 2, 5}
                cfgs = [{"n": nn, "B": B, "T": T} for nn in (n // 4, n // 2, n) for B in (256, 2048) for T in (1.0, 2.0, 5.0)]
                for kk in (16, 64):
                    stage_sweep(name, data, meta, kk, cov, seeds, cfgs)
            if args.stage in ("all", "emit"):
                emit_swept_curves(sweep_path=SWEEP, runs_path=RUNS, track=TRACK, method=f"gmm_ema_{cov}",
                                  out_method="gmm_ema_swept",
                                  group_of=lambda g, name=name, k=k, cov=cov:
                                      f"{name}|k={k}|{cov}" if g == sweep_group(name, n, k, HEAD_B, cov) else None,
                                  metric="test_nll")


if __name__ == "__main__":
    main()
