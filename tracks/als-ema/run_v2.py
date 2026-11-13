"""T06 als-ema, v2 protocol. JSONL only; verdicts from scripts/evaluate.py.

Data: MovieLens-1M and MovieLens-10M explicit ratings, random 80/10/10 rating split, centred by
train mean. f = 32 factors. lambda (ALS-WR) chosen ONCE on val from full-batch ALS@25 over
LAMBDA_GRID (seed 0) and then shared by every method.

Budget accounting: one ALS epoch touches every rating twice (user solve + item solve); one EMA
epoch also touches every rating twice (user solve + item statistics). The EMA's finalisation
(re-solve every user against final items) touches each rating once = 0.5 epoch and is charged
inside the budget: an EMA run with budget T trains for T - 0.5 epochs, then finalises.

Methods: als_wr_full_batch (reference, 25 it), stepwise_als_tuned ((tau0+t)^-kappa on the EMA
stats, tuned on val per T), als_ema_swept, als_ema_kmlaw (n = #users in the law).
NOTE: SGD-MF and the `implicit` library baselines are not run (implicit targets implicit feedback,
not explicit RMSE); documented gap.
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

from core.harness import results_path  # noqa: E402
from core.plugin import run_ema_engine  # noqa: E402
from core.schedule import CosineSchedule  # noqa: E402
from core.trackkit import Problem, run_all  # noqa: E402
from als_v2 import ALSPlugin, CSR, als_full_epoch, init_factors, rmse, solve_rows  # noqa: E402

TRACK = "als-ema"
RUNS = results_path(TRACK)
SWEEP = results_path(TRACK, "sweep.jsonl")
DATA = ROOT / "data"
F = 32
LAMBDA_GRID = (0.01, 0.03, 0.05, 0.1, 0.2)
HEAD_B = {"ml1m": 256, "ml10m": 2048}


def load(name):
    cache = DATA / "processed" / f"{name}_split.npz"
    if not cache.exists():
        path = DATA / "raw" / "movielens" / {"ml1m": "ml-1m", "ml10m": "ml-10M100K"}[name] / "ratings.dat"
        rows = [l.split("::")[:3] for l in open(path, encoding="latin-1")]
        arr = np.array(rows, dtype=np.float64)
        _, u = np.unique(arr[:, 0], return_inverse=True)
        _, i = np.unique(arr[:, 1], return_inverse=True)
        r = arr[:, 2]
        p = np.random.default_rng(0).permutation(len(r))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, u=u[p], i=i[p], r=r[p])
    z = np.load(cache)
    u, i, r = z["u"], z["i"], z["r"]
    n = len(r)
    a, b = int(0.8 * n), int(0.9 * n)
    mu = r[:a].mean()
    split = lambda s: (u[s].astype(np.int64), i[s].astype(np.int64), (r[s] - mu).astype(np.float64))
    return split(slice(0, a)), split(slice(a, b)), split(slice(b, n)), int(u.max() + 1), int(i.max() + 1)


def make_problem(name):
    tr, va, te, nu, ni = load(name)
    R_u = CSR(tr[0], tr[1], tr[2], nu)
    R_i = CSR(tr[1], tr[0], tr[2], ni)
    splits = {"val": va, "test": te}
    users = np.arange(nu, dtype=np.float64)[:, None]

    lam_file = results_path(TRACK, f"lambda_{name}.json")
    if lam_file.exists():
        lam = json.loads(lam_file.read_text())["lambda"]
    else:
        scores = {}
        for lam_c in LAMBDA_GRID:
            U, V = init_factors(nu, ni, F, 0)
            for _ in range(25):
                U, V = als_full_epoch(R_u, R_i, U, V, lam_c)
            scores[lam_c] = rmse(va, U, V)
            print(f"[lambda] {name} lam={lam_c} val_rmse@25={scores[lam_c]:.4f}", flush=True)
        lam = min(scores, key=scores.get)
        lam_file.parent.mkdir(parents=True, exist_ok=True)
        lam_file.write_text(json.dumps({"lambda": lam, "val_rmse_at_25": scores, "selected_on": "val, seed 0"}))

    class M:
        def __init__(self, U, V):
            self.U, self.V = U, V

    def fit(alpha0, T, B, seed, schedule=None, revival=True, n=None, k=None):
        f = k or F
        plug = ALSPlugin(R_u, ni, f, lam, seed=seed)
        T_train = max(T - 0.5, 0.0)
        if T_train > 0:
            steps = max(int(round(T_train * max(nu // B, 1))), 1)
            run_ema_engine(plug, users, epochs=T_train, batch_size=B, schedule=schedule or CosineSchedule(alpha0, steps, eps=0.1),
                           revival_every=0, finalize_mode="none", seed=seed)
        else:
            plug.init(users)
        U = solve_rows(np.arange(nu), R_u.indptr, R_u.indices, R_u.data, plug.V, lam)  # finalisation (0.5 epoch)
        return M(U, plug.V)

    def loss(m, split):
        return rmse(splits[split], m.U, m.V)

    def reference(seed):
        U, V = init_factors(nu, ni, F, seed)
        pts, wall = [], 0.0
        for it in range(1, 26):
            t0 = time.perf_counter()
            U, V = als_full_epoch(R_u, R_i, U, V, lam)
            wall += time.perf_counter() - t0
            pts.append({"epoch": it, "test_loss": rmse(te, U, V), "val": rmse(va, U, V), "wall_s": wall})
        return pts, M(U, V)

    return Problem(track=TRACK, dataset=name, k=F, d=ni, n=nu, metric="test_rmse", fit=fit, loss=loss,
                   reference=reference, config={"lambda": lam, "n_ratings_train": len(tr[2]), "n_items": ni})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ml1m,ml10m")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    for name in args.datasets.split(","):
        p = make_problem(name)
        print(f"\n##### {name} users={p.n} items={p.d} lambda={p.config['lambda']}", flush=True)
        hb = HEAD_B[name]
        law = [{"n": p.n, "B": B, "T": T} for B in (hb // 4, hb, hb * 4) for T in (1.0, 2.0, 5.0)]
        run_all(p, RUNS, SWEEP, seeds, ref_method="als_wr_full_batch", ema_method="als_ema", stepwise_method="stepwise_als",
                head_B=hb, stage=args.stage, law_sweep_configs=law, law_ks=(16, 32))


if __name__ == "__main__":
    main()
