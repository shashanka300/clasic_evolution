"""Paper figures, plotted ONLY from results JSONL (no analytic curves, no synthetic noise).

    python scripts/make_figures.py            # writes results/figures/*.pdf|png

Every plotted value is read from results/<track>/runs.jsonl or sweep.jsonl; each figure's caption
stub (written next to it as .txt) names the files and record counts it came from.
Colour: validated categorical palette (dataviz reference, slots in fixed order); a method keeps its
colour across all figures; the reference baseline is neutral ink, dashed line = reference@25.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.harness import read_runs  # noqa: E402
from core.sweep import sweep_cells  # noqa: E402
from evaluate import summarise_track  # noqa: E402

OUT = ROOT / "results" / "figures"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#9ec5f4", "#5e9fe8", "#2a78d6", "#1a5aa8", "#0e3d75"]  # one hue light->dark, for ordinal budgets T
_COLOR: dict[str, str] = {}

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 4,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


FAMILIES = (("_swept", 0), ("emeans", 0), ("rewarm", 0), ("balanced_ivf_priced", 4), ("balanced_ivf", 0),
            ("joint_rotation", 0), ("_kmlaw", 1), ("cosine_law", 1), ("_rev_tuned", 6), ("_tuned", 2), ("hoffman", 3), ("sklearn", 4),
            ("frozen", 1), ("periodic", 3), ("const_alpha", 2), ("faiss", 3), ("emeans_ivf", 1))


def color(method: str, ref: bool = False) -> str:
    """Colour by method FAMILY so the same role has the same colour in every track."""
    if ref:
        return INK
    if method == "emeans_ivf":
        return SLOTS[1]
    for suffix, slot in FAMILIES:
        if suffix in method:
            return SLOTS[slot]
    if method not in _COLOR:
        if len(_COLOR) >= len(SLOTS):
            return INK2  # >8 entities fold to neutral (never generate a 9th hue)
        _COLOR[method] = SLOTS[len(_COLOR)]
    return _COLOR[method]


_T_ORDER = (1.0, 2.0, 3.0, 5.0, 10.0)


def tcol(T):
    i = min(range(len(_T_ORDER)), key=lambda j: abs(_T_ORDER[j] - T))
    return SEQ[i]


def save(fig, name, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    (OUT / f"{name}.txt").write_text("Plotted from:\n" + "\n".join(f"  {s}" for s in sources) + "\n")
    plt.close(fig)
    print(f"wrote {name}")


def fig_convergence(tracks=("gmm-ema", "hmm-ema", "als-ema", "lda-scaling", "dictlearn-scaling", "core")):
    """Seed-mean test loss vs epochs, one panel per comparison group; dashed line = reference@25."""
    panels = []
    for t in tracks:
        recs = read_runs(ROOT / "results" / t / "runs.jsonl")
        if not recs:
            continue
        s = summarise_track(recs)
        for g, grp in s["groups"].items():
            if grp["reference"] and (t != "core" or g.endswith("|train")):
                panels.append((t, g, grp, len(recs)))
    if not panels:
        return
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.1 * rows), squeeze=False)
    srcs = []
    for ax, (t, g, grp, nrec) in zip(axes.flat, panels):
        ref = grp["reference"]
        for m, mm in grp["methods"].items():
            xs = sorted(float(e) for e in mm["curve_mean"])
            if len(xs) < 2:
                continue
            ys = [mm["curve_mean"][e] if e in mm["curve_mean"] else mm["curve_mean"][str(e)] for e in xs]
            is_ref = m == ref["method"]
            ax.plot(xs, ys, color=color(m, is_ref), lw=1.5 if is_ref else 2.0, marker=None if is_ref else "o",
                    label=m)
        ax.axhline(ref["loss25_mean"], color=INK2, lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_title(f"{t}: {g}", fontsize=7, color=INK)
        ax.set_xlabel("epochs (log)")
        srcs.append(f"results/{t}/runs.jsonl ({nrec} records), group {g}")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fam = [("full-batch reference (25 it)", INK), ("EMA recipe, alpha0 swept on val", SLOTS[0]),
           ("EMA recipe, k-means law alpha0 (untuned)", SLOTS[1]), ("prior-art online/stepwise, tuned on val", SLOTS[2]),
           ("Hoffman default online LDA (tau0=1024, kappa=0.7)", SLOTS[3]),
           ("prior-art online/stepwise + recipe revival, tuned on val", SLOTS[6])]
    fig.legend([plt.Line2D([], [], color=c, marker=None if c == INK else "o") for _, c in fam], [f for f, _ in fam],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, 1.0, "dashed: reference@25 (seed mean)", ha="center", color=INK2, fontsize=7)
    fig.tight_layout()
    save(fig, "convergence", srcs)


def fig_alpha_sweeps(tracks=("gmm-ema", "hmm-ema", "als-ema", "lda-scaling", "dictlearn-scaling")):
    """Val-loss penalty vs alpha0 for each sweep cell (relative to the cell's best), one panel per track."""
    data = []
    for t in tracks:
        p = ROOT / "results" / t / "sweep.jsonl"
        if p.exists():
            cells = {k: v for k, v in sweep_cells(p).items() if not k[2].startswith(("online", "stepwise"))}
            if cells:
                data.append((t, cells))
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(1.8 * len(data) + 0.6, 2.0), squeeze=False, sharey=True)
    for ax, (t, cells) in zip(axes[0], data):
        for (g, T, m), c in cells.items():
            a = sorted(c["val_curve"])
            L = np.array([c["val_curve"][x] for x in a])
            Ls = L.min()
            ax.plot(a, (L - Ls) / max(abs(Ls), 1e-12) * 100, color=tcol(T), lw=1.0, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(t, fontsize=7)
        ax.set_xlabel("alpha0")
    axes[0][0].set_ylabel("val penalty vs best (%)")
    Ts = sorted({k[1] for _, c in data for k in c})
    fig.legend([plt.Line2D([], [], color=tcol(T)) for T in Ts], [f"T={T:g}" for T in Ts], loc="upper center",
               ncol=len(Ts), bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, "alpha0_sweeps", [f"results/{t}/sweep.jsonl ({len(c)} cells)" for t, c in data])


def fig_ivf():
    recs = read_runs(ROOT / "results" / "balanced-ivf" / "runs.jsonl")
    if not recs:
        return
    groups = sorted({r["extra"]["group"] for r in recs})
    fig, axes = plt.subplots(1, len(groups), figsize=(2.2 * len(groups), 2.1), squeeze=False)
    for ax, g in zip(axes[0], groups):
        by = defaultdict(list)
        for r in recs:
            if r["extra"]["group"] == g:
                by[r["method"]].append(r["extra"]["latency"])
        for m, lats in sorted(by.items()):
            npb = sorted({p["nprobe"] for L in lats for p in L})
            rec = [np.mean([p["recall10"] for L in lats for p in L if p["nprobe"] == n]) for n in npb]
            p99 = [np.mean([p["p99_us"] for L in lats for p in L if p["nprobe"] == n]) for n in npb]
            ax.plot(rec, p99, color=color(m), marker="o", label=m)
        ax.axvline(0.9, color=INK2, lw=1.0, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("recall@10")
        ax.set_title(g, fontsize=7)
    axes[0][0].set_ylabel("p99 latency (us)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, "ivf_recall_latency", [f"results/balanced-ivf/runs.jsonl ({len(recs)} records)"])


def fig_drift():
    recs = read_runs(ROOT / "results" / "drift-rewarm" / "runs.jsonl")
    if not recs:
        return
    scen = sorted({r["extra"]["group"] for r in recs})
    fig, axes = plt.subplots(1, len(scen), figsize=(2.2 * len(scen), 2.1), squeeze=False)
    for ax, g in zip(axes[0], scen):
        rows = [r for r in recs if r["extra"]["group"] == g]
        consts = {r["method"] for r in rows if r["method"].startswith("const_alpha_")}
        best = min(consts, key=lambda m: np.mean([r["test_loss"] for r in rows if r["method"] == m])) if consts else None
        show = ["frozen_emeans", "rewarm", "periodic_retrain"] + ([best] if best else [])
        for m in show:
            cs = [r["extra"]["stream_curve"] for r in rows if r["method"] == m]
            if not cs:
                continue
            t = [p["epoch"] for p in cs[0]]
            y = np.mean([[p["test_loss"] for p in c] for c in cs], axis=0)
            ax.plot(t, y, color=color("const_alpha (best)" if m == best else m), lw=1.5,
                    label="const_alpha (best, oracle-tuned)" if m == best else m)
        orc = [p["oracle"] for p in rows[0]["extra"]["stream_curve"]]
        ax.plot(t, orc, color=INK, lw=1.0, ls="--", label="oracle")
        ax.set_title(g, fontsize=7)
        ax.set_xlabel("stream step")
    axes[0][0].set_ylabel("window MSE")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, "drift_window_mse", [f"results/drift-rewarm/runs.jsonl ({len(recs)} records)"])


def fig_law():
    recs = read_runs(ROOT / "results" / "universal-scaling-law" / "runs.jsonl")
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    labels, data = [], []
    for ev in ("loco", "loao"):
        for m in ("M0", "M1", "M2", "KM"):
            r = next((r for r in recs if r["method"] == m and r["extra"]["group"] == ev), None)
            if r:
                labels.append(f"{ev}\n{m}")
                data.append(np.maximum(np.array(r["extra"]["penalties"]) * 100, 1e-4))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True, widths=0.5)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(color(lab.split("\n")[1]))
        patch.set_edgecolor(INK2)
    ax.set_yscale("log")
    ax.set_ylabel("val penalty of predicted alpha0 (%)")
    save(fig, "law_penalties", [f"results/universal-scaling-law/runs.jsonl ({len(recs)} records)"])


FAM_STYLE = {"gmm-ema": ("GMM", "o"), "hmm-ema": ("HMM", "s"), "lda-scaling": ("LDA", "^"),
             "dictlearn-scaling": ("dictionary learning", "D"), "als-ema": ("ALS", "v")}


def fig_law_coords():
    """Fig 4: swept alpha0* vs observations per statistic cell (q_eff), one global law (Qe0, exploratory)."""
    sys.path.insert(0, str(ROOT / "tracks" / "universal-scaling-law"))
    import refit_coords as rc
    cells = rc.load_cells(rc.dataset_stats())
    if not cells:
        return
    fitc = [c for c in cells if not c["edge"]]
    c0, a, b, d = rc.fit(fitc, "Qe0", sorted({c["alg"] for c in cells}))
    nb_ref = float(np.median([c["n"] / c["B"] for c in cells]))
    # partial residual: alpha0* moved to the reference n/B and T = 1 along the fitted exponents
    adj = lambda c: c["a_star"] * (nb_ref / (c["n"] / c["B"])) ** b * (1.0 / c["T"]) ** d
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for i, (alg, (lab, mk)) in enumerate(FAM_STYLE.items()):
        cs = [c for c in cells if c["alg"] == alg]
        if not cs:
            continue
        for cens in (False, True):
            sel = [c for c in cs if c["edge"] == cens]
            if sel:
                ax.scatter([c["qe"] for c in sel], [adj(c) for c in sel], s=14, marker=mk, linewidths=0.8,
                           facecolors="none" if cens else SLOTS[i], edgecolors=SLOTS[i], alpha=0.85,
                           label=lab if not cens else None, zorder=3)
    xs = np.logspace(np.log10(min(c["qe"] for c in cells)), np.log10(max(c["qe"] for c in cells)), 50)
    ax.plot(xs, np.minimum(np.exp(c0) * xs ** a * nb_ref ** b, 1.0), color=INK, lw=1.5, zorder=4,
            label=f"global fit, slope {a:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("observations per statistic cell per step, q_eff")
    ax.set_ylabel(f"alpha0* (adjusted to n/B={nb_ref:.0f}, T=1)")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), handletextpad=0.3)
    save(fig, "law_coords", [f"results/<track>/sweep.jsonl via refit_coords.load_cells ({len(cells)} cells, "
                             f"{len(cells) - len(fitc)} censored = hollow)",
                             f"Qe0 refit here: log a* = {c0:.3f} + {a:.3f} log q_eff + {b:.3f} log(n/B) + {d:.3f} log T"])


def fig_timing(tracks=("dictlearn-scaling", "hmm-ema", "gmm-ema", "lda-scaling", "als-ema")):
    """Fig 5: isolated-timing wall-clock speed-up to reach the full-batch reference@25 (CPU, T <= 5 only)."""
    from evaluate import timing_summary
    rows = []
    for tr in tracks:
        t = timing_summary(tr) or {}
        for g, v in t.items():
            sp = {kind: next((m["wall_speedup"] for name, m in v["methods"].items()
                              if name.endswith(kind) and m["wall_speedup"]), None) for kind in ("_swept", "_kmlaw")}
            if sp["_swept"] or sp["_kmlaw"]:
                rows.append((f"{'DL' if tr == 'dictlearn-scaling' else FAM_STYLE[tr][0]} {g.split('|')[0]}" +
                             (f" {g.split('|')[-1]}" if tr == "gmm-ema" else ""), sp))
    if not rows:
        return
    rows.sort(key=lambda r: r[1]["_swept"] or r[1]["_kmlaw"])
    fig, ax = plt.subplots(figsize=(3.6, 0.22 * len(rows) + 0.6))
    y = np.arange(len(rows))
    for kind, col, lab, off in (("_swept", SLOTS[0], "recipe, alpha0 swept on val", 0.12),
                                ("_kmlaw", SLOTS[1], "recipe, k-means law (untuned)", -0.12)):
        xv = [r[1][kind] for r in rows]
        ax.scatter([x for x in xv if x], [yy + off for yy, x in zip(y, xv) if x], s=22, color=col, label=lab,
                   zorder=3, edgecolors="white", linewidths=0.8)
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock speed-up to reach full-batch@25 (CPU, isolated)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.2)
    save(fig, "timing_speedup", [f"results/_timing/{tr}/runs.jsonl via evaluate.timing_summary" for tr in tracks])


if __name__ == "__main__":
    fig_convergence()
    fig_alpha_sweeps()
    fig_ivf()
    fig_drift()
    fig_law()
    fig_law_coords()
    fig_timing()
