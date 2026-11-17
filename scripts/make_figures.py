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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.harness import read_runs  # noqa: E402
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
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# order matters: the first suffix contained in the method name wins ("frozen_emeans" must not match "emeans")
FAMILIES = (("frozen", 1), ("_swept", 0), ("emeans", 0), ("rewarm", 0), ("balanced_ivf_priced", 4),
            ("balanced_ivf", 0), ("joint_rotation", 0), ("_kmlaw", 1), ("cosine_law", 1), ("_rev_tuned", 6),
            ("_tuned", 2), ("hoffman", 3), ("sklearn", 4), ("periodic", 3), ("const_alpha", 2), ("faiss", 3))
NICE = {"frozen_emeans": "frozen (no updates)", "rewarm": "re-warming on drift (ours)",
        "periodic_retrain": "periodic retrain", "balanced_ivf": "balanced IVF (ours)",
        "balanced_ivf_priced": "balanced IVF, priced (ours)", "emeans_ivf": "e-means IVF", "faiss_ivf": "FAISS IVF"}


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


def save(fig, name, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    (OUT / f"{name}.txt").write_text("Plotted from:\n" + "\n".join(f"  {s}" for s in sources) + "\n")
    plt.close(fig)
    print(f"wrote {name}")


def small_ticks(ax):
    ax.tick_params(labelsize=6, length=2, pad=1.5)


DS_NAME = {"agnews_pca64": "AG News", "dinov2_pca64": "DINOv2", "sift1m_pca64": "SIFT1M", "synthetic": "synthetic",
           "brown_pos": "Brown POS", "uci_har_vq": "UCI HAR", "ml10m": "ML-10M", "ml1m": "ML-1M", "20ng": "20NG",
           "enron": "Enron", "nytimes": "NYTimes", "cifar_p12": "CIFAR 12x12", "cifar_p8": "CIFAR 8x8",
           "nmf_20ng": "NMF 20NG", "agnews_minilm": "AG News", "dinov2s_cifar100": "DINOv2", "sift1m": "SIFT1M", "dinov2": "DINOv2"}
TRACK_NAME = {"gmm-ema": "GMM", "hmm-ema": "HMM", "als-ema": "ALS", "lda-scaling": "LDA", "dictlearn-scaling": "DL",
              "core": "k-means"}
# Fig 1 roles: (name fragment, legend label, colour, linestyle, marker); the reference is neutral ink
ROLES = [("ref", "full-batch EM, by iteration", INK, "-", None),
         ("_swept", r"ours, $\alpha_0$ tuned on validation", SLOTS[0], "-", "o"),
         ("emeans", r"ours, $\alpha_0$ tuned on validation", SLOTS[0], "-", "o"),
         ("_kmlaw", r"ours, $\alpha_0$ from the $k$-means law (untuned)", SLOTS[1], "-", "o"),
         ("_tuned", "tuned stepwise EM", SLOTS[2], "-", "o"),
         ("_rev_tuned", "tuned stepwise EM + revival", SLOTS[6], "-", "o"),
         ("hoffman_default", "online LDA, default settings", SLOTS[3], "-", "o"),
         ("sklearn_minibatch", "scikit-learn mini-batch default", INK2, ":", None)]
TARGET_LABEL = "target: full-batch EM after 25 iterations"
TARGET_STYLE = dict(color=SLOTS[7], lw=0.9, ls=(0, (1.5, 1.5)))
# y-axis label per family (the test metric of Table 2)
YLAB = {"gmm-ema": "test NLL / point", "hmm-ema": "test NLL / token", "als-ema": "test RMSE",
        "lda-scaling": "test perplexity", "dictlearn-scaling": "test objective"}
# the six configurations shown in the main text (one or two per family); the appendix shows all 19
MAIN_PANELS = ("gmm-ema|sift1m_pca64|k=64|full", "gmm-ema|dinov2_pca64|k=64|diag", "hmm-ema|brown_pos|k=12",
               "lda-scaling|20ng|k=50", "dictlearn-scaling|cifar_p12|k=192", "als-ema|ml10m|k=32")


def role(m, is_ref):
    if is_ref:
        return ROLES[0]
    if "_rev_tuned" in m:
        return ROLES[5]
    return next((r for r in ROLES[1:] if r[0] in m), None)  # cosine_law and full-iteration library checks: not drawn


def panel_title(t, g):
    parts = g.split("|")
    k = next(p for p in parts if p.startswith("k="))
    extra = f" {parts[2]}" if t == "gmm-ema" else ""
    return f"{TRACK_NAME[t]} {DS_NAME.get(parts[0], parts[0])}{extra}, {k}"


def yfmt(ax):
    ax.yaxis.set_major_locator(MaxNLocator(3))
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.yaxis.set_major_formatter(FormatStrFormatter(
        "%.0f" if span > 20 else "%.1f" if span > 1.5 else "%.2f" if span > 0.15 else "%.3f"))


def fig_convergence(tracks=("gmm-ema", "hmm-ema", "als-ema", "lda-scaling", "dictlearn-scaling"), only=None,
                    name="convergence", cols=4):
    """Seed-mean test objective vs epochs, one panel per configuration; dotted line = target (reference after 25
    iterations). ``only``: group keys "track|group" to draw, in that order (main-text figure)."""
    panels = []
    for t in tracks:
        recs = read_runs(ROOT / "results" / t / "runs.jsonl")
        if not recs:
            continue
        s = summarise_track(recs)
        for g, grp in s["groups"].items():
            if grp["reference"] and (only is None or f"{t}|{g}" in only):
                panels.append((t, g, grp, len(recs)))
    if only is not None:
        panels.sort(key=lambda p: only.index(f"{p[0]}|{p[1]}"))
    if not panels:
        return
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, (1.55 if only else 1.35) * rows + 0.8), squeeze=False,
                             layout="constrained")
    fig.get_layout_engine().set(h_pad=0.05, w_pad=0.03, hspace=0.06, wspace=0.03)
    srcs, used = [], set()
    for i, (ax, (t, g, grp, nrec)) in enumerate(zip(axes.flat, panels)):
        ref = grp["reference"]
        for m, mm in grp["methods"].items():
            r = role(m, m == ref["method"])
            xs = sorted(float(e) for e in mm["curve_mean"])
            if r is None or len(xs) < 2:
                continue
            ys = [mm["curve_mean"][e] if e in mm["curve_mean"] else mm["curve_mean"][str(e)] for e in xs]
            ax.plot(xs, ys, color=r[2], ls=r[3], marker=r[4], ms=2.2, lw=1.0 if r[0] == "ref" else 1.2,
                    zorder=2 if r[0] == "ref" else 3)
            used.add(r[1])
        ax.axhline(ref["loss25_mean"], zorder=1, **TARGET_STYLE)
        ax.set_xscale("log")
        ax.set_xlim(0.2, 32)
        ax.set_xticks([0.25, 1, 5, 25], ["0.25", "1", "5", "25"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        yfmt(ax)
        small_ticks(ax)
        ax.set_title(panel_title(t, g), fontsize=6.5, color=INK, pad=2)
        ax.set_ylabel(YLAB[t], fontsize=5.5, labelpad=1)
        if i + cols >= len(panels):
            ax.set_xlabel("epochs (log scale)", fontsize=6.5, labelpad=1)
        srcs.append(f"results/{t}/runs.jsonl ({nrec} records), group {g}")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    seen, order = set(), []
    for r in ROLES:
        if r[1] in used and r[1] not in seen:
            seen.add(r[1]); order.append(r)
    h = [plt.Line2D([], [], color=r[2], ls=r[3], marker=r[4], ms=3, lw=1.2) for r in order]
    h.append(plt.Line2D([], [], **TARGET_STYLE))
    fig.legend(h, [r[1] for r in order] + [TARGET_LABEL], loc="outside lower center", ncol=3,
               fontsize=6.5, handlelength=2.2, columnspacing=1.2)
    save(fig, name, srcs + ["not drawn: cosine_law (LDA leave-one-corpus law), full-iteration library checks"])


FAM_STYLE = {"gmm-ema": ("GMM", "o"), "hmm-ema": ("HMM", "s"), "lda-scaling": ("LDA", "^"),
             "dictlearn-scaling": ("dictionary learning", "D"), "als-ema": ("ALS", "v")}
_LAW = {}


def law_cells():
    """Sweep cells with q_eff, plus the global Qe0 law fitted on uncensored cells (refit_coords, exploratory)."""
    if not _LAW:
        sys.path.insert(0, str(ROOT / "tracks" / "universal-scaling-law"))
        import refit_coords as rc
        cells = rc.load_cells(rc.dataset_stats())
        fitc = [c for c in cells if not c["edge"]]
        algs = sorted({c["alg"] for c in cells})
        _LAW.update(rc=rc, cells=cells, fitc=fitc, algs=algs, qe0=rc.fit(fitc, "Qe0", algs))
    return _LAW


T_GROUPS = (("0.25-1", lambda T: T <= 1, SEQ[0]), ("2-5", lambda T: 2 <= T <= 5, SEQ[2]),
            ("10-25", lambda T: T >= 10, SEQ[4]))


def fig_alpha_sweeps(tracks=("gmm-ema", "hmm-ema", "lda-scaling", "dictlearn-scaling", "als-ema")):
    """Fig 2: val penalty vs alpha0 relative to the global law's prediction; median over cells per budget group."""
    L = law_cells()
    rc, beta = L["rc"], L["qe0"]
    fig, axes = plt.subplots(1, len(tracks), figsize=(7.2, 1.95), squeeze=False, sharey=True, layout="constrained")
    bins = np.arange(-2.0, 2.01, 0.25)
    for ax, t in zip(axes[0], tracks):
        cs = [c for c in L["cells"] if c["alg"] == t]
        for lab, sel, col in T_GROUPS:
            pts = []
            for c in (c for c in cs if sel(c["T"])):
                a_law = rc.predict(beta, c, "Qe0", L["algs"])
                pts += [(np.log10(a / a_law), (v - c["L_star"]) / max(abs(c["L_star"]), 1e-12) * 100)
                        for a, v in c["curve"].items()]
            if not pts:
                continue
            pts = np.array(pts)
            idx = np.digitize(pts[:, 0], bins)
            xm, ym = [], []
            for b in np.unique(idx):
                sb = pts[idx == b]
                if len(sb) >= 3:
                    xm.append(10 ** np.median(sb[:, 0]))
                    ym.append(max(np.median(sb[:, 1]), 1e-3))
            ax.plot(xm, ym, color=col, marker="o", ms=2.5, lw=1.3, label=lab)
        ax.axvline(1.0, color=INK, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(10 ** -2.1, 10 ** 2.1)
        ax.set_xticks([0.01, 1, 100], ["0.01", "1", "100"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        small_ticks(ax)
        ax.set_title(f"{FAM_STYLE[t][0]} ({len(cs)} cells)", fontsize=7, pad=2)
    axes[0][0].set_ylabel("median validation penalty (%)", fontsize=7)
    fig.supxlabel(r"$\alpha_0$ / global-law prediction (dashed line = law)", fontsize=7)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, [f"T = {x} epochs" for x in lab], loc="outside upper center", ncol=3, fontsize=6.5)
    save(fig, "alpha0_sweeps", [f"results/<track>/sweep.jsonl via refit_coords.load_cells ({len(L['cells'])} cells)",
                                "x: alpha0 over the in-sample Qe0 prediction; y: median over cells in 0.25-decade bins",
                                "penalties floored at 1e-3 % for the log axis"])


def fig_ivf():
    recs = read_runs(ROOT / "results" / "balanced-ivf" / "runs.jsonl")
    if not recs:
        return
    groups = sorted({r["extra"]["group"] for r in recs})
    fig, axes = plt.subplots(1, len(groups), figsize=(7.2, 1.9), squeeze=False, layout="constrained")
    for ax, g in zip(axes[0], groups):
        by = defaultdict(list)
        for r in recs:
            if r["extra"]["group"] == g:
                by[r["method"]].append(r["extra"]["latency"])
        for m, lats in sorted(by.items()):
            npb = sorted({p["nprobe"] for L in lats for p in L})
            rec = [np.mean([p["recall10"] for L in lats for p in L if p["nprobe"] == n]) for n in npb]
            p99 = [np.mean([p["p99_us"] for L in lats for p in L if p["nprobe"] == n]) for n in npb]
            ax.plot(rec, p99, color=color(m), marker="o", ms=2.2, lw=1.2, label=NICE.get(m, m))
        ax.axvline(0.9, color=INK2, lw=0.7, ls="--")
        ax.set_yscale("log")
        small_ticks(ax)
        ds, nl = g.split("|")
        ax.set_title(f"{DS_NAME.get(ds, ds)}, {nl}", fontsize=6.5, pad=2)
    axes[0][0].set_ylabel("p99 latency (us)", fontsize=7)
    fig.supxlabel("recall@10", fontsize=7)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=4, fontsize=6.5)
    save(fig, "ivf_recall_latency", [f"results/balanced-ivf/runs.jsonl ({len(recs)} records)"])


def fig_drift():
    recs = read_runs(ROOT / "results" / "drift-rewarm" / "runs.jsonl")
    if not recs:
        return
    scen = sorted({r["extra"]["group"] for r in recs})
    fig, axes = plt.subplots(1, len(scen), figsize=(7.2, 1.9), squeeze=False, layout="constrained")
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
            ax.plot(t, y, color=color("const_alpha" if m == best else m), lw=1.1,
                    label="constant alpha (best, oracle-tuned)" if m == best else NICE.get(m, m))
        orc = [p["oracle"] for p in rows[0]["extra"]["stream_curve"]]
        ax.plot(t, orc, color=INK, lw=0.8, ls="--", label="oracle")
        yfmt(ax)
        small_ticks(ax)
        ax.set_title(g.replace("_", " "), fontsize=6.5, pad=2)
    axes[0][0].set_ylabel("window MSE", fontsize=7)
    fig.supxlabel("stream step", fontsize=7)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=5, fontsize=6.5)
    save(fig, "drift_window_mse", [f"results/drift-rewarm/runs.jsonl ({len(recs)} records)"])


LAW_MODELS = (("KM", r"$k$-means law (unchanged)"), ("M1", r"M1: $n/k$, $B$, $T$"),
              ("Qe1", r"Qe1: $q_{\mathrm{eff}}$, $n/B$, $T$"), ("Qe0", r"Qe0: one global law in $q_{\mathrm{eff}}$"))


def fig_law():
    """Fig 3: val penalty of the predicted alpha0, leave-one-cell-out and leave-one-algorithm-out (as refit_coords)."""
    from core.scaling import emeans_alpha0
    L = law_cells()
    rc, cells, fitc, algs = L["rc"], L["cells"], L["fitc"], L["algs"]
    res = {}
    for model, _ in LAW_MODELS:
        loco, loao = [], []
        if model == "KM":
            loco = loao = [rc.penalty(c, emeans_alpha0(int(c["n"]), int(c["k"]), c["T"], int(c["B"]))) for c in cells]
        else:
            for c in cells:
                train = [x for x in fitc if x is not c]
                loco.append(rc.penalty(c, rc.predict(rc.fit(train, model, algs), c, model, algs)))
            for held in algs:
                train = [c for c in fitc if c["alg"] != held]
                others = sorted({c["alg"] for c in train})
                beta = rc.fit(train, model, others)
                const = float(np.mean(beta[: len(others)])) if rc.MODELS[model][1] else None
                loao += [rc.penalty(c, rc.predict(beta, c, model, others, held_const=const))
                         for c in cells if c["alg"] == held]
        res[model] = (np.maximum(np.array(loco) * 100, 1e-3), np.maximum(np.array(loao) * 100, 1e-3))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.75), sharex=True, layout="constrained")
    fills = (INK2, SLOTS[1], SLOTS[2], SLOTS[0])
    for j, (ax, ev) in enumerate(zip(axes, ("leave one cell out", "leave one family out"))):
        data = [res[m][j] for m, _ in LAW_MODELS]
        bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55, whis=(5, 95),
                        flierprops=dict(marker=".", ms=2, markeredgecolor=INK2, alpha=0.5),
                        medianprops=dict(color=INK, lw=1.2))
        for patch, col in zip(bp["boxes"], fills):
            patch.set_facecolor(col); patch.set_alpha(0.5); patch.set_edgecolor(INK2)
        for k, d in enumerate(data):
            ax.text(1.01, k + 1, f"median {np.median(d):.3f}%", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=6, color=INK)
        ax.set_yticks(range(1, len(LAW_MODELS) + 1), [lab for _, lab in LAW_MODELS] if j == 0 else [])
        ax.invert_yaxis()
        ax.set_xscale("log")
        small_ticks(ax)
        ax.set_title(ev, fontsize=7, pad=2)
        ax.grid(axis="y", visible=False)
    fig.supxlabel(r"validation-loss penalty of the predicted $\alpha_0$ (%; whiskers 5th–95th percentile)",
                  fontsize=7)
    save(fig, "law_penalties", [f"results/<track>/sweep.jsonl via refit_coords ({len(cells)} cells, "
                                f"{len(cells) - len(fitc)} censored, excluded from fitting)",
                                "medians: " + "; ".join(f"{m} LOCO {np.median(res[m][0]):.3g}% LOAO "
                                                        f"{np.median(res[m][1]):.3g}%" for m, _ in LAW_MODELS),
                                "penalties floored at 1e-3 % for the log axis"])


def fig_law_coords():
    """Fig 4: swept alpha0* vs observations per statistic cell (q_eff), one global law (Qe0, exploratory)."""
    L = law_cells()
    rc, cells, fitc = L["rc"], L["cells"], L["fitc"]
    c0, a, b, d = rc.fit(fitc, "Qe0", L["algs"])
    nb_ref = float(np.median([c["n"] / c["B"] for c in cells]))
    # partial residual: alpha0* moved to the reference n/B and T = 1 along the fitted exponents
    adj = lambda c: c["a_star"] * (nb_ref / (c["n"] / c["B"])) ** b * (1.0 / c["T"]) ** d
    fig, ax = plt.subplots(figsize=(4.6, 2.6), layout="constrained")
    for i, (alg, (lab, mk)) in enumerate(FAM_STYLE.items()):
        cs = [c for c in cells if c["alg"] == alg]
        for cens in (False, True):
            sel = [c for c in cs if c["edge"] == cens]
            if sel:
                ax.scatter([c["qe"] for c in sel], [adj(c) for c in sel], s=12, marker=mk, linewidths=0.8,
                           facecolors="none" if cens else SLOTS[i], edgecolors=SLOTS[i], alpha=0.8,
                           label=lab if not cens else None, zorder=3)
    xs = np.logspace(np.log10(min(c["qe"] for c in cells)), np.log10(max(c["qe"] for c in cells)), 50)
    ax.plot(xs, np.minimum(np.exp(c0) * xs ** a * nb_ref ** b, 1.0), color=INK, lw=1.4, zorder=4,
            label=f"global fit, slope {a:.2f}")
    ax.scatter([], [], s=12, facecolors="none", edgecolors=INK2, label="censored at grid edge")
    ax.set_xscale("log"); ax.set_yscale("log")
    small_ticks(ax)
    ax.set_xlabel(r"observations per statistic cell per step, $q_{\mathrm{eff}}$", fontsize=7)
    ax.set_ylabel(rf"best $\alpha_0^\star$, adjusted to $n/B = {nb_ref:.0f}$, $T = 1$", fontsize=7)
    fig.legend(fontsize=6.5, loc="outside right upper", handletextpad=0.3)
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
                parts = g.split("|")
                rows.append((f"{TRACK_NAME[tr]} {DS_NAME.get(parts[0], parts[0])}" +
                             (f" {parts[-1]}" if tr == "gmm-ema" else ""), sp))
    if not rows:
        return
    rows.sort(key=lambda r: r[1]["_swept"] or r[1]["_kmlaw"])
    fig, ax = plt.subplots(figsize=(3.6, 0.2 * len(rows) + 0.7), layout="constrained")
    y = np.arange(len(rows))
    for kind, col, lab, off in (("_swept", SLOTS[0], r"$\alpha_0$ tuned on validation", 0.12),
                                ("_kmlaw", SLOTS[1], r"$\alpha_0$ from the $k$-means law (untuned)", -0.12)):
        xv = [r[1][kind] for r in rows]
        ax.scatter([x for x in xv if x], [yy + off for yy, x in zip(y, xv) if x], s=20, color=col, label=lab,
                   zorder=3, edgecolors="white", linewidths=0.8)
    ax.axvline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xscale("log")
    small_ticks(ax)
    ax.set_xlabel("wall-clock speed-up to reach batch@25 (CPU, isolated)", fontsize=7)
    ax.grid(axis="y", visible=False)
    fig.legend(fontsize=6.5, loc="outside upper center", ncol=1, handletextpad=0.2, title="our method",
              title_fontsize=6.5)
    save(fig, "timing_speedup", [f"results/_timing/{tr}/runs.jsonl via evaluate.timing_summary" for tr in tracks])


if __name__ == "__main__":
    fig_convergence()
    fig_convergence(only=MAIN_PANELS, name="convergence_main", cols=3)
    fig_alpha_sweeps()
    fig_ivf()
    fig_drift()
    fig_law()
    fig_law_coords()
    fig_timing()
