"""Reproduce the five figures: python plots/generate.py (numpy, pandas, matplotlib).

Read only ../runs/*/benchmark_per_sample.csv; no repository imports or network.
Average observations within image and method before pairing: two per method for
images 1–500, one for 501–2000. Retain all 2,000 images for end-to-end time;
exclude predicted-label skips from modeled phases and calls (1,951 pairs).
Runtime phases use measured formulation and main-solve wall timers.
Absolute call plots retain true zeros; ratios omit only zero baseline values.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / "runs"
SKIPPED = "SKIPPED_PREDICTED_IN_TARGETED"
COLORS = ["#0072B2", "#D08000", "#009E73", "#B15B9B"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.titlesize": 14, "axes.labelsize": 12,
                     "legend.fontsize": 11, "savefig.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})

data, skipped = {}, {}
for method in ("lp", "target_lp"):
    files = sorted(ROOT.glob(f"*-{method}/benchmark_per_sample.csv"))
    assert len(files) == 12, (method, len(files))
    rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    counts = rows.groupby("sample_index").size()
    assert counts.index.tolist() == list(range(1, 2001))
    assert (counts.loc[1:500] == 2).all() and (counts.loc[501:2000] == 1).all()
    assert set(rows.adversarial_example_objective) == {"feasibility"}
    status = rows.solve_status.eq(SKIPPED).groupby(rows.sample_index)
    assert (status.min() == status.max()).all(), "skip state changed between observations"
    skipped[method] = status.max()
    data[method] = rows.groupby("sample_index").mean(numeric_only=True)
assert skipped["lp"].equals(skipped["target_lp"])
modeled = ~skipped["lp"]
assert modeled.sum() == 1951

specs = [
    ("build_tighten", "Model construction", "modeled pairs", "s"),
    ("main_solve", "Final MIP solve", "modeled pairs", "s"),
    ("total_time", "Total end-to-end time", "all images", "s"),
    ("bound_solver_calls", "Bound solver calls", "modeled pairs", "calls"),
]
frames, summary = {}, {}
for key, label, scope, unit in specs:
    values = []
    for method in ("lp", "target_lp"):
        d = data[method]
        if key == "build_tighten":
            v = d.formulation_time_seconds
        else:
            column = {"main_solve": "main_solve_wall_time_seconds", "total_time": "total_time_seconds",
                      "bound_solver_calls": "bound_solver_call_count"}[key]
            v = d[column]
        values.append(v if key == "total_time" else v[modeled])
    frame = pd.concat(values, axis=1)
    frame.columns = ["LP baseline", "Target LP"]
    assert not frame.isna().any().any() and (frame >= 0).all().all()
    frames[key] = frame
    a, b = frame.to_numpy().T
    eligible = a > 0
    r = b[eligible] / a[eligible]
    summary[key] = dict(n=len(a), ratio_n=int(eligible.sum()),
                        omitted_zero_baseline=int((~eligible).sum()),
                        baseline_zero=int((a == 0).sum()), target_zero=int((b == 0).sum()),
                        baseline_sum=float(a.sum()), target_sum=float(b.sum()),
                        baseline_median=float(np.median(a)), target_median=float(np.median(b)),
                        ratio_median=float(np.median(r)), ratio_p10=float(np.quantile(r, .1)),
                        ratio_p90=float(np.quantile(r, .9)), pooled_ratio=float(b.sum()/a.sum()),
                        improved=int((r < .99).sum()), unchanged=int(((r >= .99)&(r <= 1.01)).sum()),
                        regressed=int((r > 1.01).sum()), strictly_lower=int((b < a).sum()),
                        equal=int((b == a).sum()), strictly_higher=int((b > a).sum()))

def finish(fig, name, title, foot):
    fig.suptitle(title, fontsize=18, weight="bold", y=.98)
    fig.text(.5, .025, foot, ha="center", va="bottom", fontsize=11, color="#444444")
    fig.subplots_adjust(top=.82, bottom=.19, left=.085, right=.98, wspace=.30)
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)

def ecdf(ax, values, **kwargs):
    x = np.sort(values)
    ax.step(np.r_[x[0], x], np.r_[0, np.arange(1, len(x)+1)/len(x)], where="post", **kwargs)

def tidy(ax):
    ax.grid(alpha=.16, which="major")
    ax.set_axisbelow(True)

fig, ax = plt.subplots(figsize=(11, 6.8))
for (key, label, scope, unit), color in zip(specs, COLORS):
    a, b = frames[key].to_numpy().T
    eligible = a > 0
    ecdf(ax, b[eligible]/a[eligible], color=color, lw=2.4,
         label=f"{label} (n = {eligible.sum():,})")
ax.set_xscale("symlog", linthresh=.01, linscale=.4)
ax.set_xticks([0, .01, .1, 1, 10])
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
ax.axvline(1, color="#555555", ls="--", lw=1.2)
ax.set(xlabel="Target LP / LP baseline ratio  (lower = cheaper)",
       ylabel="Cumulative fraction of paired images", ylim=(0, 1.01), xlim=(0, None))
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=.95)
tidy(ax)
finish(fig, "ratio_ecdf.png", "Relative cost · Target LP vs LP baseline",
       "Per-image means; zeros retained using a symmetric-log axis. No zero baseline denominators omitted.\n"
       "Phases and calls: 1,951 modeled pairs. End-to-end: all 2,000 images (49 predicted-label skips included).")

for unit, absolute_name, scatter_name in [
    ("s", "absolute_runtime_ecdf.png", "magnitude_scatter.png"),
    ("calls", "absolute_calls_ecdf.png", "calls_scatter.png"),
]:
    selected = [(s, c) for s, c in zip(specs, COLORS) if s[3] == unit]
    size = (16, 6.6) if unit == "s" else (8.5, 6.6)
    fig, axes = plt.subplots(1, len(selected), figsize=size, squeeze=False)
    for ax, ((key, label, scope, _), color) in zip(axes.flat, selected):
        frame = frames[key]
        for side, c, ls in [("LP baseline", "#707070", "--"), ("Target LP", color, "-")]:
            ecdf(ax, frame[side].to_numpy(), color=c, lw=2.3, ls=ls, label=side)
        ax.set_xscale("log" if unit == "s" else "symlog", **({} if unit == "s" else {"linthresh": 1}))
        if unit == "calls":
            ax.set_xlim(left=0)
            ax.set_xticks([0, 1, 10, 100])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        if key == "build_tighten":
            ax.set_xticks([.3, .5, 1, 2])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set(title=f"{label}\nn = {len(frame):,} · {scope}",
               xlabel="Seconds per image" if unit == "s" else "Bound solver calls per image",
               ylabel="Cumulative fraction", ylim=(0, 1.01))
        ax.legend(frameon=False, loc="lower right")
        tidy(ax)
    foot = ("Per-image means: two observations per method for images 1–500; one for images 501–2,000.\n"
            "Construction and final solve use wall timers; phases exclude 49 predicted-label skips.")
    if unit == "calls":
        foot = ("True zeros retained: Target LP = 1,174 / 1,951; LP baseline = 0 / 1,951.\n"
                "Symmetric-log scale is linear from 0 to 1; fractional calls are per-image observation means.")
    finish(fig, absolute_name, "Runtime distributions" if unit == "s" else "Bound-call distributions", foot)

    fig, axes = plt.subplots(1, len(selected), figsize=size, squeeze=False)
    for ax, ((key, label, scope, _), color) in zip(axes.flat, selected):
        frame = frames[key]
        a, b = frame.to_numpy().T
        lo = min(a.min(), b.min()) * .8 if unit == "s" else -.08
        hi = max(a.max(), b.max()) * 1.15
        scale = "log" if unit == "s" else "symlog"
        opts = {} if unit == "s" else {"linthresh": 1}
        ax.set_xscale(scale, **opts)
        ax.set_yscale(scale, **opts)
        ax.fill_between([lo, hi], [lo, hi], hi, color="#eeeeee", zorder=0)
        ax.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=1)
        ax.scatter(a, b, s=14, color=color, alpha=.35, edgecolors="none", zorder=2)
        ax.set(xlim=(lo, hi), ylim=(lo, hi),
               title=f"{label}\nn = {len(frame):,} · {scope}",
               xlabel=f"LP baseline ({unit})", ylabel=f"Target LP ({unit})")
        ax.set_aspect("equal", adjustable="box")
        if key == "build_tighten":
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_ticks([.3, .5, 1, 2])
                axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
                axis.set_minor_formatter(plt.NullFormatter())
        if unit == "calls":
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_ticks([0, 1, 10, 100])
                axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.text(.04, .96, f"Higher: {(b>a).sum():,}\nEqual: {(b==a).sum():,}", transform=ax.transAxes,
                va="top", fontsize=11)
        ax.text(.96, .04, f"Lower: {(b<a).sum():,}", transform=ax.transAxes,
                ha="right", fontsize=11)
        tidy(ax)
    finish(fig, scatter_name, "Paired runtimes" if unit == "s" else "Paired bound solver calls",
           ("Each point is one image's mean; below the diagonal = lower cost with Target LP.\n" +
            ("End-to-end includes all 2,000 images; modeled phases exclude 49 predicted-label skips."
             if unit == "s" else "True zero counts retained on symmetric-log axes; no values replaced by a floor.")))

(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
