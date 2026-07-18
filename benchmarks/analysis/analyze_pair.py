#!/usr/bin/env python3
"""Paired before/after benchmark analysis: the distribution of per-sample improvements.

Reads ``benchmark_per_sample.csv`` from a baseline and a candidate run directory, joins on
``sample_index``, and reports how the improvement is distributed across samples rather than only
aggregate totals. Outputs:

- a sorted per-sample relative-cost plot (one panel per series);
- an ECDF / performance-profile plot (cumulative fraction of samples by ratio);
- a stats table with the ratio distribution AND absolute time saved plus its concentration.

For each series the per-sample ratio is ``candidate / baseline`` (lower is better for every series
here: run time and solver-call counts). The improvement is ``baseline - candidate`` (positive =
saved). A pooled aggregate (sum candidate / sum baseline) is reported alongside so the reader can
see how far the typical sample diverges from the total.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SKIPPED_STATUS = "SKIPPED_PREDICTED_IN_TARGETED"
# A sample counts as improved/regressed only past this relative band; inside it is "unchanged".
TOLERANCE = 0.01
TIGHTENING_LABELS = {
    "interval_arithmetic": "interval-arithmetic",
    "lp": "LP",
    "mip": "MIP",
}

# Color encodes the phase/series consistently across every plot (blue = build + bound tightening,
# orange = main solve, green = total, purple = solver calls). The baseline is a neutral grey, and
# direction of change is read from geometry -- the ratio = 1 line and the y = x diagonal -- not hue.
# Colors are the Okabe-Ito colorblind-safe palette.
SERIES_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]  # blue, orange, green, purple
COLOR_BASELINE = "#8c8c8c"  # neutral grey for the "before" side
COLOR_REFERENCE = "#555555"  # reference lines (ratio = 1, y = x)
# Preferred fonts, first available wins; DejaVu Sans always ships with matplotlib.
FONT_STACK = ["Lato", "Ubuntu", "Source Sans Pro", "DejaVu Sans"]


@dataclass(frozen=True)
class Series:
    """One analysable per-sample quantity, lower-is-better."""

    key: str
    label: str
    unit: str
    builder: Callable[[pd.DataFrame], "pd.DataFrame | None"]


def _col(df: pd.DataFrame, name: str, side: str):
    full = f"{name}_{side}"
    if full not in df.columns:
        return None
    return pd.to_numeric(df[full], errors="coerce")


def _pair(df, name):
    base, cand = _col(df, name, "base"), _col(df, name, "cand")
    if base is None or cand is None:
        return None
    return pd.DataFrame({"sample_index": df["sample_index"], "base": base, "cand": cand})


def _build_and_tighten(df):
    """Everything before the main solve, defined as (total - main solve) so that this component
    and the main solve sum exactly to the end-to-end time. Dominated by model construction and
    ReLU bound tightening -- the phase this kind of change targets."""
    total = _pair(df, "total_time_seconds")
    solve = _pair(df, "solve_time_seconds")
    if total is None or solve is None:
        return None
    return pd.DataFrame(
        {
            "sample_index": df["sample_index"],
            "base": total["base"] - solve["base"],
            "cand": total["cand"] - solve["cand"],
        }
    )


# Ordered so the two components come first and their sum (the end-to-end total) follows:
# Total end-to-end = Build + bound tightening + Main solve.
SERIES: list[Series] = [
    Series("build_tighten", "Build + bound tightening", "s", _build_and_tighten),
    Series("main_solve", "Main solve time", "s", lambda d: _pair(d, "solve_time_seconds")),
    Series("total_time", "Total end-to-end time", "s", lambda d: _pair(d, "total_time_seconds")),
    Series(
        "bound_solver_calls",
        "Bound solver calls",
        "calls",
        lambda d: _pair(d, "bound_solver_call_count"),
    ),
]

# One fixed color per phase/series, reused across all three plots.
COLOR_BY_KEY = {s.key: SERIES_COLORS[i % len(SERIES_COLORS)] for i, s in enumerate(SERIES)}


def load_per_sample(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "benchmark_per_sample.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)


def joined_frame(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on sample_index, dropping inputs skipped on either side."""
    for frame, side in ((baseline, "baseline"), (candidate, "candidate")):
        if "sample_index" not in frame.columns:
            raise ValueError(f"{side} per-sample CSV lacks a sample_index column")
    merged = baseline.merge(candidate, on="sample_index", suffixes=("_base", "_cand"))
    return merged[
        (merged["solve_status_base"] != SKIPPED_STATUS)
        & (merged["solve_status_cand"] != SKIPPED_STATUS)
    ].copy()


def build_series_frame(merged: pd.DataFrame, series: Series) -> "pd.DataFrame | None":
    frame = series.builder(merged)
    if frame is None:
        return None
    frame = frame.dropna(subset=["base", "cand"])
    # A ratio needs a strictly positive baseline; a zero-baseline sample carries no signal.
    frame = frame[frame["base"] > 0].copy()
    if frame.empty:
        return None
    frame["ratio"] = frame["cand"] / frame["base"]
    frame["saved"] = frame["base"] - frame["cand"]  # positive = candidate saved time/calls
    return frame


def series_stats(series: Series, frame: pd.DataFrame) -> dict:
    ratios = frame["ratio"].to_numpy()
    saved = frame["saved"].to_numpy()
    base_sum, cand_sum = float(frame["base"].sum()), float(frame["cand"].sum())
    n = len(ratios)
    improved = int((ratios < 1 - TOLERANCE).sum())
    regressed = int((ratios > 1 + TOLERANCE).sum())
    # Concentration: share of the total absolute movement contributed by the biggest movers.
    abs_saved = np.abs(saved)
    order = np.argsort(abs_saved)[::-1]
    cum = np.cumsum(abs_saved[order])
    total_abs = cum[-1] if len(cum) else 0.0
    # With no per-sample movement, no samples contribute to concentration. Report 0 rather than
    # an undefined NaN so the Markdown table remains numeric and matches the report convention.
    top10_share = 0.0 if total_abs == 0.0 else float(cum[min(10, n) - 1] / total_abs)
    k5 = max(1, math.ceil(0.05 * n))
    top5pct_share = 0.0 if total_abs == 0.0 else float(cum[k5 - 1] / total_abs)

    def q(x):
        return float(np.quantile(ratios, x))

    return {
        "series": series.key,
        "label": series.label,
        "unit": series.unit,
        "n": n,
        "median_ratio": float(np.median(ratios)),
        "p10": q(0.10),
        "p25": q(0.25),
        "p75": q(0.75),
        "p90": q(0.90),
        "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()),
        "pct_improved": 100.0 * improved / n,
        "pct_unchanged": 100.0 * (n - improved - regressed) / n,
        "pct_regressed": 100.0 * regressed / n,
        "pooled_ratio": cand_sum / base_sum if base_sum > 0 else float("nan"),
        "base_sum": base_sum,
        "cand_sum": cand_sum,
        "net_saved": base_sum - cand_sum,
        "top10_share": top10_share,
        "top5pct_share": top5pct_share,
        "top5pct_k": k5,
    }


def stats_markdown(
    rows: list[dict], baseline_label: str, candidate_label: str, tightening_algorithm: str
) -> str:
    out = [
        f"Paired per-sample analysis: **{candidate_label}** vs **{baseline_label}**",
        "",
        "### Per-sample ratio distribution",
        "",
        "| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        out.append(
            "| {label} | {n} | {mn:.2f} | {p10:.2f} | {p25:.2f} | {med:.2f} | {p75:.2f} | {p90:.2f} "
            "| {mx:.2f} | {imp:.0f}% | {reg:.0f}% |".format(
                label=r["label"],
                n=r["n"],
                mn=r["min_ratio"],
                p10=r["p10"],
                p25=r["p25"],
                med=r["median_ratio"],
                p75=r["p75"],
                p90=r["p90"],
                mx=r["max_ratio"],
                imp=r["pct_improved"],
                reg=r["pct_regressed"],
            )
        )
    out += [
        "",
        f"- `ratio` = candidate ÷ baseline; < 1 = candidate faster. `improved`/`regressed` use a ±{TOLERANCE * 100:.0f}% band.",
        "- `build` = constructing the MIP model; `tightening` = the "
        f"{TIGHTENING_LABELS[tightening_algorithm]} bound-tightening pass; `main solve` = the final "
        "verification MIP.",
        "- `total` = `build` + `tightening` + `main solve`.",
        "",
        "### Aggregate saving and concentration",
        "",
        "- `net saved` = baseline − candidate total; positive = candidate cheaper.",
        "- `pooled ratio` = candidate total ÷ baseline total (aggregate counterpart to the per-sample median).",
        "- `top-10 concentration` = the 10 samples with the largest absolute change account for this "
        "share of the total absolute per-sample change (0–100%; higher = a few samples dominate).",
        "",
        "| series | baseline | candidate | net saved | pooled ratio | top-10 concentration |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        u = r["unit"]
        out.append(
            "| {label} | {b:.0f} {u} | {c:.0f} {u} | {net:+.0f} {u} | {pooled:.2f} | {t10:.0f}% |".format(
                label=r["label"],
                b=r["base_sum"],
                c=r["cand_sum"],
                net=r["net_saved"],
                u=u,
                pooled=r["pooled_ratio"],
                t10=100 * r["top10_share"],
            )
        )
    return "\n".join(out) + "\n"


def load_tightening_algorithm(run_dir: Path) -> str:
    metrics_path = run_dir / "benchmark_metrics.csv"
    if not metrics_path.is_file():
        raise ValueError(f"missing benchmark metadata: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    if "tightening_algorithm" not in metrics.columns:
        raise ValueError(f"missing tightening_algorithm column in {metrics_path}")

    algorithms = {
        str(value).strip()
        for value in metrics["tightening_algorithm"]
        if pd.notna(value) and str(value).strip()
    }
    if len(algorithms) != 1:
        raise ValueError(f"expected one tightening_algorithm value in {metrics_path}")

    algorithm = algorithms.pop()
    if algorithm not in TIGHTENING_LABELS:
        choices = ", ".join(TIGHTENING_LABELS)
        raise ValueError(
            f"unsupported tightening_algorithm {algorithm!r} in {metrics_path}; use {choices}"
        )
    return algorithm


def paired_tightening_algorithm(baseline_dir: Path, candidate_dir: Path) -> str:
    baseline_algorithm = load_tightening_algorithm(baseline_dir)
    candidate_algorithm = load_tightening_algorithm(candidate_dir)
    if baseline_algorithm != candidate_algorithm:
        raise ValueError(
            "paired runs use different tightening algorithms: "
            f"baseline={baseline_algorithm}, candidate={candidate_algorithm}"
        )
    return baseline_algorithm


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#333333",
            "text.color": "#222222",
            "figure.dpi": 130,
        }
    )


def _log2_axis(ax, values, axis):
    setter = ax.set_yscale if axis == "y" else ax.set_xscale
    ticks_setter = ax.set_yticks if axis == "y" else ax.set_xticks
    fmt_axis = ax.get_yaxis() if axis == "y" else ax.get_xaxis()
    setter("log", base=2)
    lo = min(0.25, float(np.min(values)))
    hi = max(4.0, float(np.max(values)))
    candidates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    ticks_setter([t for t in candidates if lo / 1.5 <= t <= hi * 1.5])
    fmt_axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))


def _frames_for_unit(frames, unit):
    """(key, label, frame) for series with the given unit, in canonical SERIES order."""
    return [(s.key, s.label, frames[s.key]) for s in SERIES if s.unit == unit and s.key in frames]


# Per-unit presentation: axis label, metric name for parallel titles, log-axis floor, and the
# words for the lower-is-better / higher-is-worse directions.
_UNIT_META = {
    "s": dict(value_label="seconds", metric="Runtime", floor=1e-3, lower="faster", higher="slower"),
    "calls": dict(
        value_label="calls", metric="Solver calls", floor=1.0, lower="fewer", higher="more"
    ),
}


def plot_ratio_ecdf(frames, out_path, baseline_label, candidate_label):
    """Paired relative view: ECDF of per-sample candidate/baseline ratios, all series overlaid."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    all_ratios = []
    for key, frame in frames.items():
        label = next(s.label for s in SERIES if s.key == key)
        r = np.sort(frame["ratio"].to_numpy())
        all_ratios.append(r)
        y = np.arange(1, len(r) + 1) / len(r)
        ax.step(
            np.concatenate([r, r[-1:]]),
            np.concatenate([[0], y]),
            where="post",
            color=COLOR_BY_KEY[key],
            linewidth=2.0,
            label=label,
        )
    ax.axvline(1.0, color=COLOR_REFERENCE, linestyle="--", linewidth=1.0)
    _log2_axis(ax, np.concatenate(all_ratios), "x")
    ax.set_xlabel("candidate / baseline ratio  (← faster / fewer)")
    ax.set_ylabel("cumulative fraction of samples")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="best")
    fig.suptitle("Relative cost — paired ECDF (all series)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def plot_absolute_ecdf(frames, out_path, baseline_label, candidate_label, unit="s"):
    """Absolute magnitude view: per-side distribution ECDFs (baseline vs candidate), small
    multiples per series. Unpaired -- it compares the two distributions, not sample by sample --
    so it shows where the mass lives and whether the whole distribution shifted. Returns True if a
    figure was written (False when no series of this unit is present)."""
    apply_style()
    series = _frames_for_unit(frames, unit)
    if not series:
        return False
    meta = _UNIT_META[unit]
    fig, axes = plt.subplots(
        1, len(series), figsize=(max(5.0 * len(series), 6.5), 4.9), squeeze=False
    )
    for ax, (key, label, frame) in zip(axes.flat, series):
        for side, color, name in (
            ("base", COLOR_BASELINE, "baseline"),
            ("cand", COLOR_BY_KEY[key], "candidate"),
        ):
            v = np.sort(np.maximum(frame[side].to_numpy(), meta["floor"]))
            y = np.arange(1, len(v) + 1) / len(v)
            ax.step(
                np.concatenate([v, v[-1:]]),
                np.concatenate([[0], y]),
                where="post",
                color=color,
                linewidth=2.0,
                label=name,
            )
        ax.set_xscale("log")
        ax.set_title(label)
        ax.set_xlabel(f"{meta['value_label']}  (← {meta['lower']})")
        ax.set_ylabel("cumulative fraction")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle(
        f"{meta['metric']} — absolute distribution per side (ECDF)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_magnitude_scatter(frames, out_path, baseline_label, candidate_label, unit="s"):
    """Paired magnitude view: log-log candidate-vs-baseline scatter, small multiples per series.
    Each point is one sample; the y=x diagonal is no change, below it the candidate is lower.
    Shows magnitude and pairing together. Returns True if a figure was written."""
    apply_style()
    series = _frames_for_unit(frames, unit)
    if not series:
        return False
    meta = _UNIT_META[unit]
    fig, axes = plt.subplots(
        1, len(series), figsize=(max(4.9 * len(series), 6.5), 5.3), squeeze=False
    )
    for ax, (key, label, frame) in zip(axes.flat, series):
        base = np.maximum(frame["base"].to_numpy(), meta["floor"])
        cand = np.maximum(frame["cand"].to_numpy(), meta["floor"])
        better = int((cand < base).sum())
        worse = len(base) - better
        lo = min(base.min(), cand.min())
        hi = max(base.max(), cand.max())
        # Shade the upper half-plane -- candidate above baseline = worse -- in neutral grey as a
        # reminder of which side is the regression; hue stays reserved for the series.
        ax.fill_between([lo, hi], [lo, hi], hi, color=COLOR_BASELINE, alpha=0.12, zorder=0)
        ax.scatter(
            base, cand, s=12, alpha=0.4, color=COLOR_BY_KEY[key], edgecolors="none", zorder=2
        )
        ax.plot([lo, hi], [lo, hi], color=COLOR_REFERENCE, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel(f"baseline {meta['value_label']}")
        ax.set_ylabel(f"candidate {meta['value_label']}")
        # Label each half in its own region: worse in the shaded upper-left, better below.
        ax.annotate(
            f"{meta['higher']} · {worse}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=9,
            color="#555555",
        )
        ax.annotate(
            f"{meta['lower']} · {better}",
            xy=(0.95, 0.05),
            xycoords="axes fraction",
            va="bottom",
            ha="right",
            fontsize=9,
            color="#555555",
        )
    fig.suptitle(
        f"{meta['metric']} — per-sample paired (scatter)", fontsize=13.5, fontweight="bold", y=0.98
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    return True


def status_markdown(baseline_df, candidate_df, baseline_label, candidate_label, max_per_group=40):
    """Aggregate solve-status counts per side, plus flips grouped by transition (before → after)
    with the affected samples listed, joined on sample_index."""
    b = baseline_df["solve_status"].value_counts().to_dict()
    c = candidate_df["solve_status"].value_counts().to_dict()
    out = [
        "### Solve status (all samples)",
        "",
        f"| status | {baseline_label} | {candidate_label} |",
        "|---|--:|--:|",
    ]
    for status in sorted(set(b) | set(c)):
        out.append(f"| {status} | {b.get(status, 0)} | {c.get(status, 0)} |")

    merged = baseline_df.merge(candidate_df, on="sample_index", suffixes=("_base", "_cand"))
    has_outcome = "semantic_outcome_base" in merged.columns

    # Group flips by transition and list the affected samples -- scales to many flips far better
    # than one row per sample.
    def grouped_flips(before_col, after_col, heading):
        changed = merged[merged[before_col] != merged[after_col]]
        lines = ["", heading, ""]
        if changed.empty:
            lines.append("_None._")
            return lines
        lines += ["| transition | n | samples |", "|---|--:|---|"]
        groups = changed.groupby([before_col, after_col])["sample_index"].apply(list)
        for (before, after), samples in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ids = sorted(int(s) for s in samples)
            shown = ", ".join(str(s) for s in ids[:max_per_group])
            if len(ids) > max_per_group:
                shown += f", … (+{len(ids) - max_per_group})"
            lines.append(f"| `{before}` → `{after}` | {len(ids)} | {shown} |")
        return lines

    out += grouped_flips(
        "solve_status_base", "solve_status_cand", "### Verdict flips — solve status"
    )
    if has_outcome:
        out += grouped_flips(
            "semantic_outcome_base", "semantic_outcome_cand", "### Verdict flips — semantic outcome"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, type=Path, help="baseline run directory")
    ap.add_argument("--candidate", required=True, type=Path, help="candidate run directory")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--candidate-label", default="candidate")
    args = ap.parse_args()

    try:
        tightening_algorithm = paired_tightening_algorithm(args.baseline, args.candidate)
    except ValueError as error:
        ap.error(str(error))

    args.out.mkdir(parents=True, exist_ok=True)
    baseline_df = load_per_sample(args.baseline)
    candidate_df = load_per_sample(args.candidate)
    merged = joined_frame(baseline_df, candidate_df)

    frames, rows, skipped = {}, [], []
    for series in SERIES:
        frame = build_series_frame(merged, series)
        if frame is None:
            skipped.append(series.label)
            continue
        frames[series.key] = frame
        rows.append(series_stats(series, frame))
    if not frames:
        raise SystemExit("no analysable series found in the given run directories")

    md = stats_markdown(rows, args.baseline_label, args.candidate_label, tightening_algorithm)
    status_md = status_markdown(
        baseline_df, candidate_df, args.baseline_label, args.candidate_label
    )
    full_md = md + "\n" + status_md
    (args.out / "improvement_stats.md").write_text(full_md)
    pd.DataFrame(rows).to_csv(args.out / "improvement_stats.csv", index=False)

    # (filename, plot fn) — plots return False and are skipped when their series/unit is absent.
    outputs = [
        ("ratio_ecdf.png", lambda f, p, b, c: plot_ratio_ecdf(f, p, b, c) or True),
        ("absolute_runtime_ecdf.png", lambda f, p, b, c: plot_absolute_ecdf(f, p, b, c, unit="s")),
        ("magnitude_scatter.png", lambda f, p, b, c: plot_magnitude_scatter(f, p, b, c, unit="s")),
        (
            "absolute_calls_ecdf.png",
            lambda f, p, b, c: plot_absolute_ecdf(f, p, b, c, unit="calls"),
        ),
        ("calls_scatter.png", lambda f, p, b, c: plot_magnitude_scatter(f, p, b, c, unit="calls")),
    ]
    written = []
    for name, fn in outputs:
        path = args.out / name
        if fn(frames, path, args.baseline_label, args.candidate_label):
            written.append(path)

    print(f"joined samples (solved both sides): {len(merged)}")
    if skipped:
        print("series skipped (columns absent): " + ", ".join(skipped))
    print()
    print(full_md)
    for path in written + [args.out / "improvement_stats.md", args.out / "improvement_stats.csv"]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
