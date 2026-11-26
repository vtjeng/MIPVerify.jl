"""Analyze the paired first-2,000 WK17a measurements and regenerate five figures.

Run python3 analyze.py after installing numpy, pandas and matplotlib. Every input
is a CSV under runs/: baseline-{lane}.csv or baseline-{pass}-{lane}.csv, and
candidate-{pass}-{lane}.csv. The two methods must each cover every image equally
often. Means are taken within image and method before pairing. First-observation
status and outcome tables use the lowest numbered pass, without averaging labels.
"""

from collections import Counter, defaultdict
from pathlib import Path
import json
import csv
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LABELS = ("Target LP", "Backward bounds")
METHODS = ("baseline", "candidate")
N = 2000
DEFINITIVE = {
    "certified_no_adversarial_example",
    "adversarial_example_found_or_best_known",
}
NUMERIC = (
    "total_time_seconds",
    "main_solve_wall_time_seconds",
    "bound_solver_call_count",
)


def read_runs():
    data, observations, first, runs = {}, {}, {}, {}
    for method in METHODS:
        files = sorted((ROOT / "runs").glob(f"{method}-*.csv"))
        assert files, f"No {method} runs found"
        records = []
        for path in files:
            match = re.fullmatch(rf"{method}-(?:(\d+)-)?(\d+)\.csv", path.name)
            assert match, f"Unexpected run filename {path.name}"
            frame = pd.read_csv(path, keep_default_na=False)
            assert not frame.empty and frame.sample_index.is_unique, path
            assert set(NUMERIC).issubset(frame.columns), path
            assert {"solve_status", "semantic_outcome", "source"}.issubset(
                frame.columns
            ), path
            if method == "candidate":
                required = {
                    "witness_verification",
                    "witness_repaired",
                    "perturbed_input_value",
                }
                assert required.issubset(frame.columns), (
                    path,
                    "missing returned-witness metadata",
                )
                assert set(frame.witness_verification).issubset(
                    {"none", "outward_interval", "exact_rational"}
                ), (path, "unknown witness proof")
                verified = frame.semantic_outcome.eq(
                    "adversarial_example_found_or_best_known"
                )
                assert verified.equals(frame.perturbed_input_value.ne("")), path
                assert verified.equals(frame.witness_verification.ne("none")), path
                for coords in frame.loc[verified, "perturbed_input_value"]:
                    point = np.fromstring(coords, sep=";")
                    assert len(point) == 784 and np.isfinite(point).all(), path
                    assert (point >= 0).all() and (point <= 1).all(), path
            frame["pass"] = int(match[1] or 0)
            frame["lane"] = int(match[2])
            for key in NUMERIC:
                frame[key] = pd.to_numeric(frame[key], errors="raise")
                assert np.isfinite(frame[key]).all() and (frame[key] >= 0).all(), (
                    path,
                    key,
                )
            assert (
                frame.main_solve_wall_time_seconds <= frame.total_time_seconds
            ).all(), path
            records.append(frame)
        rows = pd.concat(records, ignore_index=True).sort_values(
            ["pass", "lane", "sample_index"]
        )
        counts = rows.groupby("sample_index").size()
        assert counts.index.tolist() == list(range(1, N + 1)), (method, "image IDs")
        assert counts.nunique() == 1, (method, "unequal image weights")
        assert counts.iloc[0] == 2, (method, "two complete passes are required")
        assert not rows.duplicated(["pass", "sample_index"]).any(), (
            method,
            "duplicate pass/image",
        )
        for run, group in rows.groupby("pass"):
            assert sorted(group.sample_index) == list(range(1, N + 1)), (method, run)
        observations[method] = int(counts.iloc[0])
        data[method] = rows.groupby("sample_index")[list(NUMERIC)].mean()
        data[method]["outside_final_solve_seconds"] = (
            data[method].total_time_seconds - data[method].main_solve_wall_time_seconds
        )
        first[method] = rows.groupby("sample_index", sort=True).first()
        runs[method] = rows
    for i in range(1, N + 1):
        verdicts = {
            r
            for method in METHODS
            for r in runs[method].loc[
                runs[method].sample_index == i, "semantic_outcome"
            ]
            if r in DEFINITIVE
        }
        assert len(verdicts) <= 1, (i, "contradictory definitive outcomes", verdicts)
    return data, observations, first, runs


def transitions(first, key):
    result = defaultdict(list)
    for i in range(1, N + 1):
        a, b = (str(first[method].loc[i, key]) for method in METHODS)
        if a != b:
            result[a, b].append(i)
    return [
        {"before": a, "after": b, "n": len(ids), "samples": ids}
        for (a, b), ids in sorted(result.items())
    ]


def calculate(data, observations, first, runs):
    summary = {
        "samples": N,
        "labels": list(LABELS),
        "observations_per_image": observations,
        "timing_scope": "Per-image means. All 2,000 images, including zero final solves and zero bound calls.",
        "status_scope": "First observation of each image and method.",
        "metrics": {},
    }
    metrics = [
        ("outside_final_solve_seconds", "Outside final solve", "s"),
        ("main_solve_wall_time_seconds", "Final MIP solve", "s"),
        ("total_time_seconds", "Total end-to-end time", "s"),
        ("bound_solver_call_count", "Bound solver calls", "calls"),
    ]
    for key, label, unit in metrics:
        a, b = (data[method][key].to_numpy() for method in METHODS)
        eligible = a > 0
        ratio = b[eligible] / a[eligible]
        summary["metrics"][key] = {
            "label": label,
            "unit": unit,
            "n": N,
            "ratio_n": int(eligible.sum()),
            "omitted_zero_baseline": int((~eligible).sum()),
            "baseline_zero_candidate_positive": int(((a == 0) & (b > 0)).sum()),
            "zero_baseline_regression_samples": (
                np.flatnonzero((a == 0) & (b > 0)) + 1
            ).tolist(),
            "baseline_zero": int((a == 0).sum()),
            "candidate_zero": int((b == 0).sum()),
            "baseline_sum": float(a.sum()),
            "candidate_sum": float(b.sum()),
            "saved": float((a - b).sum()),
            "reduction_pct": float(100 * (1 - b.sum() / a.sum())),
            "ratio_p10": float(np.quantile(ratio, 0.1)),
            "ratio_median": float(np.median(ratio)),
            "ratio_p90": float(np.quantile(ratio, 0.9)),
            "improved": int((ratio < 0.99).sum()),
            "unchanged": int(((ratio >= 0.99) & (ratio <= 1.01)).sum()),
            "regressed": int((ratio > 1.01).sum()),
            "strictly_lower": int((b < a).sum()),
            "equal": int((b == a).sum()),
            "strictly_higher": int((b > a).sum()),
        }
    a, b = (data[method].total_time_seconds.to_numpy() for method in METHODS)
    rank = np.argsort(a - b)[::-1]
    solver_ids = []
    for i in range(1, min(N, 500) + 1):
        observations_for_image = [
            runs[method].loc[runs[method].sample_index == i] for method in METHODS
        ]
        if all(
            (r.source == "MIP").all()
            and not r.solve_status.isin(
                ["NO_SOLVER", "SKIPPED_PREDICTED_IN_TARGETED"]
            ).any()
            for r in observations_for_image
        ):
            solver_ids.append(i)
    subset_a, subset_b = (
        data[method].loc[solver_ids, "total_time_seconds"].to_numpy()
        for method in METHODS
    )
    assert len(solver_ids), (
        "No first-500 image ran the integer solver in all observations"
    )
    summary["first_500_solver_intersection"] = {
        "definition": "Images 1–500 with source MIP and a solver-run status in both observations of both methods. Times use per-image means.",
        "n": len(solver_ids),
        "samples": solver_ids,
        "baseline_sum_seconds": float(subset_a.sum()),
        "candidate_sum_seconds": float(subset_b.sum()),
        "reduction_pct": float(100 * (1 - subset_b.sum() / subset_a.sum())),
        "median_candidate_over_baseline": float(np.median(subset_b / subset_a)),
    }
    summary["top_savings"] = [
        {"sample": int(i + 1), "saved_seconds": float(a[i] - b[i])} for i in rank[:10]
    ]
    summary["top_regressions"] = [
        {"sample": int(i + 1), "saved_seconds": float(a[i] - b[i])}
        for i in rank[-10:][::-1]
    ]
    summary["reduction_excluding_largest_savings_pct"] = {
        str(k): float(100 * (1 - b[rank[k:]].sum() / a[rank[k:]].sum()))
        for k in (1, 5, 10)
    }
    rng = np.random.default_rng(20261125)
    bootstrap = []
    for _ in range(4000):
        ids = rng.integers(0, N, N)
        bootstrap.append(100 * (1 - b[ids].sum() / a[ids].sum()))
    summary["paired_image_bootstrap_95_pct"] = np.quantile(
        bootstrap, [0.025, 0.975]
    ).tolist()
    summary["returned_witnesses"] = {
        "count": int(runs["candidate"].perturbed_input_value.ne("").sum()),
        "proof_methods_by_pass": {
            str(p): dict(Counter(group.witness_verification))
            for p, group in runs["candidate"].groupby("pass")
        },
        "repaired_sample_ids_by_pass": {
            str(p): sorted(
                group.loc[
                    group.witness_repaired.astype(str).str.lower().eq("true"),
                    "sample_index",
                ].tolist()
            )
            for p, group in runs["candidate"].groupby("pass")
        },
    }
    for key in ("solve_status", "semantic_outcome", "source"):
        summary[key + "_counts"] = {
            LABELS[k]: dict(Counter(first[method][key]))
            for k, method in enumerate(METHODS)
        }
        summary[key + "_transitions"] = transitions(first, key)
    summary["run_totals_seconds"] = {
        LABELS[k]: {
            str(p): float(g.total_time_seconds.sum())
            for p, g in runs[method].groupby("pass")
        }
        for k, method in enumerate(METHODS)
    }
    summary["within_method_repeat_changes"] = {}
    for k, method in enumerate(METHODS):
        by_pass = {
            p: g.set_index("sample_index") for p, g in runs[method].groupby("pass")
        }
        first_pass, last_pass = min(by_pass), max(by_pass)
        summary["within_method_repeat_changes"][LABELS[k]] = {}
        for key in ("solve_status", "semantic_outcome", "source"):
            changes = defaultdict(list)
            for i in range(1, N + 1):
                a, b = (
                    str(by_pass[first_pass].loc[i, key]),
                    str(by_pass[last_pass].loc[i, key]),
                )
                if a != b:
                    changes[a, b].append(i)
            summary["within_method_repeat_changes"][LABELS[k]][key] = [
                {"before": a, "after": b, "n": len(ids), "samples": ids}
                for (a, b), ids in sorted(changes.items())
            ]
    summary["repeat_noise_pct"] = {}
    for k, method in enumerate(METHODS):
        totals = list(summary["run_totals_seconds"][LABELS[k]].values())
        if len(totals) > 1:
            summary["repeat_noise_pct"][LABELS[k]] = (
                100 * (max(totals) - min(totals)) / totals[0]
            )
    return summary, metrics


def compact_ids(ids, link):
    groups = []
    start = last = ids[0]
    for value in ids[1:]:
        if value == last + 1:
            last = value
            continue
        groups.append(str(start) if start == last else f"{start}–{last}")
        start = last = value
    groups.append(str(start) if start == last else f"{start}–{last}")
    result = ", ".join(groups)
    return result if len(result) <= 180 else f"[All {len(ids):,} image IDs]({link})"


def write_pairs(data, runs):
    paired = []
    for method in METHODS:
        r = runs[method].set_index(["pass", "sample_index"])[
            ["solve_status", "semantic_outcome", "source"]
        ]
        r.columns = [f"{method}_{c}" for c in r.columns]
        paired.append(r)
    joined = pd.concat(paired, axis=1)
    assert not joined.isna().any().any()
    joined.sort_index().to_csv(ROOT / "paired-statuses.csv")
    means = []
    for method in METHODS:
        frame = data[method].copy()
        frame.columns = [f"{method}_{c}" for c in frame.columns]
        means.append(frame)
    pd.concat(means, axis=1).to_csv(ROOT / "paired-means.csv")


def table(headers, rows):
    return "\n".join(
        "| " + " | ".join(str(v) for v in r) + " |"
        for r in [headers, ["---"] + ["---:"] * (len(headers) - 1), *rows]
    )


def write_tables(summary):
    counts = summary["solve_status_counts"]
    statuses = sorted({x for values in counts.values() for x in values})
    out = [
        "### Solve status and verdict flips",
        "Counts use the first observation of each image under each method.",
        table(
            ["Status", *LABELS],
            [[f"`{s}`", *[counts[l].get(s, 0) for l in LABELS]] for s in statuses],
        ),
    ]
    no_solver = {
        s
        for s in statuses
        if s in {"NO_SOLVER", "NOT_RUN", "BACKWARD_CERTIFIED", "ASCENT_WITNESS"}
    }
    if no_solver:
        out.append(
            "`NO_SOLVER` records a result reached without running the integer solver. The verification-outcome table below distinguishes robustness certificates from adversarial examples."
        )
    for key, name in (
        ("solve_status", "Solve status"),
        ("semantic_outcome", "Semantic outcome"),
    ):
        out.append("**" + name + "**")
        changes = summary[key + "_transitions"]
        result_rows = []
        for row in changes:
            filename = f"{key}-{row['before']}-to-{row['after']}.csv"
            target = ROOT / "image-ids" / filename
            target.parent.mkdir(exist_ok=True)
            with target.open("w") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["sample_index"])
                writer.writerows([i] for i in row["samples"])
            result_rows.append(
                [
                    f"`{row['before']}` → `{row['after']}`",
                    row["n"],
                    compact_ids(row["samples"], f"image-ids/{filename}"),
                ]
            )
        out.append(
            table(
                [
                    "Transition (Target LP → Backward bounds)",
                    "Count",
                    "MNIST test image IDs",
                ],
                result_rows,
            )
            if changes
            else "_None._"
        )
    (ROOT / "transitions.md").write_text("\n\n".join(out) + "\n")


def write_metrics(summary):
    m = summary["metrics"]
    rows = []
    for key in (
        "outside_final_solve_seconds",
        "main_solve_wall_time_seconds",
        "total_time_seconds",
    ):
        v = m[key]
        rows.append(
            [
                v["label"],
                f"{v['baseline_sum']:,.1f}",
                f"{v['candidate_sum']:,.1f}",
                f"{v['saved']:,.1f}",
                f"{v['reduction_pct']:.1f}%",
            ]
        )
    parts = [
        "### Timing totals",
        table(
            [
                "Phase",
                "Target LP (s)",
                "Backward bounds (s)",
                "Seconds saved",
                "Reduction",
            ],
            rows,
        ),
        "Totals sum each image's mean across its two observations.",
        "### Per-image ratios",
        table(
            [
                "Measure",
                "Eligible pairs",
                "10th percentile",
                "Median",
                "90th percentile",
                "Faster by >1%",
                "Within 1%",
                "Slower by >1%",
            ],
            [
                [
                    v["label"],
                    v["ratio_n"],
                    f"{v['ratio_p10']:.3f}",
                    f"{v['ratio_median']:.3f}",
                    f"{v['ratio_p90']:.3f}",
                    v["improved"],
                    v["unchanged"],
                    v["regressed"],
                ]
                for v in m.values()
            ],
        ),
        "Ratios are Backward bounds divided by Target LP. A zero Target LP value has no finite ratio and is omitted only from ratio statistics. Absolute plots retain all images and true zeros.",
    ]
    ci = summary["paired_image_bootstrap_95_pct"]
    parts.append(
        f"The 95% paired-image bootstrap interval for the total-time reduction is {ci[0]:.1f}–{ci[1]:.1f}%. It measures sensitivity to the image mix."
    )
    trimmed = summary["reduction_excluding_largest_savings_pct"]["10"]
    parts.append(
        f"Removing the ten images with the largest time savings leaves a {trimmed:.1f}% reduction."
    )
    for label, totals in summary["run_totals_seconds"].items():
        parts.append(
            label
            + " complete-run totals were "
            + ", ".join(f"{v:,.1f} s" for v in totals.values())
            + "."
        )
    subset = summary["first_500_solver_intersection"]
    parts.extend(
        [
            "### First 500 images requiring the integer solver in every observation",
            "The intersection requires source `MIP` and an actual solver-run status in both observations of both methods. Each image contributes its mean time across those two observations.",
            table(
                [
                    "Images",
                    "Target LP (s)",
                    "Backward bounds (s)",
                    "Reduction",
                    "Median Backward bounds / Target LP ratio",
                ],
                [
                    [
                        subset["n"],
                        f"{subset['baseline_sum_seconds']:.3f}",
                        f"{subset['candidate_sum_seconds']:.3f}",
                        f"{subset['reduction_pct']:.1f}%",
                        f"{subset['median_candidate_over_baseline']:.3f}",
                    ]
                ],
            ),
            "Image IDs: " + ", ".join(map(str, subset["samples"])) + ".",
        ]
    )
    (ROOT / "metrics.md").write_text("\n\n".join(parts) + "\n")


def plot(data, summary, specs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    out = ROOT / "plots"
    out.mkdir(exist_ok=True)
    colors = ["#0072B2", "#D08000", "#009E73", "#B15B9B"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def ecdf(ax, values, **kwargs):
        x = np.sort(values)
        ax.step(
            np.r_[x[0], x],
            np.r_[0, np.arange(1, len(x) + 1) / len(x)],
            where="post",
            **kwargs,
        )

    def finish(fig, name, title, foot):
        fig.suptitle(title, fontsize=18, weight="bold", y=0.98)
        fig.text(
            0.5, 0.025, foot, ha="center", va="bottom", fontsize=10.5, color="#444444"
        )
        fig.subplots_adjust(top=0.80, bottom=0.21, left=0.09, right=0.98, wspace=0.32)
        fig.savefig(out / name, facecolor="white")
        plt.close(fig)

    def tidy(ax):
        ax.grid(alpha=0.16, which="major")
        ax.set_axisbelow(True)

    def axis(ax, axis_name, unit, upper):
        # A linear neighborhood retains actual zeros. Positive values use logarithmic spacing.
        fn = getattr(ax, "set_" + axis_name + "scale")
        fn("symlog", linthresh=0.001 if unit == "s" else 1, linscale=0.5)
        ticks = [0, 0.001, 0.01, 0.1, 1, 10, 100] if unit == "s" else [0, 1, 10, 100]
        getattr(ax, "set_" + axis_name + "ticks")([t for t in ticks if t <= upper])
        getattr(ax, axis_name + "axis").set_major_formatter(
            FuncFormatter(lambda x, _: f"{x:g}")
        )

    fig, ax = plt.subplots(figsize=(11, 6.8))
    ratio_maximum = 0.0
    for (key, label, _), c in zip(specs, colors):
        a, b = (data[m][key].to_numpy() for m in METHODS)
        eligible = a > 0
        ratios = b[eligible] / a[eligible]
        ratio_maximum = max(ratio_maximum, float(ratios.max()))
        ecdf(ax, ratios, color=c, lw=2.4, label=f"{label} (n = {eligible.sum():,})")
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.4)
    ax.set_xticks(
        [
            x
            for x in [0, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
            if x <= ratio_maximum * 1.05
        ]
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.axvline(1, color="#555555", ls="--", lw=1.2)
    ax.set(
        xlabel="Backward bounds / Target LP ratio (lower = cheaper)",
        ylabel="Cumulative fraction of paired images",
        ylim=(0, 1.01),
        xlim=(0, None),
    )
    ax.legend(loc="lower right", framealpha=0.95)
    tidy(ax)
    n = summary["metrics"]
    foot = (
        "Per-image means. True zero candidate costs retained. Ratios omit only zero Target LP denominators.\n"
        + ". ".join(
            f"{v['label']}: {v['omitted_zero_baseline']} omitted" for v in n.values()
        )
    )
    finish(fig, "ratio_ecdf.png", "Relative cost · Backward bounds vs Target LP", foot)
    for unit, absolute, scatter in [
        ("s", "absolute_runtime_ecdf.png", "magnitude_scatter.png"),
        ("calls", "absolute_calls_ecdf.png", "calls_scatter.png"),
    ]:
        selected = [(s, c) for s, c in zip(specs, colors) if s[2] == unit]
        fig, axes = plt.subplots(
            1,
            len(selected),
            figsize=(16, 6.7) if unit == "s" else (8.5, 6.7),
            squeeze=False,
        )
        for ax, ((key, label, _), color) in zip(axes.flat, selected):
            for method, name, c, ls in [
                ("baseline", LABELS[0], "#707070", "--"),
                ("candidate", LABELS[1], color, "-"),
            ]:
                ecdf(
                    ax, data[method][key].to_numpy(), color=c, lw=2.3, ls=ls, label=name
                )
            upper = max(data[m][key].max() for m in METHODS) * 1.15
            axis(ax, "x", unit, upper)
            ax.set_xlim(0, upper)
            ax.set(
                title=f"{label}\nn = {N:,} images",
                xlabel="Seconds per image"
                if unit == "s"
                else "Bound solver calls per image",
                ylabel="Cumulative fraction",
                ylim=(0, 1.01),
            )
            ax.legend(frameon=False, loc="lower right")
            tidy(ax)
        finish(
            fig,
            absolute,
            "Runtime distributions" if unit == "s" else "Bound-call distributions",
            "Per-image means, all 2,000 images. Zero costs are retained.\nOutside final solve = total verification time minus final MIP wall time, including backward bounds and witness search.",
        )
        fig, axes = plt.subplots(
            1,
            len(selected),
            figsize=(16, 6.7) if unit == "s" else (8.5, 6.7),
            squeeze=False,
        )
        for ax, ((key, label, _), color) in zip(axes.flat, selected):
            a, b = (data[m][key].to_numpy() for m in METHODS)
            hi = max(a.max(), b.max()) * 1.15
            for direction in ("x", "y"):
                axis(ax, direction, unit, hi)
            ax.fill_between([0, hi], [0, hi], hi, color="#eeeeee", zorder=0)
            ax.plot([0, hi], [0, hi], color="#555555", ls="--", lw=1)
            ax.scatter(a, b, s=15, color=color, alpha=0.38, edgecolors="none", zorder=2)
            ax.set(
                xlim=(-0.00003 if unit == "s" else -0.05, hi),
                ylim=(-0.00003 if unit == "s" else -0.05, hi),
                title=f"{label}\nn = {N:,} images",
                xlabel=f"Target LP ({unit})",
                ylabel=f"Backward bounds ({unit})",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.text(
                0.04,
                0.96,
                f"Higher: {(b > a).sum():,}\nEqual: {(b == a).sum():,}",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
            )
            ax.text(
                0.96,
                0.04,
                f"Lower: {(b < a).sum():,}",
                transform=ax.transAxes,
                ha="right",
                fontsize=11,
            )
            tidy(ax)
        finish(
            fig,
            scatter,
            "Paired runtimes" if unit == "s" else "Paired bound solver calls",
            "Each point compares the same image under both methods. Below the diagonal = lower cost with backward bounds.\nAll 2,000 images, with true zeros retained on symmetric-log axes.",
        )


def main():
    data, observations, first, runs = read_runs()
    summary, specs = calculate(data, observations, first, runs)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_pairs(data, runs)
    write_tables(summary)
    write_metrics(summary)
    plot(data, summary, specs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
