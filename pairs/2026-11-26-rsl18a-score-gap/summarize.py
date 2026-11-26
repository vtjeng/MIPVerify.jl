"""Analyze complete, image-aligned RSL18a pairs without third-party Python packages."""
from pathlib import Path
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from statistics import mean, median, quantiles

METHODS = ("target_lp", "screen")
LABELS = {"target_lp": "Target LP", "screen": "Backward score-gap screen"}
DEFINITIVE = {"certified_no_adversarial_example", "adversarial_example_found_or_best_known"}


def analyze(root):
    records = {}
    for lane in sorted(root.glob("lane-*.csv")):
        with lane.open() as stream:
            for row in csv.DictReader(stream):
                key = (int(row["image"]), row["method"], int(row["observation"]))
                assert key not in records, f"Duplicate observation: {key}"
                assert float(row["seconds"]) > 0, f"Nonpositive duration: {key}"
                records[key] = row
    expected = {(i, m, n) for i in range(1, 101) for m in METHODS for n in (1, 2)}
    assert set(records) == expected, f"Missing: {expected-set(records)}; unexpected: {set(records)-expected}"
    summary = {"rows": len(records), "images": 100}
    alltimes = {}
    for method in METHODS:
        summary[method] = {}
        for observation in (1, 2):
            rows = [records[i, method, observation] for i in range(1, 101)]
            summary[method][observation] = {
                "seconds": sum(float(r["seconds"]) for r in rows),
                "outcomes": dict(Counter(r["outcome"] for r in rows)),
                "statuses": dict(Counter(r["status"] for r in rows)),
                "routes": dict(Counter(r["source"] for r in rows)),
                "bound_calls": sum(int(r["bound_calls"]) for r in rows),
                "main_seconds": sum(float(r["main_seconds"]) for r in rows),
                "unresolved_images": [int(r["image"]) for r in rows if r["outcome"] not in DEFINITIVE],
            }
        alltimes[method] = [mean(float(records[i, method, n]["seconds"]) for n in (1, 2)) for i in range(1, 101)]
    baseline, screen = (alltimes[m] for m in METHODS)
    summary["mean_totals"] = [sum(baseline), sum(screen)]
    summary["reduction_pct"] = 100 * (1-sum(screen)/sum(baseline))
    summary["repeat_variation_pct"] = {
        m: 100*abs(summary[m][1]["seconds"]-summary[m][2]["seconds"])/summary[m][1]["seconds"] for m in METHODS
    }
    ratio = [c/b for b, c in zip(baseline, screen, strict=True)]
    deciles = quantiles(ratio, n=10, method="inclusive")
    summary["ratio_p10_p50_p90"] = [deciles[0], median(ratio), deciles[-1]]
    summary["faster_neutral_slower_1pct"] = [sum(v < .99 for v in ratio), sum(.99 <= v <= 1.01 for v in ratio), sum(v > 1.01 for v in ratio)]
    ranked = sorted(range(100), key=lambda i: baseline[i]-screen[i], reverse=True)
    summary["trimmed_reduction_pct"] = {n: 100*(1-sum(screen[i] for i in ranked[n:])/sum(baseline[i] for i in ranked[n:])) for n in (1, 5, 10)}
    conflicts = []
    for i in range(1, 101):
        outcomes = {records[i, m, n]["outcome"] for m in METHODS for n in (1, 2)}
        if len(outcomes & DEFINITIVE) > 1:
            conflicts.append(i)
    summary["definitive_conflicts"] = conflicts
    transitions = {}
    for field in ("status", "outcome"):
        for n in (1, 2):
            changed = defaultdict(list)
            for i in range(1, 101):
                b, c = (records[i, m, n][field] for m in METHODS)
                if b != c:
                    changed[f"{b} -> {c}"].append(i)
            transitions[f"{field}_observation_{n}"] = dict(changed)
        for m in METHODS:
            changed = defaultdict(list)
            for i in range(1, 101):
                b, c = (records[i, m, n][field] for n in (1, 2))
                if b != c:
                    changed[f"{b} -> {c}"].append(i)
            transitions[f"{field}_repeat_{m}"] = dict(changed)
    summary["transitions"] = transitions
    summary["first_observation_verdict_changes"] = transitions["outcome_observation_1"]
    summary["all_runs_resolved_images"] = [i for i in range(1, 101) if all(records[i, m, n]["outcome"] in DEFINITIVE for m in METHODS for n in (1, 2))]
    def pair(i):
        return dict(image=i+1, baseline=baseline[i], screen=screen[i], saved_seconds=baseline[i]-screen[i], ratio=ratio[i])
    summary["largest_savings"] = [pair(i) for i in ranked[:10]]
    summary["largest_regressions"] = [pair(i) for i in reversed(ranked[-10:]) if screen[i] > baseline[i]]
    summary["slowest_pairs"] = sorted((pair(i) for i in range(100)), key=lambda p: p["screen"], reverse=True)[:10]
    rng = random.Random(20261126)
    bootstrap = []
    for _ in range(10000):
        selected = rng.choices(range(100), k=100)
        bootstrap.append(100*(1-sum(screen[i] for i in selected)/sum(baseline[i] for i in selected)))
    bootstrap.sort()
    summary["paired_image_bootstrap"] = {"seed": 20261126, "resamples": 10000, "percentile_95": [bootstrap[249], bootstrap[9749]], "scope": "Image-mix sensitivity, not run-to-run noise"}
    (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# RSL18a paired results", "", "Times average both observations per image. Statuses and outcomes are aligned by image and observation, never by CSV row position.", "", "| Method | Observation | Seconds | Bound calls | Main solve (s) |", "| :--- | ---: | ---: | ---: | ---: |"]
    for m in METHODS:
        for n in (1, 2):
            r = summary[m][n]
            lines.append(f"| {LABELS[m]} | {n} | {r['seconds']:.3f} | {r['bound_calls']} | {r['main_seconds']:.3f} |")
    for field, plural in (("status", "statuses"), ("outcome", "outcomes")):
        lines += ["", f"## {field.capitalize()} counts", "", "| Method | Observation | Value | Images |", "| :--- | ---: | :--- | ---: |"]
        for m in METHODS:
            for n in (1, 2):
                for value, count in sorted(summary[m][n][plural].items()):
                    lines.append(f"| {LABELS[m]} | {n} | `{value}` | {count} |")
        lines += ["", f"## {field.capitalize()} changes", "", "Cross-method rows compare Target LP to Backward score-gap screen. Repeat rows compare observation 1 to observation 2 within the named method.", "", "| Comparison | Transition | Count | Image IDs |", "| :--- | :--- | ---: | :--- |"]
        for label, changes in transitions.items():
            if not label.startswith(field+"_"):
                continue
            if not changes:
                lines.append(f"| {label} | None | 0 | — |")
            for transition, images in changes.items():
                lines.append(f"| {label} | `{transition}` | {len(images)} | {', '.join(map(str, images))} |")
    lines += ["", f"Opposing definitive outcomes across either observation: {conflicts or 'none'}.", "", "Bootstrap intervals describe image-mix sensitivity. Repeat differences measure run-to-run variation separately."]
    (root/"paired-results.md").write_text("\n".join(lines)+"\n")
    return summary


if __name__ == "__main__":
    result = analyze(Path(sys.argv[1]))
    print(json.dumps({k: result[k] for k in ("rows", "mean_totals", "reduction_pct", "repeat_variation_pct", "definitive_conflicts")}, indent=2))
    if result["definitive_conflicts"]:
        raise SystemExit("Conflicting definitive outcomes require investigation")
