# Paired benchmark performance report template

Fill-in template for the report of a paired before/after benchmark
(`benchmarks/run_pair.sh` → `analysis/analyze_pair.py` → `publish_report.sh`). Copy the skeleton,
replace every `<placeholder>`, delete the `> guidance` blockquotes, and keep the verbatim footnote
lines as-is. The section order is fixed; do not reorder.

Most of the tables and footnotes are emitted verbatim by `analyze_pair.py` into
`improvement_stats.md` — paste those, then write the prose (title, Summary, plot captions) by hand.
Optional sections and optional clauses carry a one-line include condition; drop them when the
condition does not hold.

## Two renderings

The same body is published twice: as `pairs/<slug>/report.md` next to the plots on the
`benchmark-reports` branch, and as the PR comment. Write `report.md` first and publish it — the
comment's pinned image URLs need the publish commit's SHA, which exists only once
`publish_report.sh` has run — then derive the comment. The two differ only in delivery:

|  | `report.md` (branch) | PR comment |
|---|---|---|
| Title | H1 | H2 (as in this skeleton) |
| Plot images | relative `plots/<name>.png` | absolute pinned-SHA URLs (C7) |
| Raw-data pointers | relative `baseline/` / `candidate/` | link to the published `pairs/<slug>/` folder |
| Reproduction | trailing `## Reproduce` section with the analyze command | folded into the preamble command bullet |

Everything else — section order, tables, footnotes, captions, conventions — is identical in both.

---

## Performance report: <network>, <tightening> tightening, <N> samples

> Title heading is H2 — there is no H1 in the comment. Encode the three benchmark parameters:
> network family (e.g. `WK17a`), tightening mode (e.g. `LP`), sample count (e.g. `500`).
>
> Everything from here to the `---` is the preamble (no sub-heading). Include, in order:
>
> 1. One sentence stating what the PR changes and its expected effect on runtime.
> 2. One methodology paragraph: samples `<1:N>` of the MNIST test set are verified against
>    `MNIST.<network>_linf0.1_authors` on the master and feature commits under identical settings;
>    the ratio distributions, scatter plots, and outcome-flip tables compare each sample's
>    candidate run with its own baseline run, while the absolute-runtime distributions summarize
>    each side separately; and a link to the published `pairs/<slug>/` folder, listing what it
>    holds (raw per-sample CSVs, dependency snapshots, stats, plots, and any extras such as a
>    same-commit control) — either on the `benchmark-reports` branch or pinned to the report
>    commit SHA (the pinned form is stale-proof, matching C7's image URLs).
> 3. Baseline/candidate bullet (see convention C5 below).
> 4. Reproduction bullet: the exact command
>    `benchmarks/benchmark_wk17a_first100.jl --samples <1:N> --tightening <lp> --main-time-limit <120> --norm-order <Inf>`,
>    Julia `<version>` (read from the run log) single-threaded, HiGHS named and glossed as "an open-source LP/MIP solver",
>    sequential runs on a local WSL2 workstation, the dependency provenance (either the
>    dependency-snapshot hash or the phrase "identical dependency snapshots"), any machine-contention
>    caveat that held during the run (optional — e.g. unrelated background jobs holding ~N cores
>    across both sides), and the caveat that absolute times are not comparable to the CI-hosted
>    `benchmark-results` series (C4).
> 5. Report-scope note (optional): one standalone italic sentence when the report needs a scope
>    caveat the four elements above don't carry — e.g. that the paired-analysis tooling post-dates
>    the merge, so the report is retroactive. Drop it when no such caveat applies. This is distinct
>    from the inline commit-identity qualifiers in C5, which live on the baseline/candidate bullet.

- **Baseline** `<branch-or-label>` `<sha>`<optional provenance, e.g. "(post-#209 master, this PR's base)">; **candidate** `<branch-or-label — optional>` `<sha>`<optional provenance, e.g. "(benchmarked commit; `src/` byte-identical to head `<sha>`)">.
- Command: `benchmarks/benchmark_wk17a_first100.jl --samples <1:N> --tightening <lp> --main-time-limit <120> --norm-order <Inf>`. Julia `<version>`, single-threaded; HiGHS (an open-source LP/MIP solver) for all solves; sequential runs on a local WSL2 workstation<optional: , with unrelated background jobs holding ~<3> cores throughout both sides>, <dependency-snapshot hash `<sha256>` | identical dependency snapshots>. Absolute times are not comparable to the CI-hosted `benchmark-results` series.

<optional italic report-scope note, e.g. *The paired-analysis tooling post-dates the merge, which is why this report is retroactive.*>

---

## Summary

> One bullet per finding, most important first. Each bullet leads with the headline metric and its
> `baseline → candidate` values, the absolute saving/cost, and a caveat. Standard beats to cover:
>
> - Aggregate total vs. median contrast (e.g. total `+29.8%` aggregate but `+45.8%` median), with
>   the reason the two differ for this change — which samples dominate the aggregate, and in which
>   direction relative to the median (e.g., large main-solve-bound samples that a bound-tightening
>   change barely touches dilute the aggregate below the median).
> - Which phase the change lands in (Build + bound tightening vs. Main solve), with `% improved` /
>   `% regressed`.
> - Name each phase's role — the affected phase(s) the change targets and any unaffected phase(s),
>   whichever they are. For an unaffected phase, state its near-1.00 median and attribute the small
>   aggregate wobble to the top-10 movers near the `<120>` s limit (noise). Apply that only to a
>   phase the change does not target; e.g., when the change tunes the final solve itself, Main solve
>   is the affected phase and its aggregate change is the signal, not noise.
> - Instrumentation-count change when present (e.g. bound solver calls `427,076 → 99,067`, 4.3×).
> - Outcome/status flips near the time limit: net semantic change (favorable/regressed/none), how
>   many samples flipped solve status, and whether flips changed any verdict.
>
> Keep the median-vs-pooled framing consistent with the Aggregate table below.

- <finding: metric, `baseline → candidate`, saving/cost, caveat>
- <finding>
- <finding …>

## Detailed statistics

> Container heading only — no prose directly under it. `### Plots` is always the first subsection.

### Plots

> Prose before the images, never after — the reader gets the shape before the numbers. Preferred:
> one sentence per plot immediately above that plot, describing the visual reading (direction and
> magnitude of the shift, position relative to the `y = x` diagonal), not restating table numbers.
> A single lead caption covering all plots is acceptable when they tell one story (as in the #222
> report); when two plots share a caption, place it above the first and let the second follow.
>
> Embed each PNG pinned to the publish commit's SHA so it can't be served stale:
> `https://raw.githubusercontent.com/vtjeng/MIPVerify.jl/<publish-sha>/pairs/<slug>/plots/<name>.png`
> Alt text names the plot. Two styles are in use — pick one and keep it consistent within a report:
> the descriptive name (`Paired ratio distributions`) or the file stem (`ratio ECDF`). Canonical
> files, with both alt-text forms, in order:
>
> - `ratio_ecdf.png` — `Paired ratio distributions` / `ratio ECDF`
> - `absolute_runtime_ecdf.png` — `Absolute runtime distributions` / `absolute runtime ECDF`
> - `magnitude_scatter.png` — `Paired runtime scatter` / `magnitude scatter`
> - `absolute_calls_ecdf.png` — `Absolute bound-call distributions` / `calls ECDF`  *(optional; calls series only)*
> - `calls_scatter.png` — `Paired bound-call scatter` / `calls scatter`  *(optional; calls series only)*
>
> The two `*_calls*` plots appear only when both runs carry the solver-call instrumentation columns.

<one-sentence caption reading the plot below>

![Paired ratio distributions](https://raw.githubusercontent.com/vtjeng/MIPVerify.jl/<publish-sha>/pairs/<slug>/plots/ratio_ecdf.png)

<caption>

![Absolute runtime distributions](https://raw.githubusercontent.com/vtjeng/MIPVerify.jl/<publish-sha>/pairs/<slug>/plots/absolute_runtime_ecdf.png)

<caption>

![Paired runtime scatter](https://raw.githubusercontent.com/vtjeng/MIPVerify.jl/<publish-sha>/pairs/<slug>/plots/magnitude_scatter.png)

<!-- optional, only when the solver-call series exists -->
<caption>

![Absolute bound-call distributions](https://raw.githubusercontent.com/vtjeng/MIPVerify.jl/<publish-sha>/pairs/<slug>/plots/absolute_calls_ecdf.png)

![Paired bound-call scatter](https://raw.githubusercontent.com/vtjeng/MIPVerify.jl/<publish-sha>/pairs/<slug>/plots/calls_scatter.png)

### Per-sample ratio distribution

> Paste the table `analyze_pair.py` emits. Columns and alignment are fixed. Series column
> left-aligned, every numeric column right-aligned. One row per active series; each series name is
> reused verbatim everywhere. `n` is the same for every row (the exclusion count, C3). `improved` /
> `regressed` are whole-percent shares. Include the `Bound solver calls` row only when the
> instrumentation columns exist.

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | <n> | <min> | <p10> | <p25> | <median> | <p75> | <p90> | <max> | <x>% | <x>% |
| Main solve time | <n> | … | … | … | … | … | … | … | <x>% | <x>% |
| Total end-to-end time | <n> | … | … | … | … | … | … | … | <x>% | <x>% |
| Bound solver calls | <n> | … | … | … | … | … | … | … | <x>% | <x>% |

- `ratio` = candidate ÷ baseline, < 1 = candidate faster; `improved` counts ratio below 0.99, `regressed` counts ratio above 1.01, samples within the ±1% band count as unchanged.<optional, when whole-percent rounding hides a visible outlier — a `max` above 1.01 next to `regressed` 0% (or below 0.99 next to `improved` 0%): " and shares round to whole percent (<series>'s lone regression, sample <id>, the `max` entry, rounds to 0%)">
- `build` = constructing the MIP model; `tightening` = the `<tightening>` bound-tightening pass; `main solve` = the final verification MIP.
- `total` = `build` + `tightening` + `main solve`.
- `bound solver calls` = count of HiGHS bound-tightening solves<optional parenthetical when the count moves: ", the count this PR changes" — match it to the observed direction, increase or decrease; drop it when the count is flat, as in #222>.  <!-- keep only with the calls series -->
- <n> = <N> samples minus the <k> `SKIPPED_PREDICTED_IN_TARGETED` samples, which carry no timing and are excluded from every series (see the status table below); this covers both the distribution and aggregate tables.

### Aggregate saving and concentration

> Paste the emitted table. Same series rows as above, same alignment. Cells carry units inline
> (`s`, `calls`); `net saved` cells print an explicit `+`/`−` sign; `pooled ratio` to 2 decimals;
> `top-10 concentration` a whole-percent share.

| series | baseline | candidate | net saved | pooled ratio | top-10 concentration |
|---|--:|--:|--:|--:|--:|
| Build + bound tightening | <b> s | <c> s | <±net> s | <ratio> | <x>% |
| Main solve time | <b> s | <c> s | <±net> s | <ratio> | <x>% |
| Total end-to-end time | <b> s | <c> s | <±net> s | <ratio> | <x>% |
| Bound solver calls | <b> calls | <c> calls | <±net> calls | <ratio> | <x>% |

- `net saved` = baseline − candidate total; positive = candidate cheaper.
- `pooled ratio` = candidate total ÷ baseline total (aggregate counterpart to the per-sample median).
- `top-10 concentration` = the 10 samples with the largest absolute change account for this share of the total absolute per-sample change (0–100%; higher = a few samples dominate).<optional, when a series has zero total change — e.g. equal bound-solver call counts on both sides: " <Series> made <N> calls on both sides, so their total absolute per-sample change is 0 and their concentration is reported as 0%.">

> Optional prose para: decompose the top-10 concentration for the end-to-end series (how much of the
> summed absolute per-input change the 10 largest movers hold, and whether gains and losses cancel).
> Optional paired-input bootstrap interval, stated as sensitivity to the fixed input mix — not a
> machine-to-machine or run-to-run confidence interval (C-bootstrap). When Component timings follow,
> the sentence that bridges into them may close this section as its trailing transition paragraph
> (as in the #222 report) instead of sitting under the `#### Component timings` heading — see that
> section's guidance.

#### Component timings

> Optional — include only when phase-decomposition timers were collected (formulation / final-solve /
> bound-tightening subtotals from the analyzer's dedicated timers). Bridge sentence first — or, as in
> the #222 report, as the trailing paragraph of the Aggregate section above: these timers split
> summed end-to-end time into formulation and final-solve phases, whereas the Build + bound
> tightening / Main solve rows above use a total-minus-solve split, so subtotals don't match
> one-for-one (both decompositions cover the full end-to-end time). Then, directly under this
> heading: "Rows sharing a parent add to that subtotal before rounding." Hierarchical table with
> `↳` / `↳↳` / `↳↳↳` indentation showing nesting. Bold only the subtotal/parent rows (those with
> children); leave leaf rows unbolded — bold marks a subtotal. Times to one decimal, space before
> `s`, comma thousands separators. After the table, add interpretive prose when the decomposition
> needs explaining — where certificate work lands, why two rows both fell (as in the #222 report).

| component | parent subtotal | baseline | candidate | saved | change |
|---|---|--:|--:|--:|--:|
| **Summed end-to-end sample time** | — | <b> s | <c> s | <saved> s | <x>% |
| ↳ **Formulation subtotal** | Summed sample time | … | … | … | … |
| ↳↳ **Bound-tightening subtotal** | Formulation | … | … | … | … |
| ↳↳↳ HiGHS bound-solver wall time | Bound tightening | … | … | … | … |
| ↳↳↳ Non-solver work within bound tightening | Bound tightening | … | … | … | … |
| ↳↳ Other formulation work | Formulation | … | … | … | … |
| ↳ Final-solve wall time | Summed sample time | … | … | … | … |
| ↳ Other work inside the sample call | Summed sample time | … | … | … | … |

- `Non-solver work within bound tightening` = bound-tightening time minus HiGHS wall time (interval propagation, certificate handling, bound-loop work for ReLU-scoped bounds).
- `Other formulation work` = formulation time minus bound-tightening time (target/objective construction, optimizer setup, non-solver work for bounds outside a ReLU layer).
- `Other work inside the sample call` = sample time minus formulation and final-solve wall time.

<optional interpretive prose: where certificate work lands across the rows, why the moved rows moved, and any measurement caveat>

#### Related timing views

> Optional — pairs with Component timings. Lead: "These rows use separate or overlapping timing
> views; do not add them to the table above." The `relationship` column states each row as a formula.
> After the table, add interpretive prose when it connects the result to an issue or design point
> (as with the issue #211 tie-in in the #222 report).

| measurement | relationship | baseline | candidate | saved | change |
|---|---|--:|--:|--:|--:|
| Whole-run elapsed | Summed sample time + benchmark-loop overhead | … | … | … | … |
| Formulation excluding bound-solver time | Non-solver work within bound tightening + Other formulation work | … | … | … | … |

<optional interpretive prose, e.g. an issue tie-in>

### Solve status and verdict flips

> Status-count table first (one row per status present, sorted; counts right-aligned), then a one-
> sentence prose line stating how many samples changed solve status, how many changed semantic
> outcome, and whether the two sets overlap. Then a plain `Solve status:` label above the solve-
> status transition table, then a plain `Semantic outcome:` label above the semantic-outcome
> transition table. Both transition tables share the columns `transition | n | samples`; transitions
> use the `` `A` → `B` `` form and list every affected sample index. (`analyze_pair.py` emits these
> under `### Solve status (all samples)` / `### Verdict flips — …`; consolidate them under this one
> H3 with the two label lines.) Drop the `Semantic outcome:` table when no semantic outcome changed
> (write `_None._`).

| status | <baseline-label> `<short-sha>` | <candidate-label> `<short-sha>` |
|---|--:|--:|
| INFEASIBLE | <b> | <c> |
| OPTIMAL | <b> | <c> |
| SKIPPED_PREDICTED_IN_TARGETED | <b> | <c> |
| TIME_LIMIT | <b> | <c> |

<one sentence: N samples changed solve status; M changed semantic outcome; overlap/disjoint note.>

Solve status:

| transition | n | samples |
|---|--:|---|
| `<A>` → `<B>` | <n> | <ids> |

Semantic outcome:

| transition | n | samples |
|---|--:|---|
| `<A>` → `<B>` | <n> | <ids> |

#### Model and outcome audit

> Optional — include when you ran the paired-identical-field check and/or a same-commit control
> (typically when objective values differ between sides and you need to show it is fresh-process
> solver-path variation, not a semantic regression from the change). Lead: "The paired raw rows have
> identical values for:" then a bullet list of the fields that matched (dependency snapshot and
> benchmark args; variable / constraint counts; ReLU classification counts; bound requests, solver
> calls, statuses, skips, barrier iterations, nodes). Name specific iteration kinds — barrier vs.
> simplex — rather than a generic "iterations", since they can diverge and be classified wrongly.
> When some fields differed, follow the matched list with a differed-fields sentence: name each
> differing field with its magnitude and note the two sides ran in fresh Julia/HiGHS processes (e.g.
> bound simplex iterations differed by 834 of ~2.72M; final MIP search paths differed). Then the
> objective-agreement audit: how many inputs had objectives on both sides, how many agreed within
> `1e-6`, any large optimal/optimal outlier, and a same-commit control that reran the outlier(s) in
> fresh processes to attribute the gap to solver variation rather than the change.

---

## Report-wide conventions

- **C1 — Ratio direction.** `ratio` = candidate ÷ baseline; < 1 = candidate faster. `pooled ratio` = candidate total ÷ baseline total (the aggregate counterpart to the per-sample median). Ratios are always candidate-over-baseline.
- **C2 — ±1% unchanged band.** `improved` counts samples with ratio below 0.99, `regressed` counts ratio above 1.01; samples inside the ±1% band count as unchanged. Shares are whole-percent, so a lone outlier can round into a 0% share — note it in the per-sample footnote when the `max`/`min` makes it visible.
- **C3 — n exclusion.** Every distribution and aggregate series covers `n = <N> − <k>` = the samples minus the `<k>` `SKIPPED_PREDICTED_IN_TARGETED` inputs, which carry no timing.
- **C4 — Absolute-times caveat.** Local sequential WSL2 workstation run; absolute times are not comparable to the CI-hosted `benchmark-results` series. Disclose any machine contention (concurrent load) that held during the run in the reproduction bullet.
- **C5 — Baseline/candidate identification.** Bold **Baseline** and **candidate** labels; commits as `` `<branch-or-label>` `<sha>` ``. The branch or label is optional — drop it when the benchmarked commit has none (e.g. a detached tip) — and the SHA may be short or full. Add a provenance note where it matters (base PR, byte-identical `src/`, current head not re-benchmarked).
- **C6 — Series names verbatim.** `Build + bound tightening`, `Main solve time`, `Total end-to-end time`, `Bound solver calls` — reused unchanged across every table and the prose.
- **C7 — Pinned image URLs.** `raw.githubusercontent.com/vtjeng/MIPVerify.jl/<publish-sha>/pairs/<slug>/plots/<name>.png`, pinned to the publish commit's SHA; alt text names the plot — the descriptive name or the file stem, one style per report. Prefer the same publish-SHA pinning for the methodology folder link.
- **C8 — Captions before plots.** A prose sentence reads each plot immediately above it (never after).
- **C9 — Backticked identifiers.** Column/metric names, solve statuses, and semantic-outcome labels are code spans; transitions use `` `A` → `B` ``.
- **C10 — Table alignment.** Leading label column left-aligned; every numeric/count column right-aligned (`--:`). Units inline in cells; `net saved` carries an explicit sign; negatives use a Unicode minus. In the Component timings table, bold marks subtotal/parent rows only.