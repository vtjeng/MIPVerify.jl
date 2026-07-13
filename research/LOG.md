# Cascade-aware verification research log

Record decisions, experiment definitions, results, and unresolved questions here. Do not add routine
progress notes. Link raw result artifacts instead of copying large tables into this file.

## 2026-07-10: initial research framing

Context:

The original MIPVerify work showed that proving rectified linear units (ReLUs) stable removes binary
variables and that linear programming (LP) bound tightening can repay its cost. Later dependency,
implication-graph, indirect-effect, and lookahead methods address parts of the proposed
cascade-aware branching idea.

Decision:

Study joint small-set selection with global conditional retightening and mixed-integer programming
(MIP) work-aware scoring. Treat BaBSR, Filtered Smart Branching, dependency-based branching, Bound
Implication Graph, and lookahead phase-fixing scores as baselines rather than claiming cascade-aware
branching itself as a new idea.

Rationale:

The remaining question is whether selecting nonredundant phase decisions and accounting for
constraint-induced upstream effects improves a full MIP workflow after all overhead is charged.

Open questions:

- How common are useful two-sided cascades on networks that require nontrivial branching?
- Does stable-ReLU count predict HiGHS work, or must the score weight objective influence and model
  footprint?
- Are multi-ReLU effects materially stronger than the union of pairwise implications?

## 2026-07-10: external branching boundary

Decision:

Implement the first branching harness outside HiGHS. After selecting and fixing phases, pass each
feasible child verification problem to HiGHS.

Rationale:

This gives direct control over the branch choice, makes domain coverage explicit, and permits exact
accounting of probing, rebuilding, and child-solve costs. It does not depend on a custom HiGHS
branching callback.

Consequences:

- The harness must combine child results soundly.
- Parallel child execution is an optional scheduling optimization and must not hide total compute.
- Solver-default runs remain the primary control.

## 2026-07-10: progressive bounds tightening

Decision:

Retain progressive bounds tightening in the root and every conditional child. Use interval
arithmetic first and invoke more expensive optimization only while the bound can still change the
piecewise-linear formulation.

Rationale:

Running a strong bounder indiscriminately would discard a central performance result from the
original work and could make cascade probing more expensive than the benefit being measured.

Consequences:

- Instrumentation must attribute calls and time to each tightening method.
- Branching methods must use identical tightening policies unless the tightening policy is the
  explicit ablation under study.

## 2026-07-10: baseline before heuristic development

Decision:

Build a pinned, instrumented HiGHS baseline before implementing a cascade score.

Rationale:

The nightly benchmark is useful historical evidence but floats dependencies and runner hardware. It
also lacks bound-solve counts, ReLU stability, model size, and MIP node information, and its
aggregate outcome omits already-misclassified inputs.

Initial benchmark facts:

- The archived 2026-07-10 WK17a run used 500 inputs and LP tightening.
- It took about 3,518 seconds wall time, including about 1,433 seconds in main solves.
- Median main-solve time was about 0.033 seconds; the hard tail dominates solver time.
- WK17a is a regression suite, but harder networks and samples are required for branching research.

## 2026-07-10: first instrumented LP smoke test

Experiment:

Run WK17a sample 1 with one thread, LP tightening, a 20-second limit per bound solve, and a
120-second main-solve limit. Use Julia 1.12.6 and the dependency manifest archived with the
2026-07-10 benchmark: HiGHS.jl 1.24.1, HiGHS 1.15.1, JuMP 1.30.1, and MathOptInterface 1.51.1. The
run used a local isolated copy; it did not change repository manifests. Run an interval arithmetic
control under the same main-solve settings.

Result:

- HiGHS certified the property infeasible, matching the archived result.
- Progressive tightening issued 9,626 logical bound requests but only 842 solver calls. An upper
  bound fixed the inactive phase early enough to skip 3,186 lower-bound solves.
- Of 4,804 ReLUs, 4,689 were stable and 115 remained split. The final model had 117 binary
  variables.
- Interval arithmetic left 131 ReLUs split and produced 139 binary variables. It took 14.486
  seconds, including 11.585 seconds of formulation and 0.784 seconds around the main solve. LP
  tightening reduced the main solve to 0.137 seconds but took 18.982 seconds end to end, including
  17.385 seconds of formulation, on this easy instance. HiGHS reported 0.041 seconds for the LP main
  solve.
- A paired LP call with statistics disabled had the same outcome and took 19.352 seconds; HiGHS
  reported 0.045 seconds for its main solve. The reversal in total time is evidence that a single
  process pair cannot isolate collection overhead from compilation and host variation.

Decision:

Keep statistics opt-in and charge their cost when enabled. Use repeated, order-balanced
no-statistics controls when measuring baseline variation so collection overhead can be estimated.
This single smoke test validates the fields and progressive-call accounting; it is not a performance
comparison.

## 2026-07-10: benchmark outcome schema

Decision:

Count an already-misclassified input as a zero-distance adversarial example when model construction
is skipped. Add benchmark and semantic-outcome schema versions to metrics and nightly tracking.
Treat historical files without explicit versions as schema 1 and the corrected output as schema 2.
Reject cross-schema comparisons and fail the comparison gate when semantic partition counts differ.

Rationale:

The archived 500-sample aggregate omitted eight skipped inputs from every semantic outcome. Reusing
the adversarial-count column without a schema marker would make old and new tracking rows appear
comparable when their counting rules differ.

## 2026-07-11: certify LP-derived bounds

Problem:

The bound-tightening code used the primal objective value as a lower bound after minimization and an
upper bound after maximization. Those directions are unsafe. HiGHS' reported objective bound was
within one unit in the last place of the primal value on the observed case and retained a
solve-order-dependent `6.59e-9` discrepancy, so changing the queried attribute was insufficient.

Decision:

Treat row duals as candidate Lagrange multipliers. Project inequality multipliers onto their dual
cones, form the affine Lagrangian, and minimize every stationarity residual over the variables'
declared bounds. Evaluate the calculation with outward interval arithmetic and clamp the result to
the interval-arithmetic bound. This requires no fixed tolerance or decimal grid. If no finite
certificate is available, retain the interval bound.

MIP tightening now uses the solver objective bound instead of the incumbent objective. JuMP does not
expose a complete HiGHS branch-and-bound proof, so MIP-strength bounds remain a solver trust
boundary.

Experiment:

Use the pinned 2026-07-10 environment with one thread, LP tightening, a 20-second limit per bound
solve, and a 120-second main-solve limit. Compare the existing baseline and the certified-bound
implementation on WK17a samples 1 and 9.

Result:

- Sample 1 retained all ReLU phase counts, model dimensions, and the infeasible verification result.
  Total time rose from 18.982 to 23.134 seconds. ReLU bound-phase time rose from 8.470 to 11.685
  seconds across 842 LP solves.
- Sample 9 also retained phase counts, model dimensions, and its adversarial outcome. The certified
  model changed 360 numeric fields relative to the default-tolerance baseline; its largest outward
  change was `8.36e-9`. Total time rose from 35.422 to 81.585 seconds, and the main solve rose from
  28.372 seconds and one node to 56.929 seconds and 425 nodes.

Consequence:

Treat this as a correctness fix with an expected performance regression. Measure interval and LP
screening optimizations against this corrected baseline rather than preserving optimistic
tolerance-level coefficients.

## 2026-07-11: progressive ReLU tightening

Decision:

For each ReLU layer, compute interval upper bounds and then the needed interval lower bounds before
starting an optimization solve. Send only interval-ambiguous units to the LP relaxation. When MIP
tightening is requested, send only LP-ambiguous units to MIP. Compute all LP bounds under one
temporary integrality relaxation and pass each certified LP bound to MIP as its fallback bound.

Record bound-solver statistics under the algorithm that performed each solve. A layer requested with
MIP tightening can therefore have separate LP and MIP rows.

Mechanism:

Skipping an upper-bound solve does not make a ReLU linear. Its interval lower bound has already
proved it active, so the omitted upper bound would not affect the formulation. The solve can still
affect later bounds through HiGHS' simplex basis: each LP solve ends at a feasible corner, and HiGHS
can reuse that corner when the objective changes for the next unit. Skipping a solve therefore
changes the next warm start without changing the mathematical LP. In exact arithmetic, different
starting bases give the same optimal bound. With degeneracy and feasibility tolerances, they can
produce different approximate dual solutions. Both resulting certificates are valid, but they need
not be numerically identical. The first differing bound for an ambiguous unit then changes a big-M
coefficient, so later mathematical models can also differ.

Experiment:

Use the pinned 2026-07-10 environment with one thread and LP tightening. Compare the certified
upper-first baseline with progressive tightening on WK17a samples 1 and 9.

Result:

- Sample 1 retained its ReLU phases, model dimensions, and infeasible result. LP calls fell from 842
  to 197 and simplex iterations fell from 11,750 to about 6,060. Across two runs per variant, the
  ReLU bounds phase fell from 11.685–12.994 to 3.986–4.023 seconds. Total time fell from
  23.134–24.536 to 18.813–19.162 seconds.
- Sample 9 retained its ReLU phases, model dimensions, and adversarial outcome. LP calls fell from
  888 to 204, simplex iterations fell from 13,169 to 5,293, and the ReLU bounds phase fell from
  13.519 to 3.842–4.145 seconds. Two progressive runs took 51.751 and 69.271 seconds. Their main
  solves varied from one node and 32.674 seconds to 430 nodes and 50.324 seconds, despite identical
  bound-call and bound-iteration counts. The corrected upper-first run took 81.585 seconds, 425
  nodes, and 56.929 main-solve seconds. All runs give the same semantic result, but their final
  objective values and bounds differ within the solver's termination gap.

Consequence:

Progressive screening recovers the expected solver-call reduction against the corrected baseline.
Its end-to-end gain is smaller than its bounds-phase gain, and sample 9 remains sensitive to small
coefficient changes. Run a broader paired benchmark before drawing an aggregate performance
conclusion.

## 2026-07-11: controlled 500-sample cost decomposition

Experiment:

Run WK17a samples 1–500 under three revisions: unsafe upper-first (`a741e92`), certified upper-first
(`d029b74`), and certified progressive (`2aefae1`). Use the same dependency manifest, Julia 1.12.6,
HiGHS.jl 1.22.2 with HiGHS 1.13.1, one Julia thread, LP tightening, a 20-second limit per bound
solve, and a 120-second main-solve limit. Use the schema 3 benchmark driver for all three runs. The
execution order was certified upper-first, certified progressive, then unsafe upper-first.

Result:

| Variant               |  Wall time | Formulation | Bound phase | Main solve | LP calls |
| --------------------- | ---------: | ----------: | ----------: | ---------: | -------: |
| Unsafe upper-first    | 3688.976 s |  2269.469 s |  2013.126 s | 1413.824 s |  427,076 |
| Certified upper-first | 4789.044 s |  3321.324 s |  2991.525 s | 1460.426 s |  427,076 |
| Certified progressive | 2209.804 s |   735.053 s |   397.524 s | 1468.340 s |   99,067 |

- The correctness fix added 1100.068 seconds, or 29.82%, to the upper-first run. The bound phase
  added 978.399 seconds. Time inside bound-solver calls added only 23.201 seconds; ReLU-scoped
  non-solver bound work added 955.199 seconds. This is the best available proxy for certificate
  overhead, but it also includes loop and setup work.
- Progressive tightening saved 2579.240 seconds, or 53.86%, relative to certified upper-first. It
  was 1479.172 seconds, or 40.10%, faster than unsafe upper-first, so screening more than recovered
  the correctness cost.
- LP calls fell by 328,009, or 76.80%. This exactly equals the new
  `upper_skipped_by_nonnegative_interval_lower` count. Bound simplex iterations fell by 53.74%.
  Layer 2 accounted for 324,047 removed calls and layer 3 for 3,962.
- Bound `OTHER_ERROR` statuses fell from 17 in both upper-first runs to one in the progressive run.
  Every upper-first error was a layer-2 upper solve; all used the interval fallback.
- All 500 semantic outcomes matched: 476 certified, 23 adversarial or best-known, and one
  unresolved. The same eight inputs were already misclassified. ReLU phase counts and model
  dimensions matched for every formulated sample. Main statuses and runtimes changed on a small set
  of samples, consistent with the observed branch-path sensitivity.
- Sample 9 took 62.503, 38.941, and 29.570 seconds across the three variants. Its main solve took
  56.923, 30.103, and 28.050 seconds, with 9, 17, and 11 nodes. This direction differs from earlier
  isolated runs and reinforces the need for branch-prefix replay instead of a timing-only account.

Consequence:

Use certified progressive tightening as the baseline for cascade-aware branching. Treat the large
bound-work reduction as stable because it follows exact call accounting. Do not interpret the small
aggregate main-solve differences without repeated or branch-replay experiments; the runs were
adjacent but not randomized, and the benchmark does not record branch traces or model coefficients.
