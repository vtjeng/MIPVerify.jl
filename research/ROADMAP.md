# Cascade-aware verification roadmap

## Objective

Determine whether branching on a small set of unstable rectified linear units (ReLUs), followed by
conditional bound tightening, reduces end-to-end verification cost with HiGHS. The method must
remain sound and complete, and all probing, model construction, and child solves count toward its
cost.

The main hypothesis is that a small, nonredundant set of phase decisions can stabilize other ReLUs
through both forward propagation and global constraints on the feasible input set. A useful
heuristic must find those decisions more cheaply than the solver work they avoid.

## Working principles

- Keep progressive bounds tightening. Compute interval upper and lower bounds first, then use linear
  programming (LP), and use mixed-integer programming (MIP) only for units that remain
  phase-ambiguous.
- Orchestrate the first branching prototype outside HiGHS. Each feasible child is a complete
  verification problem passed to HiGHS.
- Preserve exact coverage of the parent domain. A robust result requires every feasible child to be
  certified; an adversarial witness from any child is sufficient to refute robustness.
- Compare methods by end-to-end time and total compute, including heuristic overhead.
- Pin code, dependencies, data, solver settings, threads, random seeds, and hardware for controlled
  comparisons.
- Use the research log for decisions and experiment results, not routine progress updates.

## Phase 1: controlled baseline and instrumentation

Status: active.

Work:

- Pin a reproducible Julia and HiGHS environment from an archived benchmark run.
- Separate compilation, interval propagation, optimization-based bound tightening, formulation, and
  main-solve time.
- Record bound-solve counts and statuses, stable and unstable ReLU counts by layer, model size,
  eliminated labels, root relaxation information, branch-and-bound nodes, iterations, gap, and
  memory where the solver exposes them.
- Validate LP bounds from row duals and variable boxes with outward interval arithmetic. Treat MIP
  objective bounds as a separate solver trust boundary until a branch-and-bound proof is available.
- If LP certificate evaluation remains material after screening, test a cache scoped to one ReLU
  layer's temporary LP relaxation. Do not reuse cached rows or variable boxes after model changes.
- Compare final model coefficients across equivalent tightening orders and solver tolerances.
- Diagnose WK17a sample 9 by capturing the root relaxation and early branch decisions for the
  optimistic and certified models. Replay the old first branch and a short branch prefix on the
  certified model; if the tree remains large, compare the corresponding child LPs, bases, and
  pruning decisions.
- Count inputs that are already misclassified in the aggregate benchmark outcome.
- Verify semantic agreement with the existing 500-sample WK17a benchmark.
- Build a discovery suite containing hard WK17a samples and the local `MNIST.n1` and
  `MNIST.RSL18a_linf0.1_authors` networks.

Exit criteria:

- Repeated baseline runs are semantically identical.
- LP-derived bounds do not depend on a hardcoded numerical allowance and do not cut off known
  feasible points.
- Timing and solver-work fields are populated and documented.
- The discovery suite contains enough instances with nontrivial main-solver branching to evaluate a
  branching heuristic.

## Phase 2: single-ReLU cascade oracle

For each candidate unstable ReLU and each phase:

1. Add the active or inactive phase constraint.
2. Apply progressive conditional bounds tightening.
3. Record infeasible children, tightened widths, newly stable ReLUs, eliminated labels, relaxation
   improvement, model-size reduction, and probe cost.
4. Separate dataflow-downstream effects from constraint-induced same-layer and upstream effects.
5. Solve selected child MIPs to measure whether the structural gains reduce actual solver work.

Start exhaustively on small networks. On larger networks, preselect candidates with cheap scores and
probe only a fixed shortlist.

Exit criteria:

- Quantify how often useful cascades occur in both child phases.
- Establish which oracle measurements predict node count and end-to-end time.
- Stop this direction if conditional stability is rare or its computation consistently costs more
  than the solver work it removes.

## Phase 3: branching baselines

Implement comparable scores in the external branching harness:

- solver default with no external split;
- random unstable ReLU;
- interval width, polarity, and fixed layer orders;
- BaBSR-style relaxation score;
- Filtered Smart Branching-style shortlist and child evaluation;
- dependency-graph or bound-implication degree;
- depth-one and depth-two lookahead using balanced phase fixes or bound improvement;
- an oracle ranking for small instances.

Use the same child bounder, HiGHS settings, and total time budget for every method.

## Phase 4: joint small-set selection

Select a set of depth one to three using marginal cascade gain across the current child frontier.
Penalize:

- overlap between nodes stabilized by different decisions;
- an unbalanced child with little simplification;
- extra feasible leaves;
- probing and rebuilding cost.

Test greedy selection first. Use a small beam search to detect pair-only effects that have no useful
single-ReLU implication.

Compare three ways to consume conditional information:

1. solve the externally branched child MIPs;
2. add proven phase implications or conditional bounds to one parent MIP;
3. merge child bounds into valid parent bounds and remove any ReLU proven stable across the full
   disjunction.

## Phase 5: controlled evaluation

Primary measures:

- end-to-end wall and processor time;
- solved count and penalized runtime at a fixed timeout;
- total branch-and-bound nodes and LP iterations;
- heuristic overhead and peak memory.

Diagnostic measures:

- new stable ReLUs by layer and direction;
- binaries and constraints removed;
- root relaxation and final optimality gap;
- feasible and infeasible child count;
- cascade overlap and child balance;
- bound-tightening calls, statuses, and time.

Use paired sample-level comparisons, repeated controls, cactus plots, tail percentiles, and a
shifted geometric mean. Promote a method to the full suite only when its gain exceeds controlled
baseline variation or it solves additional instances at the same budget.

## Phase 6: scale and generalize

- Confirm successful methods on all 500 historical WK17a samples, then the full test set.
- Recover or convert a difficult same-architecture network such as the adversarially trained CNNA
  used in the original paper.
- Add shared Open Neural Network Exchange (ONNX) models and Verification of Neural Networks Library
  (VNN-LIB) properties when MIPVerify can express them faithfully.
- Compare heuristic behavior with current complete verifiers; avoid raw runtime comparisons across
  different hardware and bounding backends.

## Compute plan

The initial controlled baseline and a 30–50-instance oracle pilot should run on local processor
hardware. Use low parallelism until peak memory is measured. Request cloud compute before a broad
heuristic-by-ablation sweep, full-test-set confirmation, or experiments requiring hundreds of CPU
hours.

After progressive tightening is implemented, run a paired 500-sample benchmark against the certified
upper-first baseline. Use a short tranche to estimate local runtime before starting the full pair,
and request cloud capacity if the projected run is impractical locally.
