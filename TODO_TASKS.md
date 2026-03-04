# TODO Tasks

This file tracks the former inline TODO comments and their resolution status.

## Status Legend

- `planned`
- `in_progress`
- `done`
- `deferred_tracked`

## Tasks

| ID       | Location                                                              | Category                 | Status  | Commit | Definition of Done                                                                                               | Tests                                                              |
| -------- | --------------------------------------------------------------------- | ------------------------ | ------- | ------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| DEP-001  | `src/MIPVerify.jl:10`                                                 | dependency path          | done    | C2     | Replace dependency path heuristic with deterministic helper and remove TODO comment.                             | `test/runtests.jl`                                                 |
| DATA-002 | `src/utils/prep_data_file.jl:16`                                      | data path/url            | done    | C2     | Replace path-separator hack with explicit URL path helper and remove TODO comment.                               | `test/utils/prep_data_file.jl`                                     |
| CONV-001 | `src/net_components/layers/conv2d.jl:115`                             | conv2d API clarity       | done    | C2     | Remove misleading mutating helper naming and keep behavior unchanged.                                            | `test/net_components/layers/conv2d.jl`                             |
| API-001  | `src/batch_processing_helpers.jl:280`                                 | dataset access API       | done    | C3     | Add dataset overloads for `get_image`/`get_label`, migrate callsites, remove TODO.                               | `test/batch_processing_helpers/*.jl`                               |
| DATA-001 | `src/utils/import_datasets.jl:82`                                     | dataset type invariant   | done    | C3     | Enforce matching train/test structural type in constructor and remove TODO.                                      | `test/utils/import_datasets.jl`                                    |
| SKIP-001 | `src/net_components/layers/skip_unit.jl:43`                           | skip output shape checks | planned | C4     | Validate all skip branch outputs have same size; throw `DimensionMismatch` if not.                               | `test/net_components/layers/skip_unit.jl`                          |
| SKIP-002 | `src/net_components/nets/skip_sequential.jl:26`                       | skip type robustness     | planned | C4     | Replace brittle `typeof(layer) <: SkipBlock` with `isa`; tighten container typing.                               | `test/net_components/nets/skip_sequential.jl`                      |
| NET-001  | `src/utils/import_example_nets.jl:75`                                 | case-insensitive naming  | planned | C4     | Implement case-insensitive/canonical network name resolution with tests.                                         | `test/utils/import_example_nets.jl`                                |
| NET-002  | `src/utils/import_example_nets.jl:74`                                 | new external networks    | planned | C4     | Convert inline TODO to deferred tracked item with concrete next action and no inline TODO comment.               | N/A                                                                |
| TEST-001 | `test/integration/sequential/generated_weights/conv+fc+softmax.jl:56` | blur integration case    | planned | C4     | Timebox search for deterministic non-NaN case; if unresolved, keep tracked deferred item and remove inline TODO. | `test/integration/sequential/generated_weights/conv+fc+softmax.jl` |
| CORE-004 | `src/net_components/core_ops.jl:392`                                  | core logging             | planned | C6     | Improve maximum logging behavior and remove TODO.                                                                | `test/net_components/core_ops.jl` + benchmark gate                 |
| CORE-001 | `src/net_components/core_ops.jl:13`                                   | core constant detection  | planned | C7     | Use built-in JuMP expression checks and remove TODO.                                                             | `test/net_components/core_ops.jl` + benchmark gate                 |
| CORE-002 | `src/net_components/core_ops.jl:176`                                  | core ReLU bounds         | planned | C8     | Replace TODO with explicit tolerance and clearer fallback behavior.                                              | `test/net_components/core_ops.jl` + benchmark gate                 |
| CORE-003 | `src/net_components/core_ops.jl:375`                                  | core maximum perf        | planned | C9     | Implement lazy lower-bound computation with cutoff and remove TODO.                                              | `test/net_components/core_ops.jl` + benchmark gate                 |

## Deferred Follow-ups

### NET-002

- Description: Add additional MNIST networks (Ragunathan/Steinhardt/Liang) as importable examples.
- Blocking inputs: model files and canonical naming/metadata for distribution in `deps`.
- Next action: open follow-up issue with source links, expected architecture specs, and checksum
  plan for assets.

### TEST-001

- Description: Add deterministic integration coverage where `BlurringPerturbationFamily` yields a
  finite, non-NaN optimum.
- Blocking inputs: stable sample/network/solver configuration with reproducible non-NaN output.
- Next action: run targeted sample search with fixed solver options and record candidate cases.
