# Batch Processing

[`batch_find_untargeted_attack`](@ref) enables users to run [`find_adversarial_example`](@ref) for
multiple samples from a single dataset, writing 1) a single summary `.csv` file for the dataset,
with a row of summary results per sample, and 2) a file per sample containing the output dictionary
from [`find_adversarial_example`](@ref).

[`batch_find_untargeted_attack`](@ref) allows verification of a dataset to be resumed if the process
is interrupted by intelligently determining whether to rerun [`find_adversarial_example`](@ref) on a
sample based on the `solve_rerun_option` specified.

Resuming against a summary written by an older MIPVerify version upgrades its schema in memory; the
upgraded file is written back only when the batch appends a new result, so a batch that schedules no
work leaves the archived summary untouched. Rows from older summaries keep their historical rerun
scheduling: a recorded numeric objective counts as a completed attack for `resolve_ambiguous_cases`,
even though it is not a verified witness. Rerun with `always` to replace legacy evidence with
verified witnesses.

## Index

```@index
Pages   = ["batch_processing.md"]
Order   = [:function, :type]
```

## Public Interface

```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = ["batch_processing_helpers.jl"]
Private = false
```

## Internal

```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = ["batch_processing_helpers.jl"]
Public  = false
```
