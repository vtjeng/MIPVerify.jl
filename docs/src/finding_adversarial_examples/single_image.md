# Single Image

[`find_adversarial_example`](@ref) searches for an adversarial example to a given input image for a
particular [`NeuralNet`](@ref), using the selected closest, worst-margin, or feasibility objective.

As a sanity check, we suggest that you verify that the [`NeuralNet`](@ref) imported achieves the
expected performance on the test set. This can be done using [`frac_correct`](@ref).

## Fixed-budget verification

For a feasibility check, pass `adversarial_example_objective=MIPVerify.feasibility` and
`pp=MIPVerify.LInfNormBoundedPerturbationFamily(radius)`. On supported sequential networks the
verifier tries backward score-gap bounds before constructing a JuMP model. This screen runs with any
solver tightening setting, including `tightening_algorithm=lp`. It propagates pixel intervals
forward and uses back-substitution to tighten hidden-neuron ranges.

A target must score at least `margin` above **every** non-target class on the same input. For
`margin=0`, a tie suffices. An upper bound below the margin against any one non-target eliminates
that target. If every target is eliminated, the result records `:backward_bound` and `INFEASIBLE`. A
checked input meeting the margin is a witness. A corner or gradient-search witness records `:corner`
or `:pgd`. Each of these results has a `:SolveStatus` and zero `:SolveTime`.

If the check stays open, MIPVerify builds the JuMP model using the chosen tightening algorithm and
per-layer overrides. With the default `target_lp`, the fallback can reuse the screen’s hidden ranges
on eligible layers. A time limit can leave the answer unresolved. Pass `backward_screen=false` to
use the solver formulation directly. The default objective, `closest`, still minimizes perturbation
distance.

## Index

```@index
Pages   = ["basic_usage.md"]
Order   = [:function, :type]
```

## Public Interface

```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = ["MIPVerify.jl"]
Private = false
```

## Adversarial-example objectives

```@docs
MIPVerify.AdversarialExampleObjective
```

## Internal witness verification

```@docs
MIPVerify.PerturbationFamily
MIPVerify.get_target_margin
MIPVerify.witness_satisfies_target
MIPVerify.clear_witness_values!
MIPVerify.record_no_witness!
MIPVerify.record_witness!
MIPVerify.witness_value_in_closed_interval
MIPVerify.witness_inputs_satisfy_common_constraints
MIPVerify.witness_arrays_are_approximately_equal
MIPVerify.blur_kernel_size_is_valid
MIPVerify.verify_perturbation_witness
MIPVerify.identity_blur_kernel
MIPVerify.numeric_blur_kernel
```
