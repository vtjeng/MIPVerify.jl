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
forward. At each hidden layer after the first, it bounds each neuron’s input from above and below by
working back to the pixels. For an earlier ReLU whose input range crosses zero, the backward
calculation uses the upper secant or an adaptive lower line (zero or the input itself), according to
the coefficient’s sign. The verifier keeps the tighter endpoints and applies ReLU to each endpoint
to get the output range for the next layer. It skips further work on a neuron once its upper bound
proves its output is always zero.

A target must score at least `margin` above **every** non-target class on the same input. For
`margin=0`, a tie suffices. An upper bound below the margin against any one non-target eliminates
that target. If every target is eliminated, the result records `:backward_bound` and `INFEASIBLE`. A
checked input meeting the margin is a witness. A corner or gradient-search witness records `:corner`
or `:pgd`. Each of these results has a `:SolveStatus` and zero `:SolveTime`.

If the check stays open, the solver considers only the surviving targets and compares them with the
original non-target classes. The result’s `:TargetIndexes` records the caller’s selection. With
`interval_arithmetic`, the verifier builds the target constraints directly from those survivors. An
unresolved image still goes to the final MIP solver. MIPVerify builds that model using the chosen
tightening algorithm and per-layer overrides. Each ReLU starts with the input range established by
the backward screen. An input range entirely above or below zero fixes the ReLU’s behavior. With
`lp` or `mip` tightening, the solver tightens ranges that still cross zero. MIP tightening tries the
LP relaxation first. For a multi-target request with one non-target, `target_lp` checks only the
surviving score gaps. A time limit can leave the answer unresolved. Pass `backward_screen=false` to
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
