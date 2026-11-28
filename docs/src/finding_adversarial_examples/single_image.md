# Single Image

[`find_adversarial_example`](@ref) searches for an adversarial example to a given input image for a
particular [`NeuralNet`](@ref), using the selected closest, worst-margin, or feasibility objective.

As a sanity check, we suggest that you verify that the [`NeuralNet`](@ref) imported achieves the
expected performance on the test set. This can be done using [`frac_correct`](@ref).

## Fixed-budget verification

For a feasibility check against all rivals of one class, pass
`adversarial_example_objective=MIPVerify.feasibility` and
`pp=MIPVerify.LInfNormBoundedPerturbationFamily(radius)`. Backward score-gap screening is enabled by
default for zero-margin checks on supported sequential networks with Float64 inputs with the default
`target_lp` tightening. The verifier can rule out every rival or find a checked input before it
builds a JuMP model. In those cases the result records `:VerificationSource` and `:SolveStatus`. For
example, `:backward_bound` and `INFEASIBLE` mean the upper bound for each rival was strictly
negative throughout the pixel box.

If the screen leaves the answer open, the verifier passes the hidden-neuron bounds to Target LP and
the caller’s MIP solver. A solver-returned candidate must pass a box and score check before it
counts as a witness. A time limit can still leave the answer unresolved. To reproduce a run using
Target LP alone, pass `backward_screen=false`. The default optimization objective is `closest`,
which still solves for the minimum perturbation distance.

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
