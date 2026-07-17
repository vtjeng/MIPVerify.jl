# Single Image

[`find_adversarial_example`](@ref) searches for an adversarial example to a given input image for a
particular [`NeuralNet`](@ref), using the selected closest, worst-margin, or feasibility objective.

As a sanity check, we suggest that you verify that the [`NeuralNet`](@ref) imported achieves the
expected performance on the test set. This can be done using [`frac_correct`](@ref).

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
