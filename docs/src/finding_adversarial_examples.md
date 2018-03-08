# Finding Adversarial Examples
[`find_adversarial_example`](@ref) is the core function that you will be calling to 
find adversarial examples. To avoid spending time verifying the wrong network, we suggest
that you check that the network gets reasonable performance on the test set using
[`frac_correct`](@ref).

## Index
```@index
Pages   = ["finding_adversarial_examples.md"]
Order   = [:function, :type]
```

## Public Interface
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = ["MIPVerify.jl"]
Private = false
```