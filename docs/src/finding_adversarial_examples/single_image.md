# Single Image
[`find_adversarial_example`](@ref) finds the closest adversarial
example to a given input image for a particular [`NeuralNet`](@ref).

As a sanity check, we suggest that you verify that the [`NeuralNet`](@ref) 
imported achieves the expected performance on the test set. 
This can be done using [`frac_correct`](@ref).

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