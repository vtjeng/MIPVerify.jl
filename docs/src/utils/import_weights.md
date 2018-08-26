# Helpers for importing individual layers
You're likely to want to import parameter values from your trained neural networks from
outside of Julia. [`get_conv_params`](@ref) and [`get_matrix_params`](@ref) are helper functions enabling you to import individual layers.

## Index
```@index
Pages   = ["import_weights.md"]
Order   = [:function, :type]
```

## Public Interface
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = ["utils/import_weights.jl"]
Private = false
```

## Internal
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = ["utils/import_weights.jl"]
Public  = false
```
