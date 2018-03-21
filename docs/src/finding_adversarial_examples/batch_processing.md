# Batch Processing
When running on multiple samples from a single dataset, [`batch_find_certificate`](@ref) 
allows you to redo solves intelligently - redoing 1) no solves, 2) all solves, 3) only 
solves where the sample status is indeterminate, or 4) only solves where the best
counter-example is non-optimal.

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