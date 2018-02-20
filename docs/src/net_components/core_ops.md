# Core Operations
Our ability to cast the input-output constraints of a neural net to an efficient set of linear and integer constraints boils down to the following basic operations, over which the layers provide a convenient layer of abstraction.

## Index
```@index
Pages   = ["core_ops.md"]
Order   = [:function, :type]
```

## Internal
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = [
    "net_components/core_ops.jl"
    ]
Public  = false
```