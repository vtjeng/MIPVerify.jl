# Layers
Each layer in the neural net corresponds to a `struct` that simultaneously specifies: 1) the operation being carried out in the layer (recorded in the type of the `struct`) and 2) the parameters for the operation (recorded in the values of the fields of the `struct`).

When we pass an input array of real numbers to a layer `struct`, we get an output array of real numbers that is the result of the layer operating on the input.

Conversely, when we pass an input array of `JuMP` variables, we get an output array of `JuMP` variables, with the appropriate mixed-integer constraints (as determined
by the layer) imposed between the input and output.

## Index
```@index
Pages   = ["layers.md"]
Order   = [:function, :type]
```

## Public Interface
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = [
    "net_components/layers/conv2d.jl",
    "net_components/layers/flatten.jl",
    "net_components/layers/linear.jl",
    "net_components/layers/masked_relu.jl",
    "net_components/layers/pool.jl",
    "net_components/layers/relu.jl"
    ]
Private = false
```

## Internal
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = [
    "net_components/layers/conv2d.jl",
    "net_components/layers/flatten.jl",
    "net_components/layers/linear.jl",
    "net_components/layers/masked_relu.jl",
    "net_components/layers/pool.jl",
    "net_components/layers/relu.jl"
    ]
Public  = false
```