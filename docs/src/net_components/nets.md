# Networks
Each network corresponds to an array of layers associated with a unique string identifier. The string identifier of the network is used to store cached models, so it's important to ensure that you don't re-use names!

## Public Interface
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = [
    "net_components/nets/sequential.jl",
    ]
Private = false
```

## Internal
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = [
    "net_components/nets/sequential.jl",
    ]
Public  = false
```