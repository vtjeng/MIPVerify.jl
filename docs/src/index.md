# MIPVerify
`MIPVerify.jl` enables users to verify neural networks that are piecewise affine by finding the closest adversarial example to a selected input.

## Installation

### Installing Julia
Download links and more detailed instructions are available on the [Julia](http://julialang.org/) website. The latest release of this package requires version 0.6 of Julia.

!!! warning

    Do **not** use `apt-get` or `brew` to install Julia, as the versions provided by these package managers tend to be out of date.

### Installing MIPVerify

Once you have Julia installed, install the latest tagged release of MIPVerify by running
```
Pkg.add("MIPVerify")
```

## Getting Started
The best way to get started is to follow our [quickstart tutorial](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/00_quickstart.ipynb), which demonstrates how to find adversarial examples for a pre-trained example network on the MNIST dataset. Once you're done with that, you can explore our other tutorials depending on your needs.