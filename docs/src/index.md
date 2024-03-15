# MIPVerify

`MIPVerify.jl` enables users to verify neural networks that are piecewise affine by: 1) finding the
closest adversarial example to a selected input, or 2) proving that no adversarial example exists
for some bounded family of perturbations.

## Installation

### Prerequisites

To use our package, you require

1. The Julia programming language
2. An optimizer [supported](https://jump.dev/JuMP.jl/stable/installation/#Install-a-solver) by
   `JuMP`
3. The Julia package for working with that optimizer

We recommend either [Gurobi](http://www.gurobi.com/) (if you have a license) or `HiGHS`, but any
supported optimizer will work.

**Platform compatibility:** Julia and Gurobi are available for 32-bit and 64-bit Windows, 64-bit
macOS, and 64-bit Linux, but example code in this README is for Linux.

#### Installing Julia

The latest release of this package requires version 1.6 or above of Julia. See
[installation instructions](https://julialang.org/downloads/). To complete your installation, ensure
that you are able to call `julia` REPL from the command line.

!!! warning

    Do **not** use `apt-get` or `brew` to install Julia, as the versions provided by these package managers tend to be out of date.

#### Installing Gurobi

Download the most recent version of the
[Gurobi optimizer](https://www.gurobi.com/downloads/gurobi-software/). A license is required to use
Gurobi; free academic licenses are
[available](https://www.gurobi.com/academia/academic-program-and-licenses/).

##### On Ubuntu

```console
$ cd /your/path/here
$ wget https://packages.gurobi.com/11.0/gurobi11.0.1_linux64.tar.gz
$ tar -xvf gurobi11.0.1_linux64.tar.gz
```

Add the following environment variables to your startup file

```sh
export GUROBI_HOME="/your/path/here/gurobi801/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

Finally, install the license obtained on a terminal prompt

!!! note

    You will have to obtain your own license number from the Gurobi site.

```console
$ grbgetkey aaaa0000-0000-0000-0000-000000000000

info  : grbgetkey version 8.0.1, build v8.0.1rc0
info  : Contacting Gurobi key server...
info  : Key for license ID 000000 was successfully retrieved
info  : License expires at the end of the day on 2019-09-30
info  : Saving license key...

In which directory would you like to store the Gurobi license key file?
[hit Enter to store it in /home/ubuntu]:

info  : License 000000 written to file /home/ubuntu/gurobi.lic
```

!!! note

    If you store the license file in a non-default location, you will have to add the environment variable `GRB_LICENSE_FILE` to your startup file: `export GRB_LICENSE_FILE="/your/path/here/gurobi.lic"`

#### Installing `Gurobi.jl`

`Gurobi.jl` is a wrapper of the Gurobi optimizer accessible in Julia. Once you have installed Gurobi
_and_ activated the license, install the latest release of `Gurobi.jl`:

```julia
julia> using Pkg; Pkg.add("Gurobi")
```

You can test `Gurobi.jl` by running

```julia
julia> using Pkg; Pkg.test("Gurobi")
```

Sample output:

```sh
INFO: Testing Gurobi
Academic license - for non-commercial use only
...
Test Summary: | Pass  Total
C API         |   19     19
...
Test Summary:          | Pass  Total
MathOptInterface Tests | 1415   1415
INFO: Gurobi tests passed
```

### Installing `MIPVerify`

Once you have Julia and Gurobi installed, install the latest release of MIPVerify:

```julia
julia> using Pkg; Pkg.add("MIPVerify")
```

You can test `MIPVerify` by running

```julia
julia> using Pkg; Pkg.test("MIPVerify")
```

These tests do take a long time to run (~30 mins), but any issues generally cause early failures.

### Troubleshooting your installation

#### Invalid Gurobi License

When running `Pkg.test("Gurobi")`:

```sh
INFO: Testing Gurobi
No variables, no constraints: Error During Test
  Got an exception of type ErrorException outside of a @test
  Invalid Gurobi license
  ...
```

**FIX:** The error message indicates that you have not installed your Gurobi license. If it has been
installed, the license is saved as a file `gurobi.lic`, typically in either the `/home/ubuntu` or
`opt/gurobi` directories.

## Getting Started

The best way to get started is to follow our
[quickstart tutorial](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/00_quickstart.ipynb),
which demonstrates how to find adversarial examples for a pre-trained example network on the MNIST
dataset. Once you're done with that, you can explore our other tutorials depending on your needs.
