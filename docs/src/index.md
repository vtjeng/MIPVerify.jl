# MIPVerify
`MIPVerify.jl` enables users to verify neural networks that are piecewise affine by: 1) finding the closest adversarial example to a selected input, or 2) proving that no adversarial example exists for some bounded family of perturbations.

## Installation
### Prerequisites
To use our package, you require

   1. The Julia programming language
   2. An optimization solver [supported](http://www.juliaopt.org/JuMP.jl/0.18/installation.html#getting-solvers) by `JuMP`
   3. The Julia package for working with that solver
   
Our choice of solver is [Gurobi](http://www.gurobi.com/), but any supported optimization solver will work.

**Platform compatibility:** Julia and Gurobi are available for 32-bit and 64-bit Windows, 64-bit macOS, and 64-bit Linux, but example code in this README is for Linux.

#### Installing Julia
The latest release of this package requires [version 0.6](https://julialang.org/downloads/oldreleases.html) of Julia.

!!! note
    
    `MIPVerify` is not currently supported on version 0.7 or 1.0 of Julia.

Platform-specific instructions can be found [here](https://julialang.org/downloads/platform.html). To complete your installation, ensure that you are able to call `julia` REPL from the command line.

!!! warning

    Do **not** use `apt-get` or `brew` to install Julia, as the versions provided by these package managers tend to be out of date.

##### On Ubuntu
```console
$ cd /your/path/here
  wget https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.4-linux-x86_64.tar.gz
  tar -xvf julia-0.6.4-linux-x86_64.tar.gz
```

Julia will be extracted to a folder named `julia-hash_number`. (For example, v0.6.4 is `julia-9d11f62bcb`). 

Add the following lines to your startup file (e.g. `.bashrc` for the bash shell) to add Julia's bin folder to your system `PATH` environment variable.

```sh
export PATH="${PATH}:/your/path/here/julia-9d11f62bcb/bin"
```

#### Installing Gurobi
Download the most recent version of the [Gurobi optimizer](http://www.gurobi.com/downloads/gurobi-optimizer). A license is required to use Gurobi; free academic licenses are [available](https://user.gurobi.com/download/licenses/free-academic).

##### On Ubuntu
```console
$ cd /your/path/here
  wget https://packages.gurobi.com/8.0/gurobi8.0.1_linux64.tar.gz
  tar -xvf gurobi8.0.1_linux64.tar.gz
```

Add the following environment variables to your startup file

```sh
export GUROBI_HOME="/your/path/here/gurobi801/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

Finally, install the license obtained on a terminal prompt
```console
$ grbgetkey aaaa0000-0000-0000-0000-000000000000
```

!!! note
    
    You will have to obtain your own license number from the Gurobi site.

```sh
Sample output:

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
`Gurobi.jl` is a wrapper of the Gurobi solver accessible in Julia. Once you have installed Gurobi *and* activated the license, install the latest release of `Gurobi.jl`:
```julia
julia> Pkg.add("Gurobi")
```
You can test `Gurobi.jl` by running
```julia
julia> Pkg.test("Gurobi")
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

#### Installing `HDF5`
```julia
julia> Pkg.add("HDF5")
```

### Installing `MIPVerify`
Once you have Julia, a mathematical programming solver, and the `HDF5` package installed, install the latest release of MIPVerify:
```julia
julia> Pkg.add("MIPVerify")
```

You can test `MIPVerify` by running
```julia
julia> Pkg.test("MIPVerify")
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

**FIX:** The error message indicates that you have not installed your Gurobi license. If it has been installed, the license is saved as a file `gurobi.lic`, typically in either the `/home/ubuntu` or `opt/gurobi` directories.

#### HDF5 had build errors on Ubuntu
When running `Pkg.add("HDF5")`:

```sh
...
INFO: Building HDF5
...
=======================================[ ERROR: HDF5 ]=======================================

LoadError: failed process: Process(`sudo apt-get install hdf5-tools`, ProcessExited(1)) [1]
while loading /home/ubuntu/.julia/v0.6/HDF5/deps/build.jl, in expression starting on line 41

=============================================================================================
...
```

**FIX:**
```console
$ sudo apt-get install hdf5-tools
```
Next,
```julia
julia> Pkg.build("HDF5")
```
```sh
INFO: Building CMakeWrapper
INFO: Building Blosc
INFO: Building HDF5
```

## Getting Started
The best way to get started is to follow our [quickstart tutorial](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/00_quickstart.ipynb), which demonstrates how to find adversarial examples for a pre-trained example network on the MNIST dataset. Once you're done with that, you can explore our other tutorials depending on your needs.
