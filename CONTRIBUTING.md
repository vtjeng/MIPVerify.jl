# Contributing to `MIPVerify`

Welcome, and thank you for considering contributing to `MIPVerify`. To begin contributing, start by
[installing Julia](https://julialang.org/downloads/platform/).

## Setup

### Basic Setup

If you are making changes to only documentation or tests — or are making changes to core
functionality that will be tested with unit tests — you can simply check out this repository and
start making changes.

### Advanced Setup

If you want to make changes to the `MIPVerify` package code, _and_ run an existing Julia script that
imports the package, the most convenient approach is to use `Pkg.develop`.

```shell-session
$ julia -e 'using Pkg; Pkg.develop("MIPVerify")'
     Cloning git-repo `https://github.com/vtjeng/MIPVerify.jl.git`
  [...]
    Updating `~/.julia/environments/v1.7/Project.toml`
  [e5e5f8be] + MIPVerify v0.5.1 `~/.julia/dev/MIPVerify`
    Updating `~/.julia/environments/v1.7/Manifest.toml`
  [...]
```

The repository is now checked out at the specified location (in this case,
`~/.julia/dev/MIPVerify`).

## Running Tests

Tests for this directory are found in [`test/runtests.jl`](./test/runtests.jl). (See
[Julia language docs](https://pkgdocs.julialang.org/v1/creating-packages/#Adding-tests-to-the-package)).
They are run for each PR via the `test` job for the
[.github/workflows/CI.yml](.github/workflows/CI.yml) GitHub action.

### Running all tests locally

To run all tests for the package, run this command from the package directory:

```sh
julia --project -e 'using Pkg; Pkg.test("MIPVerify")'
```

Sample output for a successful test is shown below. Note that information on runtime and memory
allocated for each test file is included.

```sh
 ────────────────────────────────────────────────────────────────────────────────────────────────
                                                        Time                    Allocations
                                               ───────────────────────   ────────────────────────
    Tot / % measured:                            211s /  99.1%           16.8GiB /  98.9%

 Section                               ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────────────────
 batch_processing_helpers/                  1    87.9s   42.1%   87.9s   4.14GiB   24.9%  4.14GiB
   integration.jl                           1    85.5s   40.9%   85.5s   3.45GiB   20.8%  3.45GiB
   unit.jl                                  1    2.36s    1.1%   2.36s    706MiB    4.2%   706MiB
 integration/                               1    75.8s   36.3%   75.8s   4.64GiB   27.9%  4.64GiB
   sequential/                              1    75.8s   36.2%   75.8s   4.64GiB   27.9%  4.64GiB
     generated_weights/                     1    75.8s   36.2%   75.8s   4.64GiB   27.9%  4.64GiB
       conv+fc+softmax.jl                   1    50.4s   24.1%   50.4s   1.29GiB    7.8%  1.29GiB
         BlurringPerturbationFamily         1    35.0s   16.7%   35.0s   0.95GiB    5.7%  0.95GiB
         Unrestricted[...]                  1    14.8s    7.1%   14.8s    302MiB    1.8%   302MiB
           Minimizing lInf norm             1    8.57s    4.1%   8.57s    122MiB    0.7%   122MiB
           Minimizing l1 norm               1    4.27s    2.0%   4.27s    103MiB    0.6%   103MiB
           With multiple target[...]        1    2.00s    1.0%   2.00s   77.2MiB    0.5%  77.2MiB
         LInfNormBounded[...]               1    488ms    0.2%   488ms   36.9MiB    0.2%  36.9MiB
       mfc+mfc+softmax.jl                   1    25.4s   12.1%   25.4s   3.35GiB   20.2%  3.35GiB
 utils/                                     1    32.2s   15.4%   32.2s   6.50GiB   39.2%  6.50GiB
   import_example_nets.jl                   1    24.1s   11.5%   24.1s   3.04GiB   18.3%  3.04GiB
     get_example_network_params             1    24.1s   11.5%   24.1s   3.04GiB   18.3%  3.04GiB
       MNIST.WK17a_linf0.1_authors          1    17.9s    8.5%   17.9s   1.54GiB    9.3%  1.54GiB
       MNIST.RSL18a_linf0.1_authors         1    4.29s    2.1%   4.29s    786MiB    4.6%   786MiB
       MNIST.n1                             1    1.91s    0.9%   1.91s    757MiB    4.5%   757MiB
   import_datasets.jl                       1    8.08s    3.9%   8.08s   3.46GiB   20.8%  3.46GiB
 net_components/                            1    13.1s    6.3%   13.1s   1.32GiB    7.9%  1.32GiB
   layers/                                  1    6.86s    3.3%   6.86s    746MiB    4.4%   746MiB
   core_ops.jl                              1    4.05s    1.9%   4.05s    449MiB    2.6%   449MiB
   nets/                                    1   48.7ms    0.0%  48.7ms   5.17MiB    0.0%  5.17MiB
 models.jl                                  1   10.0ms    0.0%  10.0ms   25.2KiB    0.0%  25.2KiB
 ────────────────────────────────────────────────────────────────────────────────────────────────


Test Summary: | Pass  Total
MIPVerify     |  336    336
     Testing MIPVerify tests passed
```

### Running a subset of tests locally

Running all tests takes ~5 minutes. For even faster iteration, run tests in a specific file (e.g.,
[`test/vendor.jl`](./test/vendor.jl)). To do so, run this command from package directory:

```sh
$ julia -e 'using Pkg; Pkg.add("TestEnv"); using TestEnv; Pkg.activate("."); TestEnv.activate(); include("test/vendor.jl")'
    Updating registry at `~/.julia/registries/General.toml`
   Resolving package versions...
  No Changes to `~/.julia/environments/v1.7/Project.toml`
  No Changes to `~/.julia/environments/v1.7/Manifest.toml`
  Activating project at `~/development/vtjeng/MIPVerify.jl`
Test Summary: | Pass  Total
vendor/       |   22     22
```

## Preparing your PR for review

Ensure that your PR passes all required statuses.

## Contribution Types

There are many ways to contribute to this package.

> Note: If you're new to Julia, consider working on one of the issues
> [labeled "good first issue"](https://github.com/vtjeng/MIPVerify.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

### Contributing to core functionality

We're excited to hear what new ideas you have for verification, and would be happy to discuss how
best to implement it.

### Contributing datasets / networks

If you have an interesting new dataset, or a new network that you'll like to verify the robustness
of, please consider opening a PR to share it with other users of this package.

### Improving Documentation

We use [`Documenter.jl`](https://juliadocs.github.io/Documenter.jl/stable/man/guide/), and
documentation is stored in the `docs/` directory.

Documentation is generated
[as part of the build process](https://github.com/vtjeng/MIPVerify.jl/blob/2f2a0918abe28fb5f8b0b14396363c516a9c80c6/.travis.yml#L23-L28).

To generate documentation locally, run the following command from the `docs/` directory

```sh
julia make.jl
```

The generated documentation will be found in the `docs/build/` directory.

### Writing Tests

Tests are stored in the `test/` directory, which has a parallel structure to the `src/` directory.

Information on current test coverage can be found
[on codecov](https://codecov.io/github/vtjeng/MIPVerify.jl?branch=master).

## Acknowledgements

This document was adapted from Julia's
[`CONTRIBUTING.md`](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md).
