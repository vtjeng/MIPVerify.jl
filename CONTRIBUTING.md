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

The default suite skips validation of the full CIFAR10 data asset because loading it requires about
1.5 GiB of memory. CI enables the check on its Julia 1.12 Linux job. Include it locally with:

```sh
MIPVERIFY_RUN_LARGE_DATASET_TESTS=true julia --project -e 'using Pkg; Pkg.test("MIPVerify")'
```

Sample output from a warm Julia 1.12.6 run is shown below. Runtime and allocations vary by machine,
but the table identifies the expensive test groups.

```sh
 Tot / % measured:                         182s / 99.3%        9.11GiB / 99.1%

 Section                              ncalls     time    %tot     alloc    %tot
 integration/                              1     119s   65.7%   3.89GiB   43.1%
   generated_weights/                      1     115s   63.7%   3.65GiB   40.4%
     conv+fc+softmax.jl                    1    61.4s   33.9%   1.12GiB   12.5%
       BlurringPerturbationFamily          1    44.5s   24.6%    920MiB   10.0%
     mfc+mfc+softmax.jl                    1    53.7s   29.7%   2.53GiB   28.0%
 net_components/                           1    31.0s   17.1%   2.00GiB   22.1%
 utils/                                    1    17.4s    9.6%   2.11GiB   23.3%
   import_datasets.jl                      1    11.3s    6.2%   1.40GiB   15.5%
   import_example_nets.jl                  1    5.62s    3.1%    702MiB    7.6%
 batch_processing_helpers/                 1    13.7s    7.6%   1.04GiB   11.5%
   integration.jl                          1    10.6s    5.9%    848MiB    9.2%

Test Summary: | Pass  Broken  Total     Time
MIPVerify     |  513       1    514  3m02.4s
     Testing MIPVerify tests passed
```

### Running a subset of tests locally

For faster iteration, run a test file directly from the test project. For example:

```sh
julia --project=test test/models.jl
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
