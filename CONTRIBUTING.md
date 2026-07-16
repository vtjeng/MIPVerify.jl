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
 Tot / % measured:                         199s / 99.3%        9.10GiB / 99.3%

 Section                              ncalls     time    %tot     alloc    %tot
 integration/                              1     121s   61.3%   3.96GiB   43.8%
   generated_weights/                      1     113s   57.2%   3.72GiB   41.2%
     conv+fc+softmax.jl                    1    82.6s   41.9%    887MiB    9.6%
       BlurringPerturbationFamily          1    65.7s   33.3%    780MiB    8.4%
     mfc+mfc+softmax.jl                    1    30.0s   15.2%   2.85GiB   31.6%
 net_components/                           1    37.5s   19.0%   1.93GiB   21.4%
 utils/                                    1    23.0s   11.7%   2.11GiB   23.3%
   import_datasets.jl                      1    13.7s    6.9%   1.40GiB   15.5%
   import_example_nets.jl                  1    8.65s    4.4%    702MiB    7.6%
 batch_processing_helpers/                 1    15.8s    8.0%   1.04GiB   11.5%
   integration.jl                          1    12.1s    6.1%    848MiB    9.2%

Test Summary: | Pass  Broken  Total     Time
MIPVerify     |  518       1    519  3m18.6s
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
