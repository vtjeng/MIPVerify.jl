# Contributing to MIPVerify

Welcome, and thank you for considering contributing to MIPVerify.

## Getting Started

You will need to get started by [installing Julia](https://julialang.org/downloads/platform/).

### Basic Setup

If you are making changes to only documentation or tests, or are planning to add core functionality that will be tested with tests, you can simply check out this repository and start making changes.

### Advanced Setup

If you have an existing Julia script that uses the MIPVerify package, and want to run the script on the modified version of MIPVerify code via `julia my_script.jl`:

```sh
$ julia

julia>
```

Press `]` to enter the package manager

```sh
(@v1.4) pkg>
```

[Use `develop` to set up a git clone of the `MIPVerify` package](https://docs.julialang.org/en/v1/stdlib/Pkg/#Pkg).

```sh
(@v1.4) pkg> develop --local MIPVerify
    Cloning git-repo `https://github.com/vtjeng/MIPVerify.jl.git`
  Resolving package versions...
   Updating `~/.julia/environments/v1.4/Project.toml`
 [e5e5f8be] + MIPVerify v0.2.3 [`dev/MIPVerify`]
   Updating `~/.julia/environments/v1.4/Manifest.toml`
  [6e4b80f9] + BenchmarkTools v0.5.0
  [a74b3585] + Blosc v0.7.0
  
  ...
```

You should now have the `MIPVerify` repository checked out at the path `$JULIA_ENV_FOLDER/dev/MIPVerify`, where `JULIA_ENV_FOLDER` refers to the folder containing the `Project.toml` file.

## Submitting Contributions

There are many ways to contribute to this package.

> Note: If you're new to Julia, consider working on one of the issues [labeled "good first issue"](https://github.com/vtjeng/MIPVerify.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

### Contributing to core functionality

We're excited to hear what new ideas you have for verification, and would be happy to discuss how best to implement it.

### Contributing datasets / networks

If you have an interesting new dataset, or a new network that you'll like to verify the robustness of, please consider opening a PR to share it with other users of this package.

### Improving Documentation

We use [`Documenter.jl`](https://juliadocs.github.io/Documenter.jl/stable/man/guide/), and documentation is stored in the `docs/` directory.

Documentation is generated [as part of the build process](https://github.com/vtjeng/MIPVerify.jl/blob/2f2a0918abe28fb5f8b0b14396363c516a9c80c6/.travis.yml#L23-L28).

To generate documentation locally, run from the `docs/` directory

```sh
julia --color=yes make.jl
```

The generated documentation will be found in the `docs/build/` directory.

### Writing Tests

Tests are stored in the `test/` directory, which has a parallel structure to the `src/` directory.

Information on current test coverage can be found [on codecov](https://codecov.io/github/vtjeng/MIPVerify.jl?branch=master).

Tests are run as part of the build process.

To run tests locally, run `julia --project=/path/to/repo/root`, enter the package manager, then run `test MIPVerify`. (If running from the repository root, `julia --project` will suffice). Sample output:

```sh
(@v1.4) pkg> test MIPVerify
   Updating registry at `~/.julia/registries/General`
   Updating git-repo `https://github.com/JuliaRegistries/General.git`

   ...

    Testing MIPVerify
Status `/tmp/jl_rbadNV/Manifest.toml`
  [6e4b80f9] BenchmarkTools v0.5.0
  [b99e7846] BinaryProvider v0.5.10

   ...

Test Summary: | Pass  Total
MIPVerify     |  336    336
    Testing MIPVerify tests passed
```

For faster iteration, you can also run individual test files in `/test` via `julia --project=/path/to/repo/root /path/to/repo/root/test/my_basic_test.jl`. For now, you will need to replace `@timed_testset` with just `@testset` to ensure the individual test file can run.

## Getting your PR ready to merge

For your PR to be reviewed, it will need to pass two required checks:

### `JuliaFormatter / format-check (pull_request)`

This check verifies that the code formatting is aligned with our specifications. Running the script [`scripts/format.sh`](scripts/format.sh) locally will make changes to ensure that your code is up to spec.

### `Travis CI - Pull Request`

This check verifies that tests were all passing and documentation was generated successfully.

### Non-required checks

Two additional checks (`codecov/patch` and `codecov/project`) measure the change in code coverage in the project. They are generally used to verify that any new core functionality is properly tested.

## Acknowledgements

This document was adapted from Julia's [`CONTRIBUTING.md`](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md).
