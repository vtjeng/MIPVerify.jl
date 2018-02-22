using Documenter, MIPVerify

makedocs(
    modules = [MIPVerify],
    format = :html,
    sitename = "MIPVerify.jl",
    authors = "Vincent Tjeng and contributors.",
    pages = [
        "Home" => "index.md",
        "Tutorials" => "tutorials.md",
        "Finding Adversarial Examples" => "finding_adversarial_examples.md",
        "Working with Neural Net Parameters" => [
            "net_components/overview.md",
            "net_components/layers.md",
            "net_components/nets.md",
            "net_components/core_ops.md"
        ],
        "Importing" => [
            "utils/import_weights.md",
            "utils/import_datasets.md",
        ]
    ],
checkdocs = :exports
)

deploydocs(
    deps = nothing,
    repo = "github.com/vtjeng/MIPVerify.jl.git",
    target = "build",
    make = nothing,
    julia = "0.6",
    osname = "linux"
)