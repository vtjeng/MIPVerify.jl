using Documenter, MIPVerify

makedocs(
    modules = [MIPVerify],
    # See https://github.com/JuliaDocs/Documenter.jl/issues/868
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "MIPVerify",
    authors = "Vincent Tjeng and contributors.",
    pages = [
        "Home" => "index.md",
        "Tutorials" => "tutorials.md",
        "Finding Adversarial Examples" => [
            "finding_adversarial_examples/single_image.md",
            "finding_adversarial_examples/batch_processing.md",
        ],
        "Importing" =>
            ["utils/import_example_nets.md", "utils/import_weights.md", "utils/import_datasets.md"],
        "Working with Neural Net Parameters" => [
            "net_components/overview.md",
            "net_components/layers.md",
            "net_components/nets.md",
            "net_components/core_ops.md",
        ],
    ],
    checkdocs = :exports,
)

deploydocs(repo = "github.com/vtjeng/MIPVerify.jl.git")
