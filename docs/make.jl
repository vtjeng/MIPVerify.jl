using Documenter, MIPVerify

makedocs(
    modules = [MIPVerify],
    format = :html,
    sitename = "MIPVerify.jl",
    authors = "Vincent Tjeng and contributors.",
    pages = [
        "Home" => "index.md",
        "Tutorials" => "tutorials.md"
    ]
)

deploydocs(
    deps = nothing,
    repo = "github.com/vtjeng/MIPVerify.jl.git",
    target = "build",
    make = nothing,
    julia = "0.6",
    osname = "linux"
)