using Base.Test

@testset "utils" begin
    include("utils/import_datasets.jl")
    include("utils/import_weights.jl")
end