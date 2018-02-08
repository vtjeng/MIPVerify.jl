using Base.Test
using MIPVerify: read_datasets
using MIPVerify: get_example_network_params, num_correct

@testset "utils" begin
    include("import_datasets.jl")
    include("import_weights.jl")
end