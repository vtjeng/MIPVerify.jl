using Base.Test

@testset "batch_processing_helpers.jl" begin
    include("batch_processing_helpers/unit.jl")
    include("batch_processing_helpers/integration.jl")
end