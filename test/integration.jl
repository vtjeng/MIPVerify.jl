using Base.Test

@testset "integration tests" begin
    include("integration/generated_weights.jl")
    include("integration/trained_weights.jl")
end