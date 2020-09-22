using Test

@timed_testset "sequential/" begin
    include("sequential/generated_weights.jl")
    include("sequential/trained_weights.jl")
end
