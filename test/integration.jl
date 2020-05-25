using Test

@timed_testset "integration/" begin
    include("integration/sequential.jl")
end
