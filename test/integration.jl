using Test
@isdefined(TestHelpers) || include("TestHelpers.jl")

TestHelpers.@timed_testset "integration/" begin
    include("integration/sequential.jl")
end
