using Test
@isdefined(TestHelpers) || include("TestHelpers.jl")

TestHelpers.@timed_testset "vendor/" begin
    include("vendor/ConditionalJuMP.jl")
end
