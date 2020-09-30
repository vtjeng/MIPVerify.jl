using Test
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "nets/" begin
    include("nets/sequential.jl")
end
