
using Test
@isdefined(TestHelpers) || include("TestHelpers.jl")

TestHelpers.@timed_testset "batch_processing_helpers/" begin
    include("batch_processing_helpers/unit.jl")
    include("batch_processing_helpers/integration.jl")
end
