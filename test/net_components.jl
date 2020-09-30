using Test
@isdefined(TestHelpers) || include("TestHelpers.jl")

TestHelpers.@timed_testset "net_components/" begin
    include("net_components/core_ops.jl")
    include("net_components/layers.jl")
    include("net_components/nets.jl")
end
