using Test
@isdefined(TestHelpers) || include("../../TestHelpers.jl")

TestHelpers.@timed_testset "trained_weights/" begin
    include("trained_weights/MNIST.n1.jl")
    include("trained_weights/MNIST.WK17a_linf0.1_authors.jl")
end
