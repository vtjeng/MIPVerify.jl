using Test
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "sequential/" begin
    include("sequential/generated_weights.jl")
    if Base.find_package("Gurobi") !== nothing
        # Skip these tests if Gurobi is not installed.
        # The corresponding networks are too large for CBC to deal with.
        include("sequential/trained_weights.jl")
    end
end
