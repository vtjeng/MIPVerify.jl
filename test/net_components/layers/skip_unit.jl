using Test
using MIPVerify
using JuMP
@isdefined(TestHelpers) || include("../../TestHelpers.jl")

@testset "skip_unit.jl" begin
    @testset "SkipBlock" begin
        @testset "Base.show" begin
            linear1 = Linear(ones(3, 2), zeros(2))
            linear2 = Linear(ones(4, 2), zeros(2))
            skip_block = SkipBlock([linear1, linear2])

            io = IOBuffer()
            Base.show(io, skip_block)
            actual_output = String(take!(io))
            expected_output = """SkipBlock
    (1) Linear(3 -> 2)
    (2) Linear(4 -> 2)
"""
            @test actual_output == expected_output
        end

        @testset "forward pass with array input" begin
            # Create two linear layers that transform to the same output size
            linear1 = Linear(ones(3, 2), [1.0, 2.0])  # 3->2 with bias
            linear2 = Linear(ones(4, 2), [3.0, 4.0])  # 4->2 with bias
            skip_block = SkipBlock([linear1, linear2])

            # Test with numeric arrays
            x1 = ones(3)  # Input for first layer
            x2 = ones(4)  # Input for second layer
            output = skip_block([x1, x2])

            # Expected computation:
            # Layer 1: (3 ones * 1) + bias -> [4.0, 5.0]
            # Layer 2: (4 ones * 1) + bias -> [7.0, 8.0]
            # Sum: [11.0, 13.0]
            @test size(output) == (2,)
            @test output ≈ [11.0, 13.0]
        end

        @testset "optimization with JuMP variables" begin
            model = TestHelpers.get_new_model()

            # Create two linear layers
            linear1 = Linear(ones(2, 1), [0.0])  # 2->1
            linear2 = Linear(ones(3, 1), [0.0])  # 3->1
            skip_block = SkipBlock([linear1, linear2])

            # Create input variables
            x1 = @variable(model, [1:2], lower_bound = -1, upper_bound = 1)
            x2 = @variable(model, [1:3], lower_bound = -1, upper_bound = 1)

            output = skip_block([x1, x2])
            @test size(output) == (1,)

            # Test maximization
            @objective(model, Max, sum(output))
            optimize!(model)
            @test termination_status(model) == OPTIMAL
            # Maximum should be 5 (when all inputs are 1):
            # Layer 1: 2*1 + 0 = 2
            # Layer 2: 3*1 + 0 = 3
            # Sum: 5
            @test objective_value(model) ≈ 5.0

            # Test minimization
            @objective(model, Min, sum(output))
            optimize!(model)
            @test termination_status(model) == OPTIMAL
            # Minimum should be -5 (when all inputs are -1)
            @test objective_value(model) ≈ -5.0
        end
    end
end
