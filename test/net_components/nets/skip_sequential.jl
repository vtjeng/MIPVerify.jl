using Test
using MIPVerify
using JuMP
using HiGHS
@isdefined(TestHelpers) || include("../../TestHelpers.jl")

@testset "skip_sequential.jl" begin
    @testset "SkipSequential" begin
        # Create network components
        input_size = 8
        hidden_size = 6
        intermediate_size = 5
        output_size = 4
        
        # Layer 1: Linear transformation from input_size to hidden_size
        linear1 = Linear(ones(input_size, hidden_size), [0, 0, 0, -16, -16, -16])
        
        # Layer 2: ReLU activation (maintains hidden_size)
        relu = ReLU()
        
        # Layer 3: Linear transformation to intermediate_size
        linear2 = Linear(ones(hidden_size, intermediate_size), zeros(intermediate_size))
        
        # Layer 4: Skip block that combines outputs from Layer 1 and Layer 3
        skip_block = SkipBlock([
            Linear(ones(hidden_size, output_size), zeros(output_size)),  # Transform Layer 1 output
            Linear(ones(intermediate_size, output_size), zeros(output_size))   # Transform Layer 3 output
        ])
        
        # Create the skip sequential network
        nnparams = SkipSequential([linear1, relu, linear2, skip_block], "test_skip_net")
        
        @testset "Base.show" begin
            io = IOBuffer()
            Base.show(io, nnparams)
            actual_output = String(take!(io))
            expected_output = """
            skip sequential net test_skip_net
            (1) Linear(8 -> 6)
            (2) ReLU()
            (3) Linear(6 -> 5)
            (4) SkipBlock
                (1) Linear(6 -> 4)
                (2) Linear(5 -> 4)

            """
            @test actual_output == expected_output
        end
        
        @testset "forward pass with array input" begin
            x_num = ones(input_size)
            output_num = nnparams(x_num)
            @test size(output_num) == (output_size,)
            @test length(output_num) == output_size
            
            # Layer 1: 8 ones with biases [0,0,0,-16,-16,-16] -> [8,8,8,-8,-8,-8]
            # Layer 2: ReLU([8,8,8,-8,-8,-8]) -> [8,8,8,0,0,0]
            # Layer 3: Linear([8,8,8,0,0,0]) -> 5 twenty-fours (each output sums three 8s)
            # Layer 4: SkipBlock combines:
            #   Path 1: Takes ReLU output [8,8,8,0,0,0] -> 4 twenty-fours (each output sums three 8s)
            #   Path 2: Takes Linear output [24,24,24,24,24] -> 4 one-twenties (each output sums five 24s)
            # Final output: 4 one-forty-fours (sum of both paths)
            @test all(output_num .≈ fill(144.0, output_size))
        end
        
        @testset "optimization with JuMP variables" begin
            model = TestHelpers.get_new_model()
            x_var = @variable(model, [1:input_size], lower_bound=-1, upper_bound=1)
            output_var = nnparams(x_var)
            @test size(output_var) == (output_size,)
            @test length(output_var) == output_size
            
            # Find maximum possible output value
            @objective(model, Max, sum(output_var))
            optimize!(model)
            @test termination_status(model) == OPTIMAL
            println("Optimal input: ", value.(x_var))
            println("Optimal output: ", value.(output_var))
            println("Objective value: ", objective_value(model))
            # When all inputs are 1, we get output of 144 (as shown in previous test)
            # Maximum should be 144 * output_size (when all inputs are 1)
            @test objective_value(model) ≈ 576.0 # 144 * 4
        end
    end
end
