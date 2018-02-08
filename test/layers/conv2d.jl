using Base.Test
using JuMP
using MIPVerify: Conv2DParameters
using MIPVerify: increment!
using MIPVerify.TestHelpers: get_new_model

@testset "conv2d.jl" begin

    @testset "increment!" begin
        @testset "Real * Real" begin
            @test 7 == increment!(1, 2, 3)
        end
        @testset "JuMP.AffExpr * Real" begin
            m = get_new_model()
            x = @variable(m, start=100)
            y = @variable(m, start=1)
            s = 5*x+3*y
            t = 3*x+2*y
            increment!(s, 2, t)
            @test getvalue(s)==1107
            increment!(s, t, -1)
            @test getvalue(s)==805
            increment!(s, x, 3)
            @test getvalue(s)==1105
            increment!(s, y, 2)
            @test getvalue(s)==1107
        end
    end

    @testset "conv2d" begin
        srand(1)
        input_size = (1, 4, 4, 2)
        input = rand(0:5, input_size)
        filter_size = (3, 3, 2, 1)
        filter = rand(0:5, filter_size)
        bias_size = (1, )
        bias = rand(0:5, bias_size)
        true_output_raw = [
            49 74 90 56;
            67 118 140 83;
            66 121 134 80;
            56 109 119 62            
        ]
        true_output = reshape(transpose(true_output_raw), (1, 4, 4, 1))
        p = MIPVerify.Conv2DParameters(filter, bias)
        @testset "Numerical Input, Numerical Layer Parameters" begin
            evaluated_output = MIPVerify.conv2d(input, p)
            @test evaluated_output == true_output
        end
        @testset "Numerical Input, Variable Layer Parameters" begin
            m = get_new_model()
            filter_v = map(_ -> @variable(m), CartesianRange(filter_size))
            bias_v = map(_ -> @variable(m), CartesianRange(bias_size))
            p_v = MIPVerify.Conv2DParameters(filter_v, bias_v)
            output_v = MIPVerify.conv2d(input, p_v)
            @constraint(m, output_v .== true_output)
            solve(m)

            p_solve = MIPVerify.Conv2DParameters(getvalue(filter_v), getvalue(bias_v))
            solve_output = MIPVerify.conv2d(input, p_solve)
            @test solve_output≈true_output
        end
        @testset "Variable Input, Numerical Layer Parameters" begin
            m = get_new_model()
            input_v = map(_ -> @variable(m), CartesianRange(input_size))
            output_v = MIPVerify.conv2d(input_v, p)
            @constraint(m, output_v .== true_output)
            solve(m)

            solve_output = MIPVerify.conv2d(getvalue(input_v), p)
            @test solve_output≈true_output
        end
    end
end