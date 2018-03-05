using Base.Test
using JuMP
using MIPVerify
using MIPVerify: check_size, increment!
isdefined(:TestHelpers) || include("../../TestHelpers.jl")
using TestHelpers: get_new_model

@testset "conv2d.jl" begin

    @testset "Conv2d" begin
        @testset "Base.show" begin
            filter = rand(3, 3, 2, 5)
            p = Conv2d(filter)
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "Conv2d(2, 5, kernel_size=(3, 3), stride=(1, 1), padding=same)"
        end
        @testset "With Bias" begin
            @testset "Matched Size" begin
                out_channels = 5
                filter = rand(3, 3, 2, out_channels)
                bias = rand(out_channels)
                p = Conv2d(filter, bias)
                @test p.filter == filter
                @test p.bias == bias
            end
            @testset "Unmatched Size" begin
                filter_out_channels = 4
                bias_out_channels = 5
                filter = rand(3, 3, 2, filter_out_channels)
                bias = rand(bias_out_channels)
                @test_throws AssertionError Conv2d(filter, bias)
            end
        end
        @testset "No Bias" begin
            filter = rand(3, 3, 2, 5)
            p = Conv2d(filter)
            @test p.filter == filter
        end
        @testset "JuMP Variables" begin
            m = Model()
            filter_size = (3, 3, 2, 5)
            filter = map(_ -> @variable(m), CartesianRange(filter_size))
            p = Conv2d(filter)
            @test p.filter == filter
        end
        @testset "check_size" begin
            filter = rand(3, 3, 2, 5)
            p = Conv2d(filter)
            @test check_size(p, (3, 3, 2, 5)) === nothing
            @test_throws AssertionError check_size(p, (3, 3, 2, 4))
        end
    end

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
        p = Conv2d(filter, bias)
        @testset "Numerical Input, Numerical Layer Parameters" begin
            evaluated_output = MIPVerify.conv2d(input, p)
            @test evaluated_output == true_output
        end
        @testset "Numerical Input, Variable Layer Parameters" begin
            m = get_new_model()
            filter_v = map(_ -> @variable(m), CartesianRange(filter_size))
            bias_v = map(_ -> @variable(m), CartesianRange(bias_size))
            p_v = Conv2d(filter_v, bias_v)
            output_v = MIPVerify.conv2d(input, p_v)
            @constraint(m, output_v .== true_output)
            solve(m)

            p_solve = MIPVerify.Conv2d(getvalue(filter_v), getvalue(bias_v))
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