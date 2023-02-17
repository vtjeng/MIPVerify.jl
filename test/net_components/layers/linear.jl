using Test
using JuMP
using MIPVerify: Linear, check_size
@isdefined(TestHelpers) || include("../../TestHelpers.jl")

@testset "linear.jl" begin

    @testset "Linear" begin
        @testset "With Bias" begin
            @testset "Matched Size" begin
                height = 10
                matrix = ones(2, height)
                bias = ones(height)
                p = Linear(matrix, bias)
                @test p.matrix == matrix
                @test p.bias == bias
                @testset "check_size" begin
                    @test check_size(p, (2, 10)) === nothing
                    @test_throws AssertionError check_size(p, (2, 9))
                end
            end
            @testset "Unmatched Size" begin
                matrix_height = 10
                bias_height = 5
                matrix = ones(2, matrix_height)
                bias = ones(bias_height)
                @test_throws AssertionError Linear(matrix, bias)
            end
            @testset "Multiplying by >1-dimensional array" begin
                height = 15
                matrix = ones(2, height)
                bias = ones(height)
                p = Linear(matrix, bias)
                @test_throws ArgumentError p(ones(3, 5))
            end
        end

        @testset "Base.show" begin
            height = 13
            matrix = ones(37, height)
            bias = ones(height)
            p = Linear(matrix, bias)
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "Linear(37 -> 13)"
        end

        @testset "matmul" begin
            matrix = [[1, 2] [3, 4]]
            bias = [5, 6]
            p = Linear(matrix, bias)
            @testset "Real * Real" begin
                @test p([7, 8]) == [28, 59]
            end
            @testset "JuMP.AffExpr * Real" begin
                m = TestHelpers.get_new_model()
                x = @variable(m)
                y = @variable(m)
                @constraint(m, x == 7)
                @constraint(m, y == 8)
                optimize!(m)
                @test JuMP.value.(p([x, y])) == [28, 59]
            end
        end

    end

end
