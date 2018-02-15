using Base.Test
using MIPVerify: MatrixMultiplicationParameters, check_size

@testset "matmul.jl" begin

@testset "MatrixMultiplicationParameters" begin
    @testset "With Bias" begin
        @testset "Matched Size" begin
            height = 10
            matrix = rand(2, height)
            bias = rand(height)
            p = MatrixMultiplicationParameters(matrix, bias)
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
            matrix = rand(2, matrix_height)
            bias = rand(bias_height)
            @test_throws AssertionError MatrixMultiplicationParameters(matrix, bias)
        end
    end
end

end