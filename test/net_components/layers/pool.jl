using Test
using JuMP
using MIPVerify
@isdefined(TestHelpers) || include("../../TestHelpers.jl")

@testset "pool.jl" begin

    @testset "Pool" begin
        strides = (1, 2, 2, 1)
        p = Pool(strides, MIPVerify.maximum)
        @test p.strides == strides
        @testset "Base.show" begin
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "max pooling with a 2x2 filter and a stride of (2, 2)"
        end
    end

    @testset "getsliceindex" begin
        @testset "inbounds" begin
            @test MIPVerify.getsliceindex(10, 2, 3) == [5, 6]
            @test MIPVerify.getsliceindex(10, 3, 4) == [10]
        end
        @testset "out of bounds" begin
            @test MIPVerify.getsliceindex(10, 5, 4) == []
        end
    end

    input_size = (6, 6)
    input_array = reshape(1:*(input_size...), input_size)
    @testset "getpoolview" begin
        @testset "inbounds" begin
            @test MIPVerify.getpoolview(input_array, (2, 2), (3, 3)) == [29 35; 30 36]
            @test MIPVerify.getpoolview(input_array, (1, 1), (3, 3)) == reshape([15], (1, 1))
        end
        @testset "out of bounds" begin
            @test length(MIPVerify.getpoolview(input_array, (1, 1), (7, 7))) == 0
        end
    end

    @testset "maxpool" begin
        true_output = [
            8 20 32
            10 22 34
            12 24 36
        ]
        @testset "Numerical Input" begin
            @test MIPVerify.pool(input_array, MaxPool((2, 2))) == true_output
        end
        @testset "Variable Input" begin
            m = TestHelpers.get_new_model()
            input_array_v =
                map(i -> @variable(m, lower_bound = i - 2, upper_bound = i), input_array)
            pool_v = MIPVerify.pool(input_array_v, MaxPool((2, 2)))
            # elements of the input array are made to take their maximum value
            @objective(m, Max, sum(input_array_v))
            optimize!(m)

            solve_output = JuMP.value.(pool_v)
            @test solve_output ≈ true_output
        end
    end

end
