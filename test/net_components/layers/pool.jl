using Base.Test
using JuMP
using MIPVerify: Pool, MaxPool, AveragePool
using MIPVerify: getsliceindex, getpoolview
isdefined(:TestHelpers) || include("../../TestHelpers.jl")
using TestHelpers: get_new_model


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
            @test getsliceindex(10, 2, 3)==[5, 6]
            @test getsliceindex(10, 3, 4)==[10]
        end
        @testset "out of bounds" begin
            @test getsliceindex(10, 5, 4)==[]
        end
    end
    
    input_size = (6, 6)
    input_array = reshape(1:*(input_size...), input_size)
    @testset "getpoolview" begin
        @testset "inbounds" begin
            @test getpoolview(input_array, (2, 2), (3, 3)) == [29 35; 30 36]
            @test getpoolview(input_array, (1, 1), (3, 3)) == cat(2, [15])
        end
        @testset "out of bounds" begin
            @test length(getpoolview(input_array, (1, 1), (7, 7))) == 0
        end
    end

    @testset "maxpool" begin
        true_output = [
            8 20 32;
            10 22 34;
            12 24 36
        ]
        @testset "Numerical Input" begin
            @test MIPVerify.pool(input_array, MaxPool((2, 2))) == true_output
        end
        @testset "Variable Input" begin
            m = get_new_model()
            input_array_v = map(
                i -> @variable(m, lowerbound=i-2, upperbound=i), 
                input_array
            )
            pool_v = MIPVerify.pool(input_array_v, MaxPool((2, 2)))
            # elements of the input array are made to take their maximum value
            @objective(m, Max, sum(input_array_v))
            solve(m)

            solve_output = getvalue.(pool_v)
            @test solve_output≈true_output
        end
    end

    @testset "avgpool" begin
        @testset "Numerical Input" begin
            true_output = [
                4.5 16.5 28.5;
                6.5 18.5 30.5;
                8.5 20.5 32.5
            ]
            @test MIPVerify.pool(input_array, AveragePool((2, 2))) == true_output
        end
    end

end

