using Test
using JuMP
using MIPVerify
using Random

@testset "sequential.jl" begin

    @testset "Sequential" begin
        nnparams = Sequential([
            Conv2d(rand(Float64, 1, 4, 4, 16), rand(Float64, 16), 2),
            ReLU(),
            Flatten([3, 4, 1, 2]),
            Linear(rand(Float64, 1000, 50), rand(Float64, 50)),
            MaskedReLU(rand(Random.MersenneTwister(0), [-1, 0, 1], 50)),
            Linear(rand(Float64, 50, 10), rand(Float64, 10))
        ], "testnet")
        io = IOBuffer()
        Base.show(io, nnparams)
        @test String(take!(io)) == """
        sequential net testnet
          (1) Conv2d(4, 16, kernel_size=(1, 4), stride=(2, 2), padding=same)
          (2) ReLU()
          (3) Flatten(): flattens 4 dimensional input, with dimensions permuted according to the order [3, 4, 1, 2]
          (4) Linear(1000 -> 50)
          (5) MaskedReLU with expected input size (50,). (16 zeroed, 17 as-is, 17 rectified).
          (6) Linear(50 -> 10)
        """
    end

end