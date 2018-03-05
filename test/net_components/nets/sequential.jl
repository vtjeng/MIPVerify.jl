using Base.Test
using JuMP
using MIPVerify

@testset "sequential.jl" begin

    @testset "Sequential" begin
        nnparams = Sequential([
            Conv2d(rand(1, 4, 4, 16), rand(16), 2),
            ReLU(),
            Flatten([3, 4, 1, 2]),
            Linear(rand(1000, 50), rand(50)),
            MaskedReLU(rand(MersenneTwister(0), [-1, 0, 1], 50)),
            Linear(rand(50, 10), rand(10))
        ], "testnet")
        io = IOBuffer()
        Base.show(io, nnparams)
        @test String(take!(io)) == """
        sequential net testnet
          (1) Conv2d(4, 16, kernel_size=(1, 4), stride=(2, 2), padding=same)
          (2) ReLU()
          (3) Flatten(): flattens 4 dimensional input, with dimensions permuted according to the order [3, 4, 1, 2]
          (4) Linear(1000 -> 50)
          (5) MaskedReLU with expected input size (50,). (18 zeroed, 17 as-is, 15 rectified).
          (6) Linear(50 -> 10)
        """
    end

end