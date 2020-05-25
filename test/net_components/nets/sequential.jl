using Test
using MIPVerify

@testset "sequential.jl" begin

    @testset "Sequential" begin
        nnparams = Sequential(
            [
                Conv2d(ones(1, 4, 4, 16), ones(16), 2),
                ReLU(),
                Flatten([3, 4, 1, 2]),
                Linear(ones(1000, 20), ones(20)),
                MaskedReLU([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
                Linear(ones(20, 10), ones(10)),
            ],
            "testnet",
        )
        io = IOBuffer()
        Base.show(io, nnparams)
        @test String(take!(io)) == """
        sequential net testnet
          (1) Conv2d(4, 16, kernel_size=(1, 4), stride=(2, 2), padding=same)
          (2) ReLU()
          (3) Flatten(): flattens 4 dimensional input, with dimensions permuted according to the order [3, 4, 1, 2]
          (4) Linear(1000 -> 20)
          (5) MaskedReLU with expected input size (20,). (10 zeroed, 7 as-is, 3 rectified).
          (6) Linear(20 -> 10)
        """
    end

end
