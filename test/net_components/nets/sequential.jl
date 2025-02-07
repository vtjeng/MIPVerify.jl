using Test
using MIPVerify

@testset "sequential.jl" begin

    @testset "Sequential" begin
        nnparams = Sequential(
            [
                Conv2d(ones(4, 4, 7, 16), ones(16), 2),  # [height=4, width=4, in_channels=5, out_channels=16]
                ReLU(),
                Flatten([3, 4, 1, 2]),
                Linear(-ones(400, 20), ones(20)),
                MaskedReLU([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
                Linear(ones(20, 10), ones(10)),
            ],
            "testnet",
        )

        @testset "Base.show" begin
            io = IOBuffer()
            Base.show(io, nnparams)
            @test String(take!(io)) == """
            sequential net testnet
              (1) Conv2d(7, 16, kernel_size=(4, 4), stride=(2, 2), padding=same)
              (2) ReLU()
              (3) Flatten(): flattens 4 dimensional input, with dimensions permuted according to the order [3, 4, 1, 2]
              (4) Linear(400 -> 20)
              (5) MaskedReLU with expected input size (20,). (10 zeroed, 7 as-is, 3 rectified).
              (6) Linear(20 -> 10)
            """
        end

        @testset "forward pass with array input" begin
            # Create input: [batch=1, height=10, width=10, channels=7]
            x_num = ones(1, 10, 10, 7)
            output_num = nnparams(x_num)

            # Test output dimensions
            @test size(output_num) == (10,)  # Final linear layer outputs 10 values

            # Test forward pass computation
            # Layer 1 (Conv2d):
            #   - Input: [1,10,10,7]
            #   - Kernel: [4,4,7,16] (all ones)
            #   - Bias: ones
            #   - Stride: (2,2)
            #   - Output: [1,5,5,16] (64 sixty-fours, 192 eight-fives, 144 one-hundred-and-thirteens)
            #   NOTE: The differences in output values is due to the padding behavior.
            # Layer 2 (ReLU):
            #   - All values positive, no change
            #   - Output: [1,5,5,16] (64 sixty-fours, 192 eight-fives, 144 one-hundred-and-thirteens)
            # Layer 3 (Flatten with permutation [3,4,1,2]):
            #   - Input: [1,5,5,16]
            #   - Permuted: [5,16,1,5]
            #   - Flattens to 400 (5*16*1*5)
            # Layer 4 (Linear 400->20):
            #   - Matrix of minus ones, bias of ones
            #   - Each output = -(64*64+192*85+144*113) + 1 = -36687
            # Layer 5 (MaskedReLU):
            #   - First 10 values: 0 (zeroed due to mask=-1)
            #   - Next 7 values: -36687 (pass-through due to mask=1)
            #   - Last 3 values: 0 (ReLU applied due to mask=0) 
            # Layer 6 (Linear 20->10):
            #   - Matrix of ones, bias of ones
            #   - Each output = sum of (0*10 + -36687*7 + 0*3) + 1 = -256808
            @test all(output_num .≈ fill(-256808.0, 10))
        end
    end

end
