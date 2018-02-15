using Base.Test

using MIPVerify: MaskedFullyConnectedLayerParameters
using MIPVerify: SoftmaxParameters, MaskedFullyConnectedNetParameters

@testset "MaskedFullyConnectedNetParameters" begin
    
    A_height = 40
    A_width = 200
    A_mask = rand(MersenneTwister(0), [-1, 0, 1], A_height)
            
    B_height = 20
    B_width = A_height
    B_mask = rand(MersenneTwister(0), [-1, 0, 1], B_height)
            
    C_height = 10
    C_width = B_height
    
    fc1params = MaskedFullyConnectedLayerParameters(rand(A_width, A_height), rand(A_height), A_mask)
    fc2params = MaskedFullyConnectedLayerParameters(rand(B_width, B_height), rand(B_height), B_mask)
    softmaxparams = SoftmaxParameters(rand(C_width, C_height), rand(C_height))

    nnparams = MaskedFullyConnectedNetParameters(
        [fc1params, fc2params], 
        softmaxparams,
        "testnet1"
    )
    @testset "Base.show" begin
        io = IOBuffer()
        Base.show(io, nnparams)
        @test String(take!(io)) == """
            masked fully-connected net testnet1
              `masked_fclayer_params` [2]:
                masked fully connected layer with 200 inputs and 40 output units (14 zeroed, 15 as-is, 11 rectified).
                masked fully connected layer with 40 inputs and 20 output units (6 zeroed, 10 as-is, 4 rectified).
              `softmax_params`:
                softmax layer with 20 inputs and 10 output units."""
    end
end