using Base.Test
using MIPVerify: MaskedFullyConnectedLayerParameters, SoftmaxParameters
using MIPVerify: MaskedFullyConnectedNetParameters
using MIPVerify: AdditivePerturbationParameters, BlurPerturbationParameters
using MIPVerify.TestHelpers: batch_test_adversarial_example

@testset "mfc + mfc + softmax" begin

    srand(5)

    x0 = rand(1, 4, 8, 1)

    A_height = 16
    A_width = length(x0)
    A_mask = rand([-1, 0, 1], A_height)
            
    B_height = 8
    B_width = A_height
    B_mask = rand([-1, 0, 1], B_height)
            
    C_height = 4
    C_width = B_height
    
    fc1params = MaskedFullyConnectedLayerParameters(rand(A_width, A_height)-0.5, rand(A_height)-0.5, A_mask)
    fc2params = MaskedFullyConnectedLayerParameters(rand(B_width, B_height)-0.5, rand(B_height)-0.5, B_mask)
    softmaxparams = SoftmaxParameters(rand(C_width, C_height), rand(C_height))

    nnparams = MaskedFullyConnectedNetParameters(
        [fc1params, fc2params], 
        softmaxparams,
        "tests.integration.masked_fc_net.generated_weights.mfc+mfc+softmax"
    )

    pp_blur = BlurPerturbationParameters((5, 5))
    pp_additive = AdditivePerturbationParameters()

    expected_objective_values = Dict(
        (1, pp_additive, 1, 0) => 1.93413,
        (2, pp_additive, 1, 0) => 0,
        (3, pp_additive, 1, 0) => NaN,
        (4, pp_additive, 1, 0) => 5.69694,
        (1, pp_additive, 1, 0.1) => 2.39582,
        (1, pp_additive, 1, 10) => NaN,
        (2, pp_blur, 1, 0) => 0,
        (3, pp_blur, 1, 0) => NaN,
        (2, pp_blur, 1, 0.5) => 2.44867,
        (2, pp_blur, 1, 1) => NaN
        
    )

    batch_test_adversarial_example(nnparams, x0, expected_objective_values)

end