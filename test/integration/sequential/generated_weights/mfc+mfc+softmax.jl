using Base.Test
using MIPVerify
using MIPVerify: BlurringPerturbationFamily, UnrestrictedPerturbationFamily
using MIPVerify: LInfNormBoundedPerturbationFamily
isdefined(:TestHelpers) || include("../../../TestHelpers.jl")
using TestHelpers: batch_test_adversarial_example

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
    
    nn = Sequential(
        [
            Flatten(4),
            Linear(rand(A_width, A_height)-0.5, rand(A_height)-0.5), 
            MaskedReLU(A_mask),
            Linear(rand(B_width, B_height)-0.5, rand(B_height)-0.5),
            MaskedReLU(B_mask),
            Linear(rand(C_width, C_height), rand(C_height))
        ],
        "tests.integration.generated_weights.mfc+mfc+softmax"
    )

    pp_blur = BlurringPerturbationFamily((5, 5))
    pp_unrestricted = UnrestrictedPerturbationFamily()

    expected_objective_values = Dict(
        (1, pp_unrestricted, 1, 0) => 1.93413,
        (1, pp_unrestricted, Inf, 0) => 0.153096,
        (1, LInfNormBoundedPerturbationFamily(0.15), Inf, 0) => NaN,
        (1, LInfNormBoundedPerturbationFamily(0.1531), Inf, 0) => 0.153096,
        (1, LInfNormBoundedPerturbationFamily(0.3), Inf, 0) => 0.153096,
        (2, pp_unrestricted, 1, 0) => 0,
        (3, pp_unrestricted, 1, 0) => NaN,
        (4, pp_unrestricted, 1, 0) => 5.69694,
        (1, pp_unrestricted, 1, 0.1) => 2.39582,
        (1, pp_unrestricted, 1, 10) => NaN,
        (2, pp_blur, 1, 0) => 0,
        (3, pp_blur, 1, 0) => NaN,
        (2, pp_blur, 1, 0.5) => 2.44867,
        (2, pp_blur, 1, 1) => NaN
        
    )

    batch_test_adversarial_example(nn, x0, expected_objective_values)

end