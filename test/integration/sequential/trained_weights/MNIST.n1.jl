using Base.Test
using MIPVerify
using MIPVerify: UnrestrictedPerturbationFamily, BlurringPerturbationFamily
using MIPVerify: get_example_network_params, read_datasets, get_image
isdefined(:TestHelpers) || include("../../../TestHelpers.jl")
using TestHelpers: batch_test_adversarial_example

@testset "MNIST.n1" begin
    nn = get_example_network_params("MNIST.n1")
    mnist = read_datasets("mnist")

    pp_blur = BlurringPerturbationFamily((5, 5))
    pp_unrestricted = UnrestrictedPerturbationFamily()

    sample_index = 1
    @testset "sample index = $sample_index" begin
        x0 = get_image(mnist.test.images, sample_index)

        expected_objective_values = Dict(
            (1, pp_unrestricted, 1, 0) => 13.8219,
            (1, pp_unrestricted, Inf, 0) => 0.0964639,
            (3, pp_blur, 1, 0) => 88.0038,
            (3, pp_blur, Inf, 0) => 0.973011,
            (10, pp_unrestricted, 1, 0) => 4.64186,
            (10, pp_unrestricted, Inf, 0) => 0.0460847,
            (2, pp_blur, 1, 0) => NaN,
            (2, pp_blur, Inf, 0) => NaN,
            (5, pp_blur, 1, 0) => NaN,
            (5, pp_blur, Inf, 0) => NaN,
            (6, pp_blur, 1, 0) => NaN,
            (6, pp_blur, Inf, 0) => NaN
        )
    
        batch_test_adversarial_example(nn, x0, expected_objective_values)
    end

    sample_index = 3
    @testset "sample index = $sample_index" begin
        x0 = get_image(mnist.test.images, sample_index)
        
        expected_objective_values = Dict(
            (8, pp_unrestricted, 1, 0) => 0.582587,
            (8, pp_unrestricted, Inf, 0) => 0.00363594,
            (8, pp_blur, 1, 0) => 9.77784,
            (8, pp_blur, Inf, 0) => 0.277525
        )
        
        batch_test_adversarial_example(nn, x0, expected_objective_values)
    end
    
end