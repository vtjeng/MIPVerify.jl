using MIPVerify: PerturbationParameters, AdditivePerturbationParameters, BlurPerturbationParameters
using MIPVerify: get_example_network_params, read_datasets, get_image
using MIPVerify.IntegrationTestHelpers: batch_test_adversarial_example
using MIPVerify: remove_cached_models
using MIPVerify: get_max_index
using Base.Test

@testset "MNIST.n1" begin
    # remove_cached_models()
    nnparams = get_example_network_params("MNIST.n1")
    mnist = read_datasets("MNIST")

    pp_blur = BlurPerturbationParameters((5, 5))
    pp_additive = AdditivePerturbationParameters()

    sample_index = 1
    @testset "sample index = $sample_index" begin
        x0 = get_image(mnist.test.images, sample_index)

        expected_objective_values::Dict{Tuple{Int, PerturbationParameters, Real, Real}, Float64} = Dict(
            (1, pp_additive, 1, 0) => 13.8219,
            (1, pp_additive, Inf, 0) => 0.0964639,
            (3, pp_blur, 1, 0) => 88.0038,
            (3, pp_blur, Inf, 0) => 0.973011,
            (10, pp_additive, 1, 0) => 4.64186,
            (10, pp_additive, Inf, 0) => 0.0460847,
            (2, pp_blur, 1, 0) => NaN,
            (2, pp_blur, Inf, 0) => NaN,
            (5, pp_blur, 1, 0) => NaN,
            (5, pp_blur, Inf, 0) => NaN,
            (6, pp_blur, 1, 0) => NaN,
            (6, pp_blur, Inf, 0) => NaN
        )
    
        batch_test_adversarial_example(nnparams, x0, expected_objective_values)
    end

    sample_index = 3
    @testset "sample index = $sample_index" begin
        x0 = get_image(mnist.test.images, sample_index)
        
        expected_objective_values::Dict{Tuple{Int, PerturbationParameters, Real, Real}, Float64} = Dict(
            (8, pp_additive, 1, 0) => 0.582587,
            (8, pp_additive, Inf, 0) => 0.00363594,
            (8, pp_blur, 1, 0) => 9.77784,
            (8, pp_blur, Inf, 0) => 0.277525
        )
        
        batch_test_adversarial_example(nnparams, x0, expected_objective_values)
    end
    
end