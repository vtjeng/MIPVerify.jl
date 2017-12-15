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

    sample_index = 1
    @testset "sample index = $sample_index" begin
        x0 = get_image(mnist.test.images, sample_index)
    
        expected_objective_values::Dict{Int, Dict{PerturbationParameters, Dict{Real, Dict{Real, Float64}}}} = Dict(
            1 => Dict(
                AdditivePerturbationParameters() => Dict(
                    1 => Dict(
                        0 => 13.8219
                    ),
                    Inf => Dict(
                        0 => 0.0964639
                    ),
                ),
            ),
            3 => Dict(
                BlurPerturbationParameters((5, 5)) => Dict(
                    1 => Dict(
                        0 => 88.0038
                    ),
                    Inf => Dict(
                        0 => 0.973011
                    ),
                ),
            ),
            10 => Dict(
                AdditivePerturbationParameters() => Dict(
                    1 => Dict(
                        0 => 4.64186
                    ),
                    Inf => Dict(
                        0 => 0.0460847
                    ),
                ),
            ),
            2 => Dict(BlurPerturbationParameters((5, 5)) => Dict(1 => Dict(0 => NaN), Inf => Dict(0 => NaN),),),
            5 => Dict(BlurPerturbationParameters((5, 5)) => Dict(1 => Dict(0 => NaN), Inf => Dict(0 => NaN),),),
            6 => Dict(BlurPerturbationParameters((5, 5)) => Dict(1 => Dict(0 => NaN), Inf => Dict(0 => NaN),),),
        )

        batch_test_adversarial_example(nnparams, x0, expected_objective_values)
    end

    sample_index = 3
    @testset "sample index = $sample_index" begin
        x0 = get_image(mnist.test.images, sample_index)
        expected_objective_values::Dict{Int, Dict{PerturbationParameters, Dict{Real, Dict{Real, Float64}}}} = Dict(
            8 => Dict(
                AdditivePerturbationParameters() => Dict(
                    1 => Dict(
                        0 => 0.582587
                    ),
                    Inf => Dict(
                        0 => 0.00363594
                    ),
                ),
                BlurPerturbationParameters((5, 5)) => Dict(
                    1 => Dict(
                        0 => 9.77784
                    ),
                    Inf => Dict(
                        0 => 0.277525
                    ),
                ),
            ),
        )
        
        batch_test_adversarial_example(nnparams, x0, expected_objective_values)
    end
    
end