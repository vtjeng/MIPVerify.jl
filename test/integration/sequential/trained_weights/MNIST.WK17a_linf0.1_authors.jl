using Test
using MIPVerify
using MIPVerify: LInfNormBoundedPerturbationFamily
using MIPVerify: get_example_network_params, read_datasets, get_image, get_label
@isdefined(TestHelpers) || include("../../../TestHelpers.jl")

@testset "MNIST.WK17a_linf0.1_authors.jl" begin
    nn = get_example_network_params("MNIST.WK17a_linf0.1_authors")
    mnist = read_datasets("mnist")

    test_cases = [(1, NaN), (2, NaN), (9, 0.0940014207), (248, 0)]

    for test_case in test_cases
        (index, expected_objective_value) = test_case
        @testset "Sample $index (1-indexed) with expected objective value $expected_objective_value" begin
            input = get_image(mnist.test.images, index)
            label = get_label(mnist.test.labels, index)
            TestHelpers.test_find_adversarial_example(
                nn,
                input,
                label + 1,
                LInfNormBoundedPerturbationFamily(0.1),
                Inf,
                expected_objective_value,
                invert_target_selection = true,
            )
        end
    end
end
