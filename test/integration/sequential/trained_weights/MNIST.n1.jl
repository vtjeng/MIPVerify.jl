using Test
using MIPVerify
using MIPVerify:
    UnrestrictedPerturbationFamily, BlurringPerturbationFamily, LInfNormBoundedPerturbationFamily
using MIPVerify: get_example_network_params, read_datasets, get_image
@isdefined(TestHelpers) || include("../../../TestHelpers.jl")

@testset "MNIST.n1.jl" begin
    nn = get_example_network_params("MNIST.n1")
    mnist = read_datasets("mnist")

    pp_blur = BlurringPerturbationFamily((5, 5))
    pp_unrestricted = UnrestrictedPerturbationFamily()

    sample_index = 1
    input = get_image(mnist.test.images, sample_index)

    @testset "Basic integration test demonstrating success on trained (non-robust) network." begin
        test_cases = [
            ((10, pp_unrestricted, Inf), 0.0460847),
            ((10, LInfNormBoundedPerturbationFamily(0.04), Inf), NaN),
            ((10, LInfNormBoundedPerturbationFamily(0.05), Inf), 0.0460847),
        ]

        TestHelpers.batch_test_adversarial_example(nn, input, test_cases)
    end

end
