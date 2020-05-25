using Test
using MIPVerify
using MIPVerify: UnrestrictedPerturbationFamily, BlurringPerturbationFamily, LInfNormBoundedPerturbationFamily
@isdefined(TestHelpers) || include("../../../TestHelpers.jl")

@timed_testset "conv+fc+softmax.jl" begin
    ### Parameters for neural net
    batch = 1
    c1_in_height = 6
    c1_in_width = 6
    p1_stride_height = 2
    p1_stride_width = 2
    p1_strides = (1, p1_stride_height, p1_stride_width, 1)
    p1_height = round(Int, c1_in_height/p1_stride_height, RoundUp)
    p1_width = round(Int, c1_in_width/p1_stride_width, RoundUp)
    c1_in_channels = 1
    c1_filter_height = 2
    c1_filter_width = 2
    c1_out_channels = 2

    l1_height = 4
    l1_width = p1_height*p1_width*c1_out_channels

    l2_height = 3
    l2_width = l1_height

    ### Choosing data to be used
    input = gen_array(
        (batch, c1_in_height, c1_in_width, c1_in_channels),
        0, 1
    )

    c1_kernel = gen_array(
        (c1_filter_height, c1_filter_width, c1_in_channels, c1_out_channels),
        -1, 1
    )
    c1_bias = gen_array((c1_out_channels,), -1, 1)

    l1_kernel = gen_array((l1_width, l1_height), -1, 1)
    l1_bias = gen_array((l1_height,), -1, 1)

    l2_kernel = gen_array((l2_width, l2_height), -1, 1)
    l2_bias = gen_array((l2_height,), -1, 1)

    nn = Sequential(
        [
            Conv2d(c1_kernel, c1_bias), MaxPool(p1_strides), ReLU(),
            Flatten(4),
            Linear(l1_kernel, l1_bias), ReLU(),
            Linear(l2_kernel, l2_bias)
        ],
        "tests.integration.generated_weights.conv+fc+softmax"
    )

    @timed_testset "BlurringPerturbationFamily" begin
        pp_blur = BlurringPerturbationFamily((5, 5))
        # TODO (vtjeng): Add example where blurring perturbation generates non-NaN results
        test_cases = [
            ((2, pp_blur, 1, 0), 6.959316),
            ((3, pp_blur, 1, 0), NaN),
        ]

        TestHelpers.batch_test_adversarial_example(nn, input, test_cases)
    end

    @timed_testset "UnrestrictedPerturbationFamily" begin
        pp_unrestricted = UnrestrictedPerturbationFamily()
        @timed_testset "Minimizing l1 norm" begin
            test_cases = [
                ((1, pp_unrestricted, 1, 0), 0),
                ((2, pp_unrestricted, 1, 0), 0.9187638),
            ]

            TestHelpers.batch_test_adversarial_example(nn, input, test_cases)
        end

        @timed_testset "Minimizing lInf norm" begin
            test_cases = [
                ((1, pp_unrestricted, Inf, 0), 0),
                ((2, pp_unrestricted, Inf, 0), 0.06688736),
                ((3, pp_unrestricted, Inf, 0), 0.4270584),
            ]

            TestHelpers.batch_test_adversarial_example(nn, input, test_cases)
        end

        @timed_testset "Increasing margin increases required distance" begin
            test_cases = [
                ((2, pp_unrestricted, 1, 0.1), 0.98717927),
            ]

            TestHelpers.batch_test_adversarial_example(nn, input, test_cases)
        end

        @timed_testset "With multiple target labels specified, minimum target label found" begin
            test_cases = [
                (([2, 3], pp_unrestricted, Inf, 0), 0.06688736),
            ]

            TestHelpers.batch_test_adversarial_example(nn, input, test_cases)
        end
    end

    @timed_testset "LInfNormBoundedPerturbationFamily" begin
        test_cases = [
            ((2, LInfNormBoundedPerturbationFamily(0.06), Inf, 0), NaN),  # restricting maximum perturbation to below minimum distance causes optimization problem to be infeasible
            ((2, LInfNormBoundedPerturbationFamily(0.07), Inf, 0), 0.06688736),  # restricting maximum perturbation to above minimum distance does not affect optimal value of problem
        ]

        TestHelpers.batch_test_adversarial_example(nn, input, test_cases)
    end
end