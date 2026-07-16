using Test
using MIPVerify
using MIPVerify: UnrestrictedPerturbationFamily, LInfNormBoundedPerturbationFamily
@isdefined(TestHelpers) || include("../../../TestHelpers.jl")

TestHelpers.@timed_testset "mfc+mfc+softmax.jl" begin
    input = gen_array((1, 4, 8, 1), 0, 1)

    l1_height = 16
    l1_width = length(input)
    l1_kernel = gen_array((l1_width, l1_height), -1, 1)
    l1_bias = gen_array((l1_height,), -1, 1)

    # Include always-off (-1), ordinary ReLU (0), and always-on (1) units.
    m1 = [0, 0, 0, 0, -1, -1, -1, 1, 1, -1, 1, 0, 1, 0, -1, 1]

    l2_height = 8
    l2_width = l1_height
    l2_kernel = gen_array((l2_width, l2_height), -1, 1)
    l2_bias = gen_array((l2_height,), -1, 1)


    # Repeat all three mask states after the second linear layer.
    m2 = [0, -1, 1, 1, -1, -1, 0, 1]

    l3_height = 4
    l3_width = l2_height
    l3_kernel = gen_array((l3_width, l3_height), -1, 1)
    l3_bias = gen_array((l3_height,), -1, 1)

    nn = Sequential(
        [
            Flatten(4),
            Linear(l1_kernel, l1_bias),
            MaskedReLU(m1),
            Linear(l2_kernel, l2_bias),
            MaskedReLU(m2),
            Linear(l3_kernel, l3_bias),
        ],
        "tests.integration.generated_weights.mfc+mfc+softmax",
    )

    pp_unrestricted = UnrestrictedPerturbationFamily()

    @testset "Basic integration test for MaskedReLU layer." begin
        # Target 1 needs a nonzero perturbation through both MaskedReLU layers.
        unrestricted_test_cases = [((1, pp_unrestricted, Inf), 0.08112308)]

        TestHelpers.batch_test_adversarial_example(
            nn,
            input,
            unrestricted_test_cases,
            tightening_algorithm = interval_arithmetic,
        )

        bounded_test_cases = [
            # A bound just below that optimum makes the combined formulation infeasible.
            ((1, LInfNormBoundedPerturbationFamily(0.08), Inf), NaN),
            # A bound just above the optimum preserves the unrestricted objective.
            ((1, LInfNormBoundedPerturbationFamily(0.085), Inf), 0.08112308),
        ]

        # Preserve end-to-end LP tightening through MaskedReLU for the bounded formulation.
        TestHelpers.batch_test_adversarial_example(
            nn,
            input,
            bounded_test_cases,
            tightening_algorithm = lp,
        )
    end
end
