using Test
using MIPVerify
using MIPVerify:
    UnrestrictedPerturbationFamily, BlurringPerturbationFamily, LInfNormBoundedPerturbationFamily
@isdefined(TestHelpers) || include("TestHelpers.jl")

struct UncheckedCustomPerturbationFamily <: MIPVerify.PerturbationFamily end

TestHelpers.@timed_testset "models.jl" begin
    @testset "UnrestrictedPerturbationFamily" begin
        @testset "Base.show" begin
            p = UnrestrictedPerturbationFamily()
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "unrestricted"
        end
    end
    @testset "BlurringPerturbationFamily" begin
        @testset "Base.show" begin
            p = BlurringPerturbationFamily((5, 5))
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "blur-(5,5)"
        end
    end
    @testset "LInfNormBoundedPerturbationFamily" begin
        @testset "Base.show" begin
            p = LInfNormBoundedPerturbationFamily(0.1)
            io = IOBuffer()
            Base.show(io, p)
            @test String(take!(io)) == "linf-norm-bounded-0.1"
        end
    end

    @testset "independent witness constraint checks" begin
        @testset "unrestricted" begin
            # The candidate endpoints exercise both inclusive [0, 1] bounds.
            input = [0.25, 0.75]
            candidate = [0.0, 1.0]
            verified, _ = MIPVerify.verify_perturbation_witness(
                UnrestrictedPerturbationFamily(),
                input,
                candidate,
                Dict(),
            )
            @test verified

            # A value 2e-8 below zero exceeds the documented 1e-8 absolute tolerance.
            below_domain, _ = MIPVerify.verify_perturbation_witness(
                UnrestrictedPerturbationFamily(),
                input,
                [-2e-8, 1.0],
                Dict(),
            )
            @test !below_domain

            # A one-element candidate exercises exact shape validation.
            wrong_shape, _ = MIPVerify.verify_perturbation_witness(
                UnrestrictedPerturbationFamily(),
                input,
                [0.25],
                Dict(),
            )
            @test !wrong_shape

            # A non-finite candidate must never establish perturbation membership.
            nonfinite, _ = MIPVerify.verify_perturbation_witness(
                UnrestrictedPerturbationFamily(),
                input,
                [NaN, 0.75],
                Dict(),
            )
            @test !nonfinite
        end

        @testset "L-infinity bounded" begin
            # Differences of exactly 0.1 exercise both signs at the perturbation boundary.
            pp = LInfNormBoundedPerturbationFamily(0.1)
            input = [0.5, 0.5]
            boundary_candidate = [0.6, 0.4]
            verified, _ =
                MIPVerify.verify_perturbation_witness(pp, input, boundary_candidate, Dict())
            @test verified

            # A 1e-4 budget violation is well outside rounding tolerance.
            outside_radius, _ =
                MIPVerify.verify_perturbation_witness(pp, input, [0.6001, 0.4], Dict())
            @test !outside_radius

            # This candidate stays within radius 0.1 but exceeds the input domain by 0.01.
            outside_domain, _ =
                MIPVerify.verify_perturbation_witness(pp, [0.95, 0.5], [1.01, 0.5], Dict())
            @test !outside_domain

            # A 1e-9 change exceeds a 1e-12 budget. This catches accidental use of the fixed
            # 1e-8 absolute target tolerance for the perturbation radius.
            tiny_pp = LInfNormBoundedPerturbationFamily(1e-12)
            tiny_budget_violation, _ =
                MIPVerify.verify_perturbation_witness(tiny_pp, [0.5], [0.5 + 1e-9], Dict())
            @test !tiny_budget_violation
        end

        @testset "blur" begin
            # Twelve distinct in-domain values across two channels make spatial shifts and
            # cross-channel mistakes observable. The even 2x4 kernel exercises asymmetric SAME
            # padding in the identity construction.
            input = reshape(collect(range(0.05, 0.6; length = 12)), 1, 2, 3, 2)
            pp = BlurringPerturbationFamily((2, 4))
            verified, values = MIPVerify.verify_perturbation_witness(pp, input, copy(input), Dict())
            @test verified
            @test size(values[:WitnessBlurKernel]) == (2, 4, 2, 2)
            @test sum(values[:WitnessBlurKernel]) == 2.0
            @test input |> MIPVerify.Conv2d(values[:WitnessBlurKernel]) == input

            identity_kernel = values[:WitnessBlurKernel]
            # Raising one pixel by 0.01 leaves it in [0, 1] but breaks reconstruction by the
            # otherwise valid identity kernel.
            altered_candidate = copy(input)
            altered_candidate[1] += 0.01
            bad_reconstruction, _ = MIPVerify.verify_perturbation_witness(
                pp,
                input,
                altered_candidate,
                Dict(:WitnessBlurKernel => identity_kernel),
            )
            @test !bad_reconstruction

            # Removing one identity coefficient makes the global kernel sum 1 instead of the
            # required channel count 2. Zero input isolates the sum check because every kernel
            # reconstructs the zero candidate.
            zero_input = zeros(size(input))
            wrong_sum_kernel = copy(identity_kernel)
            wrong_sum_kernel[1, 2, 1, 1] = 0.0
            wrong_sum, _ = MIPVerify.verify_perturbation_witness(
                pp,
                zero_input,
                copy(zero_input),
                Dict(:WitnessBlurKernel => wrong_sum_kernel),
            )
            @test !wrong_sum

            # Raising one identity coefficient to 1.0001 isolates the upper kernel bound. Reducing
            # the other channel's coefficient to 0.9999 preserves the required global sum.
            above_bound_kernel = copy(identity_kernel)
            above_bound_kernel[1, 2, 1, 1] = 1.0001
            above_bound_kernel[1, 2, 2, 2] -= 0.0001
            above_bound, _ = MIPVerify.verify_perturbation_witness(
                pp,
                zero_input,
                copy(zero_input),
                Dict(:WitnessBlurKernel => above_bound_kernel),
            )
            @test !above_bound

            # Four 0.5 coefficients form an otherwise valid sum-2 kernel. Moving 0.0001 from a
            # zero coefficient to another coefficient isolates the lower kernel bound.
            below_bound_kernel = zeros(size(identity_kernel))
            below_bound_kernel[1, 2, 1, 1] = 0.5
            below_bound_kernel[1, 2, 2, 2] = 0.5
            below_bound_kernel[1, 1, 1, 1] = 0.5
            below_bound_kernel[1, 1, 2, 2] = 0.5
            below_bound_kernel[1, 1, 1, 2] = -0.0001
            below_bound_kernel[1, 1, 1, 1] += 0.0001
            below_bound, _ = MIPVerify.verify_perturbation_witness(
                pp,
                zero_input,
                copy(zero_input),
                Dict(:WitnessBlurKernel => below_bound_kernel),
            )
            @test !below_bound

            # Replacing one identity coefficient with NaN exercises the finite-kernel check.
            nonfinite_kernel = copy(identity_kernel)
            nonfinite_kernel[1, 2, 1, 1] = NaN
            nonfinite_kernel_verified, _ = MIPVerify.verify_perturbation_witness(
                pp,
                zero_input,
                copy(zero_input),
                Dict(:WitnessBlurKernel => nonfinite_kernel),
            )
            @test !nonfinite_kernel_verified

            # A 1x1 kernel has the wrong spatial shape for this 2x4 blur family.
            wrong_kernel_shape, _ = MIPVerify.verify_perturbation_witness(
                pp,
                input,
                copy(input),
                Dict(:WitnessBlurKernel => zeros(1, 1, 2, 2)),
            )
            @test !wrong_kernel_shape

            # Non-integer dimensions are accepted by the legacy constructor type but cannot
            # define a convolution kernel; the identity fast path must fail instead of throwing.
            invalid_pp = BlurringPerturbationFamily((2.0, 4.0))
            invalid_configuration, _ =
                MIPVerify.verify_perturbation_witness(invalid_pp, input, copy(input), Dict())
            @test !invalid_configuration
        end

        @testset "custom family fails closed" begin
            # The generic hook has no semantic knowledge of this family and must not trust a
            # numerically plausible candidate.
            verified, values = MIPVerify.verify_perturbation_witness(
                UncheckedCustomPerturbationFamily(),
                [0.5],
                [0.5],
                Dict(),
            )
            @test !verified
            @test isempty(values)
        end
    end
end
