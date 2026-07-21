using HiGHS
using JuMP
using MIPVerify
using Random

const PGD_MOI = JuMP.MOI

function toy_worst_margin_problem()
    model = JuMP.Model()
    # Two pixels are enough to distinguish original and PGD sparse starts while keeping the
    # full-start completion solve tiny.
    @variable(model, 0 <= perturbed_input[1:2] <= 1)
    @variable(model, -1 <= perturbation[1:2] <= 1)
    @variable(model, hidden)
    @variable(model, output[1:2])
    @variable(model, margin)
    @variable(model, phase, Bin)
    original = [0.2, 0.4]
    @constraint(model, perturbation .== perturbed_input .- original)
    @constraint(model, hidden == perturbed_input[1] + phase)
    @constraint(model, output[1] == hidden)
    @constraint(model, output[2] == perturbed_input[2])
    @constraint(model, margin == output[2] - output[1])
    @objective(model, Max, margin)
    return WorstMarginProblem(
        model,
        perturbed_input,
        perturbation,
        output,
        margin,
        1,
        0.0,
        PGDWarmStart.model_signature(model),
        Dict{Symbol,Any}(),
    )
end

function synthetic_wk17a_style_network()
    # Even 4x4 kernels, stride two, and asymmetric SAME padding reproduce the layout-sensitive
    # convolution pattern in WK17a on a much smaller input.
    conv1_filter = reshape(collect(-16.0:15.0), 4, 4, 1, 2) ./ 100
    conv2_filter = reshape(sin.(collect(1.0:64.0)), 4, 4, 2, 2) ./ 20
    dense1 = reshape(cos.(collect(1.0:32.0)), 8, 4) ./ 10
    dense2 = reshape(sin.(collect(1.0:12.0)), 4, 3) ./ 10
    layers = MIPVerify.Layer[
        MIPVerify.Conv2d(conv1_filter, [0.01, -0.02], 2),
        MIPVerify.ReLU(),
        MIPVerify.Conv2d(conv2_filter, [0.03, -0.01], 2),
        MIPVerify.ReLU(),
        # This is the TensorFlow-to-Julia flatten permutation used by the real WK17a model.
        MIPVerify.Flatten([1, 3, 2, 4]),
        MIPVerify.Linear(dense1, [0.01, -0.02, 0.03, -0.04]),
        MIPVerify.ReLU(),
        MIPVerify.Linear(dense2, [0.01, 0.0, -0.01]),
    ]
    return MIPVerify.Sequential(layers, "synthetic-wk17a-layout")
end

function scalar_margin_network(true_class_bias)
    # Class 2 equals the scalar input while class 1 is constant. Moving the class-1 bias across
    # the perturbation upper bound creates matched certified and attackable exact-maximum cases.
    weights = reshape([0.0, 1.0, 0.0], 1, 3)
    return MIPVerify.Sequential(
        MIPVerify.Layer[MIPVerify.Linear(weights, [true_class_bias, 0.0, 0.0])],
        "scalar-worst-margin",
    )
end

@testset "PGD warm-start benchmark treatments" begin
    @testset "variant definitions and rotated block order" begin
        @test [variant.name for variant in WARM_START_VARIANTS] == [:cold, :random_full, :pgd_full]
        @test [(variant.source, variant.coverage) for variant in WARM_START_VARIANTS] == [(:none, :none), (:random, :all_variables), (:pgd, :all_variables)]
        @test [variant.name for variant in DIAGNOSTIC_WARM_START_VARIANTS] == [:original_sparse, :pgd_sparse, :original_full]
        @test [(variant.source, variant.coverage) for variant in DIAGNOSTIC_WARM_START_VARIANTS] == [
            (:original, :input_and_perturbation),
            (:pgd, :input_and_perturbation),
            (:original, :all_variables),
        ]
        # Adjacent blocks rotate the first treatment so persistent machine drift is not always
        # charged to the same variant; the fourth block intentionally repeats the first order.
        @test [variant.name for variant in ordered_variants(1)] == [:cold, :random_full, :pgd_full]
        @test [variant.name for variant in ordered_variants(2)] == [:random_full, :pgd_full, :cold]
        @test ordered_variants(4) == ordered_variants(1)
        @test_throws ArgumentError ordered_variants(0)
    end

    @testset "sparse and full starts set exactly their declared variables" begin
        # The original point and PGD near-miss differ in both coordinates so a source mix-up is
        # visible in both the input starts and their derived perturbation starts.
        original = [0.2, 0.4]
        pgd_candidate = [0.3, 0.35]

        cold = toy_worst_margin_problem()
        apply_variant_start!(cold, :cold, original, pgd_candidate)
        @test all(isnothing ∘ JuMP.start_value, JuMP.all_variables(cold.model))

        original_sparse = toy_worst_margin_problem()
        apply_variant_start!(original_sparse, :original_sparse, original, pgd_candidate)
        @test JuMP.start_value.(original_sparse.perturbed_input) == original
        @test JuMP.start_value.(original_sparse.perturbation) == [0.0, 0.0]
        @test isnothing(JuMP.start_value(original_sparse.margin_variable))

        pgd_sparse = toy_worst_margin_problem()
        apply_variant_start!(pgd_sparse, :pgd_sparse, original, pgd_candidate)
        @test JuMP.start_value.(pgd_sparse.perturbed_input) == pgd_candidate
        @test JuMP.start_value.(pgd_sparse.perturbation) ≈ [0.1, -0.05]
        @test isnothing(JuMP.start_value(pgd_sparse.margin_variable))

        base = toy_worst_margin_problem()
        full_start = complete_full_start(base, pgd_candidate; time_limit = 5.0)
        pgd_full = copy_problem(base)
        apply_variant_start!(pgd_full, :pgd_full, original, pgd_candidate; full_start = full_start)
        @test all(!isnothing ∘ JuMP.start_value, JuMP.all_variables(pgd_full.model))
        @test JuMP.start_value.(pgd_full.perturbed_input) ≈ pgd_candidate
        # This unselected in-box point differs from both named candidates, catching accidental
        # reuse of the PGD completion for the random control.
        random_candidate = [0.25, 0.38]
        random_full = copy_problem(base)
        random_full_start = complete_full_start(base, random_candidate; time_limit = 5.0)
        apply_variant_start!(
            random_full,
            :random_full,
            original,
            pgd_candidate;
            full_start = random_full_start,
        )
        @test all(!isnothing ∘ JuMP.start_value, JuMP.all_variables(random_full.model))
        @test JuMP.start_value.(random_full.perturbed_input) ≈ random_candidate
        original_full = copy_problem(base)
        original_full_start = complete_full_start(base, original; time_limit = 5.0)
        apply_variant_start!(
            original_full,
            WarmStartVariant(:original_full, :original, :all_variables),
            original,
            pgd_candidate;
            full_start = original_full_start,
        )
        @test all(!isnothing ∘ JuMP.start_value, JuMP.all_variables(original_full.model))
        @test JuMP.start_value.(original_full.perturbed_input) ≈ original
        @test_throws ArgumentError apply_variant_start!(
            copy_problem(base),
            :pgd_full,
            original,
            pgd_candidate,
        )
    end

    @testset "PGD projection, layout parity, and determinism" begin
        # A nonsquare four-dimensional box catches the historical bug where `rand(rng, size(x))`
        # sampled a dimension value instead of an array with the requested shape.
        lower = zeros(1, 2, 3, 4)
        upper = fill(0.25, 1, 2, 3, 4)
        random_start = uniform_box_start(MersenneTwister(7), lower, upper)
        @test size(random_start) == (1, 2, 3, 4)
        @test all(lower .<= random_start .<= upper)

        input = reshape(collect(range(0.2, 0.8; length = 30)), 1, 5, 6, 1)
        projected = project_linf(input .+ 0.2, input, 0.1)
        @test maximum(abs.(projected .- input)) <= 0.1 + eps(Float64)
        @test all(0.0 .<= projected .<= 1.0)
        # The solver copy uses an intentionally inward point so a floating representation such
        # as 0.10000000000000003 cannot make full-start completion look infeasible at the boundary.
        inward = inward_project_linf(input .+ 0.2, input, 0.1)
        @test maximum(abs.(inward .- input)) < 0.1
        @test maximum(abs.(inward .- input)) <= 0.1 - 1e-9 + 10eps(Float64)

        nn = synthetic_wk17a_style_network()
        optimized = vec(optimized_logits(nn, permutedims(input, (3, 2, 4, 1))))
        reference = vec(input |> nn)
        @test optimized ≈ reference atol = 1e-12 rtol = 1e-12

        # Three restarts and two steps exercise random initialization, gradient updates, and
        # trajectory-best retention without making this unit test a performance benchmark.
        first = projected_gradient_attack(
            nn,
            Array(input),
            1;
            epsilon = 0.1,
            step_size = 0.05,
            steps = 2,
            restarts = 3,
            seed = 11,
        )
        second = projected_gradient_attack(
            nn,
            Array(input),
            1;
            epsilon = 0.1,
            step_size = 0.05,
            steps = 2,
            restarts = 3,
            seed = 11,
        )
        @test first.candidate == second.candidate
        @test first.margin == second.margin
        @test size(first.candidate) == size(input)
        @test maximum(abs.(first.candidate .- input)) <= 0.1 + 10eps(Float64)
        @test all(0.0 .<= first.candidate .<= 1.0)
        @test first.margin ≈ worst_margin(vec(first.candidate |> nn), 1)
    end

    @testset "exact worst-margin solve distinguishes certificates and attacks" begin
        # From input 0.5 with epsilon 0.1, class 2 can reach 0.6. A true-class logit of 0.7 leaves
        # the exact worst margin at -0.1 and must produce a robustness certificate.
        input = [0.5]
        perturbation = MIPVerify.LInfNormBoundedPerturbationFamily(0.1)
        robust_nn = scalar_margin_network(0.7)
        robust_base = build_worst_margin_problem(
            robust_nn,
            input,
            1,
            HiGHS.Optimizer,
            Dict("output_flag" => false);
            pp = perturbation,
            tightening_algorithm = MIPVerify.interval_arithmetic,
        )
        @test copy_problem(robust_base).signature == robust_base.signature
        robust_result = solve_worst_margin!(
            copy_problem(robust_base),
            robust_nn,
            input,
            perturbation;
            time_limit = 5.0,
        )
        @test robust_result.outcome == "certified_robust"
        @test robust_result.objective_bound ≈ -0.1 atol = 1e-8

        # Lowering the true-class logit to 0.55 makes x = 0.6 a positive-margin witness. This
        # exercises the objective-limit path and independent witness verification.
        attackable_nn = scalar_margin_network(0.55)
        attackable_base = build_worst_margin_problem(
            attackable_nn,
            input,
            1,
            HiGHS.Optimizer,
            Dict("output_flag" => false);
            pp = perturbation,
            tightening_algorithm = MIPVerify.interval_arithmetic,
        )
        attack_result = solve_worst_margin!(
            copy_problem(attackable_base),
            attackable_nn,
            input,
            perturbation;
            time_limit = 5.0,
        )
        @test attack_result.outcome == "verified_attack"
        @test attack_result.verification.verified_attack
        @test attack_result.verification.margin >= 0
    end

    @testset "resource observations are safe on unsupported platforms" begin
        @test isnan(process_rss_mb()) || process_rss_mb() >= 0
        @test isnan(system_available_mb()) || system_available_mb() >= 0
        # Synthetic available-memory values make the guard deterministic without allocating a
        # large buffer or depending on what else is running on the machine.
        @test require_available_memory(4_096; available_mb = 8_192) == 8_192
        @test_throws ErrorException require_available_memory(4_096; available_mb = 2_048)
        @test_throws ArgumentError require_available_memory(0; available_mb = 8_192)
    end
end
