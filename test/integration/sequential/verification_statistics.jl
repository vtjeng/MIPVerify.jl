using Test
using MIPVerify
using MIPVerify: UnrestrictedPerturbationFamily
using JuMP

include("../../TestHelpers.jl")

@testset "verification statistics" begin
    main_solve_options = Dict()

    @testset "default tightening screens rival logits" begin
        # On the unit interval the reference x + 1 always beats x and x - 1.
        # The separate ranges of x + 1 and x touch, so ruling out x needs a margin LP.
        nn = MIPVerify.Sequential([MIPVerify.Linear([1.0 1.0 1.0], [1.0, 0.0, -1.0])], "Rivals")
        d = find_adversarial_example(
            nn,
            [0.5], # An interior input classified as the first of the three logits.
            [2, 3],
            TestHelpers.get_optimizer(),
            TestHelpers.get_main_solve_options(),
            collect_stats = true,
        )
        @test d[:TighteningApproach] == "target_lp"
        @test d[:BoundSolverCallCount] == 1
        @test d[:SolveStatus] == MOI.INFEASIBLE
    end

    @testset "interval arithmetic records no bound solves" begin
        # SimpleNet: 2 -> 2 (ReLU) -> 2. The identity first layer sends the raw inputs into the
        # ReLU, so interval arithmetic alone bounds every unit and no solver is called.
        w1 = [1.0 0.0; 0.0 1.0]
        b1 = [0.0, 0.0]
        w2 = [1.0 -1.0; -1.0 1.0]
        b2 = [0.0, 0.0]
        nn = MIPVerify.Sequential(
            [MIPVerify.Linear(w1, b1), MIPVerify.ReLU(), MIPVerify.Linear(w2, b2)],
            "SimpleNet",
        )
        input = [1.0, 0.5]

        d = find_adversarial_example(
            nn,
            input,
            2,
            TestHelpers.get_optimizer(),
            main_solve_options,
            norm_order = 1,
            pp = UnrestrictedPerturbationFamily(),
            tightening_algorithm = MIPVerify.interval_arithmetic,
            collect_stats = true,
        )

        @test d[:SolveStatus] == MOI.OPTIMAL
        @test d[:BoundSolverCallCount] == 0
        @test d[:BoundSolverWallTime] == 0
        @test d[:ReLUStableCount] + d[:ReLUSplitCount] == 2
        @test d[:NumVariables] == JuMP.num_variables(d[:Model])
        @test d[:NumBinaryVariables] ==
              JuMP.num_constraints(d[:Model], JuMP.VariableRef, MOI.ZeroOne)
        @test d[:NumStructuralConstraints] ==
              JuMP.num_constraints(d[:Model]; count_variable_in_set_constraints = false)
        @test d[:NumTotalConstraints] ==
              JuMP.num_constraints(d[:Model]; count_variable_in_set_constraints = true)
        @test d[:FormulationTime] >= d[:BoundSolverWallTime]
        @test d[:MainSolveWallTime] >= 0
        @test MIPVerify.get_verification_stats(d[:Model]) !== nothing
    end

    @testset "progressive bound tightening statistics" begin
        # A net whose ReLU units span every phase class, so MIP tightening drives the full
        # progressive path end-to-end through `find_adversarial_example`: interval screening
        # skips the stable units, the LP relaxation tightens the split unit, and only the
        # still-split unit progresses to a MIP solve.
        #
        # `Linear` computes `transpose(matrix) * x + bias`, so column j of `pw1` is unit j's
        # weight vector. `UnrestrictedPerturbationFamily` bounds each input to [0, 1], so with
        # x in [0, 1]^2 the pre-activation z_j = (column j) . x + b_j and its interval fix each
        # unit's phase:
        #   unit 1: [1, 1] . x + 0.5   in [ 0.5, 2.5]  -> always active   (interval lower >= 0)
        #   unit 2: [-1,-1] . x - 0.5  in [-2.5,-0.5]  -> always inactive (interval upper <= 0)
        #   unit 3: [1,-1] . x + 0.0   in [-1.0, 1.0]  -> split           (LP solve, then MIP solve)
        #   unit 4: [2, 0] . x + 0.1   in [ 0.1, 2.1]  -> always active
        # `pw2` routes h1 -> output 1 and (h4 + 0.5*h3) -> output 2, so both outputs read stable
        # units plus the split unit; the always-inactive unit 2 is dropped.
        pw1 = [1.0 -1.0 1.0 2.0; 1.0 -1.0 -1.0 0.0]
        pb1 = [0.5, -0.5, 0.0, 0.1]
        pw2 = [1.0 0.0; 0.0 0.0; 0.0 0.5; 0.0 1.0]
        pb2 = [0.0, 0.0]
        pnn = MIPVerify.Sequential(
            [MIPVerify.Linear(pw1, pb1), MIPVerify.ReLU(), MIPVerify.Linear(pw2, pb2)],
            "MixedPhaseNet",
        )

        d = find_adversarial_example(
            pnn,
            [0.5, 0.5],
            2,
            TestHelpers.get_optimizer(),
            main_solve_options,
            norm_order = 1,
            pp = UnrestrictedPerturbationFamily(),
            tightening_algorithm = MIPVerify.mip,
            tightening_options = TestHelpers.get_tightening_options(),
            collect_stats = true,
        )

        @test d[:SolveStatus] == MOI.OPTIMAL
        @test d[:BoundSolverCallCount] > 0
        @test d[:ReLUStableCount] == 3
        @test d[:ReLUSplitCount] == 1

        stats = MIPVerify.get_verification_stats(d[:Model])
        lp_requests =
            sum(g.request_count for (k, g) in stats.bound_tightening if k[2] == "lp"; init = 0)
        mip_requests =
            sum(g.request_count for (k, g) in stats.bound_tightening if k[2] == "mip"; init = 0)
        # LP screening certifies the stable units, so only the split unit reaches the MIP stage.
        @test mip_requests > 0
        @test lp_requests > mip_requests

        summary = MIPVerify.summarize_verification_stats(d[:Model])
        # The always-active units have their upper solve skipped by the interval lower bound;
        # the always-inactive unit has its lower solve skipped by the nonpositive upper bound.
        @test summary[:BoundUpperSkippedCount] >= 1
        @test summary[:BoundLowerSkippedCount] >= 1
    end
end

@testset "fixed-budget feasibility uses the backward screen by default" begin
    # The first score is x. The second is 0, so a tie is possible only at x = 0.
    nn = MIPVerify.Sequential([MIPVerify.Linear([1.0 0.0], zeros(2))], "endpoint tie")
    function verify(x; options...)
        return find_adversarial_example(
            nn,
            [x],
            2,
            TestHelpers.get_optimizer(),
            TestHelpers.get_main_solve_options();
            pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
            adversarial_example_objective = MIPVerify.feasibility,
            collect_stats = true,
            options...,
        )
    end

    # The central box excludes zero. Both methods establish infeasibility.
    certificate = verify(0.5)
    @test certificate[:VerificationSource] == :backward_bound
    @test certificate[:SolveStatus] == MOI.INFEASIBLE
    @test !certificate[:WitnessAvailable]
    @test !haskey(certificate, :Model)
    baseline = verify(0.5; backward_screen = false)
    @test haskey(baseline, :Model)
    @test baseline[:SolveStatus] == certificate[:SolveStatus]

    # This box includes zero. Screening accepts the exact corner tie.
    corner = verify(0.05)
    @test corner[:VerificationSource] == :corner
    @test corner[:PerturbedInputValue] == [0.0]
    @test corner[:WitnessVerified]
    @test corner[:SolveTime] == 0

    # The endpoint tie cannot achieve a positive margin. The closest objective
    # still requires optimizing distance even when a corner witness is available.
    @test verify(0.05; margin = 0.01)[:VerificationSource] == :backward_bound
    closest = verify(0.05; adversarial_example_objective = MIPVerify.closest)
    @test haskey(closest, :Model)
    @test closest[:WitnessDistance] ≈ 0.05
end

@testset "screened solver fallback rejects a near miss" begin
    # These equal hidden activations feed different logits. Their relaxations leave the
    # target open, although its exact score always trails by 1e-9 (inside solver tolerance).
    nn = MIPVerify.Sequential(
        [
            MIPVerify.Linear([1.0 1.0], [-0.5, -0.5]),
            MIPVerify.ReLU(),
            MIPVerify.Linear([1.0 0.0; 0.0 1.0], [1e-9, 0.0]),
        ],
        "duplicate hidden units",
    )
    d = find_adversarial_example(
        nn,
        [0.5],
        2,
        TestHelpers.get_optimizer(),
        TestHelpers.get_main_solve_options();
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
        adversarial_example_objective = MIPVerify.feasibility,
        collect_stats = true,
    )
    @test haskey(d, :Model)
    @test d[:ReLULayerCount] == 1
    @test !d[:WitnessVerified]
    if d[:WitnessAvailable]
        @test !(d[:WitnessTargetVerified] && d[:WitnessPerturbationVerified])
    else
        @test d[:SolveStatus] == MOI.INFEASIBLE
    end
end
