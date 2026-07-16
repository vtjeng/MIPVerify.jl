using Test

using HiGHS
using JuMP
using MathOptInterface: MathOptInterface
using MIPVerify:
    MIPVerify,
    MIPVerifyExt,
    certified_lp_bound,
    constraint_dual_or_nothing,
    lower_bound,
    lower_bound_type,
    lp,
    mip,
    projected_dual_and_reference,
    tight_lowerbound,
    tight_upperbound,
    upper_bound,
    upper_bound_type,
    variable_interval_or_nothing

@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "lp_certification.jl" begin
    @testset "repairs stationarity residuals over variable bounds" begin
        m_lower = Model()
        @variable(m_lower, -2 <= x <= 5)
        lower_constraint = @constraint(m_lower, x >= 1)
        lower_dual = Dict(lower_constraint => 1.9)
        lower = certified_lp_bound(
            m_lower,
            lower_bound_type,
            2x + 3,
            -1.0;
            dual_value = constraint -> lower_dual[constraint],
        )
        @test lower <= 4.7
        @test lower ≈ 4.7

        m_upper = Model()
        @variable(m_upper, -2 <= x <= 5)
        upper_constraint = @constraint(m_upper, x <= 1)
        upper_dual = Dict(upper_constraint => -1.9)
        upper = certified_lp_bound(
            m_upper,
            upper_bound_type,
            2x + 3,
            13.0;
            dual_value = constraint -> upper_dual[constraint],
        )
        @test upper >= 5.4
        @test upper ≈ 5.4
    end

    @testset "uses the correct residual endpoint" begin
        m_lower = Model()
        @variable(m_lower, -2 <= x <= 5)
        lower_constraint = @constraint(m_lower, x >= 1)
        lower = certified_lp_bound(
            m_lower,
            lower_bound_type,
            2x + 3,
            -1.0;
            dual_value = constraint -> constraint == lower_constraint ? 2.1 : 0.0,
        )
        @test lower <= 4.6
        @test lower ≈ 4.6

        m_upper = Model()
        @variable(m_upper, -2 <= x <= 5)
        upper_constraint = @constraint(m_upper, x <= 1)
        upper = certified_lp_bound(
            m_upper,
            upper_bound_type,
            2x + 3,
            13.0;
            dual_value = constraint -> constraint == upper_constraint ? -2.1 : 0.0,
        )
        @test upper >= 5.3
        @test upper ≈ 5.3
    end

    @testset "projects inequality duals onto their cones" begin
        m = Model()
        @variable(m, 0 <= x <= 3)
        useful = @constraint(m, x <= 1)
        wrong_sign = @constraint(m, -x <= 0)
        duals = Dict(useful => -1.0, wrong_sign => 0.25)

        bound = certified_lp_bound(
            m,
            upper_bound_type,
            x,
            3.0;
            dual_value = constraint -> duals[constraint],
        )

        @test bound == 1.0
    end

    @testset "supports equality and interval constraints" begin
        m_equal = Model()
        @variable(m_equal, 0 <= x <= 1)
        equality = @constraint(m_equal, x == 0.5)
        @test certified_lp_bound(
            m_equal,
            lower_bound_type,
            x,
            0.0;
            dual_value = constraint -> constraint == equality ? 1.0 : 0.0,
        ) == 0.5
        @test certified_lp_bound(
            m_equal,
            upper_bound_type,
            x,
            1.0;
            dual_value = constraint -> constraint == equality ? -1.0 : 0.0,
        ) == 0.5

        m_interval = Model()
        @variable(m_interval, 0 <= y <= 1)
        interval_constraint = @constraint(m_interval, 0.25 <= y <= 0.75)
        @test certified_lp_bound(
            m_interval,
            lower_bound_type,
            y,
            0.0;
            dual_value = constraint -> constraint == interval_constraint ? 1.0 : 0.0,
        ) == 0.25
        @test certified_lp_bound(
            m_interval,
            upper_bound_type,
            y,
            1.0;
            dual_value = constraint -> constraint == interval_constraint ? -1.0 : 0.0,
        ) == 0.75
    end

    @testset "clamps the certificate to the interval bound" begin
        m_upper = Model()
        @variable(m_upper, 0 <= x <= 1)
        upper_constraint = @constraint(m_upper, x <= 100)
        @test certified_lp_bound(
            m_upper,
            upper_bound_type,
            x,
            1.0;
            dual_value = constraint -> constraint == upper_constraint ? -1.0 : 0.0,
        ) == 1.0

        m_lower = Model()
        @variable(m_lower, 0 <= x <= 1)
        lower_constraint = @constraint(m_lower, x >= -100)
        @test certified_lp_bound(
            m_lower,
            lower_bound_type,
            x,
            0.0;
            dual_value = constraint -> constraint == lower_constraint ? 1.0 : 0.0,
        ) == 0.0
    end

    @testset "handles fixed and unbounded variables" begin
        m_fixed = Model()
        @variable(m_fixed, x)
        fix(x, 0.5)
        equality = @constraint(m_fixed, x == 0.5)
        @test certified_lp_bound(
            m_fixed,
            lower_bound_type,
            x,
            -1.0;
            dual_value = constraint -> constraint == equality ? 0.9 : 0.0,
        ) <= 0.5
        @test certified_lp_bound(
            m_fixed,
            lower_bound_type,
            x,
            -1.0;
            dual_value = constraint -> constraint == equality ? 0.9 : 0.0,
        ) ≈ 0.5

        m_unbounded = Model()
        @variable(m_unbounded, x <= 5)
        upper_constraint = @constraint(m_unbounded, x <= 1)
        @test certified_lp_bound(
            m_unbounded,
            upper_bound_type,
            x,
            5.0;
            dual_value = constraint -> constraint == upper_constraint ? -2.0 : 0.0,
        ) == 5.0
        @test certified_lp_bound(
            m_unbounded,
            upper_bound_type,
            x,
            5.0;
            dual_value = constraint -> constraint == upper_constraint ? -1.0 : 0.0,
        ) == 1.0

        m_lower_unbounded = Model()
        @variable(m_lower_unbounded, x >= -5)
        lower_constraint = @constraint(m_lower_unbounded, x >= -1)
        @test certified_lp_bound(
            m_lower_unbounded,
            lower_bound_type,
            x,
            -5.0;
            dual_value = constraint -> constraint == lower_constraint ? 2.0 : 0.0,
        ) == -5.0
        @test certified_lp_bound(
            m_lower_unbounded,
            lower_bound_type,
            x,
            -5.0;
            dual_value = constraint -> constraint == lower_constraint ? 1.0 : 0.0,
        ) == -1.0
    end

    @testset "outward-rounds large objective constants" begin
        m_upper = Model()
        @variable(m_upper, 0 <= x <= 2)
        upper_constraint = @constraint(m_upper, x <= 1)
        upper_objective = x + 1.0e16
        upper = certified_lp_bound(
            m_upper,
            upper_bound_type,
            upper_objective,
            upper_bound(upper_objective);
            dual_value = constraint -> constraint == upper_constraint ? -1.0 : 0.0,
        )
        @test BigFloat(upper) >= BigFloat(1.0e16) + 1

        m_lower = Model()
        @variable(m_lower, -2 <= x <= 0)
        lower_constraint = @constraint(m_lower, x >= -1)
        lower_objective = x - 1.0e16
        lower = certified_lp_bound(
            m_lower,
            lower_bound_type,
            lower_objective,
            lower_bound(lower_objective);
            dual_value = constraint -> constraint == lower_constraint ? 1.0 : 0.0,
        )
        @test BigFloat(lower) <= BigFloat(-1.0e16) - 1
    end

    @testset "integrates with HiGHS tightening after presolve" begin
        m = TestHelpers.get_new_model()
        m.ext[:MIPVerify] = MIPVerifyExt(lp)
        @variable(m, 0 <= x <= 1)
        @constraint(m, x >= 0.3)

        lower = tight_lowerbound(x; nta = lp)
        upper = tight_upperbound(x; nta = lp)

        # The certified bounds are sound (one-sided) and match the true extrema
        # only up to solver tolerances and outward rounding.
        @test lower <= 0.3
        @test lower ≈ 0.3 atol = 1e-6
        @test upper >= 1.0
        @test upper ≈ 1.0 atol = 1e-6
    end

    @testset "does not use an unsafe primal objective" begin
        upper_mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_variable_constraint_dual = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            upper_mock,
            optimizer -> MathOptInterface.Utilities.mock_optimize!(
                optimizer,
                MathOptInterface.OPTIMAL,
                [0.999],
                (
                    MathOptInterface.ScalarAffineFunction{Float64},
                    MathOptInterface.LessThan{Float64},
                ) => [-1.0],
            ),
        )
        m_upper = Model(() -> upper_mock)
        m_upper.ext[:MIPVerify] = MIPVerifyExt(lp)
        @variable(m_upper, 0 <= x <= 2)
        @constraint(m_upper, x <= 1)
        @test tight_upperbound(x; nta = lp) == 1.0

        lower_mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_variable_constraint_dual = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            lower_mock,
            optimizer -> MathOptInterface.Utilities.mock_optimize!(
                optimizer,
                MathOptInterface.OPTIMAL,
                [-0.999],
                (
                    MathOptInterface.ScalarAffineFunction{Float64},
                    MathOptInterface.GreaterThan{Float64},
                ) => [1.0],
            ),
        )
        m_lower = Model(() -> lower_mock)
        m_lower.ext[:MIPVerify] = MIPVerifyExt(lp)
        @variable(m_lower, -2 <= x <= 0)
        @constraint(m_lower, x >= -1)
        @test tight_lowerbound(x; nta = lp) == -1.0
    end

    @testset "falls back when the solver returns no dual solution" begin
        mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_variable_constraint_dual = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            mock,
            optimizer -> MathOptInterface.Utilities.mock_optimize!(
                optimizer,
                MathOptInterface.OPTIMAL,
                [0.999],
            ),
        )
        m = Model(() -> mock)
        m.ext[:MIPVerify] = MIPVerifyExt(lp)
        @variable(m, 0 <= x <= 2)
        @constraint(m, x <= 1)

        @test tight_upperbound(x; nta = lp) == 2.0
    end

    @testset "uses the solver objective bound for MIP tightening" begin
        upper_mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_objective_value = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            upper_mock,
            optimizer -> begin
                MathOptInterface.Utilities.mock_optimize!(optimizer, MathOptInterface.OPTIMAL, [0.999])
                MathOptInterface.set(optimizer, MathOptInterface.ObjectiveValue(), 0.999)
                MathOptInterface.set(optimizer, MathOptInterface.ObjectiveBound(), 1.0)
            end,
        )
        m_upper = Model(() -> upper_mock)
        m_upper.ext[:MIPVerify] = MIPVerifyExt(mip)
        @variable(m_upper, 0 <= x <= 2)
        @constraint(m_upper, x <= 1)

        @test tight_upperbound(x; nta = mip) == 1.0

        lower_mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_objective_value = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            lower_mock,
            optimizer -> begin
                MathOptInterface.Utilities.mock_optimize!(
                    optimizer,
                    MathOptInterface.OPTIMAL,
                    [-0.999],
                )
                MathOptInterface.set(optimizer, MathOptInterface.ObjectiveValue(), -0.999)
                MathOptInterface.set(optimizer, MathOptInterface.ObjectiveBound(), -1.0)
            end,
        )
        m_lower = Model(() -> lower_mock)
        m_lower.ext[:MIPVerify] = MIPVerifyExt(mip)
        @variable(m_lower, -2 <= y <= 0)
        @constraint(m_lower, y >= -1)

        @test tight_lowerbound(y; nta = mip) == -1.0
    end

    @testset "falls back when the MIP objective bound is not finite" begin
        mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_objective_value = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            mock,
            optimizer -> begin
                MathOptInterface.Utilities.mock_optimize!(optimizer, MathOptInterface.OPTIMAL, [0.999])
                MathOptInterface.set(optimizer, MathOptInterface.ObjectiveValue(), 0.999)
                MathOptInterface.set(optimizer, MathOptInterface.ObjectiveBound(), Inf)
            end,
        )
        m = Model(() -> mock)
        m.ext[:MIPVerify] = MIPVerifyExt(mip)
        @variable(m, 0 <= x <= 2)
        @constraint(m, x <= 1)

        @test tight_upperbound(x; nta = mip) == 2.0
    end

    @testset "falls back when reading the MIP objective bound throws unexpectedly" begin
        # MockOptimizer throws a KeyError when ObjectiveBound was never set. Unexpected
        # errors are logged at warn level and treated as an unavailable attribute, so the
        # bound falls back to b_0 instead of crashing the run.
        mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_objective_value = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            mock,
            optimizer -> MathOptInterface.Utilities.mock_optimize!(
                optimizer,
                MathOptInterface.OPTIMAL,
                [0.999],
            ),
        )
        m = Model(() -> mock)
        m.ext[:MIPVerify] = MIPVerifyExt(mip)
        @variable(m, 0 <= x <= 2)
        @constraint(m, x <= 1)

        MIPVerify.Memento.TestUtils.@test_log(
            MIPVerify.LOGGER,
            "warn",
            "Unexpected error reading the solver objective bound",
            @test(tight_upperbound(x; nta = mip) == 2.0)
        )
    end

    @testset "errors when a feasible point lies outside the interval-arithmetic bound" begin
        # A feasible point can never beat a bound that interval arithmetic computed for the
        # same model; if the solver reports one that does, the solver and our view of the
        # model disagree and no bound from the run can be trusted.
        upper_mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_objective_value = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            upper_mock,
            optimizer -> begin
                MathOptInterface.Utilities.mock_optimize!(optimizer, MathOptInterface.OPTIMAL, [3.0])
                MathOptInterface.set(optimizer, MathOptInterface.ObjectiveValue(), 3.0)
            end,
        )
        m_upper = Model(() -> upper_mock)
        m_upper.ext[:MIPVerify] = MIPVerifyExt(mip)
        @variable(m_upper, 0 <= x <= 2)

        @test_throws ErrorException tight_upperbound(x; nta = mip)

        lower_mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_objective_value = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            lower_mock,
            optimizer -> begin
                MathOptInterface.Utilities.mock_optimize!(optimizer, MathOptInterface.OPTIMAL, [-3.0])
                MathOptInterface.set(optimizer, MathOptInterface.ObjectiveValue(), -3.0)
            end,
        )
        m_lower = Model(() -> lower_mock)
        m_lower.ext[:MIPVerify] = MIPVerifyExt(mip)
        @variable(m_lower, -2 <= y <= 0)

        @test_throws ErrorException tight_lowerbound(y; nta = mip)
    end

    @testset "projected_dual_and_reference ignores unsupported sets" begin
        @test projected_dual_and_reference(MathOptInterface.ZeroOne(), 1.0) === nothing
    end

    @testset "constraint_dual_or_nothing classifies retrieval errors" begin
        # expected MOI attribute-unavailable errors fall back quietly
        unsupported = MathOptInterface.UnsupportedAttribute(MathOptInterface.ConstraintDual())
        @test constraint_dual_or_nothing(:constraint, _ -> throw(unsupported)) === nothing
        # unexpected errors are logged at warn level and treated as unavailable
        MIPVerify.Memento.TestUtils.@test_log(
            MIPVerify.LOGGER,
            "warn",
            "Unexpected error reading a constraint dual",
            @test(constraint_dual_or_nothing(:constraint, _ -> error("boom")) === nothing)
        )
        # interrupts still propagate
        @test_throws InterruptException constraint_dual_or_nothing(
            :constraint,
            _ -> throw(InterruptException()),
        )
    end

    @testset "variable_interval_or_nothing clamps binaries and rejects empty intervals" begin
        m = Model()
        b = @variable(m, binary = true)
        binary_interval = variable_interval_or_nothing(b)
        @test lower_bound(binary_interval) == 0.0
        @test upper_bound(binary_interval) == 1.0

        inverted = @variable(m)
        set_lower_bound(inverted, 1.0)
        set_upper_bound(inverted, 0.0)
        @test variable_interval_or_nothing(inverted) === nothing
    end

    @testset "falls back when a residual variable has an invalid or unbounded interval" begin
        m_invalid = Model()
        @variable(m_invalid, 0 <= x <= 1)
        y = @variable(m_invalid)
        set_lower_bound(y, 1.0)
        set_upper_bound(y, 0.0)
        invalid_result =
            certified_lp_bound(m_invalid, lower_bound_type, x + y, -2.5; dual_value = _ -> 0.0)
        @test invalid_result == -2.5

        m_unbounded = Model()
        @variable(m_unbounded, 0 <= z <= 1)
        free = @variable(m_unbounded)
        unbounded_result =
            certified_lp_bound(m_unbounded, lower_bound_type, z + free, -2.5; dual_value = _ -> 0.0)
        @test unbounded_result == -2.5
    end

    @testset "retains the LP certificate when MIP tightening times out" begin
        call_count = Ref(0)
        mock = MathOptInterface.Utilities.MockOptimizer(
            MathOptInterface.Utilities.Model{Float64}();
            eval_variable_constraint_dual = false,
        )
        MathOptInterface.Utilities.set_mock_optimize!(
            mock,
            optimizer -> begin
                call_count[] += 1
                if call_count[] == 1
                    MathOptInterface.Utilities.mock_optimize!(
                        optimizer,
                        MathOptInterface.OPTIMAL,
                        [1.499],
                        (
                            MathOptInterface.ScalarAffineFunction{Float64},
                            MathOptInterface.LessThan{Float64},
                        ) => [-1.0],
                    )
                else
                    MathOptInterface.Utilities.mock_optimize!(optimizer, MathOptInterface.TIME_LIMIT)
                end
            end,
        )
        m = Model(() -> mock)
        m.ext[:MIPVerify] = MIPVerifyExt(mip)
        # The LP dual certifies the row bound 1.5; that certificate must survive the following
        # MIP solve's time limit instead of falling back to the variable upper bound 2.
        @variable(m, 0 <= x <= 2)
        @constraint(m, x <= 1.5)

        @test tight_upperbound(x; nta = mip) == 1.5
        @test call_count[] == 2
    end

    @testset "batches ordered duals by homogeneous constraint type" begin
        m = Model()
        # These ranges make the interval lower bound of x - y equal to -4 while leaving room for
        # the selected x >= 2 and y <= 2 rows to certify the exact lower bound 0.
        @variable(m, 0 <= x <= 3)
        @variable(m, 0 <= y <= 4)
        lower_first = @constraint(m, x >= 1)
        lower_second = @constraint(m, x >= 2)
        upper_first = @constraint(m, y <= 3)
        upper_second = @constraint(m, y <= 2)

        # A zero dual ignores the first row in each group; the signed unit dual selects the second.
        # This makes swapped or otherwise misordered batch values produce a different certificate.
        duals =
            Dict(lower_first => 0.0, lower_second => 1.0, upper_first => 0.0, upper_second => -1.0)
        batch_groups = Vector{Vector{Any}}()
        dual_values = constraints -> begin
            push!(batch_groups, Any[constraints...])
            return [duals[constraint] for constraint in constraints]
        end

        bound = certified_lp_bound(m, lower_bound_type, x - y, -4.0; dual_values = dual_values)

        # The selected rows prove x - y >= 2 - 2 = 0.
        @test bound == 0.0
        # GreaterThan and LessThan rows must be fetched as two separate homogeneous batches.
        @test length(batch_groups) == 2
        @test Any[lower_first, lower_second] in batch_groups
        @test Any[upper_first, upper_second] in batch_groups
    end

    @testset "default dual retrieval returns duals in constraint order" begin
        m = Model(HiGHS.Optimizer)
        set_silent(m)
        @variable(m, x)
        @variable(m, y)
        constraints = [@constraint(m, x >= 1), @constraint(m, y >= 2)]
        # Both rows are active at the optimum, so each row's dual equals its variable's objective
        # coefficient; the distinct coefficients 2 and 3 make the returned order observable.
        @objective(m, Min, 2x + 3y)

        # Before any solve there are no duals, so the batch read reports them unavailable.
        @test MIPVerify.default_constraint_duals(m, constraints) === nothing

        optimize!(m)
        @test MIPVerify.default_constraint_duals(m, constraints) == [2.0, 3.0]
    end

    @testset "retries scalar duals when a batch fails or has the wrong length" begin
        m = Model(HiGHS.Optimizer)
        set_silent(m)
        # Finite [0, 4] bounds let the certificate absorb residuals, while the two independent
        # rows x >= 1 and y >= 2 require both recovered scalar duals to prove the lower bound 3.
        @variable(m, 0 <= x <= 4)
        @variable(m, 0 <= y <= 4)
        @constraint(m, x >= 1)
        @constraint(m, y >= 2)
        @objective(m, Min, x + y)
        optimize!(m)

        for make_bad_batch in (
            # An erroring batch read exercises the exception fallback.
            _ -> error("batch retrieval failed"),
            # One value cannot represent the two affine rows, so retrieval must retry scalars.
            _ -> [0.0],
        )
            batch_calls = Ref(0)
            bound = certified_lp_bound(
                m,
                lower_bound_type,
                x + y,
                0.0;
                dual_values = constraints -> begin
                    batch_calls[] += 1
                    make_bad_batch(constraints)
                end,
            )
            # The failed batch is discarded; both scalar retries recover the solver duals and
            # certify 1 + 2 = 3.
            @test bound == 3.0
            # Both affine rows share one GreaterThan batch, attempted once before scalar fallback.
            @test batch_calls[] == 1
        end
    end

    @testset "treats invalid batch elements as independently unavailable" begin
        m = Model()
        # The four distinct right-hand sides identify which batch element remains usable, and the
        # [0, 5] variable range keeps the interval fallback finite.
        @variable(m, 0 <= x <= 5)
        @constraint(m, x >= 1)
        @constraint(m, x >= 2)
        @constraint(m, x >= 3)
        @constraint(m, x >= 4)

        bound = certified_lp_bound(
            m,
            lower_bound_type,
            x,
            0.0;
            dual_values = _ -> begin
                # NaN, infinity, and a non-Real value are ignored independently; the final unit
                # dual must remain available to select x >= 4.
                return Any[NaN, Inf, "unavailable", 1.0]
            end,
        )

        # The one valid element certifies x >= 4 despite the three invalid neighbors.
        @test bound == 4.0
    end

    @testset "propagates fatal batch errors" begin
        m = Model()
        # This single finite row is enough to enter the batch path without unrelated fallbacks.
        @variable(m, 0 <= x <= 2)
        @constraint(m, x >= 1)

        for fatal_error in (InterruptException(), OutOfMemoryError(), StackOverflowError())
            @test_throws typeof(fatal_error) certified_lp_bound(
                m,
                lower_bound_type,
                x,
                0.0;
                dual_values = _ -> throw(fatal_error),
            )
        end
    end

    @testset "keeps the custom scalar dual callback path" begin
        m = Model()
        # The second row is stricter, so assigning it the unit dual proves x >= 2 and makes the
        # callback order and returned values observable in the certificate.
        @variable(m, 0 <= x <= 3)
        first_constraint = @constraint(m, x >= 1)
        second_constraint = @constraint(m, x >= 2)
        scalar_calls = Any[]
        scalar_duals = Dict(first_constraint => 0.0, second_constraint => 1.0)

        bound = certified_lp_bound(
            m,
            lower_bound_type,
            x,
            0.0;
            dual_value = constraint -> begin
                push!(scalar_calls, constraint)
                return scalar_duals[constraint]
            end,
            dual_values = _ -> error("custom scalar duals must bypass batch retrieval"),
        )

        # The custom scalar values select x >= 2, and each row is queried once in model order.
        @test bound == 2.0
        @test scalar_calls == Any[first_constraint, second_constraint]
    end
end
