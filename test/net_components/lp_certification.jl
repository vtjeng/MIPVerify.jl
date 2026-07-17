using Test

using HiGHS
using JuMP
using MathOptInterface: MathOptInterface
using MIPVerify:
    MIPVerify,
    MIPVerifyExt,
    certified_lp_bound,
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

# Homogeneous affine constraint groups, in the (function, set) form that
# `MathOptInterface.Utilities.mock_optimize!` uses to attach dual values.
const AFFINE_GT =
    (MathOptInterface.ScalarAffineFunction{Float64}, MathOptInterface.GreaterThan{Float64})
const AFFINE_LT =
    (MathOptInterface.ScalarAffineFunction{Float64}, MathOptInterface.LessThan{Float64})
const AFFINE_EQ =
    (MathOptInterface.ScalarAffineFunction{Float64}, MathOptInterface.EqualTo{Float64})
const AFFINE_INTERVAL =
    (MathOptInterface.ScalarAffineFunction{Float64}, MathOptInterface.Interval{Float64})

# Mock-backed models exercise the production dual-read path end to end: duals are attached to
# the mock solve per homogeneous constraint group and read back through the vectorized
# MathOptInterface interface. `eval_variable_constraint_dual = false` keeps the mock from
# deriving variable-bound duals out of the attached affine duals: the certificate never reads
# them, so such a read should fail rather than answer with synthesized values.
function certification_mock()
    return MathOptInterface.Utilities.MockOptimizer(
        MathOptInterface.Utilities.Model{Float64}();
        eval_variable_constraint_dual = false,
    )
end

# Makes the next solve of `m` report OPTIMAL with the given constraint duals per homogeneous
# group, assigned in constraint creation order. Primal values are irrelevant to the
# certificate and default to zero. Calling this again replaces the previous dual configuration.
function optimize_with_mock_duals!(mock, m, dual_groups...)
    MathOptInterface.Utilities.set_mock_optimize!(
        mock,
        optimizer -> MathOptInterface.Utilities.mock_optimize!(
            optimizer,
            MathOptInterface.OPTIMAL,
            zeros(num_variables(m)),
            dual_groups...,
        ),
    )
    optimize!(m)
    return nothing
end

TestHelpers.@timed_testset "lp_certification.jl" begin
    @testset "repairs stationarity residuals over variable bounds" begin
        lower_mock = certification_mock()
        m_lower = Model(() -> lower_mock)
        @variable(m_lower, -2 <= x <= 5)
        @constraint(m_lower, x >= 1)
        optimize_with_mock_duals!(lower_mock, m_lower, AFFINE_GT => [1.9])
        lower = certified_lp_bound(m_lower, lower_bound_type, 2x + 3, -1.0)
        @test lower <= 4.7
        @test lower ≈ 4.7

        upper_mock = certification_mock()
        m_upper = Model(() -> upper_mock)
        @variable(m_upper, -2 <= x <= 5)
        @constraint(m_upper, x <= 1)
        optimize_with_mock_duals!(upper_mock, m_upper, AFFINE_LT => [-1.9])
        upper = certified_lp_bound(m_upper, upper_bound_type, 2x + 3, 13.0)
        @test upper >= 5.4
        @test upper ≈ 5.4
    end

    @testset "uses the correct residual endpoint" begin
        lower_mock = certification_mock()
        m_lower = Model(() -> lower_mock)
        @variable(m_lower, -2 <= x <= 5)
        @constraint(m_lower, x >= 1)
        optimize_with_mock_duals!(lower_mock, m_lower, AFFINE_GT => [2.1])
        lower = certified_lp_bound(m_lower, lower_bound_type, 2x + 3, -1.0)
        @test lower <= 4.6
        @test lower ≈ 4.6

        upper_mock = certification_mock()
        m_upper = Model(() -> upper_mock)
        @variable(m_upper, -2 <= x <= 5)
        @constraint(m_upper, x <= 1)
        optimize_with_mock_duals!(upper_mock, m_upper, AFFINE_LT => [-2.1])
        upper = certified_lp_bound(m_upper, upper_bound_type, 2x + 3, 13.0)
        @test upper >= 5.3
        @test upper ≈ 5.3
    end

    @testset "projects inequality duals onto their cones" begin
        mock = certification_mock()
        m = Model(() -> mock)
        @variable(m, 0 <= x <= 3)
        @constraint(m, x <= 1)
        @constraint(m, -x <= 0)
        # The useful dual -1.0 selects x <= 1; the wrong-sign dual 0.25 on -x <= 0 must be
        # projected onto the LessThan cone (to zero) instead of corrupting the certificate.
        optimize_with_mock_duals!(mock, m, AFFINE_LT => [-1.0, 0.25])

        bound = certified_lp_bound(m, upper_bound_type, x, 3.0)

        @test bound == 1.0
    end

    @testset "supports equality and interval constraints" begin
        equal_mock = certification_mock()
        m_equal = Model(() -> equal_mock)
        @variable(m_equal, 0 <= x <= 1)
        @constraint(m_equal, x == 0.5)
        optimize_with_mock_duals!(equal_mock, m_equal, AFFINE_EQ => [1.0])
        @test certified_lp_bound(m_equal, lower_bound_type, x, 0.0) == 0.5
        optimize_with_mock_duals!(equal_mock, m_equal, AFFINE_EQ => [-1.0])
        @test certified_lp_bound(m_equal, upper_bound_type, x, 1.0) == 0.5

        interval_mock = certification_mock()
        m_interval = Model(() -> interval_mock)
        @variable(m_interval, 0 <= y <= 1)
        @constraint(m_interval, 0.25 <= y <= 0.75)
        optimize_with_mock_duals!(interval_mock, m_interval, AFFINE_INTERVAL => [1.0])
        @test certified_lp_bound(m_interval, lower_bound_type, y, 0.0) == 0.25
        optimize_with_mock_duals!(interval_mock, m_interval, AFFINE_INTERVAL => [-1.0])
        @test certified_lp_bound(m_interval, upper_bound_type, y, 1.0) == 0.75
    end

    @testset "clamps the certificate to the interval bound" begin
        upper_mock = certification_mock()
        m_upper = Model(() -> upper_mock)
        @variable(m_upper, 0 <= x <= 1)
        @constraint(m_upper, x <= 100)
        optimize_with_mock_duals!(upper_mock, m_upper, AFFINE_LT => [-1.0])
        @test certified_lp_bound(m_upper, upper_bound_type, x, 1.0) == 1.0

        lower_mock = certification_mock()
        m_lower = Model(() -> lower_mock)
        @variable(m_lower, 0 <= x <= 1)
        @constraint(m_lower, x >= -100)
        optimize_with_mock_duals!(lower_mock, m_lower, AFFINE_GT => [1.0])
        @test certified_lp_bound(m_lower, lower_bound_type, x, 0.0) == 0.0
    end

    @testset "handles fixed and unbounded variables" begin
        fixed_mock = certification_mock()
        m_fixed = Model(() -> fixed_mock)
        @variable(m_fixed, x)
        fix(x, 0.5)
        @constraint(m_fixed, x == 0.5)
        optimize_with_mock_duals!(fixed_mock, m_fixed, AFFINE_EQ => [0.9])
        @test certified_lp_bound(m_fixed, lower_bound_type, x, -1.0) <= 0.5
        @test certified_lp_bound(m_fixed, lower_bound_type, x, -1.0) ≈ 0.5

        unbounded_mock = certification_mock()
        m_unbounded = Model(() -> unbounded_mock)
        @variable(m_unbounded, x <= 5)
        @constraint(m_unbounded, x <= 1)
        optimize_with_mock_duals!(unbounded_mock, m_unbounded, AFFINE_LT => [-2.0])
        @test certified_lp_bound(m_unbounded, upper_bound_type, x, 5.0) == 5.0
        optimize_with_mock_duals!(unbounded_mock, m_unbounded, AFFINE_LT => [-1.0])
        @test certified_lp_bound(m_unbounded, upper_bound_type, x, 5.0) == 1.0

        lower_unbounded_mock = certification_mock()
        m_lower_unbounded = Model(() -> lower_unbounded_mock)
        @variable(m_lower_unbounded, x >= -5)
        @constraint(m_lower_unbounded, x >= -1)
        optimize_with_mock_duals!(lower_unbounded_mock, m_lower_unbounded, AFFINE_GT => [2.0])
        @test certified_lp_bound(m_lower_unbounded, lower_bound_type, x, -5.0) == -5.0
        optimize_with_mock_duals!(lower_unbounded_mock, m_lower_unbounded, AFFINE_GT => [1.0])
        @test certified_lp_bound(m_lower_unbounded, lower_bound_type, x, -5.0) == -1.0
    end

    @testset "outward-rounds large objective constants" begin
        upper_mock = certification_mock()
        m_upper = Model(() -> upper_mock)
        @variable(m_upper, 0 <= x <= 2)
        @constraint(m_upper, x <= 1)
        upper_objective = x + 1.0e16
        optimize_with_mock_duals!(upper_mock, m_upper, AFFINE_LT => [-1.0])
        upper = certified_lp_bound(
            m_upper,
            upper_bound_type,
            upper_objective,
            upper_bound(upper_objective),
        )
        @test BigFloat(upper) >= BigFloat(1.0e16) + 1

        lower_mock = certification_mock()
        m_lower = Model(() -> lower_mock)
        @variable(m_lower, -2 <= x <= 0)
        @constraint(m_lower, x >= -1)
        lower_objective = x - 1.0e16
        optimize_with_mock_duals!(lower_mock, m_lower, AFFINE_GT => [1.0])
        lower = certified_lp_bound(
            m_lower,
            lower_bound_type,
            lower_objective,
            lower_bound(lower_objective),
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

    @testset "falls back when reading constraint duals throws unexpectedly" begin
        # The mock reports a feasible dual solution but stores no dual values, so the group
        # read and the single-constraint retries all throw. The errors are logged at warn
        # level and the certificate keeps a zero multiplier for x >= 1, proving only x's
        # declared lower bound 0 instead of crashing the run.
        mock = certification_mock()
        m = Model(() -> mock)
        @variable(m, 0 <= x <= 2)
        constraint = @constraint(m, x >= 1)
        MathOptInterface.Utilities.set_mock_optimize!(
            mock,
            optimizer -> begin
                MathOptInterface.Utilities.mock_optimize!(optimizer, MathOptInterface.OPTIMAL, [1.0])
                MathOptInterface.set(
                    optimizer,
                    MathOptInterface.DualStatus(),
                    MathOptInterface.FEASIBLE_POINT,
                )
            end,
        )
        optimize!(m)

        # Guard the premise: the warn assertion below requires the mock's failed read to
        # throw outside `UNAVAILABLE_ATTRIBUTE_ERRORS`, which the certificate treats as an
        # expected miss and skips quietly.
        read_error = try
            dual(constraint)
            nothing
        catch error
            error
        end
        @test read_error isa Exception
        @test !(read_error isa MIPVerify.UNAVAILABLE_ATTRIBUTE_ERRORS)

        MIPVerify.Memento.TestUtils.@test_log(
            MIPVerify.LOGGER,
            "warn",
            "Unexpected error reading a constraint dual",
            @test(certified_lp_bound(m, lower_bound_type, x, -1.0) == 0.0)
        )
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

    @testset "single_constraint_dual_or_nothing classifies retrieval errors" begin
        # expected MOI attribute-unavailable errors fall back quietly
        unsupported = MathOptInterface.UnsupportedAttribute(MathOptInterface.ConstraintDual())
        @test MIPVerify.single_constraint_dual_or_nothing(_ -> throw(unsupported), :constraint) ===
              nothing
        # unexpected errors are logged at warn level and treated as unavailable
        MIPVerify.Memento.TestUtils.@test_log(
            MIPVerify.LOGGER,
            "warn",
            "Unexpected error reading a constraint dual",
            @test(
                MIPVerify.single_constraint_dual_or_nothing(_ -> error("boom"), :constraint) ===
                nothing
            )
        )
        # interrupts still propagate
        @test_throws InterruptException MIPVerify.single_constraint_dual_or_nothing(
            _ -> throw(InterruptException()),
            :constraint,
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
        # Neither model has affine rows, so no duals are read and the residual repair is the
        # only certificate ingredient exercised.
        invalid_result = certified_lp_bound(m_invalid, lower_bound_type, x + y, -2.5)
        @test invalid_result == -2.5

        m_unbounded = Model()
        @variable(m_unbounded, 0 <= z <= 1)
        free = @variable(m_unbounded)
        unbounded_result = certified_lp_bound(m_unbounded, lower_bound_type, z + free, -2.5)
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

    @testset "maps batched duals to constraints by group and order" begin
        mock = certification_mock()
        m = Model(() -> mock)
        # These ranges make the interval lower bound of x - y equal to -4 while leaving room for
        # the selected x >= 2 and y <= 2 rows to certify the exact lower bound 0.
        @variable(m, 0 <= x <= 3)
        @variable(m, 0 <= y <= 4)
        @constraint(m, x >= 1)
        @constraint(m, x >= 2)
        @constraint(m, y <= 3)
        @constraint(m, y <= 2)

        # A zero dual ignores the first row in each group; the signed unit dual selects the
        # second. Swapping the groups or the values within a group would produce a different
        # certificate.
        optimize_with_mock_duals!(mock, m, AFFINE_GT => [0.0, 1.0], AFFINE_LT => [0.0, -1.0])

        # The selected rows prove x - y >= 2 - 2 = 0.
        @test certified_lp_bound(m, lower_bound_type, x - y, -4.0) == 0.0
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

    @testset "resolve_row_duals retries constraints individually when a group read fails" begin
        # `resolve_row_duals` takes its dual source as a function, so the retry ladder is
        # exercised directly: a source that fails whole-group reads but answers single-element
        # retries cannot be built on top of MockOptimizer, whose vector reads broadcast to
        # scalar reads.
        m = Model()
        @variable(m, x)
        first_constraint = @constraint(m, x >= 1)
        second_constraint = @constraint(m, x >= 2)
        constraints = [first_constraint, second_constraint]
        duals = Dict(first_constraint => 1.0, second_constraint => 2.0)

        for make_bad_group_read in (
            # An erroring group read exercises the exception fallback.
            _ -> error("batch retrieval failed"),
            # One value cannot represent the two rows, so retrieval must retry.
            _ -> [0.0],
        )
            calls = Vector{Vector{Any}}()
            row_duals = MIPVerify.resolve_row_duals(
                constraints,
                cs -> begin
                    push!(calls, Any[cs...])
                    length(cs) == 1 ? [duals[only(cs)]] : make_bad_group_read(cs)
                end,
            )
            # The failed group read is discarded; each row is retried through the same source
            # as a single-element batch, in order.
            @test row_duals == [1.0, 2.0]
            @test calls == [
                Any[first_constraint, second_constraint],
                Any[first_constraint],
                Any[second_constraint],
            ]
        end

        # A retry that answers with something other than a one-element vector leaves that
        # row's dual unavailable without aborting the others.
        MIPVerify.Memento.TestUtils.@test_log(
            MIPVerify.LOGGER,
            "warn",
            "Single constraint-dual retry returned an incompatible value",
            @test(
                MIPVerify.resolve_row_duals(
                    constraints,
                    cs -> length(cs) == 1 ? "unavailable" : [0.0],
                ) == [nothing, nothing]
            )
        )
    end

    @testset "treats invalid batch elements as independently unavailable" begin
        mock = certification_mock()
        m = Model(() -> mock)
        # The four distinct right-hand sides identify which batch element remains usable. The
        # single valid unit dual cancels x's objective coefficient exactly, so the [0, 5] bounds
        # never enter the certificate.
        @variable(m, 0 <= x <= 5)
        @constraint(m, x >= 1)
        @constraint(m, x >= 2)
        @constraint(m, x >= 3)
        @constraint(m, x >= 4)

        # NaN, infinite, and zero duals are ignored independently; the final unit dual must
        # remain usable to select x >= 4. Non-`Real` elements cannot round-trip through the
        # Float64-typed mock backend, so they are covered by the unit test below.
        optimize_with_mock_duals!(mock, m, AFFINE_GT => [NaN, Inf, 0.0, 1.0])

        # The one valid element certifies x >= 4 despite the three invalid neighbors.
        @test certified_lp_bound(m, lower_bound_type, x, 0.0) == 4.0
    end

    @testset "is_usable_constraint_dual accepts only finite nonzero reals" begin
        @test MIPVerify.is_usable_constraint_dual(1.0)
        @test MIPVerify.is_usable_constraint_dual(-0.25)
        @test !MIPVerify.is_usable_constraint_dual(0.0)
        @test !MIPVerify.is_usable_constraint_dual(NaN)
        @test !MIPVerify.is_usable_constraint_dual(Inf)
        @test !MIPVerify.is_usable_constraint_dual("unavailable")
        @test !MIPVerify.is_usable_constraint_dual(nothing)
    end

    @testset "resolve_row_duals propagates fatal dual-read errors" begin
        m = Model()
        @variable(m, x)
        constraints = [@constraint(m, x >= 1), @constraint(m, x >= 2)]

        for fatal_error in (InterruptException(), OutOfMemoryError(), StackOverflowError())
            # ...thrown by the whole-group read.
            @test_throws typeof(fatal_error) MIPVerify.resolve_row_duals(
                constraints,
                _ -> throw(fatal_error),
            )
            # ...thrown by a single-constraint retry after a wrong-length group read.
            @test_throws typeof(fatal_error) MIPVerify.resolve_row_duals(
                constraints,
                cs -> length(cs) == 1 ? throw(fatal_error) : [0.0],
            )
        end
    end
end
