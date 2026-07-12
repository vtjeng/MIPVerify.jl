using Test

using HiGHS
using JuMP
using MathOptInterface: MathOptInterface
using MIPVerify:
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

    @testset "propagates unexpected errors when reading the MIP objective bound" begin
        # MockOptimizer throws a KeyError when ObjectiveBound was never set; only the three
        # whitelisted MOI attribute errors are converted into a fallback to b_0.
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

        @test_throws KeyError tight_upperbound(x; nta = mip)
    end

    @testset "projected_dual_and_reference ignores unsupported sets" begin
        @test projected_dual_and_reference(MathOptInterface.ZeroOne(), 1.0) === nothing
    end

    @testset "constraint_dual_or_nothing classifies retrieval errors" begin
        unsupported = MathOptInterface.UnsupportedAttribute(MathOptInterface.ConstraintDual())
        @test constraint_dual_or_nothing(:constraint, _ -> throw(unsupported)) === nothing
        @test_throws ErrorException constraint_dual_or_nothing(:constraint, _ -> error("boom"))
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
end
