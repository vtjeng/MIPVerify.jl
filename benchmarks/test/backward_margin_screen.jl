using Test
include(joinpath(@__DIR__, "..", "backward_margin_screen.jl"))

# Enumerate exact line segments for a one-dimensional ReLU network. On each
# segment every logit is affine, so its maximum margin occurs at an endpoint.
function exact_breakpoints(stages, lo, hi)
    Q = Rational{BigInt}
    segments = [(Q(lo), Q(hi), Q[1], Q[0])]
    for (k, stage) in enumerate(stages)
        out = typeof(segments)()
        for (l, u, slope, constant) in segments
            a = Q.(stage.weights)' * slope
            b = Q.(stage.weights)' * constant + Q.(stage.bias)
            knots = [l, u]
            if k < length(stages)
                for j in eachindex(a)
                    iszero(a[j]) && continue
                    root = -b[j]/a[j]
                    l < root < u && push!(knots, root)
                end
            end
            sort!(unique!(knots))
            for (left, right) in zip(knots, knots[2:end])
                active = k == length(stages) ? trues(length(a)) : a .* ((left+right)/2) + b .> 0
                push!(out, (left, right, a .* active, b .* active))
            end
        end
        segments = out
    end
    return maximum((a[2]-a[1])*t+b[2]-b[1] for (l, u, a, b) in segments for t in (l, u))
end

@testset "Backward margin upper bound" begin
    rng = MersenneTwister(20261125)
    # Two hidden layers exercise intermediate tightening. Offset rectifiers
    # cross inside the box, and mixed signs exercise both ReLU envelopes.
    for _ in 1:40
        s = [
            AffineStage(sparse(randn(rng, inputs, units)), randn(rng, units) ./ 2) for
            (inputs, units) in [(1, 4), (4, 3), (3, 2)]
        ]
        x = [0.5] # Centered box avoids clipping when checking exact extrema.
        upper = margin_bounds(s, x, 1)[2]
        @test Rational{BigInt}(upper) >= exact_breakpoints(s, 2//5, 3//5)
    end
    # An exact tie is feasible for MIPVerify's zero-margin target condition.
    tied = [AffineStage(sparse(reshape([1.0, 1.0], 1, 2)), zeros(2))]
    @test margin_bounds(tied, [0.5], 1)[2] >= 0
    @test !confirms_witness(tied, [0.5], 1, 2)
    # Check padding, channel order, asymmetric flatten permutation and logits
    # against the independently implemented WK17a forward pass.
    # An asymmetric convolution and flatten permutation expose channel-order
    # mistakes without loading weights or datasets from the network.
    nn = MIPVerify.Sequential(
        [
            MIPVerify.Conv2d(reshape(collect(-3.0:4.0), 2, 2, 1, 2), [0.25, -0.5], 1),
            MIPVerify.ReLU(),
            MIPVerify.Flatten([1, 3, 2, 4]),
            MIPVerify.Linear(reshape(collect(-12.0:11.0), 12, 2) ./ 10, [0.5, -0.25]),
        ],
        "synthetic convolution",
    )
    input = rand(rng, 1, 2, 3, 1)
    stages = compile_stages(nn, size(input))
    @test last(forward(stages, input)) ≈ nn(input) rtol = 1e-12
    @test_throws ArgumentError hidden_bounds(stages, fill(NaN, size(input)))
    # A nontrivial leading permutation catches a swap of the input pixel axes.
    # RSL18a flattens its image before the first dense layer.
    flatten = MIPVerify.Flatten([1, 3, 2, 4])
    pixels = reshape(collect(1.0:6.0), 1, 2, 3, 1)
    # Distinct weights expose a wrong pixel order in each basis direction.
    dense = MIPVerify.Linear(reshape(collect(1.0:12.0), 6, 2), zeros(2))
    flat_net = MIPVerify.Sequential([flatten, dense], "input permutation")
    flat_stages = compile_stages(flat_net, size(pixels))
    for index in eachindex(pixels)
        basis = zeros(size(pixels))
        basis[index] = 1.0
        @test last(forward(flat_stages, basis)) == flat_net(basis)
    end
    @test_throws ArgumentError compile_stages(MIPVerify.Sequential([], "empty"), (1,))

end

@testset "Pass certified bounds to the integer fallback" begin
    # The box crosses zero for the first unit, lies below zero for the
    # second and above zero for the third. Each ReLU case enters the model.
    network = MIPVerify.Sequential(
        [
            MIPVerify.Linear(reshape([1.0, 1.0, 1.0], 1, 3), [-0.5, -1.0, 0.5]),
            MIPVerify.ReLU(),
            MIPVerify.Linear([1.0 0.0; 1.0 0.0; 0.0 1.0], [0.0, 0.0]),
        ],
        "bounded fallback check",
    )
    center = [0.5]
    stages = compile_stages(network, size(center))
    hidden, _, _ = hidden_bounds(stages, center)
    bounded = bounded_fallback_network(network, hidden)
    for pixel in [0.4, 0.5, 0.6]
        @test bounded([pixel]) == network([pixel])
    end
    model = Model(HiGHS.Optimizer)
    # HiGHS shares its thread scheduler across models in this process. Match
    # the verifier's one-thread setting before the first optimizer runs.
    set_optimizer_attribute(model, "threads", 1)
    set_silent(model)
    pixel = @variable(model, 0.4 <= p <= 0.6)
    scores = bounded([pixel])
    @objective(model, Max, scores[1] - scores[2])
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ -0.9 # At the lower pixel endpoint, ReLU(−0.1) is zero.
end

@testset "Verification routes and solver counters" begin
    # Constant scores [1, 0] certify the true first class immediately; reversed
    # scores make the unperturbed center a strictly adversarial input.
    for (bias, source, outcome) in [
        ([1.0, 0.0], "backward", "certified_no_adversarial_example"),
        ([0.0, 1.0], "ascent", "adversarial_example_found_or_best_known"),
    ]
        network = MIPVerify.Sequential([MIPVerify.Linear(zeros(1, 2), bias)], "constant scores")
        center = [0.5] # Interior point gives a nonempty radius-0.1 box.
        stages = compile_stages(network, size(center))
        result = verify_image(network, stages, center, 1)
        @test result.source == source
        @test result.semantic_outcome == outcome
        @test result.solve_status == "NO_SOLVER"
        @test result.main_solve_wall_time_seconds == 0
        @test result.bound_solver_call_count == 0
        @test (result.witness !== nothing) == (source == "ascent")
    end
    # Identical constant scores attain a tie. Strict witness search cannot
    # resolve it, so the MIP must retain the feasible zero-margin input.
    tied = MIPVerify.Sequential([MIPVerify.Linear(zeros(1, 2), zeros(2))], "tied scores")
    result = verify_image(tied, compile_stages(tied, (1,)), [0.5], 1)
    @test result.source == "MIP"
    @test result.solve_status == "OPTIMAL"
    @test result.semantic_outcome == "adversarial_example_found_or_best_known"
    @test result.witness !== nothing
    @test result.main_solve_wall_time_seconds >= 0
end

@testset "Strict witness validation and repair" begin
    Q = Rational{BigInt}
    # Byte-normalized centers can differ from their exact byte/255 values.
    # The inward endpoints must fit both interpretations at radius 1/10.
    center = [0.0, 1 / 255, 51 / 255, 1.0]
    lower, upper = witness_box(center)
    for i in eachindex(center)
        original = round(Int, 255 * center[i]) // 255
        @test 0 <= Q(lower[i]) <= Q(upper[i]) <= 1
        @test abs(Q(lower[i]) - Q(center[i])) <= 1//10
        @test abs(Q(upper[i]) - Q(center[i])) <= 1//10
        @test abs(Q(lower[i]) - original) <= 1//10
        @test abs(Q(upper[i]) - original) <= 1//10
    end
    # Identical nonzero columns produce an exact tie with uncertain rounded scores.
    tied = [AffineStage(sparse(reshape([1.0, 1.0], 1, 2)), zeros(2))]
    lo, hi = witness_box([0.5])
    @test check_witness(tied, [0.5], 1, lo, hi) == "exact_rational"
    @test check_witness(tied, [0.7], 1, lo, hi) == "invalid"
    @test check_witness(tied, [NaN], 1, lo, hi) == "invalid"
    @test check_witness(tied, Float64[], 1, lo, hi) == "invalid"
    # Solving cannot make a strictly losing constant target a valid witness.
    losing = [AffineStage(sparse(zeros(1, 2)), [1.0, 0.0])]
    @test repair_witness(losing, [0.5], [0.5], 1) === nothing
    @test repair_witness(losing, [0.5], nothing, 1) === nothing
    # The exact target-minus-true difference is positive although the native
    # affine accumulation loses the cancellation between the unit coefficients.
    cancellation = [AffineStage(sparse([1.0 1.0; 0.75eps() -1.0; -1.0 0.9eps()]), zeros(2))]
    point = ones(3)
    lo, hi = witness_box(point)
    @test exact_scores(cancellation, point)[2] > exact_scores(cancellation, point)[1]
    @test check_witness(cancellation, point, 1, lo, hi) == "exact_rational"
    # Reverse the uncertain cancellation to make the exact target score lower.
    reversed = [AffineStage(cancellation[1].weights[:, [2, 1]], zeros(2))]
    @test check_witness(reversed, point, 1, lo, hi) == "invalid"
    # Binary Float64(0.1) lies above the exact decimal budget from a zero center.
    # Its previous representable neighbor is inside that budget.
    lo_zero, hi_zero = witness_box([0.0])
    @test check_witness(tied, [0.1], 1, lo_zero, hi_zero) == "invalid"
    @test check_witness(tied, [hi_zero[1]], 1, lo_zero, hi_zero) == "exact_rational"
    # At x=1/2 the target trails by one ulp. A nearby gradient step recovers
    # a strict witness, exercising repair without a sample-specific branch.
    local_stages = [AffineStage(sparse(reshape([0.0, 1.0], 1, 2)), [0.0, -nextfloat(0.5)])]
    repaired = repair_witness(local_stages, [0.5], [0.5], 1)
    @test repaired !== nothing
    @test repaired.witness[1] > 0.5
    @test check_witness(local_stages, vec(repaired.witness), 1, witness_box([0.5])...) != "invalid"
    # A tie attainable only at the lower endpoint still enters and exits the
    # MIP as a verified witness, with its original solver status intact.
    network = MIPVerify.Sequential(
        [MIPVerify.Linear(reshape([1.0, 0.0], 1, 2), zeros(2))],
        "endpoint tie",
    )
    result = verify_image(network, compile_stages(network, (1,)), [0.05], 1)
    @test result.solve_status == "OPTIMAL"
    @test result.semantic_outcome == "adversarial_example_found_or_best_known"
    @test result.witness[1] == 0
    @test result.witness_verification == "exact_rational"
end

@testset "An optimal solver status does not verify a witness" begin
    # Duplicate rectifiers are exactly equal on the whole box. Their separate
    # relaxations leave a positive upper bound, so this reaches the MIP. A
    # target score 1e-9 below true is within typical feasibility tolerances.
    network = MIPVerify.Sequential(
        [
            MIPVerify.Linear(reshape([1.0, 1.0], 1, 2), [-0.5, -0.5]),
            MIPVerify.ReLU(),
            MIPVerify.Linear([1.0 0.0; 0.0 1.0], [1e-9, 0.0]),
        ],
        "near-tie duplicate rectifiers",
    )
    result = verify_image(network, compile_stages(network, (1,)), [0.5], 1)
    @test result.source == "MIP"
    @test result.witness === nothing
    @test result.witness_verification == "none"
    @test result.semantic_outcome in
          ("witness_verification_failed", "certified_no_adversarial_example")
    # A solver may establish infeasibility exactly enough to reject this model.
    # If it accepts a primal point, preserve OPTIMAL but reject that point.
    if result.solve_status == "OPTIMAL"
        @test result.semantic_outcome == "witness_verification_failed"
    end
end
