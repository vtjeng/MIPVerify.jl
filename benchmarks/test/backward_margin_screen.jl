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
