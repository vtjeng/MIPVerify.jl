using Test
using JuMP
using MIPVerify

@testset "Production backward bounds" begin
    BB = MIPVerify.BackwardBounds
    # Three output columns distinguish the requested target from another class
    # that wins everywhere. The second class crosses the first at pixel 1/2.
    network = MIPVerify.Sequential(
        [MIPVerify.Linear(reshape([0.0, 1.0, 0.0], 1, 3), [0.0, -0.5, 1.0])],
        "requested target",
    )
    for radius in (0.0, 0.03125, 0.1, 0.25, 2.0)
        # Binary-exact radii exercise the singleton, interior, clipped and full boxes.
        result = BB.screen(network, [0.5], 1, [2]; radius)
        @test result !== nothing
        @test result.bounds[1] == result.bounds[3] == -Inf
        lower, upper = BB.witness_box([0.5]; radius)
        @test Rational{BigInt}(lower[1]) >= max(0, 1//2 - Rational{BigInt}(radius))
        @test Rational{BigInt}(upper[1]) <= min(1, 1//2 + Rational{BigInt}(radius))
        witness = BB.search_checked_witness(result, [0.5])
        @test witness !== nothing
        @test witness.witness[1] >= 0.5
    end
    # At a zero radius, class 3 wins but the requested class 2 strictly loses.
    isolated = BB.screen(network, [0.4], 1, [2]; radius = 0.0)
    @test isolated.bounds[2] < 0
    @test BB.repair_witness(isolated, [0.4], [0.4]) === nothing
    @test BB.screen(network, Float32[0.5], 1, [2]) === nothing
    @test BB.screen(network, [0.5], 1, [1]) === nothing
    @test BB.screen(network, [0.5], 1, [4]) === nothing
    @test BB.screen(network, [0.5], 1, [2.0]) === nothing
    @test BB.screen(network, [NaN], 1, [2]) === nothing
    @test BB.screen(network, [0.5], 1, [2]; radius = Inf) === nothing
    @test BB.screen(network, [0.5], 1, [2]; radius = -0.1) === nothing
    # Float32 parameters widen exactly to the Float64 coefficients used by JuMP.
    reduced =
        MIPVerify.Sequential([MIPVerify.Linear(zeros(Float32, 1, 2), Float32[1, 0])], "float32")
    @test BB.screen(reduced, [0.5], 1, [2]) !== nothing
    # BigFloat parameters can lose information when compiled into Float64.
    precise =
        MIPVerify.Sequential([MIPVerify.Linear(zeros(BigFloat, 1, 2), BigFloat[1, 0])], "bigfloat")
    @test BB.screen(precise, [0.5], 1, [2]) === nothing
    # Extra matrix rows must not be silently discarded by the compiler.
    mismatch = MIPVerify.Sequential([MIPVerify.Linear(ones(2, 2), zeros(2))], "wrong shape")
    @test BB.screen(mismatch, [0.5], 1, [2]) === nothing

    # The public perturbation family includes its Float64 radius endpoint, even
    # though Float64(0.1) is larger than the exact decimal value 1/10.
    boundary = MIPVerify.Sequential(
        [MIPVerify.Linear(reshape([0.0, 1.0], 1, 2), [0.1, 0.0])],
        "public radius endpoint tie",
    )
    result = BB.screen(boundary, [0.0], 1, [2])
    repaired = BB.repair_witness(result, [0.0], [0.1])
    @test repaired !== nothing
    @test repaired.witness == [0.1]
    # Explicit tightening choices remain authoritative in the regular formulation.
    custom_relu = MIPVerify.Sequential(
        [
            MIPVerify.Linear(ones(1, 1), [-0.5]),
            MIPVerify.ReLU(MIPVerify.lp),
            MIPVerify.Linear(ones(1, 2), zeros(2)),
        ],
        "custom tightening",
    )
    screened_custom = BB.screen(custom_relu, [0.5], 1, [2])
    @test screened_custom !== nothing
    @test BB.bounded_fallback_network(custom_relu, screened_custom.hidden).layers[2] ===
          custom_relu.layers[2]
    # Built-in MNIST networks explicitly request interval arithmetic at the first ReLU.
    # This matches the first stage of the backward screen exactly.
    first_interval = MIPVerify.Sequential(
        [
            MIPVerify.Linear(ones(1, 1), [-0.5]),
            MIPVerify.ReLU(MIPVerify.interval_arithmetic),
            MIPVerify.Linear(ones(1, 2), zeros(2)),
        ],
        "first-layer interval override",
    )
    @test BB.screen(first_interval, [0.5], 1, [2]) !== nothing
    # A later interval override would bypass backward tightening at that layer.
    later_interval = MIPVerify.Sequential(
        [
            MIPVerify.Linear(ones(1, 1), [-0.5]),
            MIPVerify.ReLU(),
            MIPVerify.Linear(ones(1, 1), [-0.05]),
            MIPVerify.ReLU(MIPVerify.interval_arithmetic),
            MIPVerify.Linear(ones(1, 2), zeros(2)),
        ],
        "later-layer interval override",
    )
    screened_later = BB.screen(later_interval, [0.5], 1, [2])
    @test screened_later !== nothing
    @test BB.bounded_fallback_network(later_interval, screened_later.hidden).layers[4] ===
          later_interval.layers[4]
    # Asymmetric shape, padding, stride and flatten order exercise convolution compilation.
    image = reshape(collect(1.0:12.0) ./ 12, 1, 3, 4, 1)
    for padding in (MIPVerify.SamePadding(), MIPVerify.ValidPadding(), (1, 0, 2, 0))
        conv = MIPVerify.Conv2d(reshape(collect(-3.0:4.0), 2, 2, 1, 2), [0.25, -0.5], 2, padding)
        flatten = MIPVerify.Flatten([1, 3, 2, 4])
        units = length(conv(image))
        output = MIPVerify.Linear(reshape(collect(1.0:2units), units, 2), zeros(2))
        compiled =
            MIPVerify.Sequential([conv, MIPVerify.ReLU(), flatten, output], "padded convolution")
        result = BB.screen(compiled, image, 1, [2])
        @test result !== nothing
        @test last(BB.forward(result.stages, image)) ≈ compiled(image) rtol = 1e-12
    end

    # The three units respectively cross zero, remain negative and remain positive.
    relu_net = MIPVerify.Sequential(
        [
            MIPVerify.Linear(ones(1, 3), [-0.5, -1.0, 0.5]),
            MIPVerify.ReLU(),
            MIPVerify.Linear([1.0 0.0; 1.0 0.0; 0.0 1.0], zeros(2)),
        ],
        "instrumented supplied bounds",
    )
    result = BB.screen(relu_net, [0.5], 1, [2])
    bounded = BB.bounded_fallback_network(relu_net, result.hidden)
    model = Model()
    stats = MIPVerify.VerificationStats()
    model.ext[:MIPVerify] = MIPVerify.MIPVerifyExt(MIPVerify.target_lp, stats)
    pixel = @variable(model, 0.4 <= p <= 0.6)
    bounded([pixel])
    summary = MIPVerify.summarize_verification_stats(stats)
    @test summary[:ReLULayerCount] == 1
    @test summary[:ReLUSplitCount] == 1
    @test summary[:ReLUZeroOutputCount] == 1
    @test summary[:ReLULinearInInputCount] == 1
    @test summary[:BoundSolverCallCount] == 0
    @test stats.relu_layers[1].tightening_algorithm == "backward"
    @test stats.current_relu_layer_index == 0
end
