# WK17a fixed-radius feasibility verifier using affine bounds on score differences.
using MIPVerify
using SparseArrays
using Random
using CSV, HiGHS, JuMP
if !isdefined(@__MODULE__, :BenchmarkHelpers)
    include("BenchmarkHelpers.jl")
end
using .BenchmarkHelpers

# WK17a alternates affine layers and ReLUs. Convolutions have few input
# connections per output pixel. Store a column per output so backward
# propagation skips the full receptive field when its coefficient is zero.
struct AffineStage
    weights::SparseMatrixCSC{Float64,Int} # one column per output
    bias::Vector{Float64}
end

function affine_stage(p::MIPVerify.Conv2d, ids)
    n, h, w, channels = size(ids)
    fh, fw, _, outputs = size(p.filter)
    (oh, ow), (dy, dx) = MIPVerify.compute_output_parameters(h, w, fh, fw, p.stride, p.padding)
    out = LinearIndices((n, oh, ow, outputs))
    rows, cols, weights = Int[], Int[], Float64[]
    for pos in CartesianIndices(size(out))
        batch, y, x, channel = Tuple(pos)
        for c in 1:channels, j in 1:fw, i in 1:fh
            yy = (y-1)*p.stride+i-dy
            xx = (x-1)*p.stride+j-dx
            if 1 <= yy <= h && 1 <= xx <= w
                push!(rows, ids[batch, yy, xx, c])
                push!(cols, out[pos])
                push!(weights, p.filter[i, j, c, channel])
            end
        end
    end
    return AffineStage(
        sparse(rows, cols, weights, length(ids), length(out)),
        repeat(Float64.(p.bias); inner = n*oh*ow),
    ),
    size(out)
end

function affine_stage(p::MIPVerify.Linear, ids)
    return AffineStage(sparse(Float64.(p.matrix[invperm(vec(ids)), :])), Float64.(p.bias)),
    size(p.bias)
end

function compile_stages(nn, shape)
    !isempty(nn.layers) || throw(ArgumentError("network must have affine layers"))
    stages = AffineStage[]
    ids = reshape(1:prod(shape), shape)
    expect_relu = false
    flattened = false
    for p in nn.layers
        if p isa MIPVerify.Flatten
            expect_relu && throw(ArgumentError("Flatten must precede an affine layer"))
            flattened && throw(ArgumentError("only one Flatten layer is supported"))
            flattened = true
            ids = p(collect(ids))
        elseif p isa MIPVerify.ReLU
            expect_relu || error("Expected affine layer before ReLU")
            expect_relu = false
        else
            expect_relu && error("Adjacent affine layers require different input bounds")
            p isa Union{MIPVerify.Conv2d,MIPVerify.Linear} ||
                throw(ArgumentError("only Conv2d, Linear, ReLU, and Flatten layers are supported"))
            stage, shape = affine_stage(p, ids)
            push!(stages, stage)
            ids = reshape(1:prod(shape), shape)
            expect_relu = true
        end
    end
    expect_relu || error("Expected a final affine layer")
    return stages
end

# Outward rounding encloses the real affine sum. fma performs one rounding.
function interval_stage(stage, lo, hi)
    W = stage.weights
    lower, upper = copy(stage.bias), copy(stage.bias)
    for j in eachindex(lower)
        l, u = lower[j], upper[j]
        @inbounds for k in nzrange(W, j)
            i, w = rowvals(W)[k], nonzeros(W)[k]
            a, b = w >= 0 ? (lo[i], hi[i]) : (hi[i], lo[i])
            l = prevfloat(fma(w, a, l))
            u = nextfloat(fma(w, b, u))
        end
        lower[j], upper[j] = l, u
    end
    return lower, upper
end

function backward_affine(stage, coeff, intercept)
    W = stage.weights
    prev = zeros(size(W, 1))
    for j in eachindex(coeff)
        a = coeff[j]
        iszero(a) && continue
        intercept = nextfloat(fma(a, stage.bias[j], intercept))
        @inbounds for k in nzrange(W, j)
            i, w = rowvals(W)[k], nonzeros(W)[k]
            # Previous activations (or input pixels) are nonnegative, so
            # rounding each coefficient upward preserves an affine upper bound.
            prev[i] = nextfloat(fma(a, w, prev[i]))
        end
    end
    return prev, intercept
end

function backward_relu!(coeff, intercept, lo, hi)
    for j in eachindex(coeff)
        a, l, u = coeff[j], lo[j], hi[j]
        if u <= 0
            coeff[j] = 0
        elseif l < 0 && !iszero(a)
            if a > 0
                # Choose a slope, then independently cover both endpoints of
                # a*ReLU. This remains valid when the secant slope rounds.
                k = clamp(a*(u/(u-l)), 0, a)
                offset = max(nextfloat(-k*l), nextfloat(nextfloat(a-k)*u))
                intercept = nextfloat(intercept+offset)
                coeff[j] = k
            elseif u < -l
                coeff[j] = 0
            end
        end
    end
    return intercept
end

function upper_at_box(coeff, intercept, lo, hi)
    for (i, a) in enumerate(coeff)
        iszero(a) && continue
        intercept = nextfloat(fma(a, a > 0 ? hi[i] : lo[i], intercept))
    end
    return intercept
end

"""
Construct an affine upper bound on a linear objective on the outputs of affine stage `last` over the pixel box.

`hidden` contains bounds on the inputs to each preceding ReLU. Propagating the
objective through affine layers and upper ReLU envelopes leaves an affine
function of the pixels, maximized at the appropriate box endpoint per pixel.
"""
function backward_plane(stages, last, coeff, hidden)
    intercept = 0.0
    for k in last:-1:1
        coeff, intercept = backward_affine(stages[k], coeff, intercept)
        if k > 1
            intercept = backward_relu!(coeff, intercept, hidden[k-1]...)
        end
    end
    return coeff, intercept
end

function backward_upper(stages, last, coeff, hidden, input_lo, input_hi)
    plane, intercept = backward_plane(stages, last, coeff, hidden)
    return upper_at_box(plane, intercept, input_lo, input_hi)
end

function margin_upper(
    stages,
    hidden,
    input_lo,
    input_hi,
    true_label,
    target_label,
    corners = nothing,
)
    coeff = zeros(length(stages[end].bias))
    coeff[target_label], coeff[true_label] = 1, -1
    plane, intercept = backward_plane(stages, length(stages), coeff, hidden)
    bound = upper_at_box(plane, intercept, input_lo, input_hi)
    if corners !== nothing && bound >= 0
        corners[target_label] = plane .> 0
    end
    return bound
end

"""
Return preactivation bounds for each hidden layer and the enclosing input box.

Start with forward interval arithmetic. For later layers, tighten each unit by
propagating its positive and negative objectives back to the input box, using
only the bounds already established for earlier layers. The pixel radius is 0.1.
"""
function hidden_bounds(stages, x)
    all(v -> isfinite(v) && 0 <= v <= 1, x) ||
        throw(ArgumentError("input pixels must be finite values in [0, 1]"))
    # The center, decimal radius and endpoint arithmetic each contribute at
    # most one ulp at unit scale on [0,1]. Widen by four before clipping.
    input_lo = max.(0.0, vec(x) .- 0.1 .- 4eps())
    input_hi = min.(1.0, vec(x) .+ 0.1 .+ 4eps())
    lo, hi = input_lo, input_hi
    hidden = Tuple{Vector{Float64},Vector{Float64}}[]
    for (k, stage) in enumerate(stages[1:(end-1)])
        l, u = interval_stage(stage, lo, hi)
        if k > 1
            for j in eachindex(u)
                u[j] <= 0 && continue
                coeff = zeros(length(u))
                coeff[j] = 1
                u[j] = min(u[j], backward_upper(stages, k, coeff, hidden, input_lo, input_hi))
                u[j] <= 0 && continue
                coeff[j] = -1
                l[j] = max(l[j], -backward_upper(stages, k, coeff, hidden, input_lo, input_hi))
            end
        end
        push!(hidden, (l, u))
        lo, hi = max.(l, 0), max.(u, 0)
    end
    return hidden, input_lo, input_hi
end

function margin_bounds(stages, y, hidden, input_lo, input_hi, corners = nothing)
    return [
        j == y ? -Inf : margin_upper(stages, hidden, input_lo, input_hi, y, j, corners) for
        j in eachindex(stages[end].bias)
    ]
end

margin_bounds(stages, x, y) = margin_bounds(stages, y, hidden_bounds(stages, x)...)

# The bounds apply only to the input box that produced them. Build a new
# network per image so a later solve cannot reuse another image's bounds.
struct PreboundedReLU <: MIPVerify.Layer
    lower::Vector{Float64}
    upper::Vector{Float64}
end

(p::PreboundedReLU)(x::Array{<:Real}) = MIPVerify.relu(x)
function (p::PreboundedReLU)(x::Array{<:MIPVerify.JuMPLinearType})
    return map(x, reshape(p.lower, size(x)), reshape(p.upper, size(x))) do value, lo, hi
        MIPVerify.relu(value, lo, hi)
    end
end

function bounded_fallback_network(nn, hidden)
    layers = copy(nn.layers)
    relus = findall(layer -> layer isa MIPVerify.ReLU, layers)
    length(relus) == length(hidden) || error("A bound pair is required for every ReLU layer")
    for (i, bounds) in zip(relus, hidden)
        layers[i] = PreboundedReLU(bounds...)
    end
    return MIPVerify.Sequential(layers, nn.UUID)
end

function forward(stages, x)
    act = Vector{Float64}[vec(x)]
    for (k, stage) in enumerate(stages)
        a = transpose(stage.weights) * act[end] + stage.bias
        push!(act, k == length(stages) ? a : max.(a, 0))
    end
    return act
end

# Verify the positive margin over a singleton box with outward rounding.
function confirms_witness(stages, z, y, j)
    lo = hi = z
    for (k, stage) in enumerate(stages)
        lo, hi = interval_stage(stage, lo, hi)
        if k < length(stages)
            lo, hi = max.(lo, 0), max.(hi, 0)
        end
    end
    return lo[j] > hi[y]
end

# Float64 endpoints rounded into the exact radius-1/10 box. MNIST centers
# also retain their original byte/255 interpretation. These endpoints constrain
# proposed witnesses only; the certification and MIP domains are untouched.
function compute_witness_box(x)
    Q = Rational{BigInt}
    pixels = vec(x)
    bytes = round.(Int, 255 .* pixels)
    byte_centers = all(pixels .== bytes ./ 255)
    lower, upper = zeros(length(pixels)), zeros(length(pixels))
    for i in eachindex(pixels)
        center = Q(pixels[i])
        l, u = max(Q(0), center - 1//10), min(Q(1), center + 1//10)
        if byte_centers
            original = Q(bytes[i]) / 255
            l, u = max(l, original - 1//10), min(u, original + 1//10)
        end
        lower[i], upper[i] = Float64(l), Float64(u)
        Q(lower[i]) < l && (lower[i] = nextfloat(lower[i]))
        Q(upper[i]) > u && (upper[i] = prevfloat(upper[i]))
    end
    return lower, upper
end

# MNIST reuses the same 256 centers. Compute their exact inward rounding once.
const BYTE_WITNESS_LOWER, BYTE_WITNESS_UPPER = compute_witness_box(collect(0:255) ./ 255)

function witness_box(x)
    pixels = vec(x)
    bytes = round.(Int, 255 .* pixels)
    if all(i -> 0 <= bytes[i] <= 255 && pixels[i] == bytes[i] / 255, eachindex(pixels))
        indices = bytes .+ 1
        return BYTE_WITNESS_LOWER[indices], BYTE_WITNESS_UPPER[indices]
    end
    return compute_witness_box(x)
end

function exact_scores(stages, z)
    Q = Rational{BigInt}
    values = Q.(z)
    for (k, stage) in enumerate(stages)
        values = transpose(Q.(stage.weights)) * values + Q.(stage.bias)
        k < length(stages) && (values = max.(values, 0))
    end
    return values
end

# Accept equality when the interval endpoints prove it. Overlapping intervals
# require exact evaluation unless every target is already ruled out.
function check_witness(stages, z, y, lower, upper)
    length(z) == length(lower) || return "invalid"
    all(isfinite, z) && all(lower .<= z .<= upper) || return "invalid"
    lo = hi = z
    for (k, stage) in enumerate(stages)
        lo, hi = interval_stage(stage, lo, hi)
        if k < length(stages)
            lo, hi = max.(lo, 0), max.(hi, 0)
        end
    end
    targets = setdiff(eachindex(lo), y)
    any(lo[j] >= hi[y] for j in targets) && return "outward_interval"
    all(hi[j] < lo[y] for j in targets) && return "invalid"
    scores = exact_scores(stages, z)
    return any(scores[j] >= scores[y] for j in targets) ? "exact_rational" : "invalid"
end

# A solver point can miss the exact box or output constraint by its feasibility
# tolerance. Clip into the exact box, then try nearby projected gradient steps.
# Every returned point is independently checked; search failure proves nothing.
function repair_witness(stages, x, proposed, y)
    proposed === nothing && return nothing
    length(proposed) == length(x) && all(isfinite, proposed) || return nothing
    lower, upper = witness_box(x)
    z = clamp.(vec(proposed), lower, upper)
    proof = check_witness(stages, z, y, lower, upper)
    proof != "invalid" && return (witness = reshape(z, size(x)), proof = proof)
    for step in (1e-9, 1e-7, 1e-5, 1e-3, 0.02)
        act = forward(stages, z)
        targets = setdiff(eachindex(act[end]), y)
        target = targets[argmax(act[end][targets])]
        gradient = zeros(length(act[end]))
        gradient[target], gradient[y] = 1, -1
        for k in reverse(eachindex(stages))
            gradient = stages[k].weights * gradient
            k > 1 && (gradient[act[k] .<= 0] .= 0)
        end
        trial = clamp.(z + step * sign.(gradient), lower, upper)
        proof = check_witness(stages, trial, y, lower, upper)
        proof != "invalid" && return (witness = reshape(trial, size(x)), proof = proof)
        current = act[end][target] - act[end][y]
        scores = last(forward(stages, trial))
        if maximum(scores[targets]) - scores[y] > current
            z = trial
        end
    end
    return nothing
end

function find_box_witness(stages, x, y, bounds)
    # Shrink the witness box by the same rounding allowance.
    lo = max.(0.0, vec(x) .- 0.1 .+ 4eps())
    hi = min.(1.0, vec(x) .+ 0.1 .- 4eps())
    rng = MersenneTwister(1)
    for j in sortperm(bounds, rev = true)
        bounds[j] < 0 && continue
        for start in 1:4
            z = start == 1 ? copy(vec(x)) : lo + (hi-lo) .* rand(rng, length(lo))
            for _ in 1:32
                act = forward(stages, z)
                if act[end][j] > act[end][y] && confirms_witness(stages, z, y, j)
                    return reshape(z, size(x))
                end
                g = zeros(length(bounds))
                g[j], g[y] = 1, -1
                for k in reverse(eachindex(stages))
                    g = stages[k].weights * g
                    if k > 1
                        g[act[k] .<= 0] .= 0
                    end
                end
                next = clamp.(z .+ 0.02 .* sign.(g), lo, hi)
                next == z && break
                z = next
            end
        end
    end
    return nothing
end

include("forward_parallelograms.jl")

function staged_margin_screen(
    stages,
    x,
    y,
    corners = nothing,
    forward_corners = nothing;
    forward_screen = true,
)
    hidden, input_lo, input_hi = hidden_bounds(stages, x)
    bounds = margin_bounds(stages, y, hidden, input_lo, input_hi, corners)
    (!forward_screen || maximum(bounds) < 0) && return hidden, bounds, "backward"
    tighten_forward_margins!(bounds, stages, y, input_lo, input_hi, forward_corners)
    return hidden, bounds, "forward_parallelogram"
end

# Check the backward corners first, then the forward corners for surviving rivals.
# Round endpoints inward for the exact witness check.
function search_checked_witness(stages, x, y, bounds, corners, forward_corners = nothing)
    lower, upper = witness_box(x)
    targets = sort(findall(>=(0), bounds); by = j -> bounds[j], rev = true)
    for (source, directions) in (("corner", corners), ("forward_corner", forward_corners))
        isnothing(directions) && continue
        for target in targets
            if source == "forward_corner" && directions[target] == corners[target]
                continue
            end
            point = ifelse.(directions[target], upper, lower)
            proof = check_witness(stages, point, y, lower, upper)
            if proof != "invalid"
                return (
                    witness = reshape(point, size(x)),
                    proof = proof,
                    source = source,
                    repaired = false,
                )
            end
        end
    end
    proposed = find_box_witness(stages, x, y, bounds)
    checked = repair_witness(stages, x, proposed, y)
    checked === nothing && return nothing
    return (; checked..., source = "ascent", repaired = vec(checked.witness) != vec(proposed))
end

function verify_image(
    nn,
    stages,
    x,
    y;
    prebound_fallback = true,
    forward_screen = true,
    second_corner = forward_screen,
)
    1 <= y <= length(stages[end].bias) || throw(ArgumentError("true label is out of range"))
    corners = Dict{Int,BitVector}()
    forward_corners = forward_screen && second_corner ? Dict{Int,BitVector}() : nothing
    hidden, bounds, screen_source = staged_margin_screen(
        stages,
        x,
        y,
        corners,
        forward_corners;
        forward_screen = forward_screen,
    )
    worst_margin = maximum(bounds)
    if worst_margin < 0
        return (
            solve_status = "NO_SOLVER",
            semantic_outcome = "certified_no_adversarial_example",
            source = screen_source,
            witness = nothing,
            witness_verification = "none",
            witness_repaired = false,
            margin_upper = worst_margin,
            main_solve_wall_time_seconds = 0.0,
            bound_solver_call_count = 0,
        )
    end
    checked = search_checked_witness(stages, x, y, bounds, corners, forward_corners)
    if checked !== nothing
        return (
            solve_status = "NO_SOLVER",
            semantic_outcome = "adversarial_example_found_or_best_known",
            source = checked.source,
            witness = checked.witness,
            witness_verification = checked.proof,
            witness_repaired = checked.repaired,
            margin_upper = worst_margin,
            main_solve_wall_time_seconds = 0.0,
            bound_solver_call_count = 0,
        )
    end
    opts = Dict("threads" => 1, "output_flag" => false, "time_limit" => 120.0)
    bt = Dict("threads" => 1, "output_flag" => false, "time_limit" => 20.0)
    fallback = prebound_fallback ? bounded_fallback_network(nn, hidden) : nn
    d = MIPVerify.find_adversarial_example(
        fallback,
        x,
        y,
        HiGHS.Optimizer,
        opts;
        invert_target_selection = true,
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
        norm_order = Inf,
        tightening_algorithm = MIPVerify.target_lp,
        tightening_options = bt,
        solve_if_predicted_in_targeted = false,
        collect_stats = true,
        adversarial_example_objective = MIPVerify.feasibility,
    )
    status = string(get(d, :SolveStatus, "SKIPPED_PREDICTED_IN_TARGETED"))
    proposed = get(d, :PerturbedInputValue, nothing)
    checked = repair_witness(stages, x, proposed, y)
    outcome = if checked !== nothing
        "adversarial_example_found_or_best_known"
    elseif proposed !== nothing
        "witness_verification_failed"
    else
        classify_semantic_outcome(status, false, false)
    end
    return (
        solve_status = status,
        semantic_outcome = outcome,
        source = "MIP",
        witness = checked === nothing ? nothing : checked.witness,
        witness_verification = checked === nothing ? "none" : checked.proof,
        witness_repaired = checked !== nothing && vec(checked.witness) != vec(proposed),
        margin_upper = worst_margin,
        main_solve_wall_time_seconds = Float64(get(d, :MainSolveWallTime, 0.0)),
        bound_solver_call_count = Int(get(d, :BoundSolverCallCount, 0)),
    )
end

function benchmark(args)
    options = parse_args(args)
    prebound_fallback = parse(Bool, get(options, "prebound-fallback", "true"))
    output = abspath(options["out"])
    mkpath(output)
    samples = parse_sample_spec(get(options, "samples", "1:100"))
    network_name = get(options, "network", "MNIST.WK17a_linf0.1_authors")
    nn = MIPVerify.get_example_network_params(network_name)
    test = MIPVerify.read_datasets("MNIST").test
    setup = @elapsed stages = compile_stages(nn, size(MIPVerify.get_image(test, 1)))
    # Keep compilation separate from the per-image cost. Report setup and
    # warm-up costs so a caller can include either in a one-off comparison.
    warm = @elapsed verify_image(
        nn,
        stages,
        MIPVerify.get_image(test, 2001),
        MIPVerify.get_label(test, 2001)+1;
        prebound_fallback = prebound_fallback,
    )
    rows = NamedTuple[]
    for i in samples
        x, y = MIPVerify.get_image(test, i), MIPVerify.get_label(test, i)+1
        t = @elapsed result = verify_image(nn, stages, x, y; prebound_fallback = prebound_fallback)
        push!(
            rows,
            (
                sample_index = i,
                network = nn.UUID,
                solve_status = result.solve_status,
                adversarial_example_objective = "feasibility",
                semantic_outcome = result.semantic_outcome,
                source = result.source,
                witness_verification = result.witness_verification,
                witness_repaired = result.witness_repaired,
                total_time_seconds = t,
                margin_upper = result.margin_upper,
                main_solve_wall_time_seconds = result.main_solve_wall_time_seconds,
                bound_solver_call_count = result.bound_solver_call_count,
                perturbed_input_value = result.witness === nothing ? "" :
                                        join(vec(result.witness), ';'),
            ),
        )
    end
    CSV.write(joinpath(output, "benchmark_per_sample.csv"), rows)
    CSV.write(joinpath(output, "setup.csv"), [(setup_seconds = setup, warmup_seconds = warm)])
end

if abspath(PROGRAM_FILE) == @__FILE__
    MIPVerify.set_log_level!("warn")
    benchmark(ARGS)
end
