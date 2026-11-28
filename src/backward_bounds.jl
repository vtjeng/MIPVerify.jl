module BackwardBounds

import ..MIPVerify
using SparseArrays
using Random

# WK17a alternates affine layers and ReLUs. Convolutions have few input
# connections per output pixel. Store a column per output so backward
# propagation skips the full receptive field when its coefficient is zero.
struct AffineStage
    weights::SparseMatrixCSC{Float64,Int} # one column per output
    bias::Vector{Float64}
end

function affine_stage(p::MIPVerify.Conv2d, ids)
    n, h, w, channels = size(ids)
    fh, fw, inputs, outputs = size(p.filter)
    inputs == channels ||
        throw(DimensionMismatch("convolution channel count does not match its filter"))
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
    size(p.matrix, 1) == length(ids) ||
        throw(DimensionMismatch("linear input size does not match its weights"))
    return AffineStage(sparse(Float64.(p.matrix[invperm(vec(ids)), :])), Float64.(p.bias)),
    size(p.bias)
end

function compile_stages(nn::MIPVerify.Sequential, shape)
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
            expect_relu || throw(ArgumentError("Expected affine layer before ReLU"))
            expect_relu = false
        else
            expect_relu &&
                throw(ArgumentError("Adjacent affine layers require different input bounds"))
            p isa Union{MIPVerify.Conv2d,MIPVerify.Linear} ||
                throw(ArgumentError("only Conv2d, Linear, ReLU, and Flatten layers are supported"))
            values = p isa MIPVerify.Linear ? (p.matrix, p.bias) : (p.filter, p.bias)
            all(v -> eltype(v) <: Union{Float32,Float64} && all(isfinite, v), values) ||
                throw(ArgumentError("affine parameters must be finite Float32 or Float64 values"))
            stage, shape = affine_stage(p, ids)
            push!(stages, stage)
            ids = reshape(1:prod(shape), shape)
            expect_relu = true
        end
    end
    expect_relu || throw(ArgumentError("Expected a final affine layer"))
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

function margin_plane(stages, hidden, reference, target)
    coeff = zeros(length(stages[end].bias))
    coeff[target], coeff[reference] = 1, -1
    return backward_plane(stages, length(stages), coeff, hidden)
end

function margin_upper(stages, hidden, input_lo, input_hi, reference, target, corners = nothing)
    plane, intercept = margin_plane(stages, hidden, reference, target)
    bound = upper_at_box(plane, intercept, input_lo, input_hi)
    if corners !== nothing && bound >= 0
        corners[target] = plane .> 0
    end
    return bound
end

"""
Return preactivation bounds for each hidden layer and the enclosing input box.

Start with forward interval arithmetic. For later layers, tighten each unit by
propagating its positive and negative objectives back to the input box, using
only the bounds already established for earlier layers.
"""
function hidden_bounds(stages, x; radius::Float64 = 0.1)
    all(v -> isfinite(v) && 0 <= v <= 1, x) ||
        throw(ArgumentError("input pixels must be finite values in [0, 1]"))
    isfinite(radius) && radius >= 0 || throw(ArgumentError("radius must be finite and nonnegative"))
    radius = min(radius, 1.0)
    # The center, decimal radius and endpoint arithmetic each contribute at
    # most one ulp at unit scale on [0,1]. Widen by four before clipping.
    input_lo = max.(0.0, vec(x) .- radius .- 4eps())
    input_hi = min.(1.0, vec(x) .+ radius .+ 4eps())
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
    stats = MIPVerify.get_verification_stats(x)
    layer = MIPVerify.begin_relu_layer!(stats, size(x), MIPVerify.interval_arithmetic)
    layer !== nothing && (layer.tightening_algorithm = "backward")
    bounds_start = time_ns()
    bounds =
        MIPVerify.consistent_relu_bounds.(x, reshape(p.lower, size(x)), reshape(p.upper, size(x)))
    lower, upper = first.(bounds), last.(bounds)
    bounds_time = MIPVerify.elapsed_seconds(bounds_start)
    constraints_start = time_ns()
    output = MIPVerify.relu.(x, lower, upper)
    phases = MIPVerify.get_relu_type.(lower, upper)
    MIPVerify.finish_relu_layer!(
        stats,
        layer,
        bounds_time,
        MIPVerify.elapsed_seconds(constraints_start),
        count(==(MIPVerify.zero_output), phases),
        count(==(MIPVerify.linear_in_input), phases),
        count(==(MIPVerify.constant_output), phases),
        count(==(MIPVerify.split), phases),
    )
    return output
end

function bounded_fallback_network(nn, hidden, algorithm = MIPVerify.target_lp)
    algorithm == MIPVerify.target_lp || return nn
    layers = copy(nn.layers)
    relus = findall(layer -> layer isa MIPVerify.ReLU, layers)
    length(relus) == length(hidden) || error("A bound pair is required for every ReLU layer")
    for (k, (i, bounds)) in enumerate(zip(relus, hidden))
        # Reuse these ranges where the model would otherwise use forward intervals.
        # Explicit LP/MIP tightening still runs on the original ReLU.
        override = layers[i].tightening_algorithm
        if isnothing(override) || (k == 1 && override == MIPVerify.interval_arithmetic)
            layers[i] = PreboundedReLU(bounds...)
        end
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

reference_labels(y::Integer) = (y,)
reference_labels(labels) = labels

# A rounded subtraction can cross the requested threshold by one ulp. For zero
# margin, comparing the endpoints directly also preserves provable exact ties.
gap_at_least(lower, upper, margin) =
    iszero(margin) ? lower >= upper : prevfloat(lower - upper) >= margin
gap_below(upper, lower, margin) = iszero(margin) ? upper < lower : nextfloat(upper - lower) < margin

# Verify the positive margin over a singleton box with outward rounding.
function confirms_witness(stages, z, y, j; margin = 0)
    lo = hi = z
    for (k, stage) in enumerate(stages)
        lo, hi = interval_stage(stage, lo, hi)
        if k < length(stages)
            lo, hi = max.(lo, 0), max.(hi, 0)
        end
    end
    reference = maximum(i -> hi[i], reference_labels(y))
    return iszero(margin) ? lo[j] > reference : gap_at_least(lo[j], reference, margin)
end

# Historical MNIST measurements used exact radius 1/10 and both stored and byte/255
# centers. Public witness validation can set decimal_radius=false to match its
# Float64 perturbation family exactly. These inward endpoints constrain witnesses only.
function compute_witness_box(x; radius::Float64 = 0.1, decimal_radius::Bool = true)
    Q = Rational{BigInt}
    isfinite(radius) && radius >= 0 || throw(ArgumentError("radius must be finite and nonnegative"))
    exact_radius = decimal_radius && radius == 0.1 ? Q(1//10) : Q(radius)
    pixels = vec(x)
    bytes = round.(Int, 255 .* pixels)
    byte_centers = all(pixels .== bytes ./ 255)
    lower, upper = zeros(length(pixels)), zeros(length(pixels))
    for i in eachindex(pixels)
        center = Q(pixels[i])
        l, u = max(Q(0), center - exact_radius), min(Q(1), center + exact_radius)
        if byte_centers && decimal_radius && radius == 0.1
            original = Q(bytes[i]) / 255
            l, u = max(l, original - exact_radius), min(u, original + exact_radius)
        end
        lower[i], upper[i] = Float64(l), Float64(u)
        Q(lower[i]) < l && (lower[i] = nextfloat(lower[i]))
        Q(upper[i]) > u && (upper[i] = prevfloat(upper[i]))
    end
    return lower, upper
end

# MNIST reuses the same 256 centers. Compute their exact inward rounding once.
const BYTE_WITNESS_LOWER, BYTE_WITNESS_UPPER = compute_witness_box(collect(0:255) ./ 255)

function witness_box(x; radius::Float64 = 0.1, decimal_radius::Bool = true)
    pixels = vec(x)
    bytes = round.(Int, 255 .* pixels)
    if decimal_radius &&
       radius == 0.1 &&
       all(i -> 0 <= bytes[i] <= 255 && pixels[i] == bytes[i] / 255, eachindex(pixels))
        indices = bytes .+ 1
        return BYTE_WITNESS_LOWER[indices], BYTE_WITNESS_UPPER[indices]
    end
    return compute_witness_box(x; radius, decimal_radius)
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
function check_witness(
    stages,
    z,
    y,
    lower,
    upper;
    targets = setdiff(eachindex(stages[end].bias), y),
    margin = 0,
)
    length(z) == length(lower) || return "invalid"
    all(isfinite, z) && all(lower .<= z .<= upper) || return "invalid"
    lo = hi = z
    for (k, stage) in enumerate(stages)
        lo, hi = interval_stage(stage, lo, hi)
        if k < length(stages)
            lo, hi = max.(lo, 0), max.(hi, 0)
        end
    end
    refs = reference_labels(y)
    reference_upper = maximum(i -> hi[i], refs)
    any(gap_at_least(lo[j], reference_upper, margin) for j in targets) && return "outward_interval"
    reference_lower = maximum(i -> lo[i], refs)
    all(gap_below(hi[j], reference_lower, margin) for j in targets) && return "invalid"
    scores = exact_scores(stages, z)
    threshold = maximum(i -> scores[i], refs) + Rational{BigInt}(margin)
    return any(scores[j] >= threshold for j in targets) ? "exact_rational" : "invalid"
end

# A solver point can miss the exact box or output constraint by its feasibility
# tolerance. Clip into the exact box, then try nearby projected gradient steps.
# Every returned point is independently checked; search failure proves nothing.
function repair_witness(
    stages,
    x,
    proposed,
    y;
    radius::Float64 = 0.1,
    decimal_radius::Bool = true,
    targets = setdiff(eachindex(stages[end].bias), y),
    margin = 0,
)
    proposed === nothing && return nothing
    length(proposed) == length(x) && all(isfinite, proposed) || return nothing
    lower, upper = witness_box(x; radius, decimal_radius)
    z = clamp.(vec(proposed), lower, upper)
    proof = check_witness(stages, z, y, lower, upper; targets, margin)
    proof != "invalid" && return (witness = reshape(z, size(x)), proof = proof)
    for step in (1e-9, 1e-7, 1e-5, 1e-3, 0.02)
        act = forward(stages, z)
        target = targets[argmax(act[end][targets])]
        gradient = zeros(length(act[end]))
        reference = argmax(i -> act[end][i], reference_labels(y))
        gradient[target], gradient[reference] = 1, -1
        for k in reverse(eachindex(stages))
            gradient = stages[k].weights * gradient
            k > 1 && (gradient[act[k] .<= 0] .= 0)
        end
        trial = clamp.(z + step * sign.(gradient), lower, upper)
        proof = check_witness(stages, trial, y, lower, upper; targets, margin)
        proof != "invalid" && return (witness = reshape(trial, size(x)), proof = proof)
        current = act[end][target] - act[end][reference]
        scores = last(forward(stages, trial))
        if maximum(scores[targets]) - maximum(i -> scores[i], reference_labels(y)) > current
            z = trial
        end
    end
    return nothing
end

function find_box_witness(
    stages,
    x,
    y,
    bounds;
    radius::Float64 = 0.1,
    decimal_radius::Bool = true,
    margin = 0,
)
    # Retain the historical starts at radius .1; other radii use exact inward endpoints.
    lo, hi = if decimal_radius && radius == 0.1
        max.(0.0, vec(x) .- radius .+ 4eps()), min.(1.0, vec(x) .+ radius .- 4eps())
    else
        witness_box(x; radius, decimal_radius)
    end
    rng = MersenneTwister(1)
    for j in sortperm(bounds, rev = true)
        bounds[j] < margin && continue
        for start in 1:4
            z = start == 1 ? copy(vec(x)) : lo + (hi-lo) .* rand(rng, length(lo))
            for _ in 1:32
                act = forward(stages, z)
                reference = argmax(i -> act[end][i], reference_labels(y))
                if act[end][j] - act[end][reference] >= margin &&
                   confirms_witness(stages, z, y, j; margin)
                    return reshape(z, size(x))
                end
                g = zeros(length(bounds))
                g[j], g[reference] = 1, -1
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

# Check the maximizing corner of each open rival before running PGD.
function search_checked_witness(
    stages,
    x,
    y,
    bounds,
    corners;
    radius::Float64 = 0.1,
    decimal_radius::Bool = true,
    targets = setdiff(eachindex(bounds), y),
    margin = 0,
)
    lower, upper = witness_box(x; radius, decimal_radius)
    for target in sortperm(bounds; rev = true)
        bounds[target] < margin && break
        point = ifelse.(corners[target], upper, lower)
        proof = check_witness(stages, point, y, lower, upper; targets, margin)
        if proof != "invalid"
            return (
                witness = reshape(point, size(x)),
                proof = proof,
                source = "corner",
                repaired = false,
            )
        end
    end
    proposed = find_box_witness(stages, x, y, bounds; radius, decimal_radius, margin)
    checked = repair_witness(stages, x, proposed, y; radius, decimal_radius, targets, margin)
    checked === nothing && return nothing
    return (; checked..., source = "ascent", repaired = vec(checked.witness) != vec(proposed))
end


"""
    screen(nn, input, non_targets, targets; radius=0.1, margin=0)

Bound each requested target score minus the highest non-target score.
Return `nothing` when the network or numeric representation is unsupported. Supported
networks alternate finite Float32/Float64 Conv2d/Linear and ReLU layers, optionally flattening
once before an affine layer. Pixels must be Float64 values in [0, 1].

The result retains the hidden bounds and corner choices for witness search or a later
solver model. Only requested targets appear in `bounds`; other entries are `-Inf`.
"""
function screen(nn, input, non_targets, targets; radius = 0.1, margin = 0)
    nn isa MIPVerify.Sequential && input isa Array{Float64} && radius isa Float64 || return nothing
    isfinite(radius) && radius >= 0 && all(v -> isfinite(v) && 0 <= v <= 1, input) || return nothing
    stages = try
        compile_stages(nn, size(input))
    catch error
        error isa Union{ArgumentError,DimensionMismatch,BoundsError,AssertionError} || rethrow()
        return nothing
    end
    count = length(stages[end].bias)
    refs = reference_labels(non_targets)
    isempty(refs) && return nothing
    all(i -> i isa Integer && 1 <= i <= count, refs) || return nothing
    selected = collect(targets)
    all(j -> j isa Integer && 1 <= j <= count && j ∉ refs, selected) || return nothing
    selected = unique(Int.(selected))
    isempty(selected) && return nothing
    hidden, input_lower, input_upper = hidden_bounds(stages, input; radius)
    all(pair -> all(isfinite, pair[1]) && all(isfinite, pair[2]), hidden) || return nothing
    corners = Dict{Int,BitVector}()
    bounds = fill(-Inf, count)
    for target in selected
        # One impossible comparison excludes the target. Otherwise keep the
        # maximizing corner of its tightest pairwise upper bound.
        upper = Inf
        for reference in refs
            plane, intercept = margin_plane(stages, hidden, reference, target)
            bound = upper_at_box(plane, intercept, input_lower, input_upper)
            isfinite(bound) || return nothing
            if bound < upper
                upper = bound
                upper < margin && break
                corners[target] = plane .> 0
            end
        end
        bounds[target] = upper
    end
    return (;
        stages,
        hidden,
        input_lower,
        input_upper,
        bounds,
        corners,
        non_targets = refs,
        margin,
        targets = selected,
        radius,
    )
end

search_checked_witness(result::NamedTuple, input) = search_checked_witness(
    result.stages,
    input,
    result.non_targets,
    result.bounds,
    result.corners;
    radius = result.radius,
    targets = result.targets,
    margin = result.margin,
)

function repair_witness(result::NamedTuple, input, proposed)
    # Keep already checked historical candidates unchanged. Only a failed repair
    # needs the public API's exact Float64 perturbation-box endpoints.
    checked = repair_witness(
        result.stages,
        input,
        proposed,
        result.non_targets;
        radius = result.radius,
        targets = result.targets,
        margin = result.margin,
    )
    checked !== nothing && return checked
    return repair_witness(
        result.stages,
        input,
        proposed,
        result.non_targets;
        radius = result.radius,
        decimal_radius = false,
        targets = result.targets,
        margin = result.margin,
    )
end

end # module BackwardBounds
