# An activation lies between two affine expressions in the same nonnegative
# variables (pixels and one [0,1] error per crossing ReLU). The two expressions
# enclose roundoff as we propagate a single shared affine form.
struct ForwardForm
    ids::Vector{Int}
    lower::Vector{Float64}
    upper::Vector{Float64}
    lower_bias::Float64
    upper_bias::Float64
end

function forward_sum(stage, j, inputs, nvars)
    W = stage.weights
    lower, upper = zeros(nvars), zeros(nvars)
    lb = ub = stage.bias[j]
    for k in nzrange(W, j)
        w = nonzeros(W)[k]
        form = inputs[rowvals(W)[k]]
        if w >= 0
            low, high = form.lower, form.upper
            l, u = form.lower_bias, form.upper_bias
        else
            low, high = form.upper, form.lower
            l, u = form.upper_bias, form.lower_bias
        end
        lb, ub = prevfloat(fma(w, l, lb)), nextfloat(fma(w, u, ub))
        for (t, id) in enumerate(form.ids)
            lower[id] = prevfloat(fma(w, low[t], lower[id]))
            upper[id] = nextfloat(fma(w, high[t], upper[id]))
        end
    end
    ids = findall(i -> lower[i] != 0 || upper[i] != 0, eachindex(lower))
    return ForwardForm(ids, lower[ids], upper[ids], lb, ub)
end

function forward_range(form, pixel_lo, pixel_hi)
    l, u = form.lower_bias, form.upper_bias
    for (t, id) in enumerate(form.ids)
        x, X = id <= length(pixel_lo) ? (pixel_lo[id], pixel_hi[id]) : (0.0, 1.0)
        a, b = form.lower[t], form.upper[t]
        l = prevfloat(fma(a, a >= 0 ? x : X, l))
        u = nextfloat(fma(b, b >= 0 ? X : x, u))
    end
    return l, u
end

function forward_relus(stages, lo, hi)
    nvars = length(lo)
    inputs = [ForwardForm([i], [1.0], [1.0], 0, 0) for i in eachindex(lo)]
    for stage in stages[1:(end-1)]
        input_vars = nvars
        outputs = ForwardForm[]
        for j in eachindex(stage.bias)
            # Each input still uses only variables introduced by earlier layers.
            form = forward_sum(stage, j, inputs, input_vars)
            l, u = forward_range(form, lo, hi)
            if u <= 0
                form = ForwardForm(Int[], Float64[], Float64[], 0, 0)
            elseif l < 0
                s = clamp(u/(u-l), 0.0, 1.0)
                # ReLU(z) = s*z + width*e. The endpoint errors bound width
                # even when the secant slope rounds; e is shared at every fanout.
                width = max(nextfloat(-s*l), nextfloat(nextfloat(1-s)*u))
                nvars += 1
                form = ForwardForm(
                    [form.ids; nvars],
                    [prevfloat.(s .* form.lower); width],
                    [nextfloat.(s .* form.upper); width],
                    prevfloat(s*form.lower_bias),
                    nextfloat(s*form.upper_bias),
                )
            end
            push!(outputs, form)
        end
        inputs = outputs
    end
    return inputs, nvars
end

function forward_gap_upper(rival, truth, pixel_lo, pixel_hi, nvars, directions = nothing)
    coeff = zeros(nvars)
    coeff[rival.ids] .= rival.upper
    for (k, id) in enumerate(truth.ids)
        coeff[id] = nextfloat(coeff[id] - truth.lower[k])
    end
    upper = nextfloat(rival.upper_bias - truth.lower_bias)
    for (i, a) in enumerate(coeff)
        iszero(a) && continue
        endpoint = if i <= length(pixel_lo)
            a > 0 ? pixel_hi[i] : pixel_lo[i]
        else
            a > 0 ? 1.0 : 0.0
        end
        upper = nextfloat(fma(a, endpoint, upper))
    end
    if directions !== nothing && upper >= 0
        # Error variables are independent in the relaxation. Only original pixels
        # define a candidate input that we can check on the actual network.
        directions .= view(coeff, eachindex(pixel_lo)) .> 0
    end
    return upper
end

function tighten_forward_margins!(bounds, stages, y, lo, hi, corners = nothing)
    open = findall(>=(0), bounds)
    isempty(open) && return bounds
    inputs, nvars = forward_relus(stages, lo, hi)
    truth = forward_sum(stages[end], y, inputs, nvars)
    for j in open
        rival = forward_sum(stages[end], j, inputs, nvars)
        directions = isnothing(corners) ? nothing : falses(length(lo))
        bounds[j] = min(bounds[j], forward_gap_upper(rival, truth, lo, hi, nvars, directions))
        if !isnothing(corners) && bounds[j] >= 0
            corners[j] = directions
        end
    end
    return bounds
end
