# Independent full-domain check adapted from the archived 25-Nov exact-rational checker.
# Run with the measured checkout's benchmarks project.
# Arguments: network name, mode, raw-run directory, audit-output directory.
using LinearAlgebra
BLAS.set_num_threads(1)
using CSV, MIPVerify, SparseArrays
using MIPVerify.BackwardBounds: compile_stages, hidden_bounds, margin_upper

const Exact = Rational{BigInt}
struct ExactLayer
    W::SparseMatrixCSC{Exact,Int}
    b::Vector{Exact}
end
function boxsum(c, b, lo, hi)
    for i in eachindex(c)
        iszero(c[i]) && continue
        b+=c[i]*(c[i]>0 ? hi[i] : lo[i])
    end
    return b
end
function affineinterval(layer, lo, hi, j)
    l=u=layer.b[j]
    for k in nzrange(layer.W, j)
        w=nonzeros(layer.W)[k]
        i=rowvals(layer.W)[k]
        l+=w*(w>0 ? lo[i] : hi[i])
        u+=w*(w>0 ? hi[i] : lo[i])
    end
    return l, u
end
function exact_upper(layers, k, c, hidden, box)
    b=zero(Exact)
    for s in k:-1:1
        W=layers[s].W
        prev=zeros(Exact, size(W, 1))
        for j in eachindex(c)
            a=c[j]
            iszero(a) && continue
            b+=a*layers[s].b[j]
            for z in nzrange(W, j)
                prev[rowvals(W)[z]]+=a*nonzeros(W)[z]
            end
        end
        if s>1
            lo, hi, frac=hidden[s-1]
            for i in eachindex(prev)
                a, l, u=prev[i], lo[i], hi[i]
                if iszero(a) || l>=0
                    continue
                elseif u<=0
                    prev[i]=0
                elseif a>0
                    # Any dyadic slope in [0,a] works. Independently enclose both
                    # endpoints in exact arithmetic (the chord dominates a convex ReLU).
                    slope=a*frac[i]
                    b+=max(-slope*l, (a-slope)*u)
                    prev[i]=slope
                else
                    prev[i]=u < -l ? 0 : a
                end
            end
        end
        c=prev
    end
    return boxsum(c, b, box...)
end

function audit_image(stages, dataset, image, require_certificate)
    input = MIPVerify.get_image(dataset, image)
    label = MIPVerify.get_label(dataset, image) + 1
    floats, lower, upper_box = hidden_bounds(stages, input)
    box = (Exact.(lower), Exact.(upper_box))
    # The public perturbation family takes a binary Float64 radius and stored Float64 centers.
    center = Exact.(vec(input))
    radius = Exact(0.1)
    @assert all(box[1] .<= max.(0, center .- radius))
    @assert all(box[2] .>= min.(1, center .+ radius))
    layers = [ExactLayer(Exact.(stage.weights), Exact.(stage.bias)) for stage in stages]
    hidden = Tuple{Vector{Exact},Vector{Exact},Vector{Exact}}[]
    current_lower, current_upper = box
    intervals = 0
    for (k, (float_lower, float_upper)) in enumerate(floats)
        lo, hi = Exact.(float_lower), Exact.(float_upper)
        for unit in eachindex(lo)
            exact_lo, exact_hi = affineinterval(layers[k], current_lower, current_upper, unit)
            if exact_hi > hi[unit]
                objective = zeros(Exact, length(lo))
                objective[unit] = 1
                exact_hi = min(exact_hi, exact_upper(layers, k, objective, hidden, box))
            end
            if exact_lo < lo[unit]
                objective = zeros(Exact, length(lo))
                objective[unit] = -1
                exact_lo = max(exact_lo, -exact_upper(layers, k, objective, hidden, box))
            end
            @assert exact_lo >= lo[unit] && exact_hi <= hi[unit] (image, k, unit)
            intervals += 1
        end
        fractions = [
            a < 0 < b ? Exact(clamp(Float64(b / (b - a)), 0.0, 1.0)) : zero(Exact) for
            (a, b) in zip(lo, hi)
        ]
        push!(hidden, (lo, hi, fractions))
        current_lower, current_upper = max.(lo, 0), max.(hi, 0)
    end
    negative = 0
    for target in setdiff(eachindex(layers[end].b), label)
        objective = zeros(Exact, length(layers[end].b))
        objective[label], objective[target] = -1, 1
        exact_bound = exact_upper(layers, length(layers), objective, hidden, box)
        floating_bound = margin_upper(stages, floats, lower, upper_box, label, target)
        @assert exact_bound <= Exact(floating_bound) (
            image,
            target,
            Float64(exact_bound),
            floating_bound,
        )
        negative += floating_bound < 0
    end
    require_certificate && @assert negative == length(layers[end].b) - 1
    return (;
        sample_index = image,
        hidden_intervals_checked = intervals,
        target_bounds_checked = length(layers[end].b) - 1,
        negative_target_bounds = negative,
        full_domain_enclosed = true,
    )
end

function main(network_name, mode, source, destination)
    mode in ("fallback", "certificates", "certificates-first-10", "all") ||
        error("Unknown selection mode")
    mkpath(destination)
    dataset = MIPVerify.read_datasets("MNIST").test
    network = MIPVerify.get_example_network_params("MNIST.$(network_name)_linf0.1_authors")
    stages = compile_stages(network, size(MIPVerify.get_image(dataset, 1)))
    selected = Dict{Int,Bool}()
    for name in sort(
            filter(
                name -> occursin(r"^candidate-[12]-[0-9]+\.csv(?:\.gz)?$", name),
                readdir(source),
            ),
        ),
        row in CSV.File(joinpath(source, name); strict = true)

        route = string(row.source)
        wanted =
            (mode in ("fallback", "all") && route == "MIP") ||
            (mode in ("certificates", "certificates-first-10", "all") && route == "backward_bound")
        wanted && (selected[row.sample_index] = route == "backward_bound")
    end
    isempty(selected) && error("No candidate rows match the selected mode")
    ordered = sort(collect(selected))
    if mode == "certificates-first-10"
        length(ordered) >= 10 || error("At least ten direct certificates are required")
        ordered = first(ordered, 10)
    end
    rows = NamedTuple[]
    for (image, certificate) in ordered
        checked = audit_image(stages, dataset, image, certificate)
        push!(
            rows,
            (; network = network_name, source = certificate ? "backward_bound" : "MIP", checked...),
        )
        CSV.write(joinpath(destination, "full-domain-$(mode).csv"), rows)
        println(
            network_name,
            " image=",
            image,
            " intervals=",
            checked.hidden_intervals_checked,
            " targets=",
            checked.target_bounds_checked,
        )
        flush(stdout)
    end
end

main(ARGS...)
