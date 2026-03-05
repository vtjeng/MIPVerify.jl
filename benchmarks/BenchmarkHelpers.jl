module BenchmarkHelpers
export parse_args,
    parse_sample_spec,
    safe_sum,
    is_infeasible_status,
    classify_semantic_outcome,
    maybe_parse_norm_order,
    regression_ratio,
    percent

function parse_args(args::Vector{String})::Dict{String,String}
    parsed = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if i == length(args) || startswith(args[i+1], "--")
                parsed[key] = "true"
                i += 1
            else
                parsed[key] = args[i+1]
                i += 2
            end
        else
            i += 1
        end
    end
    return parsed
end

function parse_sample_spec(spec::String)::Vector{Int}
    if occursin(":", spec)
        parts = split(spec, ":")
        @assert length(parts) in (2, 3) "Sample spec must be start:stop or start:step:stop."
        if length(parts) == 2
            start_idx = parse(Int, parts[1])
            stop_idx = parse(Int, parts[2])
            return collect(start_idx:stop_idx)
        else
            start_idx = parse(Int, parts[1])
            step = parse(Int, parts[2])
            stop_idx = parse(Int, parts[3])
            return collect(start_idx:step:stop_idx)
        end
    end
    return parse.(Int, split(spec, ","))
end

function safe_sum(xs)::Float64
    return sum(filter(isfinite, xs))
end

function is_infeasible_status(status::String)::Bool
    return status == "INFEASIBLE" || status == "INFEASIBLE_OR_UNBOUNDED"
end

function classify_semantic_outcome(status::String, objective_value::Union{Missing,Float64})::String
    if is_infeasible_status(status)
        return "certified_no_adversarial_example"
    elseif !ismissing(objective_value)
        return "adversarial_example_found_or_best_known"
    elseif status == "TIME_LIMIT"
        return "time_limit_unresolved"
    else
        return "no_primal_solution_other"
    end
end

function maybe_parse_norm_order(raw::String)
    lowered = lowercase(strip(raw))
    if lowered == "inf"
        return Inf
    end
    return parse(Float64, raw)
end

function regression_ratio(baseline::Float64, candidate::Float64)::Float64
    if baseline == 0
        return candidate == 0 ? 0.0 : Inf
    end
    return (candidate - baseline) / baseline
end

function percent(v::Float64)::String
    if isfinite(v)
        return string(round(v * 100; digits = 3), "%")
    end
    return "Inf%"
end

end
