using CSV
using DataFrames

include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers

# These samples were ordered before the warm-start treatment runs. The hard pool comes from the
# slowest historical LP-feasibility cases after excluding sample 63 (known PGD attack) and sample
# 212 (known MIP attack). The control pool was drawn from the class-balanced historical cohort.
const HARD_CANDIDATE_POOL = [19, 46, 246, 479, 359, 407, 444, 233, 404, 432, 194, 122]
const CONTROL_CANDIDATE_POOL = [313, 460, 4, 32, 428, 280, 36, 468]

function eligible_candidates(candidates, pool)
    indexed = Dict(Int(row.sample_index) => row for row in eachrow(candidates))
    missing_samples = filter(sample -> !haskey(indexed, sample), pool)
    isempty(missing_samples) ||
        error("candidate cache is missing predeclared samples: $(join(missing_samples, ","))")
    return [
        indexed[sample] for sample in pool if
        string(indexed[sample].status) == "near_miss" && !Bool(indexed[sample].verified_attack)
    ]
end

function select_controls(candidates, count)
    selected = DataFrameRow[]
    used_classes = Set{Int}()
    for candidate in candidates
        true_index = Int(candidate.true_index)
        if !(true_index in used_classes)
            push!(selected, candidate)
            push!(used_classes, true_index)
        end
        length(selected) == count && return selected
    end
    for candidate in candidates
        candidate in selected && continue
        push!(selected, candidate)
        length(selected) == count && return selected
    end
    return selected
end

function main()
    args = parse_args(ARGS)
    candidate_path = get(args, "candidates", "")
    output_path = get(args, "out", "")
    isempty(candidate_path) && error("--candidates is required")
    isempty(output_path) && error("--out is required")
    candidates = CSV.read(candidate_path, DataFrame)

    hard_eligible = eligible_candidates(candidates, HARD_CANDIDATE_POOL)
    control_eligible = eligible_candidates(candidates, CONTROL_CANDIDATE_POOL)
    length(hard_eligible) >= 8 || error("fewer than eight eligible hard candidates")
    length(control_eligible) >= 4 || error("fewer than four eligible control candidates")
    hard = first(hard_eligible, 8)
    controls = select_controls(control_eligible, 4)

    rows = NamedTuple[]
    for (rank, candidate) in enumerate(hard)
        push!(
            rows,
            (
                sample_index = Int(candidate.sample_index),
                stratum = "hard",
                selection_rank = rank,
                true_index = Int(candidate.true_index),
                pgd_margin = Float64(candidate.margin),
                selection_basis = "predeclared historical-hard order",
            ),
        )
    end
    for (rank, candidate) in enumerate(controls)
        push!(
            rows,
            (
                sample_index = Int(candidate.sample_index),
                stratum = "control",
                selection_rank = rank,
                true_index = Int(candidate.true_index),
                pgd_margin = Float64(candidate.margin),
                selection_basis = "predeclared control order with distinct classes first",
            ),
        )
    end
    cohort = DataFrame(rows)
    mkpath(dirname(abspath(output_path)))
    CSV.write(output_path, cohort)
    show(stdout, "text/plain", cohort)
    println()
end

main()
