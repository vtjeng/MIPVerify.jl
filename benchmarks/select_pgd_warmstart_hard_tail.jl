using CSV
using DataFrames

include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers

# Frozen before the cold-only screen. Sample 212 was historically hard but is a known MIP attack;
# the remaining samples are the predeclared reserve order from the historical LP-feasibility tail.
const HARD_TAIL_SCREEN_POOL = [
    212,
    325,
    34,
    405,
    415,
    392,
    472,
    498,
    168,
    448,
    79,
    265,
    421,
    29,
    98,
    452,
    37,
    174,
    67,
    55,
    50,
    270,
    85,
    473,
    65,
    320,
]

function eligible_screen_cohort(candidates)
    indexed = Dict(Int(row.sample_index) => row for row in eachrow(candidates))
    missing_samples = filter(sample -> !haskey(indexed, sample), HARD_TAIL_SCREEN_POOL)
    isempty(missing_samples) ||
        error("candidate cache is missing hard-tail samples: $(join(missing_samples, ","))")
    rows = NamedTuple[]
    for (rank, sample_index) in enumerate(HARD_TAIL_SCREEN_POOL)
        candidate = indexed[sample_index]
        string(candidate.status) == "near_miss" || continue
        Bool(candidate.verified_attack) && continue
        push!(
            rows,
            (
                sample_index = sample_index,
                stratum = "hard_tail_screen",
                selection_rank = rank,
                true_index = Int(candidate.true_index),
                pgd_margin = Float64(candidate.margin),
                selection_basis = "predeclared historical reserve order",
            ),
        )
    end
    return DataFrame(rows)
end

function select_hard_tail(screen_cohort, cold_results, count)
    cold = filter(row -> string(row.treatment) == "cold", cold_results)
    nrow(cold) == nrow(screen_cohort) || error("cold screen does not cover the eligible pool")
    joined = innerjoin(screen_cohort, cold; on = :sample_index, makeunique = true)
    # A verified MIP attack completes verification and is not a certification-hard follow-up case.
    filter!(row -> string(row.outcome) != "verified_attack", joined)
    joined.timeout_rank = Int.(joined.outcome .== "time_limit_unresolved")
    sort!(
        joined,
        [:timeout_rank, :simplex_iterations, :main_solve_wall_time_seconds];
        rev = [true, true, true],
    )
    nrow(joined) >= count || error("fewer than $count non-attack hard-tail cases")
    selected = first(joined, count)
    return select(
        selected,
        :sample_index,
        :stratum => ByRow(_ -> "hard_tail") => :stratum,
        :selection_rank,
        :true_index,
        :pgd_margin,
        :outcome => :cold_screen_outcome,
        :main_solve_wall_time_seconds => :cold_screen_seconds,
        :simplex_iterations => :cold_screen_simplex_iterations,
        :node_count => :cold_screen_node_count,
        :objective_bound => :cold_screen_objective_bound,
    )
end

function main()
    args = parse_args(ARGS)
    candidate_path = get(args, "candidates", "")
    output_path = get(args, "out", "")
    isempty(candidate_path) && error("--candidates is required")
    isempty(output_path) && error("--out is required")
    candidates = CSV.read(candidate_path, DataFrame)
    screen_cohort = eligible_screen_cohort(candidates)
    output = if haskey(args, "cold-results")
        count = parse(Int, get(args, "count", "4"))
        cold_results = CSV.read(args["cold-results"], DataFrame)
        select_hard_tail(screen_cohort, cold_results, count)
    else
        screen_cohort
    end
    mkpath(dirname(abspath(output_path)))
    CSV.write(output_path, output)
    show(stdout, "text/plain", output)
    println()
end

main()
