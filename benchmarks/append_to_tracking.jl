include(joinpath(@__DIR__, "BenchmarkHelpers.jl"))
using .BenchmarkHelpers: append_tracking_csv!, parse_args

function main()
    args = parse_args(ARGS)

    metrics_csv = args["metrics-csv"]
    tracking_csv = args["tracking-csv"]
    dependency_versions_csv = args["dependency-versions-csv"]
    date = args["date"]
    commit_sha = args["commit-sha"]
    run_id = args["run-id"]

    append_tracking_csv!(
        metrics_csv = metrics_csv,
        tracking_csv = tracking_csv,
        dependency_versions_csv = dependency_versions_csv,
        date = date,
        commit_sha = commit_sha,
        run_id = run_id,
    )

    println("Appended tracking row for $(date) ($(run_id), $(commit_sha[1:7]))")
end

main()
