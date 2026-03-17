module BenchmarkHelpers
using CSV
using DataFrames
using Pkg
using SHA

export parse_args,
    parse_sample_spec,
    safe_sum,
    is_infeasible_status,
    classify_semantic_outcome,
    maybe_parse_norm_order,
    regression_ratio,
    percent,
    append_tracking_csv!,
    active_manifest_path,
    collect_dependency_snapshot,
    dependency_change_summary,
    dependency_snapshot_hash,
    load_dependency_snapshot,
    write_dependency_snapshot

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

const EXCLUDED_DEPENDENCY_NAMES = Set(["MIPVerify"])
const TRACKING_COLUMNS = [
    :date,
    :run_id,
    :commit_sha,
    :julia_version,
    :dependency_snapshot_sha256,
    :dependency_change_summary,
    :wall_clock_seconds,
    :sum_total_time_seconds,
    :sum_solve_time_seconds,
    :median_solve_time_seconds,
    :p90_solve_time_seconds,
    :num_samples,
    :num_certified_no_adversarial_example,
    :num_adversarial_example_found_or_best_known,
    :num_time_limit_unresolved,
    :num_no_primal_solution_other,
]

function active_manifest_path()::String
    project_file = Base.active_project()
    @assert !isnothing(project_file) "No active project found for benchmark environment."
    manifest_path = joinpath(dirname(project_file), "Manifest.toml")
    @assert isfile(manifest_path) "Missing manifest file at $manifest_path"
    return manifest_path
end

function normalize_optional_string(value)::Union{Missing,String}
    if isnothing(value) || ismissing(value)
        return missing
    end
    normalized = strip(string(value))
    return isempty(normalized) ? missing : normalized
end

function normalize_required_string(value, field::Symbol)::String
    normalized = normalize_optional_string(value)
    @assert !ismissing(normalized) "Missing required dependency field $field"
    return normalized
end

function normalize_required_bool(value, field::Symbol)::Bool
    if value isa Bool
        return value
    end
    normalized = normalize_optional_string(value)
    @assert !ismissing(normalized) "Missing required dependency field $field"
    lowered = lowercase(normalized)
    @assert lowered == "true" || lowered == "false" "Invalid boolean value for $field: $value"
    return lowered == "true"
end

function snapshot_column(snapshot::DataFrame, column::Symbol, default)
    if column in propertynames(snapshot)
        return snapshot[!, column]
    end
    return fill(default, nrow(snapshot))
end

function normalize_dependency_snapshot(snapshot::DataFrame)::DataFrame
    num_rows = nrow(snapshot)
    names_col = snapshot_column(snapshot, :name, missing)
    uuids_col = snapshot_column(snapshot, :uuid, missing)
    versions_col = snapshot_column(snapshot, :version, missing)
    tree_hashes_col = snapshot_column(snapshot, :tree_hash, missing)
    source_kinds_col = snapshot_column(snapshot, :source_kind, missing)
    direct_dep_col = snapshot_column(snapshot, :is_direct_dep, missing)
    git_revisions_col = snapshot_column(snapshot, :git_revision, missing)

    normalized = DataFrame(
        name = [normalize_required_string(names_col[i], :name) for i in 1:num_rows],
        uuid = [normalize_required_string(uuids_col[i], :uuid) for i in 1:num_rows],
        version = [normalize_optional_string(versions_col[i]) for i in 1:num_rows],
        tree_hash = [normalize_optional_string(tree_hashes_col[i]) for i in 1:num_rows],
        source_kind = [
            normalize_required_string(source_kinds_col[i], :source_kind) for i in 1:num_rows
        ],
        is_direct_dep = [
            normalize_required_bool(direct_dep_col[i], :is_direct_dep) for i in 1:num_rows
        ],
        git_revision = [normalize_optional_string(git_revisions_col[i]) for i in 1:num_rows],
    )
    sort!(normalized, [:name, :uuid])
    return normalized
end

function dependency_source_kind(
    source,
    is_tracking_path::Bool,
    is_tracking_repo::Bool,
    is_tracking_registry::Bool,
)::String
    normalized_source = normalize_optional_string(source)
    if !ismissing(normalized_source)
        stdlib_source = replace(normalized_source, "\\" => "/")
        if occursin("/stdlib/", stdlib_source)
            return "stdlib"
        end
    end
    if is_tracking_path
        return "path"
    elseif is_tracking_repo
        return "repo"
    elseif is_tracking_registry
        return "registry"
    end
    return "unknown"
end

function collect_dependency_snapshot()::DataFrame
    rows = NamedTuple[]
    for (uuid, info) in Pkg.dependencies()
        push!(
            rows,
            (
                name = info.name,
                uuid = string(uuid),
                version = normalize_optional_string(info.version),
                tree_hash = normalize_optional_string(info.tree_hash),
                source_kind = dependency_source_kind(
                    info.source,
                    info.is_tracking_path,
                    info.is_tracking_repo,
                    info.is_tracking_registry,
                ),
                is_direct_dep = info.is_direct_dep,
                git_revision = normalize_optional_string(info.git_revision),
            ),
        )
    end
    return normalize_dependency_snapshot(DataFrame(rows))
end

function write_dependency_snapshot(path::String, snapshot::DataFrame)
    CSV.write(path, normalize_dependency_snapshot(snapshot))
end

function load_dependency_snapshot(path::String)::DataFrame
    return normalize_dependency_snapshot(CSV.read(path, DataFrame))
end

function snapshot_index(snapshot::DataFrame)::Dict{String,NamedTuple}
    index = Dict{String,NamedTuple}()
    for row in eachrow(normalize_dependency_snapshot(snapshot))
        if row.name in EXCLUDED_DEPENDENCY_NAMES
            continue
        end
        index[row.uuid] = (
            name = row.name,
            uuid = row.uuid,
            version = row.version,
            tree_hash = row.tree_hash,
            source_kind = row.source_kind,
            is_direct_dep = row.is_direct_dep,
            git_revision = row.git_revision,
        )
    end
    return index
end

function dependency_row_hash_string(row::NamedTuple)::String
    return join(
        [
            row.name,
            row.uuid,
            coalesce(row.version, ""),
            coalesce(row.tree_hash, ""),
            row.source_kind,
            row.is_direct_dep ? "true" : "false",
            coalesce(row.git_revision, ""),
        ],
        "|",
    )
end

function dependency_snapshot_hash(snapshot::DataFrame; julia_version::String = string(VERSION))
    lines = String["julia_version|$julia_version"]
    rows = sort(collect(values(snapshot_index(snapshot))); by = row -> (row.name, row.uuid))
    append!(lines, dependency_row_hash_string.(rows))
    return bytes2hex(SHA.sha256(join(lines, "\n")))
end

function dependency_row_changed(previous::NamedTuple, current::NamedTuple)::Bool
    return !isequal(previous.name, current.name) ||
           !isequal(previous.version, current.version) ||
           !isequal(previous.tree_hash, current.tree_hash) ||
           !isequal(previous.source_kind, current.source_kind) ||
           !isequal(previous.is_direct_dep, current.is_direct_dep) ||
           !isequal(previous.git_revision, current.git_revision)
end

function dependency_row_summary(row::NamedTuple)::String
    summary = coalesce(row.version, "version=missing")
    if !ismissing(row.tree_hash)
        summary = string(summary, "#", row.tree_hash)
    end
    if !ismissing(row.git_revision)
        summary = string(summary, "@", row.git_revision)
    end

    qualifiers = String[]
    if row.source_kind != "registry"
        push!(qualifiers, row.source_kind)
    end
    if row.is_direct_dep
        push!(qualifiers, "direct")
    end
    if !isempty(qualifiers)
        summary = string(summary, "[", join(qualifiers, ","), "]")
    end
    return summary
end

function dependency_change_summary(
    previous_snapshot::DataFrame,
    current_snapshot::DataFrame,
)::String
    previous_index = snapshot_index(previous_snapshot)
    current_index = snapshot_index(current_snapshot)

    package_keys = sort(
        collect(union(keys(previous_index), keys(current_index)));
        by = key ->
            haskey(current_index, key) ? current_index[key].name : previous_index[key].name,
    )

    changes = String[]
    for key in package_keys
        if !haskey(previous_index, key)
            current = current_index[key]
            push!(changes, string("+", current.name, " ", dependency_row_summary(current)))
            continue
        end
        if !haskey(current_index, key)
            previous = previous_index[key]
            push!(changes, string("-", previous.name, " ", dependency_row_summary(previous)))
            continue
        end

        previous = previous_index[key]
        current = current_index[key]
        if dependency_row_changed(previous, current)
            push!(
                changes,
                string(
                    current.name,
                    " ",
                    dependency_row_summary(previous),
                    " -> ",
                    dependency_row_summary(current),
                ),
            )
        end
    end
    return join(changes, "; ")
end

function previous_dependency_snapshot_path(
    tracking_csv::String,
    tracking::DataFrame,
)::Union{Nothing,String}
    if nrow(tracking) == 0
        return nothing
    end
    if !(:date in propertynames(tracking)) || !(:run_id in propertynames(tracking))
        return nothing
    end

    previous_row = tracking[end, :]
    date = normalize_optional_string(previous_row[:date])
    run_id = normalize_optional_string(previous_row[:run_id])
    if ismissing(date) || ismissing(run_id)
        return nothing
    end

    snapshot_path =
        joinpath(dirname(abspath(tracking_csv)), "runs", date, run_id, "dependency_versions.csv")
    return isfile(snapshot_path) ? snapshot_path : nothing
end

function current_dependency_change_summary(
    tracking::DataFrame,
    tracking_csv::String,
    current_snapshot::DataFrame,
    current_hash::String,
)::Union{Missing,String}
    if nrow(tracking) == 0 || !(:dependency_snapshot_sha256 in propertynames(tracking))
        return missing
    end

    previous_hash = normalize_optional_string(tracking[end, :dependency_snapshot_sha256])
    if ismissing(previous_hash)
        return missing
    end
    if previous_hash == current_hash
        return ""
    end

    previous_snapshot = previous_dependency_snapshot_path(tracking_csv, tracking)
    if isnothing(previous_snapshot)
        return missing
    end
    return dependency_change_summary(load_dependency_snapshot(previous_snapshot), current_snapshot)
end

function build_tracking_row(
    metrics::DataFrame,
    date::String,
    commit_sha::String,
    run_id::String,
    dependency_summary::Union{Missing,String},
)::DataFrame
    return DataFrame(
        date = [date],
        run_id = [run_id],
        commit_sha = [commit_sha],
        julia_version = [string(metrics[1, :julia_version])],
        dependency_snapshot_sha256 = [string(metrics[1, :dependency_snapshot_sha256])],
        dependency_change_summary = [dependency_summary],
        wall_clock_seconds = [metrics[1, :wall_clock_seconds]],
        sum_total_time_seconds = [metrics[1, :sum_total_time_seconds]],
        sum_solve_time_seconds = [metrics[1, :sum_solve_time_seconds]],
        median_solve_time_seconds = [metrics[1, :median_solve_time_seconds]],
        p90_solve_time_seconds = [metrics[1, :p90_solve_time_seconds]],
        num_samples = [metrics[1, :num_samples]],
        num_certified_no_adversarial_example = [metrics[1, :num_certified_no_adversarial_example]],
        num_adversarial_example_found_or_best_known = [
            metrics[1, :num_adversarial_example_found_or_best_known],
        ],
        num_time_limit_unresolved = [metrics[1, :num_time_limit_unresolved]],
        num_no_primal_solution_other = [metrics[1, :num_no_primal_solution_other]],
    )
end

function append_tracking_csv!(;
    metrics_csv::String,
    tracking_csv::String,
    dependency_versions_csv::String,
    date::String,
    commit_sha::String,
    run_id::String,
)
    metrics = CSV.read(metrics_csv, DataFrame)
    dependency_snapshot = load_dependency_snapshot(dependency_versions_csv)

    tracking = isfile(tracking_csv) ? CSV.read(tracking_csv, DataFrame) : DataFrame()
    current_hash = string(metrics[1, :dependency_snapshot_sha256])
    dependency_summary =
        current_dependency_change_summary(tracking, tracking_csv, dependency_snapshot, current_hash)
    row = build_tracking_row(metrics, date, commit_sha, run_id, dependency_summary)

    combined = nrow(tracking) == 0 ? row : vcat(tracking, row; cols = :union)
    ordered_columns = [column for column in TRACKING_COLUMNS if column in propertynames(combined)]
    extra_columns = [column for column in propertynames(combined) if !(column in ordered_columns)]
    select!(combined, vcat(ordered_columns, extra_columns))
    CSV.write(tracking_csv, combined)
    return row
end

end
