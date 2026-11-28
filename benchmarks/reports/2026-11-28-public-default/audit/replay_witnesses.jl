# Run with the measured checkout's benchmarks project.
# Arguments: network name (WK17a/RSL18a), raw-run directory, audit-output directory.
# Replays stored coordinates directly on the native network using exact rational arithmetic.
using LinearAlgebra
BLAS.set_num_threads(1)
using CSV, MIPVerify, SHA
using MIPVerify: Conv2d, Linear, ReLU, Flatten, Sequential

const Exact = Rational{BigInt}
exact_layer(layer::Conv2d) =
    Conv2d(Exact.(layer.filter), Exact.(layer.bias), layer.stride, layer.padding)
exact_layer(layer::Linear) = Linear(Exact.(layer.matrix), Exact.(layer.bias))
exact_layer(layer::Union{ReLU,Flatten}) = layer

function replay(network_name, source, destination)
    mkpath(destination)
    network = MIPVerify.get_example_network_params("MNIST.$(network_name)_linf0.1_authors")
    exact_network = Sequential(exact_layer.(network.layers), "exact $(network_name)")
    dataset = MIPVerify.read_datasets("MNIST").test
    cached = Dict()
    audited = NamedTuple[]
    route_counts = Dict{Tuple{String,String,String,String},Int}()
    observations = Dict{String,Int}()
    seen_observations = Set{Tuple{String,Int,Int}}()
    coordinates = Dict{String,Set{Tuple{Int,String}}}()
    files = sort(
        filter(
            name -> occursin(r"^(baseline|candidate)-[12]-[0-9]+\.csv(?:\.gz)?$", name),
            readdir(source),
        ),
    )
    isempty(files) && error("No benchmark CSV files found")
    for filename in files
        method, pass, lane =
            match(r"^(baseline|candidate)-([12])-([0-9]+)\.csv(?:\.gz)?$", filename).captures
        for row in CSV.File(joinpath(source, filename); strict = true)
            observation = (String(method), parse(Int, pass), Int(row.sample_index))
            observation in seen_observations && error("Duplicate observation: $(observation)")
            push!(seen_observations, observation)
            observations[method] = get(observations, method, 0) + 1
            key = (method, pass, string(row.source), string(row.semantic_outcome))
            route_counts[key] = get(route_counts, key, 0) + 1
            row.witness_available || continue
            ismissing(row.perturbed_input_value) &&
                error("Available witness has no coordinates: $(filename), $(row.sample_index)")
            coordinate_key = (row.sample_index, string(row.perturbed_input_value))
            push!(get!(coordinates, method, Set{Tuple{Int,String}}()), coordinate_key)
            checked = get!(cached, coordinate_key) do
                center = MIPVerify.get_image(dataset, row.sample_index)
                target = MIPVerify.get_label(dataset, row.sample_index) + 1
                point = reshape(
                    Exact.(parse.(Float64, split(row.perturbed_input_value, ';'))),
                    size(center),
                )
                byte_center = round.(Int, 255 .* center) .// 255
                range_excess = max(zero(Exact), -minimum(point), maximum(point) - 1)
                binary_excess =
                    max(zero(Exact), maximum(abs.(point .- Exact.(center))) - Exact(0.1))
                stored_excess = max(zero(Exact), maximum(abs.(point .- Exact.(center))) - 1//10)
                byte_excess = max(zero(Exact), maximum(abs.(point .- byte_center)) - 1//10)
                scores = exact_network(point)
                margin = maximum(scores[setdiff(eachindex(scores), target)]) - scores[target]
                (;
                    range_valid = iszero(range_excess),
                    stored_binary_budget_valid = iszero(binary_excess),
                    stored_decimal_budget_valid = iszero(stored_excess),
                    byte_budget_valid = iszero(byte_excess),
                    score_valid = margin >= 0,
                    maximum_binary_budget_excess = Float64(max(range_excess, binary_excess)),
                    maximum_decimal_budget_excess = Float64(
                        max(range_excess, stored_excess, byte_excess),
                    ),
                    score_margin = Float64(margin),
                    exact_score_margin = string(margin),
                )
            end
            valid = checked.range_valid && checked.stored_binary_budget_valid && checked.score_valid
            push!(
                audited,
                (;
                    network = network_name,
                    method,
                    observation = parse(Int, pass),
                    lane = parse(Int, lane),
                    sample_index = row.sample_index,
                    source = string(row.source),
                    coordinates_sha256 = bytes2hex(sha256(string(row.perturbed_input_value))),
                    public_verified = row.witness_verified,
                    checked...,
                    exact_witness_valid = valid,
                ),
            )
            CSV.write(joinpath(destination, "witness-replay.csv"), audited)
            println(
                network_name,
                " ",
                method,
                " observation=",
                pass,
                " image=",
                row.sample_index,
                " valid=",
                valid,
            )
            flush(stdout)
        end
    end
    routes = [
        (;
            network = network_name,
            method,
            observation = parse(Int, pass),
            source = route,
            semantic_outcome = outcome,
            count,
        ) for ((method, pass, route, outcome), count) in sort(collect(route_counts))
    ]
    CSV.write(joinpath(destination, "route-counts.csv"), routes)
    summary = [
        (;
            network = network_name,
            method,
            observations = observation_count,
            available_witnesses = count(row -> row.method == method, audited),
            public_verified_witnesses = count(
                row -> row.method == method && row.public_verified,
                audited,
            ),
            exact_valid_witnesses = count(
                row -> row.method == method && row.exact_witness_valid,
                audited,
            ),
            exact_invalid_witnesses = count(
                row -> row.method == method && !row.exact_witness_valid,
                audited,
            ),
            binary_budget_failures = count(
                row ->
                    row.method == method && (!row.range_valid || !row.stored_binary_budget_valid),
                audited,
            ),
            exact_score_failures = count(row -> row.method == method && !row.score_valid, audited),
            decimal_budget_failures = count(
                row -> row.method == method && !row.stored_decimal_budget_valid,
                audited,
            ),
            byte_center_budget_failures = count(
                row -> row.method == method && !row.byte_budget_valid,
                audited,
            ),
            original_input_witnesses = count(
                row -> row.method == method && row.source == "original_input",
                audited,
            ),
            unique_coordinates_replayed = length(coordinates[method]),
        ) for (method, observation_count) in sort(collect(observations))
    ]
    CSV.write(joinpath(destination, "witness-summary.csv"), summary)
end

replay(ARGS...)
