using CSV
using DataFrames
using Test

include(joinpath(@__DIR__, "..", "BenchmarkHelpers.jl"))
using .BenchmarkHelpers

function dependency_row(
    name::String,
    uuid::String;
    version = missing,
    tree_hash = missing,
    source_kind::String = "registry",
    is_direct_dep::Bool = false,
    git_revision = missing,
)
    return (
        name = name,
        uuid = uuid,
        version = version,
        tree_hash = tree_hash,
        source_kind = source_kind,
        is_direct_dep = is_direct_dep,
        git_revision = git_revision,
    )
end

function dependency_snapshot(rows...)
    return DataFrame(collect(rows))
end

function write_metrics_csv(path::String; dependency_snapshot_sha256::String, julia_version::String)
    metrics = DataFrame(
        wall_clock_seconds = [10.0],
        sum_total_time_seconds = [9.0],
        sum_solve_time_seconds = [4.0],
        median_solve_time_seconds = [2.0],
        p90_solve_time_seconds = [3.0],
        num_samples = [2],
        num_certified_no_adversarial_example = [1],
        num_adversarial_example_found_or_best_known = [1],
        num_time_limit_unresolved = [0],
        num_no_primal_solution_other = [0],
        julia_version = [julia_version],
        dependency_snapshot_sha256 = [dependency_snapshot_sha256],
    )
    CSV.write(path, metrics)
end

function write_legacy_tracking_csv(path::String)
    tracking = DataFrame(
        date = ["2026-03-08"],
        commit_sha = ["abc1234"],
        wall_clock_seconds = [11.0],
        sum_total_time_seconds = [10.0],
        sum_solve_time_seconds = [5.0],
        median_solve_time_seconds = [2.5],
        p90_solve_time_seconds = [3.0],
        num_samples = [2],
        num_certified_no_adversarial_example = [1],
        num_adversarial_example_found_or_best_known = [1],
        num_time_limit_unresolved = [0],
        num_no_primal_solution_other = [0],
    )
    CSV.write(path, tracking)
end

@testset "BenchmarkHelpers" begin
    @testset "parse_args" begin
        @test parse_args(String[]) == Dict{String,String}()
        @test parse_args(["--out", "/tmp"]) == Dict("out" => "/tmp")
        @test parse_args(["--samples", "1:10", "--out", "/tmp"]) ==
              Dict("samples" => "1:10", "out" => "/tmp")
        @test parse_args(["--verbose"]) == Dict("verbose" => "true")
        @test parse_args(["--verbose", "--out", "/tmp"]) ==
              Dict("verbose" => "true", "out" => "/tmp")
        @test parse_args(["positional", "--key", "val"]) == Dict("key" => "val")
    end

    @testset "parse_sample_spec" begin
        @test parse_sample_spec("1:5") == [1, 2, 3, 4, 5]
        @test parse_sample_spec("1:2:10") == [1, 3, 5, 7, 9]
        @test parse_sample_spec("3,7,11") == [3, 7, 11]
        @test parse_sample_spec("42") == [42]
        @test parse_sample_spec("1:1") == [1]
        @test_throws ErrorException parse_sample_spec("1:2:3:4")
    end

    @testset "safe_sum" begin
        @test safe_sum([1.0, 2.0, 3.0]) == 6.0
        @test safe_sum([1.0, Inf, 3.0]) == 4.0
        @test safe_sum([Inf, -Inf, NaN]) == 0.0
        @test safe_sum([1.0]) == 1.0
    end

    @testset "is_infeasible_status" begin
        @test is_infeasible_status("INFEASIBLE") == true
        @test is_infeasible_status("INFEASIBLE_OR_UNBOUNDED") == true
        @test is_infeasible_status("OPTIMAL") == false
        @test is_infeasible_status("TIME_LIMIT") == false
    end

    @testset "classify_semantic_outcome" begin
        @test classify_semantic_outcome("INFEASIBLE", missing) == "certified_no_adversarial_example"
        @test classify_semantic_outcome("INFEASIBLE_OR_UNBOUNDED", missing) ==
              "certified_no_adversarial_example"
        @test classify_semantic_outcome("OPTIMAL", 0.5) == "adversarial_example_found_or_best_known"
        @test classify_semantic_outcome("INFEASIBLE", 0.5) == "certified_no_adversarial_example"
        @test classify_semantic_outcome("TIME_LIMIT", missing) == "time_limit_unresolved"
        @test classify_semantic_outcome("OTHER", missing) == "no_primal_solution_other"
    end

    @testset "maybe_parse_norm_order" begin
        @test maybe_parse_norm_order("Inf") == Inf
        @test maybe_parse_norm_order("inf") == Inf
        @test maybe_parse_norm_order(" Inf ") == Inf
        @test maybe_parse_norm_order("1") == 1.0
        @test maybe_parse_norm_order("2.0") == 2.0
    end

    @testset "regression_ratio" begin
        @test regression_ratio(100.0, 105.0) == 0.05
        @test regression_ratio(100.0, 100.0) == 0.0
        @test regression_ratio(100.0, 95.0) == -0.05
        @test regression_ratio(0.0, 0.0) == 0.0
        @test regression_ratio(0.0, 1.0) == Inf
    end

    @testset "percent" begin
        @test percent(0.05) == "5.0%"
        @test percent(0.0) == "0.0%"
        @test percent(-0.1) == "-10.0%"
        @test percent(Inf) == "Inf%"
    end

    @testset "dependency snapshots" begin
        snapshot = collect_dependency_snapshot()
        @test [:name, :uuid, :version, :tree_hash, :source_kind, :is_direct_dep, :git_revision] ==
              collect(propertynames(snapshot))
        @test any(.!ismissing.(snapshot.tree_hash))
        @test isfile(active_manifest_path())

        fixture = dependency_snapshot(
            dependency_row(
                "CSV",
                "11111111-1111-1111-1111-111111111111";
                version = "0.10.16",
                tree_hash = "csvhash",
                is_direct_dep = true,
            ),
            dependency_row(
                "Dates",
                "22222222-2222-2222-2222-222222222222";
                version = "1.11.0",
                source_kind = "stdlib",
                is_direct_dep = true,
            ),
        )

        mktempdir() do dir
            snapshot_path = joinpath(dir, "dependency_versions.csv")
            write_dependency_snapshot(snapshot_path, fixture)
            loaded = load_dependency_snapshot(snapshot_path)
            @test isequal(loaded.tree_hash, ["csvhash", missing])
        end
    end

    @testset "dependency snapshot hash" begin
        ordered_snapshot = dependency_snapshot(
            dependency_row(
                "CSV",
                "11111111-1111-1111-1111-111111111111";
                version = "0.10.16",
                tree_hash = "csvhash",
                is_direct_dep = true,
            ),
            dependency_row(
                "DataFrames",
                "22222222-2222-2222-2222-222222222222";
                version = "1.8.1",
                tree_hash = "dfhash",
            ),
        )
        reordered_snapshot = dependency_snapshot(
            dependency_row(
                "DataFrames",
                "22222222-2222-2222-2222-222222222222";
                version = "1.8.1",
                tree_hash = "dfhash",
            ),
            dependency_row(
                "CSV",
                "11111111-1111-1111-1111-111111111111";
                version = "0.10.16",
                tree_hash = "csvhash",
                is_direct_dep = true,
            ),
        )
        ordered_snapshot.source_path = ["/tmp/one", "/tmp/two"]
        reordered_snapshot.source_path = ["/var/tmp/one", "/var/tmp/two"]

        hash_one = dependency_snapshot_hash(ordered_snapshot)
        hash_two = dependency_snapshot_hash(reordered_snapshot)
        @test hash_one == hash_two

        with_path_dep = dependency_snapshot(
            dependency_row(
                "CSV",
                "11111111-1111-1111-1111-111111111111";
                version = "0.10.16",
                tree_hash = "csvhash",
                is_direct_dep = true,
            ),
            dependency_row(
                "MIPVerify",
                "33333333-3333-3333-3333-333333333333";
                version = "0.5.4",
                source_kind = "path",
                is_direct_dep = true,
            ),
        )
        with_changed_path_dep = dependency_snapshot(
            dependency_row(
                "CSV",
                "11111111-1111-1111-1111-111111111111";
                version = "0.10.16",
                tree_hash = "csvhash",
                is_direct_dep = true,
            ),
            dependency_row(
                "MIPVerify",
                "33333333-3333-3333-3333-333333333333";
                version = "9.9.9",
                tree_hash = "changed-path-package",
                source_kind = "path",
                is_direct_dep = true,
            ),
        )

        @test dependency_snapshot_hash(with_path_dep) ==
              dependency_snapshot_hash(with_changed_path_dep)
    end

    @testset "dependency change summary" begin
        previous = dependency_snapshot(
            dependency_row(
                "CSV",
                "11111111-1111-1111-1111-111111111111";
                version = "0.10.15",
                tree_hash = "oldcsvhash",
                is_direct_dep = true,
            ),
            dependency_row(
                "OldPkg",
                "22222222-2222-2222-2222-222222222222";
                version = "2.0.0",
                tree_hash = "oldpkghash",
            ),
            dependency_row(
                "TreeOnly",
                "33333333-3333-3333-3333-333333333333";
                version = "1.0.0",
                tree_hash = "beforetreehash",
            ),
            dependency_row(
                "MIPVerify",
                "44444444-4444-4444-4444-444444444444";
                version = "0.5.4",
                source_kind = "path",
                is_direct_dep = true,
            ),
        )
        current = dependency_snapshot(
            dependency_row(
                "CSV",
                "11111111-1111-1111-1111-111111111111";
                version = "0.10.16",
                tree_hash = "newcsvhash",
                is_direct_dep = true,
            ),
            dependency_row(
                "NewPkg",
                "55555555-5555-5555-5555-555555555555";
                version = "3.0.0",
                tree_hash = "newpkghash",
            ),
            dependency_row(
                "TreeOnly",
                "33333333-3333-3333-3333-333333333333";
                version = "1.0.0",
                tree_hash = "aftertreehash",
            ),
            dependency_row(
                "MIPVerify",
                "44444444-4444-4444-4444-444444444444";
                version = "0.5.5",
                tree_hash = "shouldbeignored",
                source_kind = "path",
                is_direct_dep = true,
            ),
        )

        summary = dependency_change_summary(previous, current)
        @test occursin("CSV 0.10.15#oldcsvhash[direct] -> 0.10.16#newcsvhash[direct]", summary)
        @test occursin("TreeOnly 1.0.0#beforetreehash -> 1.0.0#aftertreehash", summary)
        @test occursin("+NewPkg 3.0.0#newpkghash", summary)
        @test occursin("-OldPkg 2.0.0#oldpkghash", summary)
        @test !occursin("MIPVerify", summary)

        @test dependency_change_summary(current, current) == ""
    end

    @testset "append_tracking_csv!" begin
        @testset "widens legacy tracking csv" begin
            mktempdir() do dir
                tracking_csv = joinpath(dir, "tracking.csv")
                write_legacy_tracking_csv(tracking_csv)

                current_snapshot = dependency_snapshot(
                    dependency_row(
                        "CSV",
                        "11111111-1111-1111-1111-111111111111";
                        version = "0.10.16",
                        tree_hash = "csvhash",
                        is_direct_dep = true,
                    ),
                )
                dependency_csv = joinpath(dir, "dependency_versions.csv")
                write_dependency_snapshot(dependency_csv, current_snapshot)

                current_hash = dependency_snapshot_hash(current_snapshot)
                metrics_csv = joinpath(dir, "benchmark_metrics.csv")
                write_metrics_csv(
                    metrics_csv;
                    dependency_snapshot_sha256 = current_hash,
                    julia_version = "1.11.5",
                )

                append_tracking_csv!(
                    metrics_csv = metrics_csv,
                    tracking_csv = tracking_csv,
                    dependency_versions_csv = dependency_csv,
                    date = "2026-03-09",
                    commit_sha = "def5678",
                    run_id = "2026-03-09T06-00-00Z-def5678",
                )

                tracking = CSV.read(tracking_csv, DataFrame)
                @test nrow(tracking) == 2
                @test ismissing(tracking[1, :run_id])
                @test tracking[2, :run_id] == "2026-03-09T06-00-00Z-def5678"
                @test tracking[2, :julia_version] == "1.11.5"
                @test tracking[2, :dependency_snapshot_sha256] == current_hash
                @test ismissing(tracking[2, :dependency_change_summary])
            end
        end

        @testset "compares against previous same-day rerun" begin
            mktempdir() do dir
                tracking_csv = joinpath(dir, "tracking.csv")

                first_run_id = "2026-03-09T06-00-00Z-abc1234"
                first_snapshot = dependency_snapshot(
                    dependency_row(
                        "CSV",
                        "11111111-1111-1111-1111-111111111111";
                        version = "0.10.15",
                        tree_hash = "oldcsvhash",
                        is_direct_dep = true,
                    ),
                    dependency_row(
                        "HiGHS",
                        "22222222-2222-2222-2222-222222222222";
                        version = "1.21.1",
                        tree_hash = "highshash",
                        is_direct_dep = true,
                    ),
                )
                first_run_dir = joinpath(dir, "runs", "2026-03-09", first_run_id)
                mkpath(first_run_dir)
                first_dependency_csv = joinpath(first_run_dir, "dependency_versions.csv")
                write_dependency_snapshot(first_dependency_csv, first_snapshot)

                first_hash = dependency_snapshot_hash(first_snapshot)
                first_metrics_csv = joinpath(dir, "first_metrics.csv")
                write_metrics_csv(
                    first_metrics_csv;
                    dependency_snapshot_sha256 = first_hash,
                    julia_version = "1.11.5",
                )

                append_tracking_csv!(
                    metrics_csv = first_metrics_csv,
                    tracking_csv = tracking_csv,
                    dependency_versions_csv = first_dependency_csv,
                    date = "2026-03-09",
                    commit_sha = "abc1234",
                    run_id = first_run_id,
                )

                second_run_id = "2026-03-09T08-00-00Z-def5678"
                second_snapshot = dependency_snapshot(
                    dependency_row(
                        "CSV",
                        "11111111-1111-1111-1111-111111111111";
                        version = "0.10.16",
                        tree_hash = "newcsvhash",
                        is_direct_dep = true,
                    ),
                    dependency_row(
                        "HiGHS",
                        "22222222-2222-2222-2222-222222222222";
                        version = "1.21.1",
                        tree_hash = "highshash",
                        is_direct_dep = true,
                    ),
                )
                second_run_dir = joinpath(dir, "runs", "2026-03-09", second_run_id)
                mkpath(second_run_dir)
                second_dependency_csv = joinpath(second_run_dir, "dependency_versions.csv")
                write_dependency_snapshot(second_dependency_csv, second_snapshot)

                second_hash = dependency_snapshot_hash(second_snapshot)
                second_metrics_csv = joinpath(dir, "second_metrics.csv")
                write_metrics_csv(
                    second_metrics_csv;
                    dependency_snapshot_sha256 = second_hash,
                    julia_version = "1.11.5",
                )

                append_tracking_csv!(
                    metrics_csv = second_metrics_csv,
                    tracking_csv = tracking_csv,
                    dependency_versions_csv = second_dependency_csv,
                    date = "2026-03-09",
                    commit_sha = "def5678",
                    run_id = second_run_id,
                )

                tracking = CSV.read(tracking_csv, DataFrame)
                @test nrow(tracking) == 2
                @test ismissing(tracking[1, :dependency_change_summary])
                @test tracking[2, :dependency_change_summary] ==
                      "CSV 0.10.15#oldcsvhash[direct] -> 0.10.16#newcsvhash[direct]"
            end
        end

        @testset "reports empty diff when only Julia version changes" begin
            mktempdir() do dir
                tracking_csv = joinpath(dir, "tracking.csv")

                first_run_id = "2026-03-09T06-00-00Z-abc1234"
                snapshot = dependency_snapshot(
                    dependency_row(
                        "CSV",
                        "11111111-1111-1111-1111-111111111111";
                        version = "0.10.16",
                        tree_hash = "csvhash",
                        is_direct_dep = true,
                    ),
                )
                first_run_dir = joinpath(dir, "runs", "2026-03-09", first_run_id)
                mkpath(first_run_dir)
                first_dependency_csv = joinpath(first_run_dir, "dependency_versions.csv")
                write_dependency_snapshot(first_dependency_csv, snapshot)

                first_hash = dependency_snapshot_hash(snapshot)
                first_metrics_csv = joinpath(dir, "first_metrics.csv")
                write_metrics_csv(
                    first_metrics_csv;
                    dependency_snapshot_sha256 = first_hash,
                    julia_version = "1.11.5",
                )

                append_tracking_csv!(
                    metrics_csv = first_metrics_csv,
                    tracking_csv = tracking_csv,
                    dependency_versions_csv = first_dependency_csv,
                    date = "2026-03-09",
                    commit_sha = "abc1234",
                    run_id = first_run_id,
                )

                second_run_id = "2026-03-09T08-00-00Z-def5678"
                second_run_dir = joinpath(dir, "runs", "2026-03-09", second_run_id)
                mkpath(second_run_dir)
                second_dependency_csv = joinpath(second_run_dir, "dependency_versions.csv")
                write_dependency_snapshot(second_dependency_csv, snapshot)

                second_hash = dependency_snapshot_hash(snapshot)
                second_metrics_csv = joinpath(dir, "second_metrics.csv")
                write_metrics_csv(
                    second_metrics_csv;
                    dependency_snapshot_sha256 = second_hash,
                    julia_version = "1.11.6",
                )

                second_row = append_tracking_csv!(
                    metrics_csv = second_metrics_csv,
                    tracking_csv = tracking_csv,
                    dependency_versions_csv = second_dependency_csv,
                    date = "2026-03-09",
                    commit_sha = "def5678",
                    run_id = second_run_id,
                )

                tracking = CSV.read(tracking_csv, DataFrame)
                @test nrow(tracking) == 2
                @test tracking[1, :dependency_snapshot_sha256] == tracking[2, :dependency_snapshot_sha256]
                @test tracking[2, :julia_version] == "1.11.6"
                @test second_row[1, :dependency_change_summary] ==
                      BenchmarkHelpers.NO_DEPENDENCY_CHANGES
                @test tracking[2, :dependency_change_summary] ==
                      BenchmarkHelpers.NO_DEPENDENCY_CHANGES
            end
        end
    end
end
