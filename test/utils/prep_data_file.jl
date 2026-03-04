using Test
using MIPVerify
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "prep_data_file.jl" begin
    @testset "resolve_dependencies_path" begin
        @test MIPVerify.resolve_dependencies_path(
            env = Dict("MIPVERIFY_DEPS_PATH" => joinpath("tmp", "custom_mipverify_deps")),
            package_root = joinpath("tmp", "pkgroot"),
        ) == abspath(joinpath("tmp", "custom_mipverify_deps"))

        @test MIPVerify.resolve_dependencies_path(
            env = Dict{String,String}(),
            package_root = joinpath("tmp", "pkgroot"),
        ) == joinpath("tmp", "pkgroot", "deps")

        @test MIPVerify.resolve_dependencies_path(
            env = Dict{String,String}(),
            package_root = nothing,
        ) == normpath(joinpath(dirname(pathof(MIPVerify)), "..", "deps"))
    end

    @testset "relative_path_to_url_path" begin
        @test MIPVerify.relative_path_to_url_path(joinpath("datasets", "mnist", "sample.mat")) ==
              "datasets/mnist/sample.mat"

        @test MIPVerify.relative_path_to_url_path("datasets\\mnist\\sample.mat") ==
              "datasets/mnist/sample.mat"

        @test MIPVerify.relative_path_to_url_path("datasets//mnist\\\\sample.mat") ==
              "datasets/mnist/sample.mat"
    end
end
