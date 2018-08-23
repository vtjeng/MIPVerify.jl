using Base.Test
using MIPVerify
using MIPVerify: BatchRunParameters, UnrestrictedPerturbationFamily, mkpath_if_not_present, create_summary_file_if_not_present, verify_target_sample_numbers

@testset "batch_processing_helpers.jl" begin
    base_test_directory = joinpath(Base.tempdir(), "julia", "MIPVerify", "test")
    ispath(base_test_directory) ? rm(base_test_directory; recursive=true) : nothing

    function find_empty_folder_path()
        index = 1
        while joinpath(base_test_directory, "$index") |> ispath
            index +=1 
        end
        return joinpath(base_test_directory, "$index")
    end

    @testset "BatchRunParameters" begin
        brp = BatchRunParameters(
            Sequential([], "name"),
            UnrestrictedPerturbationFamily(),
            1,
            0
        )
        @testset "Base.show" begin
            io = IOBuffer()
            Base.show(io, brp)
            @test String(take!(io)) == "name__unrestricted__1__0"
        end
    end
    @testset "mkpath_if_not_present" begin
        path = find_empty_folder_path()
        mkpath_if_not_present(path)
        @test ispath(path)
        mkpath_if_not_present(path)
        @test ispath(path)
    end
    @testset "create_summary_file_if_not_present" begin
        folder_path = find_empty_folder_path()
        mkpath_if_not_present(folder_path)
        file_path = joinpath(folder_path, "summary.csv")
        create_summary_file_if_not_present(file_path)
        @test isfile(file_path)
    end
    @testset "verify_target_sample_numbers" begin
        dataset = read_datasets("mnist").test
        @test_throws AssertionError verify_target_sample_numbers([0], dataset) 
        @test_throws AssertionError verify_target_sample_numbers([10001], dataset) 
    end
end