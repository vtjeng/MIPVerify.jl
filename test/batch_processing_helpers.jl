using Base.Test
using MIPVerify
using MIPVerify: BatchRunParameters, UnrestrictedPerturbationFamily, LInfNormBoundedPerturbationFamily, mkpath_if_not_present, create_summary_file_if_not_present, verify_target_indices
isdefined(:TestHelpers) || include("TestHelpers.jl")
using TestHelpers: get_main_solver, get_tightening_solver

@testset "batch_processing_helpers.jl" begin
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
        mktempdir() do dir
            path = joinpath(dir, "1")
            mkpath_if_not_present(path)
            @test ispath(path)
            mkpath_if_not_present(path)
            @test ispath(path)
        end
    end

    @testset "create_summary_file_if_not_present" begin
        mktempdir() do dir
            file_path = joinpath(dir, "summary.csv")
            create_summary_file_if_not_present(file_path)
            @test isfile(file_path)
        end
    end
    
    @testset "verify_target_indices" begin
        dataset = read_datasets("mnist").test
        @test_throws AssertionError verify_target_indices([0], dataset) 
        @test_throws AssertionError verify_target_indices([10001], dataset) 
    end

    # Remaining tests are "integration tests" of complex functionality
    @testset "batch_find_certificate" begin 
        mnist = read_datasets("MNIST")
        mktempdir() do dir
            MIPVerify.batch_find_certificate(
                get_example_network_params("MNIST.WK17a_linf0.1_authors"), 
                mnist.test, 
                1:3, 
                get_main_solver(), 
                norm_order=Inf, 
                tightening_algorithm=lp, 
                rebuild=true, 
                pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
                solve_rerun_option = MIPVerify.never,
                tightening_solver = get_tightening_solver(),
                cache_model=false,
                save_path=dir
            )
        end
    end
end