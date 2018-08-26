using Base.Test
using MIPVerify
using MIPVerify: BatchRunParameters, UnrestrictedPerturbationFamily, LInfNormBoundedPerturbationFamily, mkpath_if_not_present, create_summary_file_if_not_present, verify_target_indices
isdefined(:TestHelpers) || include("TestHelpers.jl")
using TestHelpers: get_main_solver, get_tightening_solver

@testset "batch_processing_helpers.jl" begin
    mnist = read_datasets("MNIST")
    nn_wk17a = get_example_network_params("MNIST.WK17a_linf0.1_authors")

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
        @test_throws AssertionError verify_target_indices([0], mnist.test) 
        @test_throws AssertionError verify_target_indices([10001], mnist.test) 
    end

    # Remaining tests are "integration tests" of complex functionality
    @testset "batch_find_certificate" begin 
        mktempdir() do dir
            MIPVerify.batch_find_certificate(
                nn_wk17a, 
                mnist.test, 
                [1, 9, 248], # samples selected to be robust, non-robust, and misclassified.
                get_main_solver(), 
                solve_rerun_option=MIPVerify.never,
                pp=MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
                norm_order=Inf, 
                rebuild=true, 
                tightening_algorithm=lp, 
                tightening_solver=get_tightening_solver(),
                cache_model=false,
                solve_if_predicted_in_targeted=false,
                save_path=dir
            )
        end
    end

    @testset "batch_find_targeted_attack" begin 
        mktempdir() do dir
            MIPVerify.batch_find_targeted_attack(
                nn_wk17a, 
                mnist.test, 
                [1], 
                get_main_solver(), 
                solve_rerun_option=MIPVerify.never,
                pp=MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
                norm_order=Inf,
                tightening_algorithm=lp, 
                tightening_solver=get_tightening_solver(),
                cache_model=false,
                solve_if_predicted_in_targeted=false,
                save_path=dir
            )
        end
    end

    @testset "batch_build_model" begin 
        mktempdir() do dir
            MIPVerify.batch_build_model(
                nn_wk17a, 
                mnist.test, 
                [1], 
                get_tightening_solver(),
                pp=MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
                tightening_algorithm=lp
            )
        end
    end   

end