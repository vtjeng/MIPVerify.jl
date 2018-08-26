using Base.Test
using MIPVerify
using MIPVerify: LInfNormBoundedPerturbationFamily
isdefined(:TestHelpers) || include("../TestHelpers.jl")
using TestHelpers: get_main_solver, get_tightening_solver

@testset "integration" begin
    mnist = read_datasets("MNIST")
    nn_wk17a = get_example_network_params("MNIST.WK17a_linf0.1_authors")
    @testset "batch_find_untargeted_attack" begin 
        mktempdir() do dir
            MIPVerify.batch_find_untargeted_attack(
                nn_wk17a, 
                mnist.test, 
                [1], # robust sample
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

        mktempdir() do dir
            MIPVerify.batch_find_untargeted_attack(
                nn_wk17a, 
                mnist.test, 
                [9, 248], # non-robust and misclassified sample
                get_main_solver(), 
                solve_rerun_option=MIPVerify.never,
                pp=MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
                norm_order=Inf, 
                rebuild=true, 
                tightening_algorithm=interval_arithmetic,
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
                tightening_algorithm=interval_arithmetic, 
                tightening_solver=get_tightening_solver(),
                cache_model=false,
                solve_if_predicted_in_targeted=false,
                target_labels=[1, 8],
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
                tightening_algorithm=interval_arithmetic
            )
        end
    end   
end