using Test
using MIPVerify
using MIPVerify: LInfNormBoundedPerturbationFamily
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "integration.jl" begin
    mnist = read_datasets("MNIST")
    nn_wk17a = get_example_network_params("MNIST.WK17a_linf0.1_authors")
    @testset "batch_find_untargeted_attack" begin
        mktempdir() do dir
            MIPVerify.batch_find_untargeted_attack(
                nn_wk17a,
                mnist.test,
                [1], # robust sample
                TestHelpers.get_optimizer(),
                TestHelpers.get_main_solve_options(),
                solve_rerun_option = MIPVerify.never,
                pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
                norm_order = Inf,
                tightening_algorithm = lp,
                tightening_options = TestHelpers.get_tightening_options(),
                solve_if_predicted_in_targeted = false,
                save_path = dir,
            )
        end

        mktempdir() do dir
            MIPVerify.batch_find_untargeted_attack(
                nn_wk17a,
                mnist.test,
                [9, 248], # non-robust and misclassified sample
                TestHelpers.get_optimizer(),
                TestHelpers.get_main_solve_options(),
                solve_rerun_option = MIPVerify.never,
                pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
                norm_order = Inf,
                tightening_algorithm = interval_arithmetic,
                tightening_options = TestHelpers.get_tightening_options(),
                solve_if_predicted_in_targeted = false,
                save_path = dir,
            )
        end
    end

    @testset "batch_find_targeted_attack" begin
        mktempdir() do dir
            MIPVerify.batch_find_targeted_attack(
                nn_wk17a,
                mnist.test,
                [1],
                TestHelpers.get_optimizer(),
                TestHelpers.get_main_solve_options(),
                solve_rerun_option = MIPVerify.never,
                pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1),
                norm_order = Inf,
                tightening_algorithm = interval_arithmetic,
                tightening_options = TestHelpers.get_tightening_options(),
                solve_if_predicted_in_targeted = false,
                target_labels = [1, 8],
                save_path = dir,
            )
        end
    end
end
