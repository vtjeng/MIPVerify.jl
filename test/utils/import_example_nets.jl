using Test
using MIPVerify: get_example_network_params, read_datasets, frac_correct
using MIPVerify: canonical_example_network_name
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "import_example_nets.jl" begin
    TestHelpers.@timed_testset "get_example_network_params" begin
        mnist = read_datasets("mnist")
        # The first 1,000 samples cover every digit and catch incorrect weight orientation
        # without repeating each network over the full 10,000-image test set.
        num_accuracy_samples = 1_000
        TestHelpers.@timed_testset "MNIST.n1" begin
            nn = get_example_network_params("MNIST.n1")
            @test frac_correct(nn, mnist.test, num_accuracy_samples) == 0.970
        end
        TestHelpers.@timed_testset "MNIST.WK17a_linf0.1_authors" begin
            nn = get_example_network_params("MNIST.WK17a_linf0.1_authors")
            @test frac_correct(nn, mnist.test, num_accuracy_samples) == 0.980
        end
        TestHelpers.@timed_testset "MNIST.RSL18a_linf0.1_authors" begin
            nn = get_example_network_params("MNIST.RSL18a_linf0.1_authors")
            @test frac_correct(nn, mnist.test, num_accuracy_samples) == 0.953
        end
    end

    @testset "unrecognized example network" begin
        @test_throws DomainError get_example_network_params("the social network")
    end

    @testset "case-insensitive example network names" begin
        @test canonical_example_network_name("mnist.n1") == "MNIST.n1"
        @test canonical_example_network_name("mnist.wk17a_linf0.1_authors") ==
              "MNIST.WK17a_linf0.1_authors"
        @test canonical_example_network_name("mnist.rsl18a_linf0.1_authors") ==
              "MNIST.RSL18a_linf0.1_authors"
        @test canonical_example_network_name("  Mnist.N1  ") == "MNIST.n1"
        @test canonical_example_network_name("not-real") === nothing
        @test get_example_network_params("mnist.n1").UUID == "MNIST.n1"
    end
end
