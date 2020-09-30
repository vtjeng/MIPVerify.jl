using Test
using MIPVerify: get_example_network_params, read_datasets, frac_correct
@isdefined(TestHelpers) || include("../TestHelpers.jl")

TestHelpers.@timed_testset "import_example_nets.jl" begin
    TestHelpers.@timed_testset "get_example_network_params" begin
        TestHelpers.@timed_testset "MNIST.n1" begin
            nn = get_example_network_params("MNIST.n1")
            mnist = read_datasets("mnist")
            @test frac_correct(nn, mnist.test, 10000) == 0.9695
        end
        TestHelpers.@timed_testset "MNIST.WK17a_linf0.1_authors" begin
            nn = get_example_network_params("MNIST.WK17a_linf0.1_authors")
            mnist = read_datasets("mnist")
            @test frac_correct(nn, mnist.test, 10000) == 0.9811
        end
        TestHelpers.@timed_testset "MNIST.RSL18a_linf0.1_authors" begin
            nn = get_example_network_params("MNIST.RSL18a_linf0.1_authors")
            mnist = read_datasets("mnist")
            @test frac_correct(nn, mnist.test, 10000) == 0.9582
        end
    end

    @testset "unrecognized example network" begin
        @test_throws DomainError get_example_network_params("the social network")
    end
end
