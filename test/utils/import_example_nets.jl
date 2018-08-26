using Base.Test
using MIPVerify: get_example_network_params, read_datasets, frac_correct

@testset "import_example_nets.jl" begin
    @testset "get_example_network_params" begin
        @testset "MNIST.n1" begin
            nn = get_example_network_params("MNIST.n1")
            mnist = read_datasets("mnist")
            @test frac_correct(nn, mnist.test, 10000) == 0.9695
        end
        @testset "MNIST.WK17a_linf0.1_authors" begin
            nn = get_example_network_params("MNIST.WK17a_linf0.1_authors")
            mnist = read_datasets("mnist")
            @test frac_correct(nn, mnist.test, 10000) == 0.9811
        end
        @testset "MNIST.RSL18a_linf0.1_authors" begin
            nn = get_example_network_params("MNIST.RSL18a_linf0.1_authors")
            mnist = read_datasets("mnist")
            @test frac_correct(nn, mnist.test, 10000) == 0.9582
        end
    end

    @testset "unrecognized example network" begin
        @test_throws DomainError get_example_network_params("the social network")
    end
end