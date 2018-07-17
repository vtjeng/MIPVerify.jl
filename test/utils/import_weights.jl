using Base.Test
using MIPVerify: get_example_network_params, read_datasets, frac_correct

@testset "import_weights.jl" begin
    @testset "Example network params" begin
        @testset "MNIST.n1" begin
            nn = get_example_network_params("MNIST.n1")
            mnist = read_datasets("mnist")
            @test frac_correct(nn, mnist.test, 1000) == 0.970
        end
    end

    @testset "Unrecognized example network" begin
        @test_throws DomainError get_example_network_params("the social network")
    end
end