using Base.Test
using MIPVerify: get_example_network_params, num_correct

@testset "import_weights.jl" begin
    @testset "Example network params" begin
        @testset "MNIST.n1" begin
            nnparams = get_example_network_params("MNIST.n1")
            @test num_correct(nnparams, "MNIST", 1000) == 970
        end
    end
end