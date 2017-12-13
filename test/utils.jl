using Base.Test
using MIPVerify: read_datasets
using MIPVerify: get_example_network_params, num_correct

@testset "utils" begin
@testset "import_datasets.jl" begin
    @testset "MNIST" begin
        d = read_datasets("MNIST")
        # Some sanity checks to make sure that the data looks liek we expect.
        @test d.test.labels[1:10] == [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
        @test d.train.labels[1:10] == [7, 3, 4, 6, 1, 8, 1, 0, 9, 8]
        @test d.test.images |> mean ≈ 0.13251463f0
        @test d.train.images |> mean ≈ 0.1307003f0
        @test size(d.test.images) == (10000, 28, 28, 1)
        @test size(d.train.images) == (55000, 28, 28, 1)
        @test d.test.images[1, 8, 9, 1] ≈ 0.62352943f0
        @test d.test.images[1, :, :, :] |> mean ≈ 0.09230693f0        
    end
end

@testset "import_weights.jl" begin
    @testset "Example network params" begin
        @testset "MNIST.n1" begin
            nnparams = get_example_network_params("MNIST.n1")
            @test num_correct(nnparams, "MNIST", 1000) == 970
        end
    end
end
end