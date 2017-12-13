using Base.Test
using MIPVerify: read_datasets

@testset "import_datasets.jl" begin
    @testset "MNIST" begin
        d = read_datasets("MNIST_data")
        # Some sanity checks to make sure that the data looks liek we expect.
        @test d.train.labels[1:10] == [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
        @test d.test.labels[1:10] == [7, 3, 4, 6, 1, 8, 1, 0, 9, 8]
        @test d.train.images |> mean == 0.13251463f0
        @test d.test.images |> mean == 0.1307003f0
    end
end