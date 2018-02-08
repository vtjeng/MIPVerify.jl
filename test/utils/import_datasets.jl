using Base.Test
using MIPVerify: read_datasets

@testset "import_datasets.jl" begin
    @testset "NamedTrainTestDataset" begin
        io = IOBuffer()
        mnist = read_datasets("MNIST")
        Base.show(io, mnist)
        @test String(take!(io)) == "MNIST:\n  `train`: {ImageDataset}\n    `images`: 55000 images of size (28, 28, 1), with pixels in [0.0, 1.0].\n    `labels`: 55000 corresponding labels, with 10 unique labels in [0, 9].\n  `test`: {ImageDataset}\n    `images`: 10000 images of size (28, 28, 1), with pixels in [0.0, 1.0].\n    `labels`: 10000 corresponding labels, with 10 unique labels in [0, 9]."
    end

    @testset "ImageDataset" begin
        io = IOBuffer()
        mnist = read_datasets("MNIST")
        Base.show(io, mnist.train)
        @test String(take!(io)) == "{ImageDataset}\n    `images`: 55000 images of size (28, 28, 1), with pixels in [0.0, 1.0].\n    `labels`: 55000 corresponding labels, with 10 unique labels in [0, 9]."
    end

    @testset "MNIST" begin
        d = read_datasets("MNIST")
        # Some sanity checks to make sure that the data imported from MNIST looks like we
        # expect.
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