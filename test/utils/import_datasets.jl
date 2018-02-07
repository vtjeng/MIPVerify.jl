using MIPVerify: read_datasets
using Base.Test

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
end