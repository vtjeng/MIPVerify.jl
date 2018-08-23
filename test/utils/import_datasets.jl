using Base.Test
using MIPVerify
using MIPVerify: num_samples

@testset "import_datasets.jl" begin
    @testset "NamedTrainTestDataset" begin
        @testset "Base.show" begin
            io = IOBuffer()
            mnist = read_datasets("mnist")
            Base.show(io, mnist)
            @test String(take!(io)) == """
                mnist:
                  `train`: {LabelledImageDataset}
                    `images`: 60000 images of size (28, 28, 1), with pixels in [0.0, 1.0].
                    `labels`: 60000 corresponding labels, with 10 unique labels in [0, 9].
                  `test`: {LabelledImageDataset}
                    `images`: 10000 images of size (28, 28, 1), with pixels in [0.0, 1.0].
                    `labels`: 10000 corresponding labels, with 10 unique labels in [0, 9]."""
        end
    end

    @testset "LabelledImageDataset" begin
        @testset "Base.show" begin
            io = IOBuffer()
            mnist = read_datasets("mnist")
            Base.show(io, mnist.train)
            @test String(take!(io)) == """
                {LabelledImageDataset}
                    `images`: 60000 images of size (28, 28, 1), with pixels in [0.0, 1.0].
                    `labels`: 60000 corresponding labels, with 10 unique labels in [0, 9]."""
        end
    end

    @testset "read_datasets" begin
        @testset "MNIST" begin
            d = read_datasets("mnist")
            @test num_samples(d.test) == 10000
            @test num_samples(d.train) == 60000
            # Some sanity checks to make sure that the data imported looks like we expect.
            @test d.test.labels[1:10] == [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
            @test d.train.labels[1:10] == [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]
            @test d.test.images |> mean ≈ 0.13251463f0
            @test d.train.images |> mean ≈ 0.1307003f0
            @test size(d.test.images) == (10000, 28, 28, 1)
            @test size(d.train.images) == (60000, 28, 28, 1)
            @test d.test.images[1, 8, 9, 1] ≈ 0.62352943f0
            @test d.test.images[1, :, :, :] |> mean ≈ 0.09230693f0        
        end

        @testset "CIFAR10" begin
            d = read_datasets("cifar10")
            @test num_samples(d.test) == 10000
            @test num_samples(d.train) == 50000
            # Some sanity checks to make sure that the data imported looks like we expect.
            @test d.test.labels[1:10] == [3, 8, 8, 0, 6, 6, 1, 6, 3, 1]
            @test d.train.labels[1:10] == [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]
            @test d.test.images |> mean ≈ 0.4765849205984478
            @test d.train.images |> mean ≈ 0.4733630004850899
            @test size(d.test.images) == (10000, 32, 32, 3)
            @test size(d.train.images) == (50000, 32, 32, 3)
            @test d.test.images[1, 8, 9, 1] ≈ 0.6705882352941176
            @test d.test.images[1, :, :, :] |> mean ≈ 0.42504340277777786        
        end

        @testset "Unsupported dataset" begin
            @test_throws DomainError read_datasets("the angry hippo of the nile")
        end
    end
end