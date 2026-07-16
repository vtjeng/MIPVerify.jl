using Test
using Statistics
using MIPVerify
@isdefined(TestHelpers) || include("../TestHelpers.jl")

struct DummyDatasetA <: MIPVerify.Dataset
    x::Int
end

TestHelpers.@timed_testset "import_datasets.jl" begin
    # Two small images span the pixel range and both labels needed by the display tests.
    display_train = MIPVerify.LabelledImageDataset(
        reshape(Float32[0.0, 0.25, 0.75, 1.0], (2, 2, 1, 1)),
        Int32[0, 1],
    )
    # One test image exercises singular sample and label counts in the nested display.
    display_test =
        MIPVerify.LabelledImageDataset(reshape(Float64[0.25, 0.75], (1, 2, 1, 1)), Int64[1])

    @testset "NamedTrainTestDataset" begin
        @testset "Base.show" begin
            io = IOBuffer()
            dataset = MIPVerify.NamedTrainTestDataset("synthetic", display_train, display_test)
            Base.show(io, dataset)
            @test String(take!(io)) == """
                synthetic:
                  `train`: {LabelledImageDataset}
                    `images`: 2 images of size (2, 1, 1), with pixels in [0.0, 1.0].
                    `labels`: 2 corresponding labels, with 2 unique labels in [0, 1].
                  `test`: {LabelledImageDataset}
                    `images`: 1 images of size (2, 1, 1), with pixels in [0.25, 0.75].
                    `labels`: 1 corresponding labels, with 1 unique labels in [1, 1]."""
        end

        @testset "constructor validates train/test struct type" begin
            train = MIPVerify.LabelledImageDataset(zeros(Float32, 1, 1, 1, 1), Int32[0])
            test = MIPVerify.LabelledImageDataset(zeros(Float64, 1, 1, 1, 1), Int64[0])
            @test MIPVerify.NamedTrainTestDataset("dummy", train, test) isa
                  MIPVerify.NamedTrainTestDataset
            @test_throws AssertionError MIPVerify.NamedTrainTestDataset(
                "invalid",
                train,
                DummyDatasetA(1),
            )
        end
    end

    @testset "LabelledImageDataset" begin
        @testset "Base.show" begin
            io = IOBuffer()
            Base.show(io, display_train)
            @test String(take!(io)) == """
                {LabelledImageDataset}
                    `images`: 2 images of size (2, 1, 1), with pixels in [0.0, 1.0].
                    `labels`: 2 corresponding labels, with 2 unique labels in [0, 1]."""
        end
    end

    @testset "read_datasets" begin
        @testset "MNIST" begin
            d = read_datasets("mnist")
            @test MIPVerify.num_samples(d.test) == 10000
            @test MIPVerify.num_samples(d.train) == 60000
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
            if get(ENV, "MIPVERIFY_RUN_LARGE_DATASET_TESTS", "false") == "true"
                d = read_datasets("cifar10")
                @test MIPVerify.num_samples(d.test) == 10000
                @test MIPVerify.num_samples(d.train) == 50000
                # These values identify the published CIFAR10 files and their NHWC layout.
                @test d.test.labels[1:10] == [3, 8, 8, 0, 6, 6, 1, 6, 3, 1]
                @test d.train.labels[1:10] == [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]
                @test d.test.images |> mean ≈ 0.4765849205984478
                @test d.train.images |> mean ≈ 0.4733630004850899
                @test size(d.test.images) == (10000, 32, 32, 3)
                @test size(d.train.images) == (50000, 32, 32, 3)
                @test d.test.images[1, 8, 9, 1] ≈ 0.6705882352941176
                @test d.test.images[1, :, :, :] |> mean ≈ 0.42504340277777786
            else
                @test_skip "set MIPVERIFY_RUN_LARGE_DATASET_TESTS=true to validate the full CIFAR10 asset"
            end
        end

        @testset "Unsupported dataset" begin
            @test_throws DomainError read_datasets("the angry hippo of the nile")
        end
    end
end
