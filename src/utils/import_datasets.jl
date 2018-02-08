using MAT

abstract type Dataset end

struct ImageDataset{T<:Real, U<:Int} <: Dataset
    images::Array{T, 4}
    labels::Array{U, 1}

    function ImageDataset{T, U}(images::Array{T, 4}, labels::Array{U, 1})::ImageDataset where {T<:Real, U<:Integer}
        (num_image_samples, image_height, image_width, num_channels) = size(images)
        (num_label_samples, ) = size(labels)
        @assert num_image_samples==num_label_samples
        return new(images, labels)
    end
end

function ImageDataset(images::Array{T, 4}, labels::Array{U, 1})::ImageDataset where {T<:Real, U<:Integer}
    ImageDataset{T, U}(images, labels)
end

function Base.show(io::IO, dataset::ImageDataset)
    image_size = size(dataset.images[1, :, :, :])
    num_samples = size(dataset.labels)[1]
    min_pixel = minimum(dataset.images)
    max_pixel = maximum(dataset.images)
    min_label = minimum(dataset.labels)
    max_label = maximum(dataset.labels)
    num_unique_labels = length(unique(dataset.labels))
    print(io,
        "{ImageDataset}",
        "\n    `images`: $num_samples images of size $image_size, with pixels in [$min_pixel, $max_pixel].",
        "\n    `labels`: $num_samples corresponding labels, with $num_unique_labels unique labels in [$min_label, $max_label]."
    )
end

struct NamedTrainTestDataset{T<:Dataset} <: Dataset
    name::String
    train::T
    test::T
end

function Base.show(io::IO, dataset::NamedTrainTestDataset)
    print(io, 
        "$(dataset.name):",
        "\n  `train`: $(dataset.train |> Base.string)",
        "\n  `test`: $(dataset.test |> Base.string)"
    )
end

function read_datasets(name::String)::NamedTrainTestDataset
    if name == "MNIST"

        MNIST_dir = joinpath("datasets", "mnist")

        m_train = prep_data_file(MNIST_dir, "mnist_train.mat") |> matread
        train = ImageDataset(m_train["images"], m_train["labels"][:])

        m_test = prep_data_file(MNIST_dir, "mnist_test.mat") |> matread
        test = ImageDataset(m_test["images"], m_test["labels"][:])
        return NamedTrainTestDataset(name, train, test)
    else
        throw(DomainError("Dataset $name not supported."))
    end
end
