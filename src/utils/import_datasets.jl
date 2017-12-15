using MAT

abstract type Dataset end
Base.show(io::IO, dataset::Dataset) = print(Base.string(dataset))

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

function Base.string(dataset::ImageDataset)
    image_size = size(dataset.images[1, :, :, :])
    num_samples = size(dataset.labels)[1]
    min_pixel = minimum(dataset.images)
    max_pixel = maximum(dataset.images)
    min_label = minimum(dataset.labels)
    max_label = maximum(dataset.labels)
    num_unique_labels = length(unique(dataset.labels))
    string(
        "{ImageDataset}",
        "\n    images: $num_samples images of size $image_size, with pixels in [$min_pixel, $max_pixel].",
        "\n    labels: $num_samples corresponding labels, with $num_unique_labels unique labels in [$min_label, $max_label]."
    )
end

struct NamedTrainTestDataset{T<:Dataset} <: Dataset
    name::String
    train::T
    test::T
end

function Base.string(dataset::NamedTrainTestDataset)
    string(
        "$(dataset.name):",
        "\n  train: $(dataset.train |> Base.string)",
        "\n  test: $(dataset.test |> Base.string)"
    )
end

const dependencies_path = joinpath(Pkg.dir("MIPVerify"), "deps")
const data_repo_path = "https://github.com/vtjeng/MIPVerify_data/raw/master"

function prep_data_file(relative_dir::String, filename::String)::String
    absolute_dir = joinpath(dependencies_path, relative_dir)
    if !ispath(absolute_dir)
        mkpath(absolute_dir)
    end
    
    relative_file_path = joinpath(relative_dir, filename)
    absolute_file_path = joinpath(dependencies_path, relative_file_path)
    if !isfile(absolute_file_path)
        url = joinpath(data_repo_path, relative_file_path)
        println(url)
        download(url, absolute_file_path)
    end

    return absolute_file_path
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
        throw(ArgumentError("Dataset $name not supported."))
    end
end
