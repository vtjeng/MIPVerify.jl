using MAT

export read_datasets

abstract type Dataset end

abstract type LabelledDataset<:Dataset end

"""
$(TYPEDEF)

Dataset of images stored as a 4-dimensional array of size `(num_samples, image_height, 
image_width, num_channels)`, with accompanying labels (sorted in the same order) of size
`num_samples`.
"""
struct LabelledImageDataset{T<:Real, U<:Integer} <: LabelledDataset
    images::Array{T, 4}
    labels::Array{U, 1}

    function LabelledImageDataset{T, U}(images::Array{T, 4}, labels::Array{U, 1})::LabelledImageDataset where {T<:Real, U<:Integer}
        (num_image_samples, image_height, image_width, num_channels) = size(images)
        (num_label_samples, ) = size(labels)
        @assert num_image_samples==num_label_samples
        return new(images, labels)
    end
end

function LabelledImageDataset(images::Array{T, 4}, labels::Array{U, 1})::LabelledImageDataset where {T<:Real, U<:Integer}
    LabelledImageDataset{T, U}(images, labels)
end

function num_samples(dataset::LabelledDataset)
    return length(dataset.labels)
end

function Base.show(io::IO, dataset::LabelledImageDataset)
    image_size = size(dataset.images[1, :, :, :])
    num_samples = MIPVerify.num_samples(dataset)
    min_pixel = minimum(dataset.images)
    max_pixel = maximum(dataset.images)
    min_label = minimum(dataset.labels)
    max_label = maximum(dataset.labels)
    num_unique_labels = length(unique(dataset.labels))
    print(io,
        "{LabelledImageDataset}",
        "\n    `images`: $num_samples images of size $image_size, with pixels in [$min_pixel, $max_pixel].",
        "\n    `labels`: $num_samples corresponding labels, with $num_unique_labels unique labels in [$min_label, $max_label]."
    )
end

"""
$(TYPEDEF)

Named dataset containing a training set and a test set which are expected to contain the
same kind of data.
"""
struct NamedTrainTestDataset{T<:Dataset, U<:Dataset} <: Dataset
    name::String
    train::T
    test::U
    # TODO (vtjeng): train and test should be the same type of struct (but might potentially have different parameters).
end

function Base.show(io::IO, dataset::NamedTrainTestDataset)
    print(io, 
        "$(dataset.name):",
        "\n  `train`: $(dataset.train)",
        "\n  `test`: $(dataset.test)"
    )
end

"""
$(SIGNATURES)

Makes popular machine learning datasets available as a `NamedTrainTestDataset`.

# Arguments
* `name::String`: name of machine learning dataset. Options:
    * `MNIST`: [The MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
"""
function read_datasets(name::String)::NamedTrainTestDataset
    name = lowercase(name)

    if name in ["mnist", "cifar10"]
        # TODO (vtjeng): specify in documentation that we normalize the images in these datasets to range from 0 to 1
        MNIST_dir = joinpath("datasets", name)

        m_train = prep_data_file(MNIST_dir, "$(name)_int_train.mat") |> matread
        train = LabelledImageDataset(m_train["images"]/255, m_train["labels"][:])

        m_test = prep_data_file(MNIST_dir, "$(name)_int_test.mat") |> matread
        test = LabelledImageDataset(m_test["images"]/255, m_test["labels"][:])
        return NamedTrainTestDataset(name, train, test)
    else
        throw(DomainError("Dataset $name not supported."))
    end
end
