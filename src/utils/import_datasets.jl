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

struct FullDataset{T<:Dataset}
    train::T
    test::T
end

function read_data_sets(name::String)
    if name == "MNIST_data"
        # TODO: Download files if they are not available yet.
        path = "deps/input_data/mnist"
        m_train = matread("$(path)/mnist_train.mat")
        train = ImageDataset(m_train["images"], m_train["labels"][:])
        m_test = matread("$(path)/mnist_test.mat")
        test = ImageDataset(m_test["images"], m_test["labels"][:])
        return FullDataset(train, test)
    end
end
