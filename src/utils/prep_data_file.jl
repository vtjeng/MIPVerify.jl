using Downloads

const data_repo_path = "https://storage.googleapis.com/mipverify-data"

function relative_path_to_url_path(relative_path::String)::String
    normalized = replace(relative_path, "\\" => "/")
    return join(filter(part -> !isempty(part), Base.split(normalized, "/")), "/")
end

function prep_data_file(relative_dir::String, filename::String)::String
    absolute_dir = joinpath(dependencies_path, relative_dir)
    if !ispath(absolute_dir)
        mkpath(absolute_dir)
    end

    relative_file_path = joinpath(relative_dir, filename)
    absolute_file_path = joinpath(dependencies_path, relative_file_path)
    if !isfile(absolute_file_path)
        relative_url_path = relative_path_to_url_path(relative_file_path)
        url = string(data_repo_path, "/", relative_url_path)
        Downloads.download(url, absolute_file_path)
    end

    return absolute_file_path
end
