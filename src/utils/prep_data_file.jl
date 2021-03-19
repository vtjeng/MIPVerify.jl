const data_repo_path = "https://github.com/vtjeng/MIPVerify_data/raw/master"

function prep_data_file(relative_dir::String, filename::String)::String
    absolute_dir = joinpath(dependencies_path, relative_dir)
    if !ispath(absolute_dir)
        mkpath(absolute_dir)
    end

    relative_file_path = joinpath(relative_dir, filename)
    absolute_file_path = joinpath(dependencies_path, relative_file_path)
    if !isfile(absolute_file_path)
        if Sys.iswindows()
            # On windows, the `joinpath` command uses `\` as a separator.
            # TODO: This is a bit of a hack; we might prefer rethinking `relative_dir`
            relative_url_path = replace(relative_file_path, "\\" => "/")
        else
            relative_url_path = relative_file_path
        end
        url = string(data_repo_path, "/", relative_url_path)
        download(url, absolute_file_path)
    end

    return absolute_file_path
end
